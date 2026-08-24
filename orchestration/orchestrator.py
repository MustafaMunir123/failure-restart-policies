"""Pilot orchestrator — walks the frozen grid, idempotent by run_id.

Grid: 50 tasks x 4 policies x 3 models x 3 seeds (attempt-2 runs are part of a run).
Writes append-only JSONL of RunRecords; resumes by skipping existing run_ids.
Framework interaction goes through an EnvAdapter (implemented on the host in
adapters/thinkingbox_adapter.py) — this module never imports grader code.

Usage:
    python -m orchestration.orchestrator --config configs/frozen_pilot_config.json \
        --out runs/pilot_runs.jsonl --models Qwen3-1.7B [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

from controller.budget import TOKEN_POOL, AttemptBudget, Budget
from controller.controller import dispatch, should_attempt_two
from controller.failures import VisibleError, first_visible_error

POLICIES = ["P0_stop", "P1_blind_restart", "P2_same_session", "P3_restart_note"]
MODELS = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]


class EnvAdapter:
    """Interface implemented per-framework on the host. No-op here."""

    def start_attempt(self, task_id: str, seed: int, reset_env: bool,
                      prepend_note: str | None, resume_trace: dict | None):
        raise NotImplementedError

    def step(self, attempt_budget: AttemptBudget):
        """Runs agent/simulator/tool loop until terminal or budget-blocked."""
        raise NotImplementedError

    def snapshot_totals(self):
        raise NotImplementedError


def load_done_run_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["run_id"])
    return done


def execute_run(task_id: str, domain: str, model: str, policy: str,
                base_seed: int, env: EnvAdapter, config_hash: str = "") -> dict:
    run_id = f"{model}:{policy}:{base_seed}:{task_id}"
    totals = {"tokens_agent_in": 0, "tokens_agent_out": 0, "tokens_simulator_in": 0,
              "tokens_simulator_out": 0, "gpu_seconds": 0.0, "wall_seconds": 0.0}
    t0 = time.time()

    # ---- attempt 1 ----
    budget = Budget(tokens_remaining=TOKEN_POOL)
    ab1 = budget.start_attempt()
    env.start_attempt(task_id, base_seed, reset_env=True, prepend_note=None, resume_trace=None)
    result1 = env.step(ab1)
    budget.settle_attempt(ab1, ab1.tokens_used)

    raw_err = result1.get("visible_error")
    err: VisibleError | None = (
        VisibleError(**raw_err) if isinstance(raw_err, dict) else raw_err)
    success1 = bool(result1.get("success"))

    record = {
        "run_id": run_id, "task_id": task_id, "domain": domain, "model": model,
        "policy": policy, "seed": base_seed,
        "attempt_1_hit_visible_error": err is not None,
        "visible_error": _err_to_dict(err),
        "attempt_1": _attempt_dict(result1, ab1),
        "attempt_2": None,
        "totals": totals,
        "grade": {"passed": success1, "graded_attempt": 1},
        "flags": {"budget_exhausted": False, "harness_error": result1.get("status") == "harness_error"},
        "config_hash": config_hash,
    }

    # ---- controller decision + attempt 2 ----
    if should_attempt_two(success1, err, budget.tokens_remaining):
        action = dispatch(policy, base_seed, err)
        if policy == "P0_stop":
            pass  # control group: no second attempt even though error was visible
        else:
            if not action.continue_session:
                pass  # P1/P3: fresh env; P2: keep everything
            ab2 = budget.start_attempt()
            env.start_attempt(task_id, action.second_seed or base_seed,
                              reset_env=action.reset_env,
                              prepend_note=action.inject_note,
                              resume_trace=result1.get("trace") if action.continue_session else None)
            result2 = env.step(ab2)
            budget.settle_attempt(ab2, ab2.tokens_used)
            exhausted = ab2.exhausted
            raw_err2 = result2.get("visible_error")
            err2 = VisibleError(**raw_err2) if isinstance(raw_err2, dict) else raw_err2
            record["attempt_2"] = _attempt_dict(result2, ab2)
            record["grade"] = {"passed": bool(result2.get("success")),
                               "failed_checks": result2.get("failed_checks"),
                               "rubric_results": result2.get("rubric_results"),
                               "graded_attempt": 2}
            record["flags"]["budget_exhausted"] = exhausted

    for k in totals:
        for a in ("attempt_1", "attempt_2"):
            att = record.get(a)
            if att:
                mapping = {"tokens_agent_in": "tokens_in",
                           "tokens_agent_out": "tokens_out"}
                totals[k] += att.get(mapping.get(k, k), 0) or 0
    totals["wall_seconds"] = time.time() - t0
    return record


def _err_to_dict(err: VisibleError | None):
    return None if err is None else {
        "type": err.type, "tool": err.tool,
        "error_text": err.error_text, "arg_names": err.arg_names}


def _attempt_dict(result: dict, ab: AttemptBudget) -> dict | None:
    status = result.get("status")
    if status is None:
        return None
    return {
        "status": ("success" if result.get("success") else
                   "budget_exhausted" if getattr(ab, "exhausted", False) else
                   "failed_visible_error" if result.get("visible_error") else
                   "failed_silent" if status != "harness_error" else "harness_error"),
        "n_turns": result.get("n_turns", 0),
        "tokens_in": result.get("tokens_in", 0),
        "tokens_out": result.get("tokens_out", 0),
        "tool_calls": ab.tool_calls_used,
        "gpu_seconds": result.get("gpu_seconds", 0.0),
        "wall_seconds": result.get("wall_seconds", 0.0),
        "trace_path": result.get("trace_path"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--policies", nargs="*", default=POLICIES)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    tasks = cfg["tasks"]  # [{task_id, domain}]
    seeds = cfg["seeds"]

    done = load_done_run_ids(args.out)
    todo = [(m, p, s, t) for m in args.models for p in args.policies
            for s in seeds for t in tasks]
    remaining = sum(
        1 for m, p, s, t in todo
        if f"{m}:{p}:{s}:{t['task_id'] if isinstance(t, dict) else t}" not in done)
    print(f"grid={len(todo)} done={len(done)} remaining={remaining}")

    if args.dry_run:
        for m, p, s, t in todo:
            tid = t["task_id"] if isinstance(t, dict) else t
            rid = f"{m}:{p}:{s}:{tid}"
            if rid not in done:
                print(rid)
        return

    from adapters.thinkingbox_adapter import ThinkingboxAdapter  # host-only
    env = ThinkingboxAdapter(cfg)
    with open(args.out, "a") as out:
        for m, p, s, t in todo:
            tid = t["task_id"] if isinstance(t, dict) else t
            domain = t["domain"] if isinstance(t, dict) else t.rsplit("_", 1)[0]
            rid = f"{m}:{p}:{s}:{tid}"
            if rid in done:
                continue
            rec = execute_run(tid, domain, m, p, s, env, cfg.get("config_hash", ""))
            out.write(json.dumps(rec) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
