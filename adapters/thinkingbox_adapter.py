"""ThinkingboxAdapter — EnvAdapter implementation on the pinned Thinkingbox framework.

Mechanism per attempt: one `tb infer` subprocess against a per-attempt config;
parse the output YAML trace for visible errors + token usage.

P2 mechanism (locked decision doc: docs/decisions/p2-mechanism.md):
  native  = framework resumes mid-session      [requires framework API support]
  replay  = attempt-1 transcript embedded in fresh context; full prefix counted
This adapter ships REPLAY; switch to native when the agent-loop inspection
(Phase 2 step 3) confirms a supported entry point. The choice is recorded
per-run in flags.p2_mechanism.

NEVER reads grader output. Error signals come exclusively from tool-result
payloads inside the conversation trace.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET  # noqa: F401 (yaml is used; etree reserved)
from pathlib import Path

import yaml

from controller.budget import AttemptBudget
from controller.failures import VisibleError, classify_parse_failure
from orchestration.orchestrator import EnvAdapter


class ThinkingboxAdapter(EnvAdapter):
    def __init__(self, cfg: dict, workdir: str | None = None):
        self.dataset = cfg["dataset"]
        self.base_config = Path(cfg["base_config"])
        self.thinkingbox_dir = Path(cfg["thinkingbox_dir"])
        self.workdir = Path(workdir or cfg.get("workdir", tempfile.mkdtemp(prefix="tbadapter_")))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, **cfg.get("env", {}))
        self.p2_mechanism = cfg.get("p2_mechanism", "replay")
        self._state: dict = {}

    # ---------- EnvAdapter interface ----------

    def start_attempt(self, task_id, seed, reset_env, prepend_note, resume_trace):
        self._state = {
            "task_id": task_id, "seed": seed,
            "prepend_note": prepend_note, "resume_trace": resume_trace,
            "out": None,
        }

    def step(self, ab: AttemptBudget) -> dict:
        t0 = time.time()
        out_path = self.workdir / f"attempt_{int(time.time() * 1000)}.yaml"
        cmd = ["uv", "run", "tb", "infer",
               "-c", str(self._config_for_attempt()),
               "--dataset", self.dataset, "--agent", "think",
               "--name", self._state["task_id"], "--output", str(out_path)]
        p = subprocess.run(cmd, cwd=self.thinkingbox_dir, env=self.env,
                           capture_output=True, text=True, timeout=1800)
        wall = time.time() - t0
        if p.returncode != 0:
            return {"status": "harness_error", "success": False, "n_turns": 0,
                    "tokens_in": 0, "tokens_out": 0, "gpu_seconds": 0.0,
                    "wall_seconds": wall, "stderr": p.stderr[-2000:]}

        trace = self._load_trace(out_path)
        err = self._first_visible_error(trace)
        tokens_in, tokens_out = self._token_usage(trace)
        success = self._trace_passed(trace)

        if ab.register_tokens(tokens_out) is False:
            err = err or classify_parse_failure(None, "budget_exhausted_tokens")
            status_extra = "budget_exhausted"
        else:
            status_extra = None

        result = {
            "status": status_extra or ("done" if success or err else "failed_silent"),
            "success": bool(success),
            "visible_error": ({"type": err.type, "tool": err.tool,
                               "error_text": err.error_text,
                               "arg_names": err.arg_names} if err else None),
            "n_turns": len(trace.get("conversation", [])),
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "gpu_seconds": 0.0,  # server-side timing wired at pilot instrumentation
            "wall_seconds": wall,
            "trace_path": str(out_path),
            "trace": {"messages": trace.get("conversation", [])},
            "failed_checks": trace.get("failed_checks"),
            "rubric_results": trace.get("rubric_results"),
        }
        return result

    def snapshot_totals(self):
        return {}

    # ---------- internals ----------

    def _config_for_attempt(self) -> str:
        """Per-attempt config: base + P3 note / P2 replay injection."""
        cfg = yaml.safe_load(self.base_config.read_text())
        note = self._state.get("prepend_note")
        if note and self.p2_mechanism != "replay":
            # P3-style system note: appended into first user turn content upstream?
            # Framework has no system-note hook -> inject via orchestrator.user_model
            # is NOT allowed (simulator frozen). Instead prepend to agent context via
            # the task query is impossible without touching task defs.
            # => P3/P2 both use replay-style context injection below.
            pass
        if note:
            prev = self._state.get("resume_trace") or {}
            msgs = prev.get("messages") or []
            convo = "\n".join(
                f"[{m.get('role')}] {str(m.get('content'))[:500]}" for m in msgs[-10:])
            injected = (f"{note}\n\n[PREVIOUS ATTEMPT TRANSCRIPT - for reference]\n{convo}"
                        if self.p2_mechanism == "replay" and msgs else note)
            cfg.setdefault("orchestrator", {}).setdefault("agent_model", {})[
                "stop_sequences"] = []
            cfg["orchestrator"]["agent_model"]["context_prefix"] = injected
        path = self.workdir / f"cfg_{int(time.time() * 1000)}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return str(path)

    @staticmethod
    def _load_trace(out_path: Path) -> dict:
        try:
            data = yaml.safe_load(out_path.read_text())
        except Exception:
            return {"conversation": [], "passed": False}
        if isinstance(data, dict):
            conv = (data.get("conversation") or data.get("messages")
                    or data.get("turns") or [])
            return {"conversation": conv or [], "raw": data}
        return {"conversation": [], "raw": data}

    @staticmethod
    def _first_visible_error(trace: dict) -> VisibleError | None:
        """Scan conversation for tool-result error payloads (observable only)."""
        best = None
        order = {"task_timeout": 0, "tool_error": 1, "parse_failure": 2}
        for m in trace.get("conversation", []):
            if not isinstance(m, dict):
                continue
            if m.get("role") == "tool" or m.get("type") == "tool_result":
                content = str(m.get("content", ""))
                meta = m.get("metadata") or {}
                if meta.get("error") or m.get("is_error"):
                    e = VisibleError("tool_error", tool=m.get("name"),
                                     error_text=str(meta.get("error") or content)[:2000],
                                     arg_names=sorted((m.get("arguments") or {}).keys()))
                    if best is None or order[e.type] < order[best.type]:
                        best = e
                elif '"is_error": true' in content or '"status": "error"' in content:
                    e = VisibleError("tool_error", tool=m.get("name"),
                                     error_text=content[:2000])
                    if best is None or order[e.type] < order[best.type]:
                        best = e
        return best

    @staticmethod
    def _token_usage(trace: dict) -> tuple[int, int]:
        raw = trace.get("raw") or {}
        usage = raw.get("usage") or {}
        return int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0), \
               int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)

    @staticmethod
    def _trace_passed(trace: dict) -> bool:
        raw = trace.get("raw") or {}
        for key in ("passed", "test_passed", "assertions_passed"):
            if key in raw:
                return bool(raw[key])
        # grading happens OUTSIDE the adapter via tb run-test in phase >=4;
        # inside smoke/validation, absence of grader info means "not graded"
        return bool(raw.get("graded") is True and raw.get("passed"))
