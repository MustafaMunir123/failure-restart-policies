import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.orchestrator import POLICIES, execute_run, load_done_run_ids


class FakeAdapter:
    """Scripted adapter: attempt 1 always fails visibly; attempt 2 succeeds."""

    def __init__(self):
        self.calls = []

    def start_attempt(self, task_id, seed, reset_env, prepend_note, resume_trace):
        self.calls.append({"seed": seed, "reset": reset_env, "note": prepend_note,
                           "resume": resume_trace is not None})

    def step(self, ab):
        ab.register_tool_call()
        return {"success": len(self.calls) > 1,
                "status": "done",
                "visible_error": None if len(self.calls) > 1 else {
                    "type": "tool_error", "tool": "t", "error_text": "e", "arg_names": []},
                "n_turns": 3, "tokens_in": 10, "tokens_out": 5,
                "gpu_seconds": 1.0, "wall_seconds": 2.0, "trace_path": None}


def test_grid_size():
    assert len(POLICIES) == 4


def test_execute_run_p0_single_attempt(tmp_path):
    rec = execute_run("t1", "retail", "Qwen3-4B", "P0_stop", 42, FakeAdapter())
    assert rec["attempt_2"] is None and rec["attempt_1_hit_visible_error"]
    assert not rec["grade"]["passed"]


def test_execute_run_p3_two_attempts_with_note_and_reset(tmp_path):
    env = FakeAdapter()
    rec = execute_run("t1", "retail", "Qwen3-4B", "P3_restart_note", 42, env)
    assert rec["grade"]["passed"] and rec["grade"]["graded_attempt"] == 2
    assert env.calls[1]["reset"] is True
    assert "SYSTEM NOTE" in (env.calls[1]["note"] or "")
    assert rec["totals"]["tokens_agent_out"] == 10


def test_execute_run_p2_continues_without_reset():
    env = FakeAdapter()
    execute_run("t1", "retail", "Qwen3-4B", "P2_same_session", 42, env)
    assert env.calls[1]["resume"] is True and env.calls[1]["reset"] is False


def test_idempotent_resume(tmp_path):
    out = tmp_path / "runs.jsonl"
    out.write_text(json.dumps(
        execute_run("t1", "retail", "M", "P0_stop", 1, FakeAdapter())) + "\n")
    assert load_done_run_ids(str(out)) == {"M:P0_stop:1:t1"}
