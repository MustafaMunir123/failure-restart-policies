"""Contract tests for ThinkingboxAdapter — synthetic traces, no framework needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.thinkingbox_adapter import ThinkingboxAdapter
from controller.budget import AttemptBudget


def mk_adapter(tmp_path, p2="replay"):
    base = tmp_path / "base.yaml"
    base.write_text("mcp_proxy: {endpoint_url: 'http://x'}\n")
    cfg = {"dataset": "/d", "base_config": str(base),
           "thinkingbox_dir": str(tmp_path), "workdir": str(tmp_path / "w"),
           "p2_mechanism": p2}
    a = ThinkingboxAdapter.__new__(ThinkingboxAdapter)
    ThinkingboxAdapter.__init__(a, cfg)
    return a


def fake_run(monkeypatch, adapter, trace_text, rc=0):
    out = adapter.workdir / "out.yaml"
    out.write_text(trace_text)
    calls = {}

    def fake(cmd, **kw):
        calls["cmd"] = cmd
        class P:
            returncode = rc
            stderr = ""
        return P()
    monkeypatch.setattr("subprocess.run", fake)
    monkeypatch.setattr("adapters.thinkingbox_adapter.ThinkingboxAdapter._load_trace",
                        lambda self, p: ThinkingboxAdapter._load_trace.__func__(
                            None, out) if False else __import__("yaml").safe_load(out.read_text()) or {})
    return calls


TRACE_ERR = """
conversation:
  - role: user
    content: cancel my order
  - role: assistant
    content: ""
    tool_calls: [{name: cancel_order, arguments: {order_id: "123"}}]
  - role: tool
    name: cancel_order
    metadata: {error: "record_not_found: order 123"}
    content: '{"is_error": true}'
usage: {input_tokens: 100, output_tokens: 40}
"""

TRACE_OK = """
conversation:
  - role: user
    content: hi
  - role: assistant
    content: done
usage: {input_tokens: 50, output_tokens: 20}
passed: true
"""


def _writer(text, rc=0):
    """Fake subprocess.run that writes `text` to the --output path from cmd."""
    def fake(cmd, **kw):
        if rc == 0:
            Path(cmd[cmd.index("--output") + 1]).write_text(text)
        class P:
            returncode = rc
            stderr = "boom" if rc else ""
        return P()
    return fake


def test_error_trace_yields_visible_error(tmp_path, monkeypatch):
    a = mk_adapter(tmp_path)
    a.start_attempt("t:x", 1, True, None, None)
    monkeypatch.setattr("subprocess.run", _writer(TRACE_ERR))
    ab = AttemptBudget(tokens=8000)
    r = a.step(ab)
    assert r["visible_error"]["type"] == "tool_error"
    assert r["visible_error"]["tool"] == "cancel_order"
    assert r["success"] is False and r["tokens_out"] == 40


def test_clean_trace_no_error(tmp_path, monkeypatch):
    a = mk_adapter(tmp_path)
    a.start_attempt("t", 1, True, None, None)
    monkeypatch.setattr("subprocess.run", _writer(TRACE_OK))
    r = a.step(AttemptBudget(tokens=8000))
    assert r["visible_error"] is None and r["tokens_in"] == 50


def test_harness_error_on_nonzero_rc(tmp_path, monkeypatch):
    a = mk_adapter(tmp_path)
    a.start_attempt("t", 1, True, None, None)
    monkeypatch.setattr("subprocess.run", _writer("", rc=2))
    r = a.step(AttemptBudget(tokens=8000))
    assert r["status"] == "harness_error"


def test_p3_note_lands_in_config(tmp_path):
    a = mk_adapter(tmp_path)
    a.start_attempt("t", 1, True, "[SYSTEM NOTE] failed tool x", None)
    cfg_path = a._config_for_attempt()
    text = Path(cfg_path).read_text()
    assert "[SYSTEM NOTE]" in text
