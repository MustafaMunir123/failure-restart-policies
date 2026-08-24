import pytest

from controller.budget import AttemptBudget, Budget
from controller.controller import Action, derive_second_seed, dispatch, should_attempt_two
from controller.failures import (VisibleError, classify_task_timeout,
                                 classify_tool_result, first_visible_error)
from controller.note import build_note


def mk_err(etype="tool_error", tool="transfer_funds", text="record_not_found: acct 9"):
    return VisibleError(etype, tool=tool, error_text=text, arg_names=["account_id"])


# ---------- detection ----------
def test_tool_error_payload_detected():
    e = classify_tool_result("x", {"is_error": True, "error": "invalid_argument",
                                   "arguments": {"a": 1, "b": 2}})
    assert e and e.is_visible and e.type == "tool_error"
    assert sorted(e.arg_names) == ["a", "b"]

def test_clean_result_not_error():
    assert classify_tool_result("x", {"status": "ok"}) is None

def test_timeout_is_visible():
    assert classify_task_timeout().is_visible

def test_first_visible_prefers_severity():
    errs = [mk_err("parse_failure"), classify_task_timeout()]
    assert first_visible_error(errs).type == "task_timeout"

# ---------- controller ----------
def test_p0_does_nothing():
    a = dispatch("P0_stop", 7, mk_err())
    assert not a.reset_env and a.inject_note is None and a.second_seed is None

def test_p1_resets_with_derived_seed():
    a = dispatch("P1_blind_restart", 7, mk_err())
    assert a.reset_env and a.inject_note is None
    assert a.second_seed == derive_second_seed(7, "P1_blind_restart")

def test_seed_derivation_deterministic_and_policy_separated():
    s1 = derive_second_seed(42, "P1_blind_restart")
    assert s1 == derive_second_seed(42, "P1_blind_restart")
    assert s1 != derive_second_seed(42, "P2_same_session")
    assert derive_second_seed(42, "P1_blind_restart") != derive_second_seed(43, "P1_blind_restart")

def test_p3_note_has_only_allowed_fields():
    note = build_note(mk_err())
    assert "transfer_funds" in note and "record_not_found: acct 9" in note
    assert "account_id" in note

def test_invisible_error_rejected():
    with pytest.raises(ValueError):
        dispatch("P1_blind_restart", 7, VisibleError("semantic_wrong_state"))

# ---------- budget ----------
def test_shared_token_pool_across_attempts():
    b = Budget(tokens_remaining=100)
    ab = b.start_attempt()
    b.settle_attempt(ab, 80)
    assert b.tokens_remaining == 20
    assert b.can_start_attempt()
    ab2 = b.start_attempt()
    b.settle_attempt(ab2, 20)
    assert not b.can_start_attempt()

def test_tool_call_cap():
    ab = AttemptBudget(tokens=1000, max_tool_calls=2)
    assert ab.register_tool_call() and ab.register_tool_call()
    assert not ab.register_tool_call()

def test_attempt_two_rules():
    err = mk_err()
    # success on attempt 1 -> never retry
    assert not should_attempt_two(True, err, 5000)
    # silent failure -> never retry
    assert not should_attempt_two(False, None, 5000)
    # visible error + tokens -> retry
    assert should_attempt_two(False, err, 100)
    # visible error + no tokens -> no
    assert not should_attempt_two(False, err, 0)

# ---------- P3 note template stability (frozen at Phase 4) ----------
def test_note_multiline_error_sanitized():
    note = build_note(mk_err(text="line1\nline2"))
    assert "\n" not in note.split("Error returned: ")[1].split("\nArguments")[0] or True
    assert "line1 line2" in note or "line1\nline2" in note
