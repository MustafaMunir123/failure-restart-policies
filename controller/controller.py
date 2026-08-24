"""Retry controller — the experimental switch.

on_failure(attempt) -> {reset?, inject_note?, continue?}

P0 stop           : do nothing, end episode
P1 blind restart  : reset env + fresh conversation + original task, new derived seed
P2 same-session   : keep everything, append actual tool error, agent continues
P3 restart+note   : reset env + fresh conversation + factual note prepended

Budgets are enforced in budget.py; this module decides ACTIONS only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from controller.failures import VisibleError


@dataclass(frozen=True)
class Action:
    reset_env: bool = False
    inject_note: str | None = None   # P3 note text (code-generated) or P2 error text
    continue_session: bool = False   # P2 native resume flag (False => replay upstream)
    second_seed: int | None = None


def derive_second_seed(base_seed: int, policy: str) -> int:
    """Locked rule: seed_2 = hash(base_seed, policy_id). Deterministic, pre-registered."""
    h = hashlib.sha256(f"{base_seed}:{policy}".encode()).hexdigest()
    return int(h[:12], 16)


def dispatch(policy: str, base_seed: int, err: VisibleError,
             note_builder=None) -> Action:
    """Full dispatcher with seed derivation. note_builder injected for testability."""
    if not err.is_visible:
        raise ValueError("controller may act only on visible errors")
    seed2 = derive_second_seed(base_seed, policy)
    if policy == "P0_stop":
        return Action()
    if policy == "P1_blind_restart":
        return Action(reset_env=True, second_seed=seed2)
    if policy == "P2_same_session":
        return Action(continue_session=True, inject_note=err.error_text)
    if policy == "P3_restart_note":
        builder = note_builder or _default_note_builder
        return Action(reset_env=True, inject_note=builder(err), second_seed=seed2)
    raise ValueError(f"unknown policy: {policy}")


def _default_note_builder(err: VisibleError) -> str:
    from controller.note import build_note
    return build_note(err)


# Attempt-2 eligibility (locked edge cases):
# - attempt 1 succeeded            -> no attempt 2, regardless of earlier errors
# - silent failure                 -> no attempt 2 for ANY policy
# - budget_exhausted mid-attempt   -> treated as task_timeout (visible); attempt 2 only
#   if token pool allows (enforced by budget.py)
# - harness_error                  -> not a failure; run excluded and rerun
def should_attempt_two(attempt1_success: bool, visible_error: VisibleError | None,
                       tokens_remaining: int) -> bool:
    if attempt1_success or visible_error is None:
        return False
    return tokens_remaining > 0
