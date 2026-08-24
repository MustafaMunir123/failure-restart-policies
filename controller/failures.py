"""Failure detection — observable signals only.

The classifier NEVER imports or consults grader/check modules. Enforced by
tests/test_no_grader_imports.py, which greps this package's source.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VisibleError:
    type: str  # "tool_error" | "parse_failure" | "task_timeout"
    tool: str | None = None
    error_text: str | None = None
    arg_names: list[str] = field(default_factory=list)

    @property
    def is_visible(self) -> bool:
        return self.type in ("tool_error", "parse_failure", "task_timeout")


def classify_tool_result(tool_name: str | None, result: dict) -> VisibleError | None:
    """Check one tool result for an observable error payload."""
    if not isinstance(result, dict):
        return None
    if result.get("is_error") or result.get("status") in ("error", "timeout"):
        err = result.get("error") or result.get("message") or "unknown tool error"
        args = result.get("invalid_args") or sorted((result.get("arguments") or {}).keys())
        etype = "task_timeout" if result.get("status") == "timeout" else "tool_error"
        return VisibleError(etype, tool=tool_name, error_text=str(err)[:2000],
                            arg_names=[str(a) for a in args])
    return None


def classify_parse_failure(tool_name: str | None, raw: str) -> VisibleError:
    """Malformed / unparseable tool call."""
    return VisibleError("parse_failure", tool=tool_name,
                        error_text=f"unparseable tool call: {str(raw)[:500]}")


def classify_task_timeout() -> VisibleError:
    """Task-level timeout. LOCKED DECISION: visible; triggers policy under P1-P3."""
    return VisibleError("task_timeout")


def first_visible_error(events: list[VisibleError]) -> VisibleError | None:
    """Attempt-level signal = first/most severe visible error (locked edge case).

    Severity order: task_timeout > tool_error > parse_failure.
    """
    if not events:
        return None
    for wanted in ("task_timeout", "tool_error", "parse_failure"):
        for e in events:
            if e.type == wanted:
                return e
    return None
