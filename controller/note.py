"""P3 factual error note — code-generated, no LLM calls.

Contains ONLY: failed tool name, the tool's returned error text, and failed
argument NAMES. Never expected state, hidden answers, or benchmark metadata.
Exact template frozen in Phase 4 (hash recorded in config).
"""
from __future__ import annotations

from controller.failures import VisibleError

TEMPLATE = (
    "[SYSTEM NOTE — previous attempt failed]\n"
    "A previous attempt at this task made a tool call that returned an error.\n"
    "Failed tool: {tool}\n"
    "Error returned: {error_text}\n"
    "Arguments involved: {arg_names}\n"
    "You are starting fresh. Complete the original task below."
)


def build_note(err: VisibleError) -> str:
    if err is None or not err.is_visible:
        raise ValueError("note requires a visible error")
    return TEMPLATE.format(
        tool=err.tool or "unknown",
        error_text=(err.error_text or "unspecified error").replace("\n", " "),
        arg_names=", ".join(err.arg_names) if err.arg_names else "(none identified)",
    )
