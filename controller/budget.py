"""Budget enforcement — identical for all policies (locked).

<=2 attempts, <=8,192 NEW agent tokens shared across both attempts,
<=32 tool calls per attempt. Timeouts enforced by the runtime adapter.
"""
from __future__ import annotations

from dataclasses import dataclass

TOKEN_POOL = 8_192
MAX_TOOL_CALLS_PER_ATTEMPT = 32
MAX_ATTEMPTS = 2


@dataclass
class Budget:
    tokens_remaining: int = TOKEN_POOL

    @property
    def attempts_remaining(self) -> int:
        return MAX_ATTEMPTS - self.attempts_used

    @property
    def attempts_used(self) -> int:
        return self._attempts_used

    def __post_init__(self):
        self._attempts_used = 0

    def charge(self, tokens: int) -> None:
        """Charge new agent tokens (input+output generated during the episode).
        Prefill of replayed context counts when P2 uses prefix replay."""
        if tokens < 0:
            raise ValueError("token charge must be non-negative")
        self.tokens_remaining = max(0, self.tokens_remaining - tokens)

    def can_start_attempt(self) -> bool:
        return self.attempts_remaining > 0 and self.tokens_remaining > 0

    def start_attempt(self, tool_call_cap: int = MAX_TOOL_CALLS_PER_ATTEMPT) -> "AttemptBudget":
        if not self.can_start_attempt():
            raise RuntimeError("budget exhausted: no attempts or tokens remaining")
        self._attempts_used += 1
        return AttemptBudget(tokens=self.tokens_remaining, max_tool_calls=tool_call_cap)

    def settle_attempt(self, ab: "AttemptBudget", used_tokens: int) -> bool:
        """Charge actual usage; returns True if the pool is now exhausted."""
        self.charge(used_tokens)
        return self.tokens_remaining == 0


@dataclass
class AttemptBudget:
    tokens: int
    max_tool_calls: int = MAX_TOOL_CALLS_PER_ATTEMPT
    tool_calls_used: int = 0
    tokens_used: int = 0

    def register_tool_call(self) -> bool:
        """Returns False when the cap blocks the call (agent must stop)."""
        if self.tool_calls_used >= self.max_tool_calls:
            return False
        self.tool_calls_used += 1
        return True

    def register_tokens(self, n: int) -> bool:
        """Returns False when the token cap blocks further generation."""
        self.tokens_used += n
        if self.tokens_used > self.tokens:
            return False
        return True

    @property
    def exhausted(self) -> bool:
        return self.tool_calls_used >= self.max_tool_calls or self.tokens_used >= self.tokens
