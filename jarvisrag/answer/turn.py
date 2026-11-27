"""One question's state as it moves through the answer pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    """Everything the pipeline knows about the question being answered.

    Built from the user's words and the previous turn's record, then filled in
    step by step: the evidence, the drafted answer, and the review verdicts.
    ``record`` produces the dict handed back to the caller and to the next turn.
    """

    user_query: str
    prior: dict[str, object]
    effective_query: str = ""
    is_follow_up: bool = False
    source_hints: list = field(default_factory=list)
    negative_feedback: bool = False

    # Evidence, by tier, and how it may be cited.
    domain_routed: bool = False
    routed_domain_names: list = field(default_factory=list)
    domain_evidence: list[str] = field(default_factory=list)
    domain_refs: list[str] = field(default_factory=list)
    domain_superseded: set[str] = field(default_factory=set)
    mcp_evidence: list[str] = field(default_factory=list)
    evidence_lines: list[str] = field(default_factory=list)
    home_evidence_refs: list[str] | None = None
    excluded_for_display: list[str] | None = None

    # The answer and the verdicts on it.
    response_text: str = ""
    review_passed: bool = False
    review_reason: str = ""
    answer_was_dismissive: bool = False
    answer_completeness: str = ""
    evidence_off_topic: bool = False
    # Set only when this turn ENDS by asking the user for a missing detail. The
    # next turn reads it to (a) link the reply back to this question no matter how
    # tersely it is phrased, and (b) refuse to ask a second time.
    awaiting_clarification: bool = False

    @property
    def prior_awaited_clarification(self) -> bool:
        """The previous turn ended by asking the user for a missing detail."""
        return bool(self.prior.get("awaiting_clarification"))

    @property
    def answers_clarification(self) -> bool:
        """This turn is the user's reply to that clarifying question."""
        return self.is_follow_up and self.prior_awaited_clarification

    def record(self, **fields: object) -> dict[str, object]:
        return {
            "user_query": self.user_query,
            "effective_query": self.effective_query,
            "response_text": self.response_text,
            **fields,
        }
