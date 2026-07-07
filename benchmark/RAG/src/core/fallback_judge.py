from dataclasses import dataclass
from typing import List


REFUSAL_PATTERNS = [
    "not mentioned",
    "insufficient information",
    "i don't know",
    "i do not know",
    "cannot be determined",
    "not enough information",
    "no relevant information",
    "unable to find",
    "cannot answer",
    "no information available",
]


@dataclass
class JudgeVerdict:
    should_fallback: bool
    reasoning: str


def _check_refusal_patterns(answer: str) -> JudgeVerdict:
    answer_lower = answer.strip().lower()
    for pattern in REFUSAL_PATTERNS:
        if pattern in answer_lower:
            return JudgeVerdict(
                should_fallback=True,
                reasoning=f"Refusal pattern detected: '{pattern}'",
            )
    return JudgeVerdict(should_fallback=False, reasoning="Answer looks valid")


def judge_answer(
    sufficient: bool,
    answer: str,
    reasoning: str = "",
    action: str = "answer",
    audit_needs_more: bool = False,
    audit_reason: str = "",
) -> JudgeVerdict:
    if action == "fallback":
        return JudgeVerdict(
            should_fallback=True,
            reasoning=f"Phase 1 routed to fallback: {reasoning}" if reasoning else "Phase 1 routed to fallback",
        )
    if audit_needs_more:
        return JudgeVerdict(
            should_fallback=True,
            reasoning=f"Phase 1 evidence audit routed to fallback: {audit_reason}"
            if audit_reason
            else "Phase 1 evidence audit routed to fallback",
        )
    if not sufficient:
        return JudgeVerdict(
            should_fallback=True,
            reasoning=f"LLM assessed context as insufficient: {reasoning}" if reasoning else "LLM assessed context as insufficient",
        )
    if not answer.strip():
        return JudgeVerdict(should_fallback=True, reasoning="Empty answer")
    return _check_refusal_patterns(answer)
