import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    action: str
    sufficient: bool
    answer: str
    reasoning: str
    evidence_analysis: list[str]
    missing_info: list[str]
    raw: str


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [str(value).strip()] if str(value).strip() else []


def _normalize_action(value: Any, sufficient: bool, answer: str) -> str:
    if isinstance(value, str) and value.strip().lower() == "fallback":
        return "fallback"

    if not sufficient:
        return "fallback"
    if answer.strip().lower() == "not mentioned":
        return "fallback"

    if isinstance(value, str):
        action = value.strip().lower()
        if action == "answer":
            return action

    return "answer"


def parse_llm_response(raw: str) -> LLMResponse:
    text = raw.strip()

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    def _try_parse(s: str):
        try:
            obj = json.loads(s)
            sufficient = obj.get("sufficient", True)
            if isinstance(sufficient, str):
                sufficient = sufficient.lower() in ("true", "1", "yes")
            answer = str(obj.get("answer", "")).strip()
            action = _normalize_action(obj.get("action"), sufficient, answer)
            reasoning = str(obj.get("reasoning", "")).strip()
            evidence_analysis = _coerce_string_list(obj.get("evidence_analysis"))
            missing_info = _coerce_string_list(obj.get("missing_info"))
            if not reasoning and evidence_analysis:
                reasoning = evidence_analysis[-1]
            return LLMResponse(
                action=action,
                sufficient=sufficient,
                answer=answer,
                reasoning=reasoning,
                evidence_analysis=evidence_analysis,
                missing_info=missing_info,
                raw=raw,
            )
        except (json.JSONDecodeError, ValueError):
            return None

    result = _try_parse(text)
    if result:
        return result

    match = re.search(r'\{[^{}]*"sufficient"\s*:', text)
    if match:
        brace_start = match.start()
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    result = _try_parse(text[brace_start:i + 1])
                    if result:
                        return result
                    break

    answ = raw.strip()
    if answ:
        return LLMResponse(
            action="answer",
            sufficient=True,
            answer=answ,
            reasoning="",
            evidence_analysis=[],
            missing_info=[],
            raw=raw,
        )
    return LLMResponse(
        action="fallback",
        sufficient=False,
        answer="",
        reasoning="parse failed",
        evidence_analysis=[],
        missing_info=[],
        raw=raw,
    )
