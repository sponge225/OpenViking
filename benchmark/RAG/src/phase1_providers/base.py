import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from core.response_parser import LLMResponse, parse_llm_response


REFUSAL_MARKERS = (
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
)


def default_supplemental(reason: str = "No supplemental retrieval") -> dict:
    return {
        "enabled": False,
        "triggered": False,
        "reason": reason,
        "query": "",
        "uris": [],
        "needs_more": False,
        "audit_needs_more": False,
    }


@dataclass
class Phase1ProviderResult:
    name: str
    answer: str
    parsed: LLMResponse
    should_fallback: bool
    reasoning: str
    search_res: dict
    prompt: str = ""
    raw: str = ""
    meta: dict = field(default_factory=dict)
    supplemental: dict = field(default_factory=default_supplemental)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_sec: float = 0.0
    details: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "provider": self.name,
            "answer": self.answer,
            "should_fallback": bool(self.should_fallback),
            "reasoning": self.reasoning,
            "action": self.parsed.action,
            "sufficient": bool(self.parsed.sufficient),
            "evidence_analysis": list(self.parsed.evidence_analysis or []),
            "missing_info": list(self.parsed.missing_info or []),
            "raw": self.raw,
            "parse_raw": self.parsed.raw,
            "input_tokens": int(self.input_tokens or 0),
            "output_tokens": int(self.output_tokens or 0),
            "latency_sec": float(self.latency_sec or 0.0),
            "supplemental_retrieval": dict(self.supplemental or {}),
            "details": dict(self.details or {}),
        }


class Phase1Provider:
    name = "base"
    requires_primary = False

    def __init__(self, config: dict, adapter, db, llm, logger=None):
        self.config = config
        self.adapter = adapter
        self.db = db
        self.llm = llm
        self.logger = logger

    def run(self, qa, search_res: dict, **kwargs) -> Phase1ProviderResult:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return self.db.count_tokens(text) if self.db else 0

    def context_blocks(self, search_res: dict) -> list[str]:
        context_blocks = [
            str(block).strip()
            for block in (search_res.get("context_blocks", []) or [])
            if str(block).strip()
        ]
        if context_blocks:
            return context_blocks

        recall_texts = search_res.get("recall_texts", {}) or {}
        retrieved_uris = list(search_res.get("retrieved_uris", []) or [])
        if not retrieved_uris:
            retrieved_uris = list(recall_texts.keys())
        return [
            str(recall_texts.get(uri, "") or "").strip()
            for uri in retrieved_uris
            if str(recall_texts.get(uri, "") or "").strip()
        ]

    def config_bool(self, value, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def provider_config(self) -> dict:
        execution = self.config.get("execution", {}) or {}
        phase_cfg = execution.get("phase1_providers", self.config.get("phase1_providers", {}))
        if not isinstance(phase_cfg, dict):
            return {}

        configs = phase_cfg.get("configs", {})
        if isinstance(configs, dict) and isinstance(configs.get(self.name), dict):
            return configs[self.name]

        direct = phase_cfg.get(self.name)
        return direct if isinstance(direct, dict) else {}

    def is_refusal_like(self, answer: str) -> bool:
        answer_lower = (answer or "").strip().lower()
        return not answer_lower or any(marker in answer_lower for marker in REFUSAL_MARKERS)

    def extract_json_object(self, raw: str) -> tuple[dict | None, str]:
        text = (raw or "").strip()
        if not text:
            return None, "empty output"

        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        obj, error = self._load_json_object(text)
        if obj is not None:
            return obj, error

        start = text.find("{")
        if start < 0:
            return None, "No JSON object found"

        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:idx + 1]
                    obj, error = self._load_json_object(candidate)
                    if obj is not None:
                        return obj, error
                    return None, error
        return None, "Unclosed JSON object"

    def _load_json_object(self, text: str) -> tuple[dict | None, str]:
        try:
            obj = json.loads(text)
            return (obj, "") if isinstance(obj, dict) else (None, "JSON root is not an object")
        except (json.JSONDecodeError, TypeError) as exc:
            first_error = f"{type(exc).__name__}: {exc}"

        repaired = self._escape_invalid_json_backslashes(text)
        if repaired != text:
            try:
                obj = json.loads(repaired, strict=False)
                if isinstance(obj, dict):
                    return obj, f"repaired invalid JSON escapes after {first_error}"
                return None, "JSON root is not an object"
            except (json.JSONDecodeError, TypeError) as exc:
                return None, f"{type(exc).__name__}: {exc}"
        return None, first_error

    def _escape_invalid_json_backslashes(self, text: str) -> str:
        result = []
        i = 0
        length = len(text)
        hex_digits = set("0123456789abcdefABCDEF")
        simple_escapes = set('"\\/bfnrt')
        while i < length:
            char = text[i]
            if char != "\\":
                result.append(char)
                i += 1
                continue

            if i + 1 >= length:
                result.append("\\\\")
                i += 1
                continue

            nxt = text[i + 1]
            if nxt in simple_escapes:
                result.append(char)
                result.append(nxt)
                i += 2
                continue

            if nxt == "u" and i + 5 < length and all(c in hex_digits for c in text[i + 2:i + 6]):
                result.append(char)
                result.append(nxt)
                i += 2
                continue

            result.append("\\\\")
            i += 1
        return "".join(result)

    def error_result(self, qa, search_res: dict, reason: str) -> Phase1ProviderResult:
        parsed = parse_llm_response("")
        return Phase1ProviderResult(
            name=self.name,
            answer="",
            parsed=parsed,
            should_fallback=False,
            reasoning=reason,
            search_res=search_res,
            details={"error": reason},
        )


def timed() -> float:
    return time.time()
