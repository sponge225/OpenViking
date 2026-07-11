import re
from typing import Any

from core.response_parser import LLMResponse

from .base import Phase1Provider, Phase1ProviderResult, default_supplemental, timed


EVIDENCE_SUFFICIENCY_PROMPT = """Act as a strict evidence auditor and answerer.

Before answering, break the question into its required entities, constraints, comparison items, and reasoning hops. Select direct context evidence for every requirement, then decide whether it is sufficient without guessing.

Rules:
- Relevance alone is insufficient; exact names, values, dates, scopes, and conclusions must be supported.
- If anything is missing, inferred, ambiguous, or conflicting, set "sufficient" to false.
- Use no external knowledge.
- Select the minimum set of short exact quotes. Prefer at most 3 quotes, but completeness overrides this preference.
- If "sufficient" is true, answer using only the selected evidence.
- If "sufficient" is false, set "answer" to "Not mentioned".

Question-type checklist (apply every relevant item):
- Definition: include the definition itself and any required exclusion, alias, variant, or scope limit.
- Yes/no: directly support the conclusion; a negative requires evidence covering the requested scope and supporting absence, non-existence, non-mention, or no-change.
- Location/source/section: identify the requested place/source/section and enough nearby content to verify it.
- Comparison: cover every compared item and the comparison criterion.
- Change/version/time: cover the relevant time or version and the change, or explicit absence/no-change.
- Multi-hop: cover every required hop and the links between hops.
- List/count: support the complete set or count, not only examples.
- Open-ended/information-about: cover the main requested facts and necessary conditions, limits, exceptions, or triggers.

Return JSON only:
{
  "sufficient": true/false,
  "selected_evidence": [
    {
      "quote": "<exact quote from context>",
      "source_hint": "<URI, section, or context block identifier when available>"
    }
  ],
  "missing_info": ["<missing or unsupported required part>"],
  "answer": "<final answer or Not mentioned>",
  "reasoning": "<one short sentence>"
}

Return only this JSON object with no extra fields."""


class RawContextPhase1ResultProvider(Phase1Provider):
    name = "raw_context_phase1_result"

    def run(self, qa, search_res: dict, **kwargs) -> Phase1ProviderResult:
        start = timed()
        provider_cfg = self.provider_config()

        stage1_prompt_start = timed()
        stage1_prompt, stage1_prompt_meta = self._build_evidence_sufficiency_prompt(
            qa,
            search_res,
            provider_cfg,
        )
        stage1_prompt_build_latency = timed() - stage1_prompt_start

        stage1_start = timed()
        stage1_raw = self.llm.generate(stage1_prompt)
        stage1_latency = timed() - stage1_start

        stage1_parse_start = timed()
        stage1_obj, stage1_parse_error = self.extract_json_object(stage1_raw)
        stage1_json_parse_failed = stage1_obj is None
        stage1_regex_recovered = False
        stage1_sufficient_regex_found = False
        stage1_used_raw_context_for_answer = False
        stage1 = self._normalize_stage1(stage1_obj)
        stage1_parse_latency = timed() - stage1_parse_start

        stage1_decision_start = timed()
        selected_evidence = stage1["selected_evidence"]
        missing_info = stage1["missing_info"]
        stage1_sufficient = bool(stage1["sufficient"])
        parse_trigger = stage1_obj is None
        answer = self.adapter.post_process_answer(qa, stage1["answer"], {})
        action_text = str((stage1_obj or {}).get("action", "answer") or "answer").strip().lower()
        action_trigger = (
            self.config_bool(provider_cfg.get("fallback_on_action_fallback"), True)
            and action_text == "fallback"
        )
        missing_info_trigger = bool(missing_info) and self.config_bool(
            provider_cfg.get("fallback_on_missing_info"),
            True,
        )
        no_evidence_trigger = stage1_sufficient and not selected_evidence
        insufficient_trigger = (
            self.config_bool(provider_cfg.get("fallback_on_insufficient"), True)
            and not stage1_sufficient
        )
        refusal_trigger = stage1_sufficient and self.is_refusal_like(answer)
        triggered = (
            parse_trigger
            or action_trigger
            or missing_info_trigger
            or no_evidence_trigger
            or insufficient_trigger
            or refusal_trigger
        )
        stage1_decision_latency = timed() - stage1_decision_start

        # Compatibility fields retained for historical reports.  This provider
        # now performs exactly one LLM request and never runs a second stage.
        stage2_prompt = ""
        stage2_raw = ""
        stage2_latency = 0.0
        stage2_prompt_build_latency = 0.0
        stage2_parse_postprocess_latency = 0.0
        stage2_refusal_trigger = False
        stage2_insufficient_trigger = False

        final_build_start = timed()
        if parse_trigger:
            reasoning = f"Strict evidence output could not be parsed: {stage1_parse_error}"
        elif action_trigger:
            reasoning = "Strict evidence request explicitly routed to fallback."
        elif insufficient_trigger:
            reasoning = stage1["reasoning"] or "Selected evidence is insufficient."
        elif missing_info_trigger:
            reasoning = "Strict evidence audit reported missing information."
        elif no_evidence_trigger:
            reasoning = "Strict evidence audit selected no direct evidence."
        elif refusal_trigger:
            reasoning = "Strict evidence request produced a refusal-like answer."
        else:
            reasoning = stage1["reasoning"] or "Selected evidence supports the answer."

        final_answer = "Not mentioned" if triggered else answer
        parsed = LLMResponse(
            action="fallback" if triggered else "answer",
            sufficient=not triggered,
            answer=final_answer,
            reasoning=reasoning,
            evidence_analysis=self._evidence_analysis(selected_evidence, missing_info),
            missing_info=missing_info if triggered else [],
            raw=stage1_raw,
        )

        full_prompt = stage1_prompt
        raw = stage1_raw
        final_build_latency = timed() - final_build_start

        token_count_start = timed()
        stage1_input_tokens = self.count_tokens(stage1_prompt)
        stage1_output_tokens = self.count_tokens(stage1_raw)
        stage2_input_tokens = 0
        stage2_output_tokens = 0
        token_count_latency = timed() - token_count_start
        total_latency = timed() - start
        measured_latency = (
            stage1_prompt_build_latency
            + stage1_latency
            + stage1_parse_latency
            + stage1_decision_latency
            + final_build_latency
            + token_count_latency
        )
        profile_other_latency = max(0.0, total_latency - measured_latency)
        stage1_prompt_chars = len(stage1_prompt or "")
        stage1_raw_chars = len(stage1_raw or "")
        stage2_prompt_chars = len(stage2_prompt or "")
        stage2_raw_chars = len(stage2_raw or "")
        selected_evidence_chars = sum(len(x or "") for x in selected_evidence)

        return Phase1ProviderResult(
            name=self.name,
            answer=final_answer,
            parsed=parsed,
            should_fallback=triggered,
            reasoning=reasoning,
            search_res=search_res,
            prompt=full_prompt,
            raw=raw,
            meta={
                "phase1_provider_mode": "single_stage_strict_evidence_answer",
                "stage1_sufficient": stage1_sufficient,
                "stage2_ran": False,
            },
            supplemental=default_supplemental("Single-stage strict evidence provider"),
            input_tokens=stage1_input_tokens + stage2_input_tokens,
            output_tokens=stage1_output_tokens + stage2_output_tokens,
            latency_sec=total_latency,
            details={
                "rule": "single_stage_strict_evidence_answer",
                "llm_call_count": 1,
                "fallback_on_insufficient": self.config_bool(provider_cfg.get("fallback_on_insufficient"), True),
                "fallback_on_missing_info": self.config_bool(provider_cfg.get("fallback_on_missing_info"), True),
                "fallback_on_action_fallback": self.config_bool(provider_cfg.get("fallback_on_action_fallback"), True),
                "fallback_on_stage2_insufficient": False,
                "stage1_parse_error": stage1_parse_error,
                "stage1_json_parse_failed": stage1_json_parse_failed,
                "stage1_regex_recovered": stage1_regex_recovered,
                "stage1_sufficient_regex_found": stage1_sufficient_regex_found,
                "stage1_used_raw_context_for_answer": stage1_used_raw_context_for_answer,
                "stage1_sufficient": stage1_sufficient,
                "stage1_triggered": triggered,
                "stage1_prompt_build_latency_sec": stage1_prompt_build_latency,
                "stage1_latency_sec": stage1_latency,
                "stage1_parse_latency_sec": stage1_parse_latency,
                "stage1_decision_latency_sec": stage1_decision_latency,
                "stage1_input_tokens": stage1_input_tokens,
                "stage1_output_tokens": stage1_output_tokens,
                "stage1_prompt_chars": stage1_prompt_chars,
                **stage1_prompt_meta,
                "stage1_raw_chars": stage1_raw_chars,
                "stage2_ran": False,
                "stage2_prompt_build_latency_sec": stage2_prompt_build_latency,
                "stage2_latency_sec": stage2_latency,
                "stage2_parse_postprocess_latency_sec": stage2_parse_postprocess_latency,
                "stage2_input_tokens": stage2_input_tokens,
                "stage2_output_tokens": stage2_output_tokens,
                "stage2_prompt_chars": stage2_prompt_chars,
                "stage2_raw_chars": stage2_raw_chars,
                "final_build_latency_sec": final_build_latency,
                "token_count_latency_sec": token_count_latency,
                "profile_other_latency_sec": profile_other_latency,
                "selected_evidence_count": len(selected_evidence),
                "selected_evidence_chars": selected_evidence_chars,
                "selected_evidence": selected_evidence,
                "missing_info": missing_info,
                "answer_chars": len(answer or ""),
                "parse_trigger": parse_trigger,
                "insufficient_trigger": insufficient_trigger,
                "missing_info_trigger": missing_info_trigger,
                "no_evidence_trigger": no_evidence_trigger,
                "action_trigger": action_trigger,
                "refusal_trigger": refusal_trigger,
                "stage2_insufficient_trigger": stage2_insufficient_trigger,
                "stage2_refusal_trigger": stage2_refusal_trigger,
            },
        )

    def _build_evidence_sufficiency_prompt(
        self,
        qa,
        search_res: dict,
        provider_cfg: dict | None = None,
    ) -> tuple[str, dict]:
        context_text, context_meta = self._format_search_context(
            search_res,
            question=str(qa.question or ""),
            provider_cfg=provider_cfg,
        )
        # Adapter-specific evidence guidance is intentionally disabled here so
        # the provider can be evaluated as a dataset-agnostic reviewer.
        selection_block = ""
        sufficiency_block = ""
        return (
            (
                f"Retrieved context:\n{context_text}\n\n"
                f"{selection_block}"
                f"{sufficiency_block}"
                f"{EVIDENCE_SUFFICIENCY_PROMPT}\n\n"
                f"Question: {qa.question}"
            ),
            context_meta,
        )

    def _format_search_context(
        self,
        search_res: dict,
        *,
        question: str = "",
        provider_cfg: dict | None = None,
    ) -> tuple[str, dict]:
        provider_cfg = provider_cfg or {}
        max_total_chars = self._config_int(
            provider_cfg.get("stage1_context_max_chars_total"),
            0,
        )
        max_block_chars = self._config_int(
            provider_cfg.get("stage1_context_max_chars_per_block"),
            0,
        )
        head_chars = self._config_int(
            provider_cfg.get("stage1_context_head_chars"),
            500,
        )
        window_chars = self._config_int(
            provider_cfg.get("stage1_context_window_chars"),
            900,
        )

        recall_texts = search_res.get("recall_texts", {}) or {}
        retrieved_uris = list(search_res.get("retrieved_uris", []) or [])
        if retrieved_uris and recall_texts:
            sources: list[tuple[str, str]] = []
            seen = set()
            for uri in retrieved_uris:
                if uri in seen:
                    continue
                seen.add(uri)
                content = str(recall_texts.get(uri, "") or "").strip()
                if content:
                    sources.append((str(uri), content[:8000]))
            if sources:
                return self._format_context_sources(
                    sources,
                    question=question,
                    max_total_chars=max_total_chars,
                    max_block_chars=max_block_chars,
                    head_chars=head_chars,
                    window_chars=window_chars,
                )

        context_blocks = self.context_blocks(search_res)
        if not context_blocks:
            return "No retrieved context.", {
                "stage1_context_block_count": 0,
                "stage1_context_original_chars": 0,
                "stage1_context_chars": len("No retrieved context."),
                "stage1_context_compacted": False,
                "stage1_context_reduction_pct": 0.0,
                "stage1_context_max_chars_total": max_total_chars,
                "stage1_context_max_chars_per_block": max_block_chars,
            }
        return self._format_context_sources(
            [("", block) for block in context_blocks],
            question=question,
            max_total_chars=max_total_chars,
            max_block_chars=max_block_chars,
            head_chars=head_chars,
            window_chars=window_chars,
        )

    def _format_context_sources(
        self,
        sources: list[tuple[str, str]],
        *,
        question: str,
        max_total_chars: int,
        max_block_chars: int,
        head_chars: int,
        window_chars: int,
    ) -> tuple[str, dict]:
        original_chars = sum(len(content) for _, content in sources)
        effective_block_limit = max_block_chars
        if max_total_chars > 0:
            fair_share = max(300, max_total_chars // max(len(sources), 1))
            effective_block_limit = (
                min(effective_block_limit, fair_share)
                if effective_block_limit > 0
                else fair_share
            )

        blocks = []
        any_compacted = False
        for idx, (uri, content) in enumerate(sources, start=1):
            compacted = self._compact_context_content(
                content,
                question=question,
                max_chars=effective_block_limit,
                head_chars=head_chars,
                window_chars=window_chars,
            )
            any_compacted = any_compacted or len(compacted) < len(content)
            uri_line = f"\nURI: {uri}" if uri else ""
            blocks.append(f"[Context block {idx}]{uri_line}\n{compacted}")

        context_text = "\n\n".join(blocks)
        reduction_pct = (
            max(0.0, (1.0 - len(context_text) / original_chars) * 100.0)
            if original_chars > 0 else 0.0
        )
        return context_text, {
            "stage1_context_block_count": len(sources),
            "stage1_context_original_chars": original_chars,
            "stage1_context_chars": len(context_text),
            "stage1_context_compacted": any_compacted,
            "stage1_context_reduction_pct": reduction_pct,
            "stage1_context_max_chars_total": max_total_chars,
            "stage1_context_max_chars_per_block": effective_block_limit,
        }

    def _compact_context_content(
        self,
        content: str,
        *,
        question: str,
        max_chars: int,
        head_chars: int,
        window_chars: int,
    ) -> str:
        text = str(content or "").strip()
        if max_chars <= 0 or len(text) <= max_chars:
            return text

        head_end = min(len(text), max(0, min(head_chars, max_chars)))
        spans: list[tuple[int, int]] = [(0, head_end)] if head_end else []
        terms = self._question_terms(question)
        lowered = text.lower()
        candidates = []
        half_window = max(100, window_chars // 2)
        for term in terms:
            start = 0
            while True:
                pos = lowered.find(term, start)
                if pos < 0:
                    break
                left = max(0, pos - half_window)
                right = min(len(text), pos + len(term) + half_window)
                window = lowered[left:right]
                score = sum(1 for candidate in terms if candidate in window)
                candidates.append((score, pos, left, right))
                start = pos + len(term)

        used_chars = sum(right - left for left, right in spans)
        for _, pos, left, right in sorted(candidates, key=lambda item: (-item[0], item[1])):
            if any(kept_left <= pos < kept_right for kept_left, kept_right in spans):
                continue
            for kept_left, kept_right in sorted(spans):
                if kept_right <= pos and kept_right > left:
                    left = kept_right
                elif kept_left > pos and kept_left < right:
                    right = kept_left
            remaining = max_chars - used_chars
            if remaining <= 0:
                break
            if right <= left:
                continue
            if right - left > remaining:
                left = max(left, min(pos - remaining // 2, right - remaining))
                right = left + remaining
            spans.append((left, right))
            used_chars += right - left

        if used_chars < max_chars:
            spans.append((head_end, min(len(text), head_end + (max_chars - used_chars))))

        merged = []
        for left, right in sorted(spans):
            if right <= left:
                continue
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))

        pieces = [text[left:right].strip() for left, right in merged if text[left:right].strip()]
        return "\n...\n".join(pieces)

    def _question_terms(self, question: str) -> list[str]:
        stopwords = {
            "and", "are", "did", "does", "for", "from", "has", "have", "how",
            "into", "its", "that", "the", "their", "then", "this", "was", "were",
            "what", "when", "where", "which", "who", "why", "with", "would",
        }
        return list(dict.fromkeys(
            token
            for token in re.findall(r"[a-z0-9]+", str(question or "").lower())
            if len(token) >= 3 and token not in stopwords
        ))

    def _config_int(self, value: Any, default: int = 0) -> int:
        try:
            return max(0, int(value)) if value is not None else default
        except (TypeError, ValueError):
            return default

    def _normalize_stage1(self, obj: dict | None) -> dict:
        if not isinstance(obj, dict):
            return {
                "sufficient": False,
                "selected_evidence": [],
                "missing_info": ["Evidence sufficiency output was not valid JSON."],
                "answer": "Not mentioned",
                "reasoning": "",
            }

        return {
            "sufficient": self._json_bool(obj.get("sufficient"), False),
            "selected_evidence": self._selected_evidence_list(obj.get("selected_evidence")),
            "missing_info": self._string_list(obj.get("missing_info")),
            "answer": str(obj.get("answer", "Not mentioned") or "Not mentioned").strip(),
            "reasoning": str(obj.get("reasoning", "") or "").strip(),
        }

    def _json_bool(self, value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        if value is None:
            return default
        return bool(value)

    def _string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items = value
        else:
            items = [value]
        result = []
        for item in items:
            text = str(item or "").strip()
            if text:
                result.append(text)
        return result

    def _selected_evidence_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        evidence = []
        for item in items:
            if isinstance(item, dict):
                quote = str(
                    item.get("quote")
                    or item.get("evidence")
                    or item.get("text")
                    or item.get("content")
                    or ""
                ).strip()
                source_hint = str(
                    item.get("source_hint")
                    or item.get("source")
                    or item.get("uri")
                    or item.get("section")
                    or ""
                ).strip()
                if not quote:
                    continue
                parts = []
                if source_hint:
                    parts.append(f"Source hint: {source_hint}")
                parts.append(f"Quote: {quote}")
                evidence.append("\n".join(parts))
            else:
                text = str(item or "").strip()
                if text:
                    evidence.append(text)
        return evidence

    def _evidence_analysis(
        self,
        selected_evidence: list[str],
        missing_info: list[str],
    ) -> list[str]:
        evidence_text = " | ".join(selected_evidence) if selected_evidence else "No direct evidence selected."
        missing_text = "; ".join(missing_info) if missing_info else "None."
        return [
            f"Direct support: {evidence_text}",
            f"Unsupported or inferred parts: {missing_text}",
        ]
