from core.response_parser import parse_llm_response

from .base import Phase1Provider, Phase1ProviderResult, default_supplemental, timed


class RawContextNaiveRuleProvider(Phase1Provider):
    name = "raw_context_naive_rule"

    def run(self, qa, search_res: dict, **kwargs) -> Phase1ProviderResult:
        start = timed()
        provider_cfg = self.provider_config()
        context_blocks = self.context_blocks(search_res)
        build_simple_prompt = getattr(self.adapter, "build_simple_prompt")
        full_prompt, meta = build_simple_prompt(qa, context_blocks)
        raw = self.llm.generate(full_prompt)
        parsed = parse_llm_response(raw)
        answer = self.adapter.post_process_answer(qa, parsed.answer, meta)

        action_text = str(parsed.action or "answer").strip().lower()
        action_trigger = (
            self.config_bool(provider_cfg.get("fallback_on_action_fallback"), True)
            and action_text == "fallback"
        )
        insufficient_trigger = (
            self.config_bool(provider_cfg.get("fallback_on_insufficient"), True)
            and not bool(parsed.sufficient)
        )
        refusal_trigger = (
            self.config_bool(provider_cfg.get("fallback_on_refusal_like_answer"), True)
            and self.is_refusal_like(answer)
        )
        should_fallback = action_trigger or insufficient_trigger or refusal_trigger
        if action_trigger:
            reasoning = "Simple answer prompt routed to fallback."
        elif insufficient_trigger:
            reasoning = "Simple answer prompt marked the context insufficient."
        elif refusal_trigger:
            reasoning = "Simple answer prompt produced a refusal-like answer."
        else:
            reasoning = "Simple answer prompt marked the answer sufficient."

        return Phase1ProviderResult(
            name=self.name,
            answer=answer,
            parsed=parsed,
            should_fallback=should_fallback,
            reasoning=parsed.reasoning or reasoning,
            search_res=search_res,
            prompt=full_prompt,
            raw=raw,
            meta=meta,
            supplemental=default_supplemental("Naive combined answer+sufficiency provider"),
            input_tokens=self.count_tokens(full_prompt),
            output_tokens=self.count_tokens(raw),
            latency_sec=timed() - start,
            details={
                "rule": "simple_llm_answer_and_sufficiency",
                "fallback_on_action_fallback": self.config_bool(provider_cfg.get("fallback_on_action_fallback"), True),
                "fallback_on_insufficient": self.config_bool(provider_cfg.get("fallback_on_insufficient"), True),
                "fallback_on_refusal_like_answer": self.config_bool(provider_cfg.get("fallback_on_refusal_like_answer"), True),
                "action_trigger": action_trigger,
                "insufficient_trigger": insufficient_trigger,
                "refusal_trigger": refusal_trigger,
            },
        )
