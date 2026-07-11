from .base import Phase1ProviderResult
from .raw_context_naive_rule import RawContextNaiveRuleProvider
from .raw_context_phase1_result import RawContextPhase1ResultProvider


DEFAULT_PRIMARY_PROVIDER = "raw_context_phase1_result"
DEFAULT_PROVIDER_NAMES = [
    "raw_context_phase1_result",
    "raw_context_naive_rule",
]

PROVIDER_CLASSES = {
    RawContextPhase1ResultProvider.name: RawContextPhase1ResultProvider,
    RawContextNaiveRuleProvider.name: RawContextNaiveRuleProvider,
}


class Phase1ProviderRunner:
    def __init__(self, config: dict, adapter, db, llm, logger=None):
        self.config = config
        self.adapter = adapter
        self.db = db
        self.llm = llm
        self.logger = logger
        self.primary_name, self.provider_names = self._resolve_provider_names()
        self.providers = {
            name: PROVIDER_CLASSES[name](config, adapter, db, llm, logger=logger)
            for name in self.provider_names
        }

    def _provider_config(self) -> dict:
        execution = self.config.get("execution", {}) or {}
        raw = execution.get("phase1_providers", self.config.get("phase1_providers", {}))
        return raw if isinstance(raw, dict) else {}

    def _resolve_provider_names(self) -> tuple[str, list[str]]:
        cfg = self._provider_config()
        primary = str(cfg.get("primary", DEFAULT_PRIMARY_PROVIDER) or DEFAULT_PRIMARY_PROVIDER)
        requested = cfg.get("enabled", cfg.get("providers", DEFAULT_PROVIDER_NAMES))
        if isinstance(requested, str):
            requested = [requested]
        if not isinstance(requested, list):
            requested = list(DEFAULT_PROVIDER_NAMES)

        names = []
        for name in requested:
            key = str(name or "").strip()
            if key in PROVIDER_CLASSES and key not in names:
                names.append(key)
        if primary not in PROVIDER_CLASSES or PROVIDER_CLASSES[primary].requires_primary:
            primary = DEFAULT_PRIMARY_PROVIDER
        if primary not in names:
            names.insert(0, primary)
        return primary, names

    def run_all(self, qa, search_res: dict, **kwargs) -> tuple[str, dict[str, Phase1ProviderResult]]:
        primary_provider = self.providers[self.primary_name]
        primary_result = primary_provider.run(qa, search_res, **kwargs)
        results = {self.primary_name: primary_result}

        for name in self.provider_names:
            if name == self.primary_name:
                continue
            provider = self.providers[name]
            try:
                if provider.requires_primary:
                    result = provider.run(qa, search_res, primary_result=primary_result, **kwargs)
                else:
                    result = provider.run(qa, search_res, **kwargs)
            except Exception as exc:
                if name == self.primary_name:
                    raise
                result = provider.error_result(qa, search_res, f"{type(exc).__name__}: {exc}")
            results[name] = result

        return self.primary_name, results
