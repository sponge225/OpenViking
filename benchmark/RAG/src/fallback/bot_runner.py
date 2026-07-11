import copy
import uuid
from typing import Callable


class FallbackBotRunner:
    def __init__(
        self,
        config: dict,
        run_query: Callable,
        summarize_result: Callable,
        resolve_target_uris: Callable,
    ):
        self.config = config
        self.run_query = run_query
        self.summarize_result = summarize_result
        self.resolve_target_uris = resolve_target_uris

    def run(self, task: dict, qa, bot_use_relations: bool, trace_suffix: str) -> dict:
        bot_config = copy.deepcopy(self.config)
        bot_config.setdefault("vikingbot", {})["use_relations"] = bot_use_relations

        session_id = f"{trace_suffix}_{uuid.uuid4().hex}"
        restrict_to_qa_doc = bool(self.config.get("execution", {}).get("restrict_to_qa_doc", False))
        allowed_target_uris = self.resolve_target_uris(task, qa) if restrict_to_qa_doc else None

        vikingbot_result = self.run_query(
            question=qa.question,
            config=bot_config,
            session_id=session_id,
            allowed_target_uris=allowed_target_uris,
        )
        return self.summarize_result(
            vikingbot_result,
            task["id"],
            trace_suffix=trace_suffix,
        )
