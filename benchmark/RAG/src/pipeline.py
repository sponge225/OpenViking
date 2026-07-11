import os
import json
import time
import uuid
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import sys
from typing import Set

sys.path.append(str(Path(__file__).parent))

from adapters.base import BaseAdapter, StandardQA, StandardSample
from core.logger import get_logger
from core.vector_store import VikingStoreWrapper
from core.monitor import BenchmarkMonitor
from core.metrics import MetricsCalculator
from core.judge_util import llm_grader
from core.response_parser import parse_llm_response
from core.checkpoint import CheckpointManager
from core.question_rewriter import QuestionRewriteError, QuestionRewriteStore, get_or_create_rewrites
from fallback import (
    FallbackBotRunner,
    fallback_judgment_gain_summary,
    fallback_judgment_summary,
    fallback_miss_gain_summary,
    fallback_miss_summary,
    phase1_fallback_judgment,
    recoverable_miss_gain_summary,
    recoverable_miss_summary,
)
from phase1_providers import Phase1ProviderRunner
from vikingbot_runner import run_vikingbot_query
from nanobot_runner import run_nanobot_query


class BenchmarkPipeline:
    def __init__(self, config, adapter: BaseAdapter, vector_db: VikingStoreWrapper, llm, resume: bool = False):
        self.config = config
        self.adapter = adapter
        self.db = vector_db
        self.llm = llm
        self.logger = get_logger()
        self.monitor = BenchmarkMonitor()
        self.resume = resume
        self.phase1_provider_runner = Phase1ProviderRunner(config, adapter, vector_db, llm, logger=self.logger)
        self.fallback_bot_runner = FallbackBotRunner(
            config,
            run_vikingbot_query,
            self._summarize_vikingbot_result,
            self._resolve_target_uris,
        )
        
        self.output_dir = self.config['paths']['output_dir']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.generated_file = os.path.join(self.output_dir, "generated_answers.json")
        self.eval_file = os.path.join(self.output_dir, "qa_eval_detailed_results.json")
        self.report_file = os.path.join(self.output_dir, "benchmark_metrics_report.json")

        # Checkpoint + incremental saving for resume support.
        self.checkpoint_manager = CheckpointManager(self.output_dir, self.config)
        self._file_lock = threading.Lock()
        self.save_frequency = 10
        
        self.metrics_summary = {
            "insertion": {"time": 0, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0},
            "deletion": {"time": 0, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
        }

    def _evidence_audit_needs_more(self, parsed, answer: str) -> tuple[bool, str]:
        if not parsed.sufficient:
            return True, "LLM marked context insufficient"

        answer_lower = (answer or "").strip().lower()
        refusal_markers = (
            "not mentioned",
            "insufficient information",
            "i don't know",
            "cannot be determined",
            "not enough information",
            "no relevant information",
            "unable to find",
        )
        if not answer_lower or any(marker in answer_lower for marker in refusal_markers):
            return True, "Answer is empty or refusal-like"

        if parsed.missing_info:
            return True, "LLM reported missing information"

        evidence_lines = [line for line in (parsed.evidence_analysis or []) if str(line).strip()]
        if not evidence_lines:
            return True, "Missing evidence analysis"

        evidence_text = "\n".join(str(line) for line in evidence_lines).lower()
        issue_patterns = (
            r"\bmissing\b",
            r"\bconflict(?:ing|s|ed)?\b",
            r"\binsufficient\b",
            r"\bnot supported\b",
            r"\bnot enough\b",
            r"\blacks?\b",
            r"\bcannot be determined\b",
            r"\bunable to verify\b",
        )
        negated_issue_patterns = (
            r"\bno\s+.*\bmissing\b",
            r"\bno\s+.*\bconflict(?:ing|s|ed)?\b",
            r"\bno\s+.*\binsufficient\b",
            r"\bnot\s+.*\bmissing\b",
            r"\bnot\s+.*\bconflict(?:ing|s|ed)?\b",
            r"\bwithout\s+.*\bmissing\b",
            r"\bwithout\s+.*\bconflict(?:ing|s|ed)?\b",
        )
        for line in evidence_lines:
            line_text = re.sub(r"\s+", " ", str(line).strip().lower())
            if not any(re.search(pattern, line_text) for pattern in issue_patterns):
                continue
            if any(re.search(pattern, line_text) for pattern in negated_issue_patterns):
                continue
            return True, "Evidence analysis indicates missing or conflicting information"

        has_quote_like_support = (
            '"' in evidence_text
            or "'" in evidence_text
            or "[" in evidence_text
            or "quote" in evidence_text
            or "context" in evidence_text
        )
        if not has_quote_like_support:
            return True, "Evidence analysis lacks checkable support"

        return False, "Evidence audit passed"

    def _config_bool(self, value, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _build_supplemental_query(self, question: str, parsed, retrieval_instruction: str = "") -> str:
        parts = []
        if retrieval_instruction:
            parts.append(retrieval_instruction)
        parts.append(question)

        missing = " ".join(str(x).strip() for x in (parsed.missing_info or []) if str(x).strip())
        if missing:
            parts.append(f"Find missing evidence: {missing}")

        analysis = " ".join(str(x).strip() for x in (parsed.evidence_analysis or []) if str(x).strip())
        if analysis:
            parts.append(f"Evidence audit notes: {analysis}")

        query = " ".join(parts)
        query = re.sub(r"\s+", " ", query).strip()
        return query or question

    def _merge_search_results(self, base: dict, extra: dict) -> dict:
        merged = {
            "recall_texts": dict(base.get("recall_texts", {}) or {}),
            "context_blocks": list(base.get("context_blocks", []) or []),
            "retrieved_uris": list(base.get("retrieved_uris", []) or []),
            "retrieval_tokens": int(base.get("retrieval_tokens", 0) or 0)
            + int(extra.get("retrieval_tokens", 0) or 0),
        }
        if "relations_uris" in base or "relations_uris" in extra:
            merged["relations_uris"] = list(base.get("relations_uris", []) or [])
            merged["relations_found"] = int(base.get("relations_found", 0) or 0) + int(extra.get("relations_found", 0) or 0)
            merged["relations_added"] = int(base.get("relations_added", 0) or 0)

        seen = set(merged["retrieved_uris"])
        for uri in extra.get("retrieved_uris", []) or []:
            if uri in seen:
                continue
            seen.add(uri)
            merged["retrieved_uris"].append(uri)

        for uri, content in (extra.get("recall_texts", {}) or {}).items():
            if uri not in merged["recall_texts"]:
                merged["recall_texts"][uri] = content
                merged["context_blocks"].append(str(content)[:8000])

        if "relations_uris" in merged:
            rel_seen = set(merged["relations_uris"])
            for uri in extra.get("relations_uris", []) or []:
                if uri not in rel_seen:
                    rel_seen.add(uri)
                    merged["relations_uris"].append(uri)
                    merged["relations_added"] += 1

        return merged

    def _generate_with_optional_supplemental_retrieval(
        self,
        qa,
        search_res: dict,
        retrieval_instruction: str,
        meta_target_uri: str | None = None,
        skip_supplemental: bool = False,
    ) -> dict:
        context_blocks = search_res["context_blocks"]
        full_prompt, meta = self.adapter.build_prompt(qa, context_blocks)
        ans_raw = self.llm.generate(full_prompt)
        generation_input_tokens = self.db.count_tokens(full_prompt)
        generation_output_tokens = self.db.count_tokens(ans_raw)
        parsed = parse_llm_response(ans_raw)
        answer = self.adapter.post_process_answer(qa, parsed.answer, meta)

        audit_enabled = self.config.get("execution", {}).get("enable_evidence_supplement", True)
        audit_needs_more, audit_reason = self._evidence_audit_needs_more(parsed, answer)
        supplemental = {
            "enabled": bool(audit_enabled),
            "triggered": False,
            "reason": audit_reason,
            "query": "",
            "uris": [],
            "needs_more": bool(audit_enabled and audit_needs_more and not skip_supplemental),
            "audit_needs_more": bool(audit_enabled and audit_needs_more),
        }

        if (
            audit_enabled
            and audit_needs_more
            and not skip_supplemental
        ):
            supplemental_query = self._build_supplemental_query(
                qa.question,
                parsed,
                retrieval_instruction=retrieval_instruction,
            )
            target_uri = meta_target_uri or self.config.get("execution", {}).get("retrieval_target_uri", "viking://resources")
            extra_res = self.db.retrieve(
                query=supplemental_query,
                topk=self.config["execution"]["retrieval_topk"],
                target_uri=target_uri,
            )
            search_res = self._merge_search_results(search_res, extra_res)
            supplemental.update({
                "triggered": True,
                "query": supplemental_query,
                "uris": extra_res.get("retrieved_uris", []),
            })

            full_prompt, meta = self.adapter.build_prompt(qa, search_res["context_blocks"])
            ans_raw = self.llm.generate(full_prompt)
            generation_input_tokens += self.db.count_tokens(full_prompt)
            generation_output_tokens += self.db.count_tokens(ans_raw)
            parsed = parse_llm_response(ans_raw)
            answer = self.adapter.post_process_answer(qa, parsed.answer, meta)

        return {
            "search_res": search_res,
            "prompt": full_prompt,
            "meta": meta,
            "raw": ans_raw,
            "parsed": parsed,
            "answer": answer,
            "supplemental": supplemental,
            "input_tokens": generation_input_tokens,
            "output_tokens": generation_output_tokens,
        }

    def _get_record_input_tokens(self, record: dict) -> int:
        usage = (record or {}).get("token_usage", {}) or {}
        if "prompt_tokens" in usage:
            return int(usage.get("prompt_tokens", 0) or 0)
        return int(usage.get("total_input_tokens", 0) or 0)

    def _get_record_output_tokens(self, record: dict) -> int:
        usage = (record or {}).get("token_usage", {}) or {}
        if "completion_tokens" in usage:
            return int(usage.get("completion_tokens", 0) or 0)
        return int(usage.get("llm_output_tokens", 0) or 0)

    def _score_answer_for_accuracy(self, question: str, golds, answer: str, dataset_name: str) -> tuple[float, float, dict]:
        f1 = max((MetricsCalculator.calculate_f1(answer, gt) for gt in golds), default=0.0)
        eval_record = {
            "score": 0.0,
            "reasoning": "",
            "prompt_type": "",
        }

        try:
            eval_record = llm_grader(
                self.llm.llm,
                self.config['llm']['model'],
                question,
                golds,
                answer,
                dataset_name=dataset_name,
            )
        except Exception as e:
            self.logger.error(f"Grader error: {e}")

        if MetricsCalculator.check_refusal(answer) and any(MetricsCalculator.check_refusal(gt) for gt in golds):
            f1 = 1.0
            eval_record["score"] = 4.0
            eval_record["reasoning"] = "System successfully identified Unanswerable/Refusal condition."
            eval_record["prompt_type"] = "Heuristic_Refusal_Check"

        return f1, eval_record["score"], eval_record

    def _parse_relation_reason_json(self, reason):
        if isinstance(reason, dict):
            return reason
        if not isinstance(reason, str) or not reason.strip():
            return None
        try:
            parsed = json.loads(reason)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _normalize_relation_tool_calls(self, tool_calls):
        if not isinstance(tool_calls, list):
            return tool_calls
        for tc in tool_calls:
            if not isinstance(tc, dict) or tc.get('tool_name') != 'openviking_search':
                continue
            result_data = tc.get('result')
            if not isinstance(result_data, list):
                continue
            priority_rank = 1
            for item in result_data:
                if not isinstance(item, dict):
                    continue
                mr = item.get('match_reason', '')
                if not isinstance(mr, str) or not mr.startswith('relation_from:'):
                    continue
                relation_from = mr.replace('relation_from:', '').strip()
                item['is_priority'] = True
                item['priority_rank'] = item.get('priority_rank') or priority_rank
                item['relation_from'] = item.get('relation_from') or relation_from
                item['relation_question_id'] = item.get('relation_question_id') or ""
                priority_rank += 1
                reason_json = self._parse_relation_reason_json(item.get('relation_reason', ''))
                if reason_json:
                    item['relation_reason_json'] = reason_json
                else:
                    item['relation_reason'] = ""
                    item.pop('relation_reason_json', None)
        return tool_calls

    def _tool_args_dict(self, tool_call: dict) -> dict:
        args_data = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}
        if isinstance(args_data, str):
            try:
                args_data = json.loads(args_data)
            except (json.JSONDecodeError, TypeError):
                args_data = {}
        return args_data if isinstance(args_data, dict) else {}

    def _uri_list(self, value) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value if str(v)]
        if isinstance(value, str) and value:
            return [value]
        return []

    def _dedupe_list(self, values: list[str]) -> list[str]:
        return list(dict.fromkeys(v for v in values if v))

    def _extract_build_link_stats_from_tool_calls(self, tool_calls: list) -> dict:
        """Summarize relation hits and build-link actions from VikingBot tool calls."""
        stats = {
            "relation_search_calls": 0,
            "relation_hit_search_calls": 0,
            "relations_found_per_search": [],
            "relations_found_total": 0,
            "has_relations_found": False,
            "link_tool_calls": 0,
            "link_pairs_requested": 0,
            "link_pairs_created": 0,
            "has_links_created": False,
            "skipped_relation_to_uris": [],
            "skipped_relation_to_uri_count": 0,
            "has_relation_to_uri_skipped": False,
        }

        skipped_relation_to_uris: list[str] = []
        for tc in tool_calls if isinstance(tool_calls, list) else []:
            if not isinstance(tc, dict):
                continue
            tn = tc.get("tool_name", "")
            if tn == "openviking_search":
                stats["relation_search_calls"] += 1
                rf = int(tc.get("relations_found", 0) or 0)
                result_relation_count = 0
                result_data = tc.get("result")
                if isinstance(result_data, list):
                    for item in result_data:
                        if not isinstance(item, dict):
                            continue
                        if str(item.get("match_reason", "")).startswith("relation_from:"):
                            result_relation_count += 1
                if rf <= 0:
                    rf = result_relation_count
                stats["relations_found_per_search"].append(rf)
                stats["relations_found_total"] += rf
                if rf > 0:
                    stats["relation_hit_search_calls"] += 1
            elif tn == "openviking_link":
                stats["link_tool_calls"] += 1
                args_data = self._tool_args_dict(tc)
                from_uris = self._uri_list(args_data.get("from_uris", []))
                to_uris = self._uri_list(args_data.get("to_uris", []))
                skipped_relation_to_uris.extend(self._uri_list(args_data.get("skipped_to_uris", [])))
                skipped_relation_to_uris.extend(self._uri_list(tc.get("skipped_relation_to_uris", [])))

                requested = len(from_uris) * len(to_uris)
                stats["link_pairs_requested"] += requested
                if "relations_found" in tc:
                    created = int(tc.get("relations_found", 0) or 0)
                else:
                    created = requested
                stats["link_pairs_created"] += created

        stats["has_relations_found"] = stats["relations_found_total"] > 0
        stats["has_links_created"] = stats["link_pairs_created"] > 0
        stats["skipped_relation_to_uris"] = self._dedupe_list(skipped_relation_to_uris)
        stats["skipped_relation_to_uri_count"] = len(stats["skipped_relation_to_uris"])
        stats["has_relation_to_uri_skipped"] = stats["skipped_relation_to_uri_count"] > 0
        return stats

    def _summarize_phase1_evidence_efficiency(self, records: list[dict]) -> dict:
        samples = []
        for record in records:
            fallback = record.get("fallback", {}) or {}
            provider_name = str(
                fallback.get("primary_provider", "raw_context_phase1_result")
                or "raw_context_phase1_result"
            )
            provider_result = (
                (fallback.get("provider_results", {}) or {}).get(provider_name, {}) or {}
            )
            details = provider_result.get("details", {}) or {}
            if "stage1_latency_sec" not in details:
                continue
            samples.append({
                "provider_latency_sec": float(provider_result.get("latency_sec", 0) or 0),
                "stage1_latency_sec": float(details.get("stage1_latency_sec", 0) or 0),
                "stage1_input_tokens": int(details.get("stage1_input_tokens", 0) or 0),
                "stage1_output_tokens": int(details.get("stage1_output_tokens", 0) or 0),
                "stage1_prompt_chars": int(details.get("stage1_prompt_chars", 0) or 0),
                "context_original_chars": int(details.get("stage1_context_original_chars", 0) or 0),
                "context_chars": int(details.get("stage1_context_chars", 0) or 0),
                "context_reduction_pct": float(details.get("stage1_context_reduction_pct", 0) or 0),
                "stage2_ran": bool(details.get("stage2_ran", False)),
                "llm_call_count": int(
                    details.get(
                        "llm_call_count",
                        1 + int(bool(details.get("stage2_ran", False))),
                    ) or 0
                ),
            })

        if not samples:
            return {}

        def average(key: str) -> float:
            return sum(float(sample[key]) for sample in samples) / len(samples)

        stage1_latencies = sorted(sample["stage1_latency_sec"] for sample in samples)
        p95_index = int((len(stage1_latencies) - 1) * 0.95)
        return {
            "Records": len(samples),
            "Average Phase1 Provider Time (s)": average("provider_latency_sec"),
            "Average Stage1 Evidence LLM Time (s)": average("stage1_latency_sec"),
            "P95 Stage1 Evidence LLM Time (s)": stage1_latencies[p95_index],
            "Average Stage1 Input Tokens": average("stage1_input_tokens"),
            "Average Stage1 Output Tokens": average("stage1_output_tokens"),
            "Average Stage1 Prompt Chars": average("stage1_prompt_chars"),
            "Average Original Context Chars": average("context_original_chars"),
            "Average Submitted Context Chars": average("context_chars"),
            "Average Context Reduction (%)": average("context_reduction_pct"),
            "Average LLM Calls per Query": average("llm_call_count"),
            "Stage2 Run Rate": sum(1 for sample in samples if sample["stage2_ran"]) / len(samples),
        }

    def _save_partial_results(self, results_map: dict):
        # Persist partial generation results so we can resume safely after interruption.
        with self._file_lock:
            sorted_results = [results_map[i] for i in sorted(results_map.keys())]
            dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')
            save_data = {
                "summary": {"dataset": dataset_name, "total_queries": len(sorted_results)},
                "results": sorted_results
            }
            with open(self.generated_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

    def _save_partial_eval_results(self, eval_results_map: dict):
        # Persist partial evaluation results so we can resume safely after interruption.
        with self._file_lock:
            eval_records = list(eval_results_map.values())
            with open(self.eval_file, "w", encoding="utf-8") as f:
                json.dump({"results": eval_records}, f, indent=2, ensure_ascii=False)

    def run_import(self):
        """Stage: Import documents into OV store"""
        self.logger.info(">>> Stage: Import (Data Prepare + Ingest)")

        if not self.db:
            raise RuntimeError("Cannot ingest without a vector store. Disable use_nanobot to use import.")

        doc_dir = self.config['paths'].get('doc_output_dir')
        if not doc_dir:
            doc_dir = os.path.join(self.output_dir, "docs")

        try:
            doc_info = self.adapter.data_prepare(doc_dir)
        except Exception as e:
            self.logger.exception(f"Data preparation failed: {e}")
            exit(1)

        ingest_workers = self.config['execution'].get('ingest_workers', 10)
        ingest_mode = self.config['execution'].get('ingest_mode', 'per_file')
        
        mode_desc = {
            'directory': 'Unified directory mode',
            'per_file': 'Per-file mode'
        }
        self.logger.info(f"Ingestion mode: {ingest_mode} ({mode_desc.get(ingest_mode, 'Unknown mode')})")
        self.logger.info(f"Number of documents: {len(doc_info)}")
        
        ingest_stats = self.db.ingest(
            doc_info, 
            max_workers=ingest_workers, 
            monitor=self.monitor,
            ingest_mode=ingest_mode
        )
        self.metrics_summary["insertion"] = ingest_stats
        self.logger.info(f"Import finished. Time: {ingest_stats['time']:.2f}s")

        if self.db:
            self.db.close()

        self._update_report({
            "Insertion Efficiency (Total Dataset)": {
                "Total Insertion Time (s)": self.metrics_summary["insertion"]["time"],
                "Total Input Tokens": self.metrics_summary["insertion"]["input_tokens"],
                "Total Output Tokens": self.metrics_summary["insertion"]["output_tokens"],
                "Total Embedding Tokens": self.metrics_summary["insertion"].get("embedding_tokens", 0)
            }
        })

    def run_generation(self):
        """Stage: Generate answers for QA queries"""
        self.logger.info(">>> Stage: Generation (Retrieve + Generate)")
        samples = self.adapter.load_and_transform()
        samples = self._apply_question_rewrites_to_samples(samples)
        tasks = self._prepare_tasks(samples)
        results_map = {}
        max_workers = self.config['execution']['max_workers']

        completed_tasks: Set[int] = set()
        if self.resume:
            completed_tasks = self.checkpoint_manager.get_completed_tasks("generation")
            if completed_tasks:
                self.logger.info(f"Resuming from checkpoint. {len(completed_tasks)} tasks already completed.")
                if os.path.exists(self.generated_file):
                    try:
                        with open(self.generated_file, "r", encoding="utf-8") as f:
                            saved_data = json.load(f)
                        for result in saved_data.get("results", []):
                            results_map[result["_global_index"]] = result
                    except Exception as e:
                        self.logger.warning(f"Failed to load previous generated results, continuing fresh: {e}")

        remaining_tasks = [task for task in tasks if task["id"] not in completed_tasks]
        self.logger.info(f"Total tasks: {len(tasks)}, Remaining: {len(remaining_tasks)}")
        
        mode = self.config.get("execution", {}).get("mode")
        if mode is None:
            if self.config.get("execution", {}).get("use_nanobot", False):
                mode = "nanobot"
            elif self.config.get("execution", {}).get("use_vikingbot", False):
                mode = "vikingbot"
            else:
                mode = "standard"

        mode_dispatch = {
            "standard": self._process_generation_task,
            "vikingbot": self._process_vikingbot_task,
            "nanobot": self._process_nanobot_task,
            "ov_fallback_bot": self._process_ov_fallback_bot_task,
            "ov_fallback_bot_relations": self._process_ov_fallback_bot_relations_task,
        }
        process_fn = mode_dispatch[mode]

        if remaining_tasks:
            initial_completed = len(completed_tasks)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(process_fn, task): task
                    for task in remaining_tasks
                }

                pbar = tqdm(total=len(tasks), desc="Generating Answers", unit="task", initial=len(completed_tasks))
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        res = future.result()
                        results_map[res['_global_index']] = res
                        completed_tasks.add(res['_global_index'])

                        newly_completed = len(completed_tasks) - initial_completed
                        if newly_completed % self.save_frequency == 0 or len(completed_tasks) == len(tasks):
                            self.checkpoint_manager.update_completed_tasks("generation", completed_tasks, len(tasks))
                            self._save_partial_results(results_map)
                    except Exception as e:
                        self.logger.error(f"Generation failed for task {task['id']}: {e}")
                        self.monitor.worker_end(success=False)
                    pbar.set_postfix(self.monitor.get_status_dict())
                    pbar.update(1)
                pbar.close()
        else:
            self.logger.info("All tasks already completed!")

        sorted_results = [results_map[i] for i in sorted(results_map.keys())]
        dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')
        save_data = {
            "summary": {"dataset": dataset_name, "total_queries": len(sorted_results)},
            "results": sorted_results
        }
        total = len(sorted_results)
        if total > 0:
            self._update_report({
                    "Query Efficiency (Average Per Query)": {
                        "Average Retrieval Time (s)": sum(r['retrieval']['latency_sec'] for r in sorted_results) / total,
                        "Average Input Tokens": sum(self._get_record_input_tokens(r) for r in sorted_results) / total,
                        "Average Output Tokens": sum(self._get_record_output_tokens(r) for r in sorted_results) / total,
                    }
                }
            )
        with open(self.generated_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        self.checkpoint_manager.delete_checkpoint()

    def run_evaluation(self):
        """Step 4: Evaluation"""
        self.logger.info(">>> Stage: Evaluation")

        if not os.path.exists(self.generated_file):
            self.logger.error("Generated answers file not found.")
            return

        with open(self.generated_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data.get("results", [])

        # Recompute generation-stage efficiency metrics from generated answers file.
        # This keeps report consistent even if generation was resumed/partially updated.
        total_items = len(items)
        if total_items > 0:
            avg_latency = sum((i.get("retrieval", {}) or {}).get("latency_sec", 0) for i in items) / total_items
            avg_in_tokens = (
                sum(self._get_record_input_tokens(i) for i in items) / total_items
            )
            avg_out_tokens = (
                sum(self._get_record_output_tokens(i) for i in items) / total_items
            )
            self._update_report(
                {
                    "Query Efficiency (Average Per Query)": {
                        "Average Retrieval Time (s)": avg_latency,
                        "Average Input Tokens": avg_in_tokens,
                        "Average Output Tokens": avg_out_tokens,
                    }
                }
            )

        eval_items = items
        eval_results_map = {}

        completed_eval_tasks: Set[int] = set()
        if self.resume:
            completed_eval_tasks = self.checkpoint_manager.get_completed_tasks("evaluation")
            if completed_eval_tasks:
                self.logger.info(f"Resuming from checkpoint. {len(completed_eval_tasks)} evaluations already completed.")
                if os.path.exists(self.eval_file):
                    try:
                        with open(self.eval_file, "r", encoding="utf-8") as f:
                            saved_eval_data = json.load(f)
                        for result in saved_eval_data.get("results", []):
                            eval_results_map[result["_global_index"]] = result
                    except Exception as e:
                        self.logger.warning(f"Failed to load previous eval results, continuing fresh: {e}")

        remaining_eval_items = [item for item in eval_items if item["_global_index"] not in completed_eval_tasks]
        self.logger.info(f"Total evaluations: {len(eval_items)}, Remaining: {len(remaining_eval_items)}")
        
        if remaining_eval_items:
            initial_completed_eval = len(completed_eval_tasks)
            with ThreadPoolExecutor(max_workers=self.config['execution']['max_workers']) as executor:
                future_to_item = {
                    executor.submit(self._process_evaluation_task, item): item
                    for item in remaining_eval_items
                }

                pbar = tqdm(total=len(eval_items), desc="Evaluating", unit="item", initial=len(completed_eval_tasks))
                for future in as_completed(future_to_item):
                    try:
                        res = future.result()
                        eval_results_map[res['_global_index']] = res
                        completed_eval_tasks.add(res['_global_index'])

                        newly_completed_eval = len(completed_eval_tasks) - initial_completed_eval
                        if newly_completed_eval % self.save_frequency == 0 or len(completed_eval_tasks) == len(eval_items):
                            self.checkpoint_manager.update_completed_tasks("evaluation", completed_eval_tasks, len(eval_items))
                            self._save_partial_eval_results(eval_results_map)
                    except Exception as e:
                        self.logger.error(f"Evaluation failed: {e}")
                    pbar.update(1)
                pbar.close()
        else:
            self.logger.info("All evaluations already completed!")

        eval_records = list(eval_results_map.values())
        total = len(eval_records)

        with open(self.eval_file, "w", encoding="utf-8") as f:
            json.dump({"results": eval_records}, f, indent=2, ensure_ascii=False)

        if total > 0:
            performance_metrics = {
                "Average F1 Score": sum(r['metrics']['F1'] for r in eval_records) / total,
                "Average Recall": sum(r['metrics']['Recall'] for r in eval_records) / total,
                "Average Accuracy (Hit 0-4)": sum(r['metrics']['Accuracy'] for r in eval_records) / total,
                "Average Accuracy (normalization)": (sum(r['metrics']['Accuracy'] for r in eval_records) / total)/4,
            }
            phase1_metric_records = [
                r for r in eval_records
                if "Phase1 Accuracy" in (r.get("metrics", {}) or {})
            ]
            if phase1_metric_records:
                performance_metrics["Average Phase1 F1 Score"] = (
                    sum(float((r.get("metrics", {}) or {}).get("Phase1 F1", 0.0) or 0.0) for r in phase1_metric_records)
                    / len(phase1_metric_records)
                )
                performance_metrics["Average Phase1 Accuracy (Hit 0-4)"] = (
                    sum(float((r.get("metrics", {}) or {}).get("Phase1 Accuracy", 0.0) or 0.0) for r in phase1_metric_records)
                    / len(phase1_metric_records)
                )
                performance_metrics["Average Phase1 Accuracy (normalization)"] = (
                    performance_metrics["Average Phase1 Accuracy (Hit 0-4)"] / 4
                )
            self._update_report({
                "Dataset": self.config.get('dataset_name', 'Unknown_Dataset'),
                "Total Queries Evaluated": total,
                "Performance Metrics": performance_metrics,
            })

            # Relations Usage report from vikingbot records
            vb_records = [r for r in eval_records if 'vikingbot' in r]
            if vb_records:
                relations_hits_list = [r['vikingbot'].get('relations_hits', 0) for r in vb_records]
                total_relations_list = [r['vikingbot'].get('total_relations_found', 0) for r in vb_records]
                report_relations = {
                    "Total Questions with Relations Hits": sum(1 for h in relations_hits_list if h > 0),
                    "Total Relations Found": sum(total_relations_list),
                    "Average Relations Found per Query": sum(total_relations_list) / len(total_relations_list),
                    "Relations Utilization Rate": sum(1 for h in relations_hits_list if h > 0) / len(relations_hits_list),
                }

                # Link Construction Rate
                enable_linking = self.config.get('vikingbot', {}).get('enable_linking', False)
                if enable_linking:
                    queries_with_links = sum(
                        1 for r in vb_records
                        if int(r['vikingbot'].get('link_pairs_created', r['vikingbot'].get('links_created', 0)) or 0) > 0
                    )
                    total_links = sum(
                        int(r['vikingbot'].get('link_pairs_created', r['vikingbot'].get('links_created', 0)) or 0)
                        for r in vb_records
                    )
                    total_link_pairs_requested = sum(
                        int(r['vikingbot'].get('link_pairs_requested', r['vikingbot'].get('links_created', 0)) or 0)
                        for r in vb_records
                    )
                    total_skipped_relation_to_uris = sum(
                        int(r['vikingbot'].get('skipped_relation_to_uri_count', 0) or 0)
                        for r in vb_records
                    )
                    queries_with_skipped_relation_to_uris = sum(
                        1 for r in vb_records
                        if int(r['vikingbot'].get('skipped_relation_to_uri_count', 0) or 0) > 0
                    )
                    total_link_tool_calls = sum(
                        int(r['vikingbot'].get('link_tool_calls', 0) or 0)
                        for r in vb_records
                    )
                    queries_with_link_tool_calls = sum(
                        1 for r in vb_records
                        if int(r['vikingbot'].get('link_tool_calls', 0) or 0) > 0
                    )
                    report_relations["Link Construction Rate"] = queries_with_links / len(vb_records)
                    report_relations["Queries With Links Created"] = queries_with_links
                    report_relations["Queries Without Links"] = len(vb_records) - queries_with_links
                    report_relations["Total Links Created"] = total_links
                    report_relations["Total Link Pairs Requested"] = total_link_pairs_requested
                    report_relations["Total Relation-derived ToURIs Skipped"] = total_skipped_relation_to_uris
                    report_relations["Queries With Relation-derived ToURIs Skipped"] = queries_with_skipped_relation_to_uris
                    report_relations["Total Link Tool Calls"] = total_link_tool_calls
                    report_relations["Queries With Link Tool Calls"] = queries_with_link_tool_calls

                    build_link_report = {
                        "Total Queries": len(vb_records),
                        "Queries With Relations Found": sum(1 for h in relations_hits_list if h > 0),
                        "Queries Without Relations Found": len(vb_records) - sum(1 for h in relations_hits_list if h > 0),
                        "Relations Search Rate": sum(1 for h in relations_hits_list if h > 0) / len(vb_records),
                        "Total Relation Hit Search Calls": sum(
                            int(r['vikingbot'].get('relation_hit_search_calls', r['vikingbot'].get('relations_hits', 0)) or 0)
                            for r in vb_records
                        ),
                        "Total Relations Found During Search": sum(total_relations_list),
                        "Average Relations Found per Query": sum(total_relations_list) / len(vb_records),
                        "Relations Found Per Query": [
                            int(r['vikingbot'].get('total_relations_found', 0) or 0)
                            for r in vb_records
                        ],
                        "Relations Found Per Search": [
                            r['vikingbot'].get('relations_found_per_search', [])
                            for r in vb_records
                        ],
                        "Total Link Tool Calls": total_link_tool_calls,
                        "Queries With Link Tool Calls": queries_with_link_tool_calls,
                        "Queries With Links Created": queries_with_links,
                        "Queries Without Links Created": len(vb_records) - queries_with_links,
                        "Link Construction Rate": queries_with_links / len(vb_records),
                        "Total Link Pairs Requested": total_link_pairs_requested,
                        "Total Links Created": total_links,
                        "Queries With Relation-derived ToURIs Skipped": queries_with_skipped_relation_to_uris,
                        "Total Relation-derived ToURIs Skipped": total_skipped_relation_to_uris,
                        "Average Relation-derived ToURIs Skipped per Query": total_skipped_relation_to_uris / len(vb_records),
                        "Relation-derived ToURI Skip Rate": queries_with_skipped_relation_to_uris / len(vb_records),
                    }
                    self._update_report({"Build Link Diagnostics": build_link_report})

                # Edge Coverage Rate
                use_relations = self.config.get('vikingbot', {}).get('use_relations', False)
                if use_relations:
                    strategy = self.config.get('vikingbot', {}).get('link_strategy', 'llm_review')
                    total_edges_count, all_edges = self._count_total_relations(strategy)
                    hit_edges = set()
                    for r in vb_records:
                        tc_list = r['vikingbot'].get('tool_calls', [])
                        if isinstance(tc_list, str):
                            try:
                                tc_list = json.loads(tc_list)
                            except (json.JSONDecodeError, TypeError):
                                tc_list = []
                        for tc in (tc_list if isinstance(tc_list, list) else []):
                            if not isinstance(tc, dict) or tc.get('tool_name') != 'openviking_search':
                                continue
                            result_data = tc.get('result')
                            if not isinstance(result_data, list):
                                continue
                            for item in result_data:
                                if not isinstance(item, dict):
                                    continue
                                mr = item.get('match_reason', '')
                                if mr.startswith('relation_from:'):
                                    src = mr.replace('relation_from:', '').strip()
                                    tgt = item.get('uri', '')
                                    if src and tgt:
                                        hit_edges.add((src, tgt))
                    hit_count = len(hit_edges & all_edges) if all_edges else 0
                    coverage = hit_count / total_edges_count if total_edges_count > 0 else 0.0
                    report_relations["Total Edges in Store"] = total_edges_count
                    report_relations["Unique Edges Hit"] = hit_count
                    report_relations["Edge Coverage Rate"] = round(coverage, 4)

                self._update_report({"Relations Usage": report_relations})

                # VikingBot Iteration Metrics
                vb_valid = [r for r in vb_records if r['vikingbot'].get('tool_calls')]
                records_for_iters = vb_valid if vb_valid else vb_records

                iters_total = [r['vikingbot'].get('iterations_used', 0) for r in records_for_iters]
                iters_search = [r['vikingbot'].get('search_iterations', 0) for r in records_for_iters]
                iters_read = [r['vikingbot'].get('read_iterations', 0) for r in records_for_iters]
                iters_retrieval = [r['vikingbot'].get('retrieval_iterations', r['vikingbot'].get('iterations_used', 0)) for r in records_for_iters]

                if records_for_iters:
                    self._update_report({
                        "VikingBot Iteration Metrics": {
                            "Average Total Iterations": sum(iters_total) / len(iters_total),
                            "Average Retrieval Iterations (excl. link/relations)": sum(iters_retrieval) / len(iters_retrieval),
                            "Average Search Iterations": sum(iters_search) / len(iters_search),
                            "Average Read Iterations": sum(iters_read) / len(iters_read),
                            "Min Retrieval Iterations": min(iters_retrieval),
                            "Max Retrieval Iterations": max(iters_retrieval),
                            "Excluded Anomalous Records (tc=0)": len(vb_records) - len(vb_valid),
                        }
                    })

            # Fallback judgment summaries
            fallback_records = [r for r in eval_records if 'fallback' in r]
            if fallback_records:
                fallback_mode_stats = self._fallback_mode_statistics(fallback_records)
                phase1_evidence_efficiency = self._summarize_phase1_evidence_efficiency(
                    fallback_records
                )
                fallback_report_update = {
                    "__delete_keys__": [
                        "Phase1 Fallback Judgment Metrics",
                        "Phase1 Provider Judgment Metrics",
                        "Phase1 Provider Judgment Comparison",
                        "Phase1 Provider Cost",
                        "Naive Phase1 Fallback Judgment Metrics",
                        "Naive Phase1 Result Fallback Judgment Metrics",
                        "Phase1 Fallback Judgment Comparison",
                        "Phase1 Two-Stage Evidence Metrics",
                        "Phase1 Evidence Efficiency",
                    ],
                    "Fallback Judgment (whether the fallback decision matches phase1 answer quality)": (
                        fallback_judgment_summary(fallback_records)
                    ),
                    "Fallback Miss (bad phase1 answers that were not sent to fallback)": (
                        fallback_miss_summary(fallback_records)
                    ),
                    "Recoverable Miss (missed fallback cases where shadow bot would have corrected phase1)": (
                        recoverable_miss_summary(fallback_records)
                    ),
                    "Fallback Judgment By Accuracy Gain (whether fallback decision matches bot improvement)": (
                        fallback_judgment_gain_summary(fallback_records, logger=self.logger)
                    ),
                    "Fallback Miss By Accuracy Gain (missed cases where bot would improve phase1)": (
                        fallback_miss_gain_summary(fallback_records)
                    ),
                    "Recoverable Miss By Accuracy Gain (missed fallback cases where shadow bot improved phase1)": (
                        recoverable_miss_gain_summary(fallback_records)
                    ),
                    "Fallback Mode Statistics": fallback_mode_stats,
                }
                if phase1_evidence_efficiency:
                    fallback_report_update["Phase1 Evidence Efficiency"] = phase1_evidence_efficiency
                self._update_report(fallback_report_update)
        self.checkpoint_manager.delete_checkpoint()

    def _fallback_mode_statistics(self, fallback_records):
        """Keep the legacy fallback-mode summary alongside provider judgment metrics."""
        total = len(fallback_records)
        triggered = [
            row for row in fallback_records
            if bool((row.get("fallback", {}) or {}).get("triggered"))
        ]
        not_triggered = [row for row in fallback_records if row not in triggered]

        def avg(rows, getter):
            if not rows:
                return 0.0
            return sum(float(getter(row) or 0.0) for row in rows) / len(rows)

        def metric(row, key):
            return (row.get("metrics", {}) or {}).get(key, 0)

        def fb(row, key):
            return (row.get("fallback", {}) or {}).get(key, 0)

        return {
            "Total Queries": total,
            "Fallback Trigger Rate": (len(triggered) / total) if total else 0.0,
            "Triggered Count": len(triggered),
            "Not Triggered Count": len(not_triggered),
            "Average Total Latency (s)": avg(
                fallback_records,
                lambda row: fb(row, "total_latency_sec") or (row.get("retrieval", {}) or {}).get("latency_sec"),
            ),
            "Average Total Input Tokens": avg(
                fallback_records,
                lambda row: fb(row, "total_input_tokens") or (row.get("token_usage", {}) or {}).get("total_input_tokens"),
            ),
            "Average Total Output Tokens": avg(
                fallback_records,
                lambda row: fb(row, "total_output_tokens") or (row.get("token_usage", {}) or {}).get("llm_output_tokens"),
            ),
            "Average OV Retrieval Latency (s)": avg(
                fallback_records,
                lambda row: fb(row, "ov_retrieval_sec") or (row.get("retrieval", {}) or {}).get("latency_sec"),
            ),
            "Average Bot Latency (triggered) (s)": avg(
                triggered,
                lambda row: fb(row, "bot_latency_sec"),
            ),
            "Average Bot Tokens (triggered)": avg(
                triggered,
                lambda row: int(fb(row, "bot_input_tokens") or 0) + int(fb(row, "bot_output_tokens") or 0),
            ),
            "Average Accuracy (triggered)": avg(triggered, lambda row: metric(row, "Accuracy")),
            "Average Accuracy (not triggered)": avg(not_triggered, lambda row: metric(row, "Accuracy")),
            "Average Accuracy (overall)": avg(fallback_records, lambda row: metric(row, "Accuracy")),
        }

    def run_deletion(self):
        """Step 5: Cleanup"""
        self.logger.info(">>> Stage: Deletion")
        if not self.db:
            self.logger.info("No vector store to delete (nanobot mode)")
            return
        start_time = time.time()
        self.db.clear()
        duration = time.time() - start_time
        self.metrics_summary["deletion"] = {"time": duration, "input_tokens": 0, "output_tokens": 0}
        self.logger.info(f"Deletion finished. Time: {duration:.2f}s")

        self._update_report({
            "Deletion Efficiency (Total Dataset)": {
                "Total Deletion Time (s)": duration,
                "Total Input Tokens": 0,
                "Total Output Tokens": 0
            }
        })

    def _prepare_tasks(self, samples):
        tasks = []
        global_idx = 0
        max_queries = self.config['execution'].get('max_queries')
        env_max = os.environ.get("RAG_MAX_QUERIES")
        if env_max is not None:
            env_max_stripped = env_max.strip().lower()
            if env_max_stripped in ("", "null", "none"):
                max_queries = None
            else:
                max_queries = int(env_max_stripped)
        if max_queries is not None and self._is_question_rewrite_enabled():
            original_max_queries = max_queries
            max_queries = max_queries * 2
            self.logger.info(
                f"Question rewrite enabled: max_queries={original_max_queries} original QA(s) "
                f"-> {max_queries} rewritten task(s)"
            )
        for sample in samples:
            for qa in sample.qa_pairs:
                if max_queries is not None and global_idx >= max_queries:
                    break
                tasks.append({"id": global_idx, "sample_id": sample.sample_id, "qa": qa})
                global_idx += 1
            if max_queries is not None and global_idx >= max_queries:
                break
        return tasks

    def _apply_question_rewrites_to_samples(self, samples):
        if not self._is_question_rewrite_enabled():
            return samples

        store = QuestionRewriteStore(
            self._question_rewrite_cache_dir(),
            dataset_name=self.config.get("dataset_name", ""),
        )
        model = str(self.config.get("llm", {}).get("model", ""))
        max_attempts = self._question_rewrite_max_attempts()
        max_workers = self._question_rewrite_max_workers()

        rewrite_by_id: dict[str, list[str]] = {}
        missing_by_id: dict[str, tuple[str, str]] = {}
        total_original = 0
        for sample in samples:
            for qa in sample.qa_pairs:
                total_original += 1
                rec_id = store.compute_id(sample.sample_id, qa.question)
                cached = store.get(sample.sample_id, qa.question)
                if cached and len(cached) == 2:
                    rewrite_by_id[rec_id] = cached
                elif rec_id not in missing_by_id:
                    missing_by_id[rec_id] = (sample.sample_id, qa.question)

        if missing_by_id:
            worker_count = max(1, min(max_workers, len(missing_by_id)))
            self.logger.info(
                f"Question rewrite cache miss: {len(missing_by_id)} unique QA(s). "
                f"Generating concurrently with {worker_count} worker(s), "
                f"max_attempts={max_attempts}. Cache: {self._question_rewrite_cache_dir()}"
            )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_id = {
                    executor.submit(
                        get_or_create_rewrites,
                        store,
                        sample_id,
                        question,
                        self.llm,
                        model,
                        max_attempts,
                    ): rec_id
                    for rec_id, (sample_id, question) in missing_by_id.items()
                }
                completed = 0
                for future in as_completed(future_to_id):
                    rec_id = future_to_id[future]
                    sample_id, question = missing_by_id[rec_id]
                    try:
                        rewrite_by_id[rec_id] = future.result()
                    except Exception as e:
                        raise QuestionRewriteError(
                            f"Failed to generate rewrites for sample_id={sample_id}, "
                            f"question={question!r}: {e}"
                        ) from e
                    completed += 1
                    if completed % 20 == 0 or completed == len(future_to_id):
                        self.logger.info(
                            f"Question rewrite generation progress: "
                            f"{completed}/{len(future_to_id)} completed"
                        )

        rewritten_samples = []
        total_rewritten = 0

        for sample in samples:
            rewritten_qas = []
            for qa_index, qa in enumerate(sample.qa_pairs):
                rec_id = store.compute_id(sample.sample_id, qa.question)
                rewrites = rewrite_by_id.get(rec_id)
                if not rewrites:
                    rewrites = get_or_create_rewrites(
                        store,
                        sample.sample_id,
                        qa.question,
                        self.llm,
                        model=model,
                        max_attempts=max_attempts,
                    )

                for rewrite_index, rewritten_question in enumerate(rewrites, 1):
                    metadata = dict(qa.metadata or {})
                    metadata.update({
                        "original_question": qa.question,
                        "rewrite_index": rewrite_index,
                        "rewrite_source_sample_id": sample.sample_id,
                        "rewrite_source_qa_index": qa_index,
                    })
                    rewritten_qas.append(StandardQA(
                        question=rewritten_question,
                        gold_answers=list(qa.gold_answers or []),
                        evidence=list(qa.evidence or []),
                        category=qa.category,
                        metadata=metadata,
                    ))
                    total_rewritten += 1

            sample_metadata = dict(sample.metadata or {})
            sample_metadata.update({
                "question_rewrites_enabled": True,
                "original_num_questions": len(sample.qa_pairs),
                "rewritten_num_questions": len(rewritten_qas),
            })
            rewritten_samples.append(StandardSample(
                sample_id=sample.sample_id,
                qa_pairs=rewritten_qas,
                metadata=sample_metadata,
            ))

        self.logger.info(
            f"Question rewrite expansion: {total_original} original QA(s) -> "
            f"{total_rewritten} rewritten QA task(s). Cache: {self._question_rewrite_cache_dir()}"
        )
        return rewritten_samples

    def _question_rewrite_max_workers(self) -> int:
        rewrite_config = self.config.get("question_rewrites", {})
        value = None
        if isinstance(rewrite_config, dict):
            value = rewrite_config.get("max_workers")
        if value is None:
            value = self.config.get("execution", {}).get("rewrite_workers")
        if value is None:
            value = self.config.get("execution", {}).get("max_workers", 8)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 8

    def _question_rewrite_max_attempts(self) -> int:
        rewrite_config = self.config.get("question_rewrites", {})
        value = rewrite_config.get("max_attempts", 3) if isinstance(rewrite_config, dict) else 3
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 3

    def _is_build_link_mode(self) -> bool:
        return bool(self.config.get("vikingbot", {}).get("enable_linking", False))

    def _is_question_rewrite_enabled(self) -> bool:
        rewrite_config = self.config.get("question_rewrites", None)

        if isinstance(rewrite_config, dict) and "enabled" in rewrite_config:
            return bool(rewrite_config.get("enabled", False))

        if isinstance(rewrite_config, bool):
            return rewrite_config

        return False

    def _question_rewrite_cache_dir(self) -> str:
        dataset_name = str(self.config.get("dataset_name", "") or "default")
        rewrite_config = self.config.get("question_rewrites", {})
        rag_root = Path(__file__).resolve().parents[1]

        def resolve_cache_path(path_value: str) -> Path:
            path = Path(str(path_value)).expanduser()
            if not path.is_absolute():
                path = rag_root / path
            return path

        if isinstance(rewrite_config, dict):
            cache_dir = rewrite_config.get("cache_dir")
            if cache_dir:
                return str(resolve_cache_path(cache_dir))

            cache_root = rewrite_config.get("cache_root")
            if cache_root:
                return str(resolve_cache_path(cache_root) / dataset_name)

        return str(rag_root / "rewrites" / dataset_name)

    def _save_bot_trace(self, trace: str, query_id: int, suffix: str = "") -> str:
        if not trace:
            return ""
        trace_dir = os.path.join(self.output_dir, "traces")
        os.makedirs(trace_dir, exist_ok=True)
        suffix_part = f"_{suffix}" if suffix else ""
        trace_file = os.path.join(trace_dir, f"query_{query_id}{suffix_part}_trace.txt")
        try:
            trace_data = json.loads(trace, strict=False)
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, ensure_ascii=False, indent=2, default=str)
        except json.JSONDecodeError:
            with open(trace_file, "w", encoding="utf-8") as f:
                f.write(trace)
        return trace_file

    def _summarize_vikingbot_result(self, vikingbot_result: dict, query_id: int, trace_suffix: str = "") -> dict:
        token_usage = vikingbot_result.get("token_usage", {}) or {}
        prompt_tokens = int(
            token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)) or 0
        )
        completion_tokens = int(
            token_usage.get("completion_tokens", token_usage.get("output_tokens", 0)) or 0
        )
        total_tokens = int(token_usage.get("total_tokens") or (prompt_tokens + completion_tokens))

        tools_used_raw = vikingbot_result.get("tools_used", [])
        tc_list = tools_used_raw if isinstance(tools_used_raw, list) else []
        if isinstance(tools_used_raw, str):
            try:
                tc_list = json.loads(tools_used_raw)
            except (json.JSONDecodeError, TypeError):
                tc_list = []
        tc_list = self._normalize_relation_tool_calls(tc_list)

        search_iterations = 0
        read_iterations = 0
        relations_hits = 0
        total_relations_found = 0
        relation_edges_hit = []
        read_tool_names = {"openviking_multi_read", "openviking_read"}
        for tc in tc_list:
            if not isinstance(tc, dict):
                continue
            tn = tc.get('tool_name', '')
            if tn == 'openviking_search':
                search_iterations += 1
                rf = int(tc.get('relations_found', 0) or 0)
                result_relation_count = 0
                result_data = tc.get('result')
                if isinstance(result_data, list):
                    for item in result_data:
                        if not isinstance(item, dict):
                            continue
                        mr = item.get('match_reason', '')
                        if mr.startswith('relation_from:'):
                            result_relation_count += 1
                            src = mr.replace('relation_from:', '').strip()
                            tgt = item.get('uri', '')
                            if src and tgt:
                                relation_edges_hit.append((src, tgt))
                if rf <= 0:
                    rf = result_relation_count
                total_relations_found += rf
                if rf > 0:
                    relations_hits += 1
            elif tn in read_tool_names:
                read_iterations += 1

        build_link_stats = self._extract_build_link_stats_from_tool_calls(tc_list)
        links_created = int(build_link_stats.get("link_pairs_created", 0) or 0)

        iterations_used = int(vikingbot_result.get("iterations_used", 0) or 0)
        link_tools = {'openviking_link', 'openviking_relations'}
        total_calls = len(tc_list) if tc_list else 1
        non_link_call_count = sum(
            1 for tc in tc_list
            if isinstance(tc, dict) and tc.get('tool_name', '') not in link_tools
        )
        retrieval_iterations = (
            max(1, round(iterations_used * non_link_call_count / total_calls))
            if iterations_used > 0 else iterations_used
        )
        trace_file = self._save_bot_trace(
            vikingbot_result.get("trace", ""),
            query_id,
            suffix=trace_suffix,
        )

        return {
            "answer": vikingbot_result.get("answer", ""),
            "total_time_sec": float(vikingbot_result.get("total_time_sec", 0) or 0),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tools_used_names": vikingbot_result.get("tools_used_names", []),
            "tool_calls": tc_list,
            "iterations_used": iterations_used,
            "retrieval_iterations": retrieval_iterations,
            "search_iterations": search_iterations,
            "read_iterations": read_iterations,
            "debug_log": vikingbot_result.get("debug_log", ""),
            "session_id": vikingbot_result.get("session_id", ""),
            "trace_file": trace_file,
            "relations_hits": relations_hits,
            "total_relations_found": total_relations_found,
            "links_created": links_created,
            "has_relations_found": bool(build_link_stats.get("has_relations_found", False)),
            "relations_found_per_search": build_link_stats.get("relations_found_per_search", []),
            "relation_search_calls": build_link_stats.get("relation_search_calls", 0),
            "relation_hit_search_calls": build_link_stats.get("relation_hit_search_calls", 0),
            "link_tool_calls": build_link_stats.get("link_tool_calls", 0),
            "link_pairs_requested": build_link_stats.get("link_pairs_requested", 0),
            "link_pairs_created": build_link_stats.get("link_pairs_created", 0),
            "has_links_created": bool(build_link_stats.get("has_links_created", False)),
            "skipped_relation_to_uris": build_link_stats.get("skipped_relation_to_uris", []),
            "skipped_relation_to_uri_count": build_link_stats.get("skipped_relation_to_uri_count", 0),
            "has_relation_to_uri_skipped": bool(build_link_stats.get("has_relation_to_uri_skipped", False)),
            "build_link_stats": build_link_stats,
            "relation_edges_hit": relation_edges_hit,
        }

    def _merge_vikingbot_summaries(self, summaries: list[dict]) -> dict:
        if not summaries:
            return {}
        merged_tool_calls = []
        merged_tool_names = []
        relation_edges_hit = []
        skipped_relation_to_uris = []
        relations_found_per_search = []
        for s in summaries:
            merged_tool_calls.extend(s.get("tool_calls", []) or [])
            merged_tool_names.extend(s.get("tools_used_names", []) or [])
            relation_edges_hit.extend(s.get("relation_edges_hit", []) or [])
            skipped_relation_to_uris.extend(s.get("skipped_relation_to_uris", []) or [])
            relations_found_per_search.extend(s.get("relations_found_per_search", []) or [])

        merged_build_link_stats = {
            "relation_search_calls": sum(int(s.get("relation_search_calls", 0) or 0) for s in summaries),
            "relation_hit_search_calls": sum(int(s.get("relation_hit_search_calls", 0) or 0) for s in summaries),
            "relations_found_per_search": relations_found_per_search,
            "relations_found_total": sum(int(s.get("total_relations_found", 0) or 0) for s in summaries),
            "has_relations_found": any(bool(s.get("has_relations_found", False)) for s in summaries),
            "link_tool_calls": sum(int(s.get("link_tool_calls", 0) or 0) for s in summaries),
            "link_pairs_requested": sum(int(s.get("link_pairs_requested", 0) or 0) for s in summaries),
            "link_pairs_created": sum(int(s.get("link_pairs_created", s.get("links_created", 0)) or 0) for s in summaries),
            "has_links_created": any(bool(s.get("has_links_created", False)) for s in summaries),
            "skipped_relation_to_uris": self._dedupe_list(skipped_relation_to_uris),
            "skipped_relation_to_uri_count": len(self._dedupe_list(skipped_relation_to_uris)),
            "has_relation_to_uri_skipped": any(bool(s.get("has_relation_to_uri_skipped", False)) for s in summaries),
        }

        return {
            "iterations_used": sum(int(s.get("iterations_used", 0) or 0) for s in summaries),
            "retrieval_iterations": sum(int(s.get("retrieval_iterations", 0) or 0) for s in summaries),
            "search_iterations": sum(int(s.get("search_iterations", 0) or 0) for s in summaries),
            "read_iterations": sum(int(s.get("read_iterations", 0) or 0) for s in summaries),
            "tools_used_names": merged_tool_names,
            "tool_calls": merged_tool_calls,
            "total_time_sec": sum(float(s.get("total_time_sec", 0) or 0) for s in summaries),
            "debug_log": summaries[0].get("debug_log", ""),
            "session_id": summaries[0].get("session_id", ""),
            "trace_file": summaries[0].get("trace_file", ""),
            "relations_hits": sum(int(s.get("relations_hits", 0) or 0) for s in summaries),
            "total_relations_found": sum(int(s.get("total_relations_found", 0) or 0) for s in summaries),
            "links_created": merged_build_link_stats["link_pairs_created"],
            "has_relations_found": bool(merged_build_link_stats["has_relations_found"]),
            "relations_found_per_search": relations_found_per_search,
            "relation_search_calls": merged_build_link_stats["relation_search_calls"],
            "relation_hit_search_calls": merged_build_link_stats["relation_hit_search_calls"],
            "link_tool_calls": merged_build_link_stats["link_tool_calls"],
            "link_pairs_requested": merged_build_link_stats["link_pairs_requested"],
            "link_pairs_created": merged_build_link_stats["link_pairs_created"],
            "has_links_created": bool(merged_build_link_stats["has_links_created"]),
            "skipped_relation_to_uris": merged_build_link_stats["skipped_relation_to_uris"],
            "skipped_relation_to_uri_count": merged_build_link_stats["skipped_relation_to_uri_count"],
            "has_relation_to_uri_skipped": bool(merged_build_link_stats["has_relation_to_uri_skipped"]),
            "build_link_stats": merged_build_link_stats,
            "relation_edges_hit": relation_edges_hit,
            "runs": summaries,
        }

    def _get_question_rewrites(self, sample_id: str, question: str) -> list[str]:
        store = QuestionRewriteStore(
            self._question_rewrite_cache_dir(),
            dataset_name=self.config.get("dataset_name", ""),
        )
        model = str(self.config.get("llm", {}).get("model", ""))
        return get_or_create_rewrites(
            store,
            sample_id,
            question,
            self.llm,
            model=model,
            max_attempts=self._question_rewrite_max_attempts(),
        )

    def _process_vikingbot_task(self, task):
        self.monitor.worker_start()
        try:
            qa = task['qa']
            self.logger.info(f"[Query-{task['id']}] Using VikingBot for agentic RAG")

            restrict_to_qa_doc = bool(self.config.get("execution", {}).get("restrict_to_qa_doc", False))
            allowed_target_uris = self._resolve_target_uris(task, qa) if restrict_to_qa_doc else None

            build_link_mode = self._is_build_link_mode()
            rewrite_enabled = self._is_question_rewrite_enabled()
            rewrite_index = int((qa.metadata or {}).get("rewrite_index", 0) or 0)
            trace_suffix = f"rewrite_{rewrite_index}" if rewrite_index else ""
            session_suffix = trace_suffix or "main"
            session_id = f"query_{uuid.uuid4().hex}_{session_suffix}"
            vikingbot_result = run_vikingbot_query(
                question=qa.question,
                config=self.config,
                session_id=session_id,
                allowed_target_uris=allowed_target_uris,
            )
            merged = self._summarize_vikingbot_result(
                vikingbot_result,
                task["id"],
                trace_suffix=trace_suffix,
            )
            ans = merged.get("answer", "")
            prompt_tokens = int(merged.get("prompt_tokens", 0) or 0)
            completion_tokens = int(merged.get("completion_tokens", 0) or 0)
            total_tokens = int(merged.get("total_tokens", 0) or 0)

            self.monitor.worker_end(tokens=prompt_tokens + completion_tokens)
            self.logger.info(
                f"[Query-{task['id']}] VikingBot | BuildLink={build_link_mode} | "
                f"QuestionRewrites={rewrite_enabled} | "
                f"RewriteIndex={rewrite_index} | Iterations: {merged.get('iterations_used', 0)} | "
                f"Time: {merged.get('total_time_sec', 0):.1f}s"
            )

            return {
                "_global_index": task['id'], "sample_id": task['sample_id'], "question": qa.question,
                "gold_answers": qa.gold_answers, "category": str(qa.category), "evidence": qa.evidence,
                "retrieval": {"latency_sec": merged.get("total_time_sec", 0), "uris": []},
                "llm": {"final_answer": ans},
                "vikingbot": {
                    "iterations_used": merged.get("iterations_used", 0),
                    "retrieval_iterations": merged.get("retrieval_iterations", 0),
                    "search_iterations": merged.get("search_iterations", 0),
                    "read_iterations": merged.get("read_iterations", 0),
                    "tools_used_names": merged.get("tools_used_names", []),
                    "tool_calls": merged.get("tool_calls", []),
                    "total_time_sec": merged.get("total_time_sec", 0),
                    "debug_log": merged.get("debug_log", ""),
                    "session_id": merged.get("session_id", ""),
                    "trace_file": merged.get("trace_file", ""),
                    "relations_hits": merged.get("relations_hits", 0),
                    "total_relations_found": merged.get("total_relations_found", 0),
                    "links_created": merged.get("links_created", 0),
                    "has_relations_found": merged.get("has_relations_found", False),
                    "relations_found_per_search": merged.get("relations_found_per_search", []),
                    "relation_search_calls": merged.get("relation_search_calls", 0),
                    "relation_hit_search_calls": merged.get("relation_hit_search_calls", 0),
                    "link_tool_calls": merged.get("link_tool_calls", 0),
                    "link_pairs_requested": merged.get("link_pairs_requested", 0),
                    "link_pairs_created": merged.get("link_pairs_created", merged.get("links_created", 0)),
                    "has_links_created": merged.get("has_links_created", False),
                    "skipped_relation_to_uris": merged.get("skipped_relation_to_uris", []),
                    "skipped_relation_to_uri_count": merged.get("skipped_relation_to_uri_count", 0),
                    "has_relation_to_uri_skipped": merged.get("has_relation_to_uri_skipped", False),
                    "build_link_stats": merged.get("build_link_stats", {}),
                    "relation_edges_hit": merged.get("relation_edges_hit", []),
                    "question_rewrites_enabled": rewrite_enabled,
                    "build_link_rewrites_enabled": rewrite_enabled,
                    "question_rewrite_cache_dir": self._question_rewrite_cache_dir(),
                    "original_question": (qa.metadata or {}).get("original_question", qa.question),
                    "rewrite_index": rewrite_index,
                    "rewrite_source_sample_id": (qa.metadata or {}).get("rewrite_source_sample_id", ""),
                    "rewrite_source_qa_index": (qa.metadata or {}).get("rewrite_source_qa_index", ""),
                },
                "metrics": {"Recall": 0.0},
                "token_usage": {
                    # Keep legacy simple-RAG fields present for schema compatibility,
                    # but set them to 0 for bot mode to avoid mixing semantics.
                    "total_input_tokens": 0,
                    "llm_output_tokens": 0,
                    "retrieval_embedding_tokens": 0,
                    # VikingBot-native names aligned with vikingbot JSON output.
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
        except Exception:
            self.monitor.worker_end(success=False)
            raise

    def _resolve_target_uris(self, task, qa):
        evidence = getattr(qa, 'evidence', []) or []
        if not evidence:
            return None
        uris = []
        for ev in evidence:
            if isinstance(ev, str) and ev.startswith("viking://"):
                parts = ev.rstrip("/").split("/")
                if len(parts) >= 5:
                    uris.append("/".join(parts[:5]))
        return list(set(uris)) if uris else None

    def _process_nanobot_task(self, task):
        self.monitor.worker_start()
        try:
            qa = task['qa']
            self.logger.info(f"[Query-{task['id']}] Using Nanobot (grep/glob) for RAG")

            session_id = f"query_{uuid.uuid4().hex}"

            nanobot_result = run_nanobot_query(
                question=qa.question,
                config=self.config,
                session_id=session_id,
            )

            ans = nanobot_result.get("answer", "")
            total_time_sec = nanobot_result.get("total_time_sec", 0)
            token_usage = nanobot_result.get("token_usage", {})
            tools_used_names = nanobot_result.get("tools_used_names", [])
            iterations_used = nanobot_result.get("iterations_used", 0)

            prompt_tokens = int(
                token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)) or 0
            )
            completion_tokens = int(
                token_usage.get("completion_tokens", token_usage.get("output_tokens", 0)) or 0
            )
            total_tokens = int(token_usage.get("total_tokens") or (prompt_tokens + completion_tokens))
            self.monitor.worker_end(tokens=prompt_tokens + completion_tokens)

            self.logger.info(f"[Query-{task['id']}] Nanobot | Time: {total_time_sec:.1f}s")

            return {
                "_global_index": task['id'], "sample_id": task['sample_id'], "question": qa.question,
                "gold_answers": qa.gold_answers, "category": str(qa.category), "evidence": qa.evidence,
                "retrieval": {"latency_sec": total_time_sec, "uris": []},
                "llm": {"final_answer": ans},
                "nanobot": {
                    "iterations_used": iterations_used,
                    "tools_used_names": tools_used_names,
                    "total_time_sec": total_time_sec,
                    "session_id": nanobot_result.get("session_id", ""),
                },
                "metrics": {"Recall": 0.0},
                "token_usage": {
                    "total_input_tokens": 0,
                    "llm_output_tokens": 0,
                    "retrieval_embedding_tokens": 0,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
        except Exception:
            self.monitor.worker_end(success=False)
            raise

    def _process_generation_task(self, task):
        self.monitor.worker_start()
        try:
            qa = task['qa']
            
            t0 = time.time()
            # Get retrieval instruction from config, default to empty
            retrieval_instruction = self.config['execution'].get('retrieval_instruction', '')
            # Build enhanced query with instruction if provided
            if retrieval_instruction:
                enhanced_query = f"{retrieval_instruction} {qa.question}"
                self.logger.debug(f"[Query-{task['id']}] Using retrieval instruction: {retrieval_instruction}")
                self.logger.debug(f"[Query-{task['id']}] Enhanced query: {enhanced_query}")
            else:
                enhanced_query = qa.question
                self.logger.debug(f"[Query-{task['id']}] No retrieval instruction, using raw query")
            search_res = self.db.retrieve(query=enhanced_query, topk=self.config['execution']['retrieval_topk'])
            latency = time.time() - t0

            generation = self._generate_with_optional_supplemental_retrieval(
                qa,
                search_res,
                retrieval_instruction=retrieval_instruction,
            )
            search_res = generation["search_res"]
            recall_texts = search_res["recall_texts"]
            context_blocks = search_res["context_blocks"]
            retrieved_uris = search_res["retrieved_uris"]

            retrieved_texts = list(recall_texts.values())
            recall = MetricsCalculator.check_recall(retrieved_texts, qa.evidence)
            full_prompt = generation["prompt"]
            ans_raw = generation["raw"]
            parsed = generation["parsed"]
            ans = generation["answer"]

            in_tokens = generation["input_tokens"]
            out_tokens = generation["output_tokens"]
            self.monitor.worker_end(tokens=in_tokens + out_tokens)

            self.logger.info(f"[Query-{task['id']}] Q: {qa.question[:30]}... | Recall: {recall:.2f} | Action: {parsed.action} | Sufficient: {parsed.sufficient} | Latency: {latency:.2f}s")

            return {
                "_global_index": task['id'], "sample_id": task['sample_id'], "question": qa.question,
                "gold_answers": qa.gold_answers, "category": str(qa.category), "evidence": qa.evidence,
                "retrieval": {"latency_sec": latency, "uris": retrieved_uris},
                "llm": {
                    "final_answer": ans,
                    "action": parsed.action,
                    "sufficient": parsed.sufficient,
                    "reasoning": parsed.reasoning,
                    "evidence_analysis": parsed.evidence_analysis,
                    "missing_info": parsed.missing_info,
                    "supplemental_retrieval": generation["supplemental"],
                },
                "metrics": {"Recall": recall}, "token_usage": {"total_input_tokens": in_tokens, "llm_output_tokens": out_tokens}
            }
        except Exception:
            self.monitor.worker_end(success=False)
            raise

    def _process_ov_fallback_bot_task(self, task):
        return self._process_fallback_task(task, bot_use_relations=False)

    def _process_ov_fallback_bot_relations_task(self, task):
        return self._process_fallback_task(task, bot_use_relations=True)

    def _process_fallback_task(self, task, bot_use_relations: bool):
        self.monitor.worker_start()
        try:
            qa = task['qa']

            # --- Phase 1: OV retrieval + generation ---
            t0 = time.time()
            retrieval_instruction = self.config['execution'].get('retrieval_instruction', '')
            enhanced_query = f"{retrieval_instruction} {qa.question}" if retrieval_instruction else qa.question
            search_res = self.db.retrieve(query=enhanced_query, topk=self.config['execution']['retrieval_topk'])
            ov_retrieval_sec = time.time() - t0

            if self._config_bool(self.config.get("execution", {}).get("use_oracle_evidence_context"), False):
                search_res = self.db.build_context_result(
                    qa.evidence,
                    uri_prefix="oracle_evidence",
                    base_result=search_res,
                )

            recall_texts = search_res["recall_texts"]
            context_blocks = search_res["context_blocks"]
            retrieved_uris = search_res["retrieved_uris"]

            # Extract Phase 1 relations info
            ov_relations_uris = search_res.get("relations_uris", [])
            ov_relations_found = search_res.get("relations_found", 0)
            ov_relations_added = search_res.get("relations_added", 0)

            # Calculate token counts for original docs vs relations docs
            # context_blocks = relations_blocks + original_blocks (see vector_store_with_relations.py)
            num_relations_blocks = len(ov_relations_uris)
            relations_blocks = context_blocks[:num_relations_blocks] if num_relations_blocks > 0 else []
            original_blocks = context_blocks[num_relations_blocks:] if num_relations_blocks > 0 else context_blocks

            ov_original_doc_tokens = sum(self.db.count_tokens(block) for block in original_blocks)
            ov_relations_doc_tokens = sum(self.db.count_tokens(block) for block in relations_blocks)

            t1 = time.time()
            primary_provider, provider_results = self.phase1_provider_runner.run_all(
                qa,
                search_res,
                sample_id=task["sample_id"],
                retrieval_instruction=retrieval_instruction,
                ov_original_doc_tokens=ov_original_doc_tokens,
                ov_relations_doc_tokens=ov_relations_doc_tokens,
            )
            ov_generation_sec = time.time() - t1

            primary_result = provider_results[primary_provider]
            provider_result_records = {
                name: result.to_record()
                for name, result in provider_results.items()
            }
            provider_total_input_tokens = sum(int(r.input_tokens or 0) for r in provider_results.values())
            provider_total_output_tokens = sum(int(r.output_tokens or 0) for r in provider_results.values())
            provider_total_latency_sec = sum(float(r.latency_sec or 0.0) for r in provider_results.values())
            provider_diagnostic_input_tokens = max(0, provider_total_input_tokens - int(primary_result.input_tokens or 0))
            provider_diagnostic_output_tokens = max(0, provider_total_output_tokens - int(primary_result.output_tokens or 0))
            provider_diagnostic_latency_sec = max(0.0, provider_total_latency_sec - float(primary_result.latency_sec or 0.0))

            search_res = primary_result.search_res

            retrieved_texts = list(recall_texts.values())
            recall = MetricsCalculator.check_recall(retrieved_texts, qa.evidence)

            full_prompt = primary_result.prompt
            ans_raw = primary_result.raw
            parsed = primary_result.parsed
            ov_answer = primary_result.answer

            ov_in_tokens = int(primary_result.input_tokens or 0)
            ov_out_tokens = int(primary_result.output_tokens or 0)

            # --- Phase 3: Fallback decision ---
            fallback_triggered = bool(primary_result.should_fallback)
            fallback_executed = fallback_triggered
            final_answer = ov_answer
            bot_latency_sec = 0
            bot_input_tokens = 0
            bot_output_tokens = 0
            bot_detail = None
            shadow_bot_detail = None
            shadow_bot_error = ""

            if fallback_executed:
                self.logger.info(
                    f"[Query-{task['id']}] Fallback triggered ({primary_result.reasoning}), "
                    f"calling bot (relations={bot_use_relations})"
                )
                bot_detail = self.fallback_bot_runner.run(
                    task,
                    qa,
                    bot_use_relations=bot_use_relations,
                    trace_suffix="fallback",
                )
                final_answer = bot_detail.get("answer", "")
                bot_latency_sec = float(bot_detail.get("total_time_sec", 0.0) or 0.0)
                bot_input_tokens = int(bot_detail.get("prompt_tokens", 0) or 0)
                bot_output_tokens = int(bot_detail.get("completion_tokens", 0) or 0)
            else:
                self.logger.info(
                    f"[Query-{task['id']}] Fallback not triggered; "
                    f"running shadow bot for recoverable-miss metrics (relations={bot_use_relations})"
                )
                try:
                    shadow_bot_detail = self.fallback_bot_runner.run(
                        task,
                        qa,
                        bot_use_relations=bot_use_relations,
                        trace_suffix="shadow_fallback",
                    )
                except Exception as e:
                    shadow_bot_error = str(e)
                    self.logger.error(f"[Query-{task['id']}] Shadow fallback bot failed: {e}")

            # --- Aggregate metrics ---
            total_input_tokens = ov_in_tokens + bot_input_tokens
            total_output_tokens = ov_out_tokens + bot_output_tokens
            actual_total_input_tokens = provider_total_input_tokens + bot_input_tokens
            actual_total_output_tokens = provider_total_output_tokens + bot_output_tokens
            primary_provider_latency_sec = float(primary_result.latency_sec or 0.0)
            total_latency_sec = primary_provider_latency_sec + bot_latency_sec
            actual_total_latency_sec = ov_retrieval_sec + ov_generation_sec + bot_latency_sec

            self.monitor.worker_end(tokens=total_input_tokens + total_output_tokens)
            extra_provider_status = []
            for name, result_for_provider in provider_results.items():
                if name == primary_provider:
                    continue
                status = "YES" if result_for_provider.should_fallback else "NO"
                extra_provider_status.append(f"{name}Fallback={status}")
            extra_provider_log = (
                " | " + " | ".join(extra_provider_status)
                if extra_provider_status else ""
            )
            self.logger.info(
                f"[Query-{task['id']}] Fallback={'YES' if fallback_triggered else 'NO'} | "
                f"Provider={primary_provider}{extra_provider_log} | "
                f"Executed={'YES' if fallback_executed else 'NO'} | "
                f"{primary_result.reasoning} | Total: {total_latency_sec:.1f}s"
            )

            result = {
                "_global_index": task['id'], "sample_id": task['sample_id'], "question": qa.question,
                "gold_answers": qa.gold_answers, "category": str(qa.category), "evidence": qa.evidence,
                "retrieval": {"latency_sec": total_latency_sec, "uris": retrieved_uris},
                "llm": {
                    "final_answer": final_answer,
                    "action": parsed.action,
                    "sufficient": parsed.sufficient,
                    "reasoning": parsed.reasoning,
                    "evidence_analysis": parsed.evidence_analysis,
                    "missing_info": parsed.missing_info,
                    "supplemental_retrieval": primary_result.supplemental,
                    "phase1_provider": primary_provider,
                    "phase1_provider_results": provider_result_records,
                },
                "metrics": {"Recall": recall},
                "token_usage": {
                    "total_input_tokens": total_input_tokens,
                    "llm_output_tokens": total_output_tokens,
                    "retrieval_embedding_tokens": 0,
                    "prompt_tokens": total_input_tokens,
                    "completion_tokens": total_output_tokens,
                    "total_tokens": total_input_tokens + total_output_tokens,
                },
                "fallback": {
                    "triggered": fallback_triggered,
                    "executed": fallback_executed,
                    "phase1_judge_should_fallback": fallback_triggered,
                    "primary_provider": primary_provider,
                    "provider_results": provider_result_records,
                    "provider_total_input_tokens": provider_total_input_tokens,
                    "provider_total_output_tokens": provider_total_output_tokens,
                    "provider_total_latency_sec": provider_total_latency_sec,
                    "provider_diagnostic_input_tokens": provider_diagnostic_input_tokens,
                    "provider_diagnostic_output_tokens": provider_diagnostic_output_tokens,
                    "provider_diagnostic_latency_sec": provider_diagnostic_latency_sec,
                    "actual_total_input_tokens": actual_total_input_tokens,
                    "actual_total_output_tokens": actual_total_output_tokens,
                    "actual_total_latency_sec": actual_total_latency_sec,
                    "phase1_action": parsed.action,
                    "bot_use_relations": bot_use_relations,
                    "judge_reasoning": primary_result.reasoning,
                    "ov_answer": ov_answer,
                    "supplemental_retrieval": primary_result.supplemental,
                    "ov_retrieval_sec": ov_retrieval_sec,
                    "ov_generation_sec": ov_generation_sec,
                    "primary_provider_latency_sec": primary_provider_latency_sec,
                    "bot_latency_sec": bot_latency_sec,
                    "ov_input_tokens": ov_in_tokens,
                    "ov_output_tokens": ov_out_tokens,
                    "bot_input_tokens": bot_input_tokens,
                    "bot_output_tokens": bot_output_tokens,
                    "total_latency_sec": total_latency_sec,
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    # Phase 1 relations info
                    "ov_relations_found": ov_relations_found,
                    "ov_relations_added": ov_relations_added,
                    "ov_relations_uris": ov_relations_uris,
                    "ov_original_doc_tokens": ov_original_doc_tokens,
                    "ov_relations_doc_tokens": ov_relations_doc_tokens,
                    "shadow_bot": {
                        "attempted": not fallback_executed,
                        "executed": bool(shadow_bot_detail) and not shadow_bot_error,
                        "error": shadow_bot_error,
                        "answer": (shadow_bot_detail or {}).get("answer", ""),
                        "latency_sec": float((shadow_bot_detail or {}).get("total_time_sec", 0.0) or 0.0),
                        "input_tokens": int((shadow_bot_detail or {}).get("prompt_tokens", 0) or 0),
                        "output_tokens": int((shadow_bot_detail or {}).get("completion_tokens", 0) or 0),
                        "total_tokens": int((shadow_bot_detail or {}).get("total_tokens", 0) or 0),
                        "bot_use_relations": bot_use_relations,
                        "detail": shadow_bot_detail or {},
                    },
                },
            }
            naive_rule_result = provider_results.get("raw_context_naive_rule")
            if naive_rule_result:
                result["fallback"].update({
                    "naive_rule_triggered": bool(naive_rule_result.should_fallback),
                    "naive_rule_reasoning": naive_rule_result.reasoning,
                    "naive_rule_input_tokens": int(naive_rule_result.input_tokens or 0),
                    "naive_rule_output_tokens": int(naive_rule_result.output_tokens or 0),
                    "naive_rule_latency_sec": float(naive_rule_result.latency_sec or 0.0),
                })
            if bot_detail:
                result["vikingbot"] = bot_detail
            return result
        except Exception:
            self.monitor.worker_end(success=False)
            raise

    def _process_evaluation_task(self, item):
        """
        Process a single evaluation task, computing F1 and Accuracy metrics.
        
        For multi-annotator scenarios (like Qasper dataset), a question may have multiple gold answers.
        Evaluation logic:
        - F1: Compute for each gold answer separately and take the maximum
        - Accuracy: Pass all gold answers to LLM at once for comprehensive judgment
        
        This correctly handles multi-annotator scenarios while maintaining compatibility with single-answer datasets (like Locomo).
        """
        ans, golds = item['llm']['final_answer'], item['gold_answers']

        dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')
        f1, acc, eval_record = self._score_answer_for_accuracy(
            item['question'],
            golds,
            ans,
            dataset_name,
        )

        item["metrics"].update({"F1": f1, "Accuracy": acc})
        phase1_eval_record = None
        if "fallback" in item and "ov_answer" in item["fallback"]:
            phase1_answer = item["fallback"].get("ov_answer", "")
            if str(phase1_answer or "").strip() == str(ans or "").strip():
                phase1_f1 = f1
                phase1_acc = acc
                phase1_eval_record = dict(eval_record)
            else:
                phase1_f1, phase1_acc, phase1_eval_record = self._score_answer_for_accuracy(
                    item['question'],
                    golds,
                    phase1_answer,
                    dataset_name,
                )
            item["metrics"].update({
                "Phase1 F1": phase1_f1,
                "Phase1 Accuracy": phase1_acc,
            })

        if "fallback" in item:
            fallback_info = item.get("fallback", {}) or {}
            provider_results = fallback_info.get("provider_results", {}) or {}
            answer_score_cache = {
                str(ans or "").strip(): (f1, acc, eval_record)
            }
            if phase1_eval_record is not None:
                answer_score_cache[str(fallback_info.get("ov_answer", "") or "").strip()] = (
                    item["metrics"].get("Phase1 F1", 0.0),
                    item["metrics"].get("Phase1 Accuracy", 0.0),
                    phase1_eval_record,
                )

            shadow_bot = fallback_info.get("shadow_bot", {}) or {}
            if shadow_bot.get("executed"):
                shadow_answer = str(shadow_bot.get("answer", "") or "")
                cache_key = shadow_answer.strip()
                if cache_key in answer_score_cache:
                    shadow_f1, shadow_acc, shadow_eval_record = answer_score_cache[cache_key]
                else:
                    shadow_f1, shadow_acc, shadow_eval_record = self._score_answer_for_accuracy(
                        item['question'],
                        golds,
                        shadow_answer,
                        dataset_name,
                    )
                    answer_score_cache[cache_key] = (shadow_f1, shadow_acc, shadow_eval_record)
                shadow_bot["metrics"] = {
                    "F1": shadow_f1,
                    "Accuracy": shadow_acc,
                }
                shadow_bot["llm_evaluation"] = {
                    "prompt_used": shadow_eval_record["prompt_type"],
                    "reasoning": shadow_eval_record["reasoning"],
                    "normalized_score": shadow_acc,
                }
                item["shadow_bot_evaluation"] = shadow_bot["llm_evaluation"]

            for provider_name, provider_result in provider_results.items():
                provider_answer = str(provider_result.get("answer", "") or "")
                cache_key = provider_answer.strip()
                if cache_key in answer_score_cache:
                    provider_f1, provider_acc, provider_eval_record = answer_score_cache[cache_key]
                else:
                    provider_f1, provider_acc, provider_eval_record = self._score_answer_for_accuracy(
                        item['question'],
                        golds,
                        provider_answer,
                        dataset_name,
                    )
                    answer_score_cache[cache_key] = (provider_f1, provider_acc, provider_eval_record)
                provider_result["metrics"] = {
                    "F1": provider_f1,
                    "Accuracy": provider_acc,
                }
                provider_result["llm_evaluation"] = {
                    "prompt_used": provider_eval_record["prompt_type"],
                    "reasoning": provider_eval_record["reasoning"],
                    "normalized_score": provider_acc,
                }

            for provider_name, provider_result in provider_results.items():
                provider_judgment = phase1_fallback_judgment(item, provider_name=provider_name)
                provider_result["fallback_expected_trigger"] = provider_judgment["expected_trigger"]
                provider_result["fallback_judgment_error"] = provider_judgment["error"]
                provider_result["fallback_judgment_error_type"] = provider_judgment["error_type"]

            phase1_judgment = phase1_fallback_judgment(item)
            item["fallback"]["phase1_fallback_expected_trigger"] = phase1_judgment["expected_trigger"]
            item["fallback"]["phase1_fallback_judgment_error"] = phase1_judgment["error"]
            item["fallback"]["phase1_fallback_judgment_error_type"] = phase1_judgment["error_type"]
        
        item["llm_evaluation"] = {
            "prompt_used": eval_record["prompt_type"],
            "reasoning": eval_record["reasoning"],
            "normalized_score": acc
        }
        if phase1_eval_record is not None:
            item["phase1_llm_evaluation"] = {
                "prompt_used": phase1_eval_record["prompt_type"],
                "reasoning": phase1_eval_record["reasoning"],
                "normalized_score": item["metrics"].get("Phase1 Accuracy", 0.0),
            }

        detailed_info = (
            f"\n" + "="*60 +
            f"\n[Query ID]: {item['_global_index']}"
            f"\n[Question]: {item['question']}"
            f"\n[Retrieved URIs]: {item['retrieval'].get('uris', [])}"
            f"\n[LLM Answer]: {ans}"
            f"\n[Gold Answer]: {golds}"
            f"\n[Metrics]: {item['metrics']}"
            f"\n[LLM Judge Reasoning]: {eval_record['reasoning']}"
            f"\n[VikingBot Trace File]: {item.get('vikingbot', {}).get('trace_file', '')}"
            f"\n" + "="*60
        )
        self.logger.info(detailed_info)
        return item

    def _update_report(self, data):
        """Read existing report, merge new data, and write back"""
        data = dict(data or {})
        delete_keys = data.pop("__delete_keys__", [])
        report = {}
        if os.path.exists(self.report_file):
            with open(self.report_file, "r", encoding="utf-8") as f:
                try:
                    report = json.load(f)
                except json.JSONDecodeError:
                    report = {}
        for key in delete_keys:
            report.pop(key, None)
        report.update(data)
        with open(self.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Report updated -> {self.report_file}")

    def _count_total_relations(self, strategy: str):
        """Count total unique edge pairs in all .relations_{strategy}.jsonl files."""
        vector_store_path = self.config.get('paths', {}).get('vector_store', '')
        if not vector_store_path or not os.path.isdir(vector_store_path):
            return 0, set()
        viking_dir = os.path.join(vector_store_path, "viking")
        if not os.path.isdir(viking_dir):
            return 0, set()
        filename = ".relations.jsonl" if strategy == "blind" else f".relations_{strategy}.jsonl"
        all_edges = set()
        for root, _dirs, files in os.walk(viking_dir):
            if filename in files:
                fpath = os.path.join(root, filename)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                uri1 = rec.get("uri1", "")
                                uri2 = rec.get("uri2", "")
                                if uri1 and uri2 and uri1 != uri2:
                                    all_edges.add((uri1, uri2))
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    continue
        return len(all_edges), all_edges
