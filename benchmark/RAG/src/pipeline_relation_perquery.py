import copy
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from adapters.base import StandardQA
from core.logger import get_logger
from core.vector_store import VikingStoreWrapper
from core.vector_store_with_relations import VikingStoreHTTPWithRelations
from pipeline import BenchmarkPipeline
from vikingbot_runner import (
    _ensure_openviking_server,
    _generate_temp_ov_conf,
    _load_server_url_and_key,
    _stop_openviking_server,
)


RELATION_FILENAMES = {".relations_llm_review.jsonl", ".reference_questions.jsonl"}


class RelationPerQueryPipeline:
    """VersionRAG relation-perquery experiment orchestrator.

    For each question:
    1. Use a clone-pruned single-document OV store.
    2. Clear relation files.
    3. Build relation links for this question.
    4. Run bot-relations and OV-fallback-bot-relations immediately.
    5. Clear relation files again before moving to the next question.
    """

    def __init__(self, build_config, relations_config, fallback_config, adapter, llm, resume: bool = False):
        self.build_config = build_config
        self.relations_config = relations_config
        self.fallback_config = fallback_config
        self.adapter = adapter
        self.llm = llm
        self.resume = resume
        self.logger = get_logger()

        self.project_root = Path(__file__).resolve().parent.parent
        self.dataset_name = self.build_config.get("dataset_name", "VersionRAG")
        self.execution = self.build_config.get("execution", {})
        self.store_parent_path = self.build_config["paths"]["vector_store"]
        self.records_file = os.path.join(self.store_parent_path, "_relation_perquery_records.json")
        self.base_vector_store = self.execution["base_vector_store"]
        self.clone_prune_resource_root_uri = self.execution.get(
            "clone_prune_resource_root_uri",
            "viking://resources/VersionRAG_processed_docs",
        )
        self.qa_doc_mapping_path = self.execution["qa_doc_mapping_path"]
        self.force_rebuild_cloned_stores = bool(self.execution.get("force_rebuild_cloned_stores", False))
        self.max_queries = self._resolve_max_queries()

        self.build_output_dir = self.build_config["paths"]["output_dir"]
        self.relations_output_dir = self.relations_config["paths"]["output_dir"]
        self.fallback_output_dir = self.fallback_config["paths"]["output_dir"]
        for output_dir in (self.build_output_dir, self.relations_output_dir, self.fallback_output_dir):
            os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.store_parent_path, exist_ok=True)

    def _resolve_max_queries(self):
        max_queries = self.execution.get("max_queries")
        env_max = os.environ.get("RAG_MAX_QUERIES")
        if env_max is None:
            return max_queries
        env_max = env_max.strip().lower()
        if env_max in ("", "none", "null"):
            return None
        return int(env_max)

    @staticmethod
    def _normalize_question(text: str) -> str:
        return " ".join((text or "").strip().split())

    @staticmethod
    def _normalize_source_file(source_file: str) -> str:
        return os.path.normpath(str(source_file)).replace("\\", "/")

    @staticmethod
    def _doc_name_from_source_file(source_file: str) -> str:
        stem = Path(source_file).stem
        return re.sub(r'[\\/*?:"<>|\s]+', "_", stem)

    @staticmethod
    def _store_key_from_doc_names(doc_names: List[str]) -> str:
        unique = sorted({d for d in doc_names if d})
        if len(unique) == 1:
            return unique[0][:120]
        raw = "|".join(unique)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _store_path(self, store_key: str) -> str:
        return os.path.join(self.store_parent_path, store_key)

    def _load_records(self) -> Dict[str, Any]:
        if not os.path.exists(self.records_file):
            return {}
        try:
            with open(self.records_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self.logger.warning(f"Failed to load relation-perquery records, starting fresh: {e}")
            return {}

    def _save_records(self, records: Dict[str, Any]) -> None:
        os.makedirs(self.store_parent_path, exist_ok=True)
        with open(self.records_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    def _load_mapping(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.qa_doc_mapping_path):
            raise FileNotFoundError(f"qa_doc_mapping.json not found: {self.qa_doc_mapping_path}")
        with open(self.qa_doc_mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Invalid qa_doc_mapping format: {self.qa_doc_mapping_path}")

        mapping = {}
        duplicates = []
        for item in data:
            question = self._normalize_question(item.get("question", ""))
            if not question:
                continue
            if question in mapping:
                duplicates.append(question)
            mapping[question] = item
        if duplicates:
            raise ValueError(f"Duplicate questions in qa_doc_mapping: {duplicates[:3]}")
        return mapping

    def _load_tasks(self) -> List[Dict[str, Any]]:
        mapping = self._load_mapping()
        dataset_path = self.build_config["paths"]["dataset_path"]
        tasks = []
        missing = []

        with open(dataset_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row_index, row in enumerate(reader):
                if self.max_queries is not None and len(tasks) >= self.max_queries:
                    break
                question = self._normalize_question(row.get("Question", ""))
                mapping_item = mapping.get(question)
                if not mapping_item:
                    missing.append({"row_index": row_index, "question": question})
                    continue
                source_files = [
                    self._normalize_source_file(s)
                    for s in mapping_item.get("source_files", [])
                    if s
                ]
                if not source_files:
                    missing.append({"row_index": row_index, "question": question, "reason": "empty source_files"})
                    continue
                doc_names = [self._doc_name_from_source_file(s) for s in source_files]
                store_key = self._store_key_from_doc_names(doc_names)
                qa = StandardQA(
                    question=row.get("Question", "").strip(),
                    gold_answers=[row.get("Answer", "").strip()] if row.get("Answer", "").strip() else [],
                    evidence=[],
                    category=row.get("Type", "Unknown").strip(),
                    metadata={
                        "row_index": mapping_item.get("row_index", row_index),
                        "source_files": source_files,
                    },
                )
                tasks.append(
                    {
                        "id": len(tasks),
                        "sample_id": f"qa_{mapping_item.get('row_index', row_index)}",
                        "qa": qa,
                        "source_files": source_files,
                        "doc_names": doc_names,
                        "store_key": store_key,
                        "store_path": self._store_path(store_key),
                    }
                )

        if missing:
            raise ValueError(
                "qa_doc_mapping does not match evaluation_set.csv. "
                f"Missing {len(missing)} questions, first examples: {missing[:3]}"
            )
        if not tasks:
            raise ValueError("No VersionRAG tasks loaded for relation-perquery experiment")
        self.logger.info(f"Loaded {len(tasks)} VersionRAG relation-perquery tasks")
        return tasks

    def validate(self) -> Dict[str, Any]:
        tasks = self._load_tasks()
        unique_stores = sorted({task["store_key"] for task in tasks})
        missing_docs = self._find_missing_base_docs(tasks)
        if missing_docs:
            raise FileNotFoundError(f"Mapped docs not found in base OV store: {missing_docs[:10]}")
        result = {"tasks": len(tasks), "stores": len(unique_stores), "base_vector_store": self.base_vector_store}
        self.logger.info(f"Relation-perquery validation passed: {result}")
        return result

    def _resource_uri_to_local_path(self, store_path: str, resource_uri: str) -> str:
        prefix = "viking://resources"
        if resource_uri == prefix:
            rel_path = ""
        elif resource_uri.startswith(prefix + "/"):
            rel_path = resource_uri[len(prefix) + 1:]
        else:
            raise ValueError(f"Resource uri must start with {prefix}: {resource_uri}")
        parts = rel_path.split("/") if rel_path else []
        return os.path.join(store_path, "viking", "default", "resources", *parts)

    def _resource_root_dir(self, store_path: str) -> str:
        return self._resource_uri_to_local_path(store_path, self.clone_prune_resource_root_uri)

    def _list_resource_dirs(self, store_path: str) -> List[str]:
        resource_root = self._resource_root_dir(store_path)
        if not os.path.isdir(resource_root):
            raise FileNotFoundError(f"Resource root not found: {resource_root}")
        return sorted(
            name
            for name in os.listdir(resource_root)
            if os.path.isdir(os.path.join(resource_root, name))
        )

    def _find_missing_base_docs(self, tasks: List[Dict[str, Any]]) -> List[str]:
        if not os.path.isdir(self.base_vector_store):
            raise FileNotFoundError(f"Base OV store does not exist: {self.base_vector_store}")
        available = set(self._list_resource_dirs(self.base_vector_store))
        required = {doc_name for task in tasks for doc_name in task["doc_names"]}
        return sorted(required - available)

    def _prepare_openviking_config_for_store(self, store_path: str) -> None:
        original_conf_path = os.environ.get("OPENVIKING_CONFIG_FILE")
        if not original_conf_path:
            original_conf_path = str((self.project_root / "ov.conf").resolve())
        if not os.path.exists(original_conf_path):
            return

        temp_conf_path = _generate_temp_ov_conf(
            original_conf_path,
            store_path,
            search_limit=self.build_config.get("vikingbot", {}).get("search_limit"),
            llm_config=self.build_config.get("llm"),
            server_port=self.build_config.get("execution", {}).get("server_port"),
        )
        os.environ["OPENVIKING_CONFIG_FILE"] = temp_conf_path
        try:
            from openviking_cli.utils.config.open_viking_config import OpenVikingConfigSingleton

            OpenVikingConfigSingleton.reset_instance()
        except Exception as e:
            self.logger.warning(f"Failed to reset OpenViking config singleton: {e}")

    def _clone_prune_store(self, task: Dict[str, Any]) -> Dict[str, Any]:
        store_path = task["store_path"]
        if os.path.exists(store_path):
            if self.force_rebuild_cloned_stores:
                shutil.rmtree(store_path)
            else:
                return {"time": 0, "strategy": "clone_prune", "reused": True}

        start_time = time.time()
        shutil.copytree(self.base_vector_store, store_path)
        self._prepare_openviking_config_for_store(store_path)

        kept_docs = set(task["doc_names"])
        removed_docs = []
        store = VikingStoreWrapper(store_path)
        try:
            for doc_name in self._list_resource_dirs(store_path):
                if doc_name in kept_docs:
                    continue
                store.client.rm(f"{self.clone_prune_resource_root_uri}/{doc_name}", recursive=True)
                removed_docs.append(doc_name)
        finally:
            store.close()

        relation_files_removed = self._clear_relation_files(store_path)
        return {
            "time": time.time() - start_time,
            "strategy": "clone_prune",
            "base_store_path": self.base_vector_store,
            "resource_root_uri": self.clone_prune_resource_root_uri,
            "kept_docs": sorted(kept_docs),
            "removed_docs": removed_docs,
            "relation_files_removed_after_clone": relation_files_removed,
        }

    def run_import(self) -> None:
        tasks = self._load_tasks()
        missing_docs = self._find_missing_base_docs(tasks)
        if missing_docs:
            raise FileNotFoundError(f"Mapped docs not found in base OV store: {missing_docs[:10]}")

        records = self._load_records()
        unique_tasks = OrderedDict()
        for task in tasks:
            unique_tasks.setdefault(task["store_key"], task)

        start_all = time.time()
        total_clone_time = 0.0
        for store_key, task in tqdm(unique_tasks.items(), desc="Preparing Relation PerQuery Stores", unit="store"):
            rec = records.get(store_key, {})
            if rec.get("status") == "ingested" and os.path.isdir(task["store_path"]):
                continue
            stats = self._clone_prune_store(task)
            total_clone_time += float(stats.get("time", 0) or 0)
            records[store_key] = {
                "status": "ingested",
                "store_path": task["store_path"],
                "source_files": task["source_files"],
                "doc_names": task["doc_names"],
                "stats": stats,
                "updated_at": time.time(),
            }
            self._save_records(records)

        report = {
            "Insertion Efficiency (Relation PerQuery Stores)": {
                "Total Insertion Time (s)": time.time() - start_all,
                "Total Clone-Prune Time (s)": total_clone_time,
                "Total Input Tokens": 0,
                "Total Output Tokens": 0,
                "Total Embedding Tokens": 0,
            },
            "Relation PerQuery": {
                "Total Queries": len(tasks),
                "Store Count": len(unique_tasks),
                "Base Vector Store": self.base_vector_store,
            },
        }
        self._update_report_file(self.build_output_dir, report)

    def _clear_relation_files(self, store_path: str) -> int:
        removed = 0
        if not os.path.isdir(store_path):
            return removed
        for root, _dirs, files in os.walk(store_path):
            for filename in files:
                if filename not in RELATION_FILENAMES:
                    continue
                path = os.path.join(root, filename)
                try:
                    os.remove(path)
                    removed += 1
                except FileNotFoundError:
                    continue
        return removed

    def _config_for_store(self, config: Dict[str, Any], store_path: str) -> Dict[str, Any]:
        cfg = copy.deepcopy(config)
        cfg.setdefault("paths", {})["vector_store"] = store_path
        return cfg

    def _make_task_record(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": task["id"], "sample_id": task["sample_id"], "qa": task["qa"]}

    def _attach_relation_perquery_meta(self, result: Dict[str, Any], task: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        result.setdefault("relation_perquery", {})
        result["relation_perquery"].update(
            {
                "enabled": True,
                "store_key": task["store_key"],
                "store_path": task["store_path"],
                "source_files": task["source_files"],
                "doc_names": task["doc_names"],
                **meta,
            }
        )
        return result

    def _run_build_links_for_query(self, task: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self._config_for_store(self.build_config, task["store_path"])
        pipeline = BenchmarkPipeline(cfg, self.adapter, None, self.llm, resume=False)
        return pipeline._process_vikingbot_task(self._make_task_record(task))

    def _run_bot_relations_for_query(self, task: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self._config_for_store(self.relations_config, task["store_path"])
        pipeline = BenchmarkPipeline(cfg, self.adapter, None, self.llm, resume=False)
        return pipeline._process_vikingbot_task(self._make_task_record(task))

    def _make_fallback_vector_store(self, cfg: Dict[str, Any], store_path: str):
        ov_conf_path = str((self.project_root / "ov.conf").resolve())
        temp_conf_path = _generate_temp_ov_conf(
            ov_conf_path,
            store_path,
            search_limit=cfg.get("vikingbot", {}).get("search_limit"),
            llm_config=cfg.get("llm"),
            server_port=cfg.get("execution", {}).get("server_port"),
        )
        _ensure_openviking_server(temp_conf_path)
        server_url, api_key = _load_server_url_and_key(temp_conf_path)

        embedder = None
        embedding_cfg = cfg.get("embedding", {})
        emb_api_key = embedding_cfg.get("api_key", "")
        emb_api_key = os.path.expandvars(emb_api_key) if emb_api_key else emb_api_key
        if emb_api_key and not emb_api_key.startswith("${"):
            from core.embedder import VolcengineEmbedder

            embedder = VolcengineEmbedder(
                api_key=emb_api_key,
                base_url=embedding_cfg.get("base_url", "https://ark.cn-beijing.volces.com/api/v3"),
                model=embedding_cfg.get("model", "doubao-embedding-vision-250615"),
            )

        return VikingStoreHTTPWithRelations(
            server_url=server_url,
            api_key=api_key,
            store_path=store_path,
            embedder=embedder,
            strategy=cfg.get("vikingbot", {}).get("link_strategy", "llm_review"),
            relations_topk=cfg.get("vikingbot", {}).get("relations_topk", 0),
            similarity_threshold=cfg.get("vikingbot", {}).get("relations_similarity_threshold"),
        )

    def _run_fallback_relations_for_query(self, task: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self._config_for_store(self.fallback_config, task["store_path"])
        vector_store = self._make_fallback_vector_store(cfg, task["store_path"])
        pipeline = BenchmarkPipeline(cfg, self.adapter, vector_store, self.llm, resume=False)
        try:
            return pipeline._process_ov_fallback_bot_relations_task(self._make_task_record(task))
        finally:
            if hasattr(vector_store, "close"):
                vector_store.close()

    @staticmethod
    def _load_existing_results(output_dir: str) -> Dict[int, Dict[str, Any]]:
        generated_file = os.path.join(output_dir, "generated_answers.json")
        if not os.path.exists(generated_file):
            return {}
        try:
            with open(generated_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {int(r["_global_index"]): r for r in data.get("results", [])}
        except Exception:
            return {}

    def _save_results(self, output_dir: str, results_map: Dict[int, Dict[str, Any]]) -> None:
        os.makedirs(output_dir, exist_ok=True)
        sorted_results = [results_map[i] for i in sorted(results_map.keys())]
        data = {
            "summary": {"dataset": self.dataset_name, "total_queries": len(sorted_results)},
            "results": sorted_results,
        }
        with open(os.path.join(output_dir, "generated_answers.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        total = len(sorted_results)
        if total:
            self._update_report_file(
                output_dir,
                {
                    "Query Efficiency (Average Per Query)": {
                        "Average Retrieval Time (s)": sum(r.get("retrieval", {}).get("latency_sec", 0) for r in sorted_results) / total,
                        "Average Input Tokens": sum(self._get_input_tokens(r) for r in sorted_results) / total,
                        "Average Output Tokens": sum(self._get_output_tokens(r) for r in sorted_results) / total,
                    }
                },
            )

    @staticmethod
    def _get_input_tokens(record: Dict[str, Any]) -> int:
        usage = record.get("token_usage", {}) or {}
        return int(usage.get("prompt_tokens", usage.get("total_input_tokens", 0)) or 0)

    @staticmethod
    def _get_output_tokens(record: Dict[str, Any]) -> int:
        usage = record.get("token_usage", {}) or {}
        return int(usage.get("completion_tokens", usage.get("llm_output_tokens", 0)) or 0)

    @staticmethod
    def _update_report_file(output_dir: str, data: Dict[str, Any]) -> None:
        os.makedirs(output_dir, exist_ok=True)
        report_file = os.path.join(output_dir, "benchmark_metrics_report.json")
        report = {}
        if os.path.exists(report_file):
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    report = json.load(f)
            except (json.JSONDecodeError, OSError):
                report = {}
        report.update(data)
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def run_generation(self) -> None:
        tasks = self._load_tasks()
        self.run_import()

        build_results = self._load_existing_results(self.build_output_dir) if self.resume else {}
        relations_results = self._load_existing_results(self.relations_output_dir) if self.resume else {}
        fallback_results = self._load_existing_results(self.fallback_output_dir) if self.resume else {}

        for task in tqdm(tasks, desc="Relation PerQuery Generation", unit="query"):
            task_id = task["id"]
            if self.resume and task_id in relations_results and task_id in fallback_results:
                continue

            _stop_openviking_server()
            cleared_before = self._clear_relation_files(task["store_path"])
            build_result = self._run_build_links_for_query(task)
            links_created = int((build_result.get("vikingbot", {}) or {}).get("links_created", 0) or 0)
            build_results[task_id] = self._attach_relation_perquery_meta(
                build_result,
                task,
                {
                    "build_links_created": links_created,
                    "relation_files_cleared_before": cleared_before,
                    "stage": "build_links_review",
                },
            )

            relations_result = self._run_bot_relations_for_query(task)
            relations_results[task_id] = self._attach_relation_perquery_meta(
                relations_result,
                task,
                {
                    "build_links_created": links_created,
                    "relation_files_cleared_before": cleared_before,
                    "stage": "relations_review",
                },
            )

            fallback_result = self._run_fallback_relations_for_query(task)
            cleared_after = self._clear_relation_files(task["store_path"])
            _stop_openviking_server()
            fallback_results[task_id] = self._attach_relation_perquery_meta(
                fallback_result,
                task,
                {
                    "build_links_created": links_created,
                    "relation_files_cleared_before": cleared_before,
                    "relation_files_cleared_after": cleared_after,
                    "stage": "ov_fallback_bot_relations",
                },
            )
            build_results[task_id]["relation_perquery"]["relation_files_cleared_after"] = cleared_after
            relations_results[task_id]["relation_perquery"]["relation_files_cleared_after"] = cleared_after

            self._save_results(self.build_output_dir, build_results)
            self._save_results(self.relations_output_dir, relations_results)
            self._save_results(self.fallback_output_dir, fallback_results)

    def run_evaluation(self) -> None:
        for cfg in (self.relations_config, self.fallback_config):
            pipeline = BenchmarkPipeline(cfg, self.adapter, None, self.llm, resume=self.resume)
            pipeline.run_evaluation()

    def run_deletion(self) -> None:
        start_time = time.time()
        if os.path.isdir(self.store_parent_path):
            shutil.rmtree(self.store_parent_path)
        for output_dir in (self.build_output_dir, self.relations_output_dir, self.fallback_output_dir):
            self._update_report_file(
                output_dir,
                {
                    "Deletion Efficiency (Relation PerQuery Stores)": {
                        "Total Deletion Time (s)": time.time() - start_time,
                        "Total Input Tokens": 0,
                        "Total Output Tokens": 0,
                    }
                },
            )
