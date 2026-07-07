#!/usr/bin/env python3

import importlib
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import yaml
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
sys.path.append(str(SCRIPT_DIR))

load_dotenv(SCRIPT_DIR / ".env")

VENV_BIN = Path("/home/zhanggaoyuan.225/vikingrag/.venv/bin")
if VENV_BIN.exists():
    os.environ["PATH"] = f"{VENV_BIN}:{os.environ.get('PATH', '')}"

ov_config_path = SCRIPT_DIR / "ov.conf"
if ov_config_path.exists():
    os.environ["OPENVIKING_CONFIG_FILE"] = str(ov_config_path)
    print(f"[Init] Auto-detected OpenViking config: {ov_config_path}")

from src.core.llm_client import LLMClientWrapper
from src.core.logger import setup_logging
from src.pipeline_relation_perquery import RelationPerQueryPipeline


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str, base_path: Path) -> str:
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    return str((base_path / path_str).resolve())


def render_and_resolve_config(config: dict) -> dict:
    dataset_name = config.get("dataset_name", "UnknownDataset")
    retrieval_topk = config.get("execution", {}).get("retrieval_topk", 5)
    format_vars = {
        "dataset_name": dataset_name,
        "retrieval_topk": retrieval_topk,
        "search_limit": config.get("vikingbot", {}).get("search_limit", ""),
        "max_iterations": config.get("vikingbot", {}).get("max_iterations", ""),
        "relations_topk": config.get("vikingbot", {}).get("relations_topk", ""),
        "relations_similarity_threshold": config.get("vikingbot", {}).get(
            "relations_similarity_threshold", ""
        ),
    }

    for key in ("dataset_path", "output_dir", "vector_store", "log_file", "doc_output_dir"):
        if key in config.get("paths", {}):
            rendered = str(config["paths"][key]).format(**format_vars)
            config["paths"][key] = resolve_path(rendered, PROJECT_ROOT)

    for key in ("base_vector_store", "qa_doc_mapping_path"):
        if key in config.get("execution", {}):
            rendered = str(config["execution"][key]).format(**format_vars)
            config["execution"][key] = resolve_path(rendered, PROJECT_ROOT)

    return config


def build_adapter(config: dict):
    adapter_cfg = config.get("adapter", {})
    module_path = adapter_cfg.get("module", "src.adapters.versionrag_adapter")
    class_name = adapter_cfg.get("class_name", "VersionRAGAdapter")
    mod = importlib.import_module(module_path)
    adapter_cls = getattr(mod, class_name)
    return adapter_cls(raw_file_path=config["paths"]["dataset_path"])


def build_llm(config: dict):
    llm_cfg = config.get("llm", {})
    api_key = os.environ.get(llm_cfg.get("api_key_env_var", ""), llm_cfg.get("api_key"))
    api_key = os.path.expandvars(api_key) if api_key else api_key
    return LLMClientWrapper(config=llm_cfg, api_key=api_key)


def main() -> int:
    parser = ArgumentParser(description="Run VersionRAG relation-perquery experiment")
    parser.add_argument(
        "--build-config",
        default="config/versionrag_relation_perquery/versionrag_bot_config_build_links_review.yaml",
        help="Config for per-query build_links_review stage",
    )
    parser.add_argument(
        "--relations-config",
        default="config/versionrag_relation_perquery/versionrag_bot_config_relations_review.yaml",
        help="Config for per-query bot relations_review stage",
    )
    parser.add_argument(
        "--fallback-config",
        default="config/versionrag_relation_perquery/versionrag_ov_fallback_bot_relations_config.yaml",
        help="Config for per-query OV fallback bot relations stage",
    )
    parser.add_argument(
        "--step",
        choices=["validate", "all", "import", "gen", "eval", "gen+eval", "del"],
        default="all",
        help="Execution step",
    )
    parser.add_argument("--resume", action="store_true", help="Resume generation if generated_answers.json exists")
    args = parser.parse_args()

    build_config = render_and_resolve_config(load_config(resolve_path(args.build_config, PROJECT_ROOT)))
    relations_config = render_and_resolve_config(load_config(resolve_path(args.relations_config, PROJECT_ROOT)))
    fallback_config = render_and_resolve_config(load_config(resolve_path(args.fallback_config, PROJECT_ROOT)))

    setup_logging(build_config["paths"]["log_file"])

    adapter = build_adapter(build_config)
    llm = build_llm(build_config)
    pipeline = RelationPerQueryPipeline(
        build_config=build_config,
        relations_config=relations_config,
        fallback_config=fallback_config,
        adapter=adapter,
        llm=llm,
        resume=args.resume,
    )

    if args.step == "validate":
        result = pipeline.validate()
        print(f"[Validate] OK: {result}")
        return 0
    if args.step in ("all", "import"):
        pipeline.run_import()
    if args.step in ("all", "gen", "gen+eval"):
        pipeline.run_generation()
    if args.step in ("all", "eval", "gen+eval"):
        pipeline.run_evaluation()
    if args.step == "del":
        pipeline.run_deletion()
    print("[Done] Relation perquery experiment finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
