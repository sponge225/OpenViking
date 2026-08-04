#!/usr/bin/env python3
"""Probe official BookRAG PDF tree node counts for benchmark PDFs.

This script intentionally calls the original PDF path:

    build_tree_from_pdf(cfg)

It does not use the benchmark Markdown tree builder and does not build KG/GBC.
By default node summaries are disabled so the result measures the PDF/MinerU
tree granularity without paying per-node summary cost.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


RAG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG_ROOT / "src"))
sys.path.insert(0, str(RAG_ROOT))

from bookrag_core.Index.Tree import NodeType
from bookrag_runner import build_bookrag_system_config


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _resolve_path(value: str | None, base: Path) -> str | None:
    if not value:
        return value
    rendered = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(rendered)
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def _resolve_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    dataset_name = config.get("dataset_name", "UnknownDataset")
    execution = config.get("execution") or {}
    format_vars = {
        "dataset_name": dataset_name,
        "retrieval_topk": execution.get("retrieval_topk", 5),
        "search_limit": (config.get("vikingbot") or {}).get("search_limit", ""),
        "max_iterations": (config.get("vikingbot") or {}).get("max_iterations", ""),
    }

    paths = config.get("paths") or {}
    for key in ["dataset_path", "output_dir", "vector_store", "log_file", "doc_output_dir"]:
        if key in paths and paths[key]:
            paths[key] = _resolve_path(str(paths[key]).format(**format_vars), RAG_ROOT)

    bookrag = config.get("bookrag") or {}
    if bookrag.get("index_dir"):
        bookrag["index_dir"] = _resolve_path(
            str(bookrag["index_dir"]).format(**format_vars), RAG_ROOT
        )
    return config


def _slug(path: Path) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return value[:160] or "pdf"


def _content(node: Any) -> str:
    meta = getattr(node, "meta_info", None)
    if meta is None:
        return ""
    pieces = [
        getattr(meta, "content", None),
        getattr(meta, "caption", None),
        getattr(meta, "footnote", None),
        getattr(meta, "table_body", None),
    ]
    return "\n".join(str(item) for item in pieces if item)


def _token_counter(tokenizer: str):
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(tokenizer)
        return lambda text: len(encoding.encode(text or ""))
    except Exception:
        return lambda text: len((text or "").split())


def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, int(len(values) * ratio))
    return sorted(values)[index]


def _summarize_tree(tree: Any, tokenizer: str) -> dict[str, Any]:
    count_tokens = _token_counter(tokenizer)
    rows = []
    for node in getattr(tree, "nodes", []) or []:
        node_type = str(getattr(node, "type", None))
        text = _content(node)
        rows.append(
            {
                "type": node_type,
                "tokens": count_tokens(text),
                "chars": len(text),
                "depth": int(getattr(node, "depth", 0) or 0),
            }
        )

    tokens = [row["tokens"] for row in rows]
    by_type = Counter(row["type"] for row in rows)
    token_stats_by_type = {}
    for node_type in sorted(by_type):
        type_tokens = [row["tokens"] for row in rows if row["type"] == node_type]
        token_stats_by_type[node_type] = {
            "nodes": len(type_tokens),
            "tokens": sum(type_tokens),
            "avg_tokens": round(sum(type_tokens) / len(type_tokens), 2)
            if type_tokens
            else 0,
            "p50_tokens": _percentile(type_tokens, 0.50),
            "p90_tokens": _percentile(type_tokens, 0.90),
            "max_tokens": max(type_tokens) if type_tokens else 0,
        }

    non_root = [row for row in rows if row["type"] != str(NodeType.ROOT)]
    return {
        "nodes": len(rows),
        "non_root_nodes": len(non_root),
        "max_depth": max((row["depth"] for row in rows), default=0),
        "tokens": sum(tokens),
        "avg_tokens_per_node": round(sum(tokens) / len(rows), 2) if rows else 0,
        "p50_tokens": _percentile(tokens, 0.50),
        "p90_tokens": _percentile(tokens, 0.90),
        "max_tokens": max(tokens) if tokens else 0,
        "node_type_counts": dict(sorted(by_type.items())),
        "token_stats_by_type": token_stats_by_type,
    }


def _build_pdf_tree(
    config: dict[str, Any],
    pdf_path: Path,
    save_path: Path,
    *,
    force: bool,
    mineru_backend: str | None,
    mineru_method: str | None,
    mineru_lang: str | None,
) -> Any:
    local_config = json.loads(json.dumps(config))
    local_config.setdefault("bookrag", {})["index_dir"] = str(save_path)
    cfg = build_bookrag_system_config(local_config)
    cfg.pdf_path = str(pdf_path.resolve())
    cfg.save_path = str(save_path.resolve())
    cfg.tree.node_summary = False
    cfg.tree.node_keywords = False
    cfg.tree.use_vlm = False
    if mineru_backend:
        cfg.mineru.backend = mineru_backend
    if mineru_method:
        cfg.mineru.method = mineru_method
    if mineru_lang:
        cfg.mineru.lang = mineru_lang
    from bookrag_core.pipelines.doc_tree_builder import build_tree_from_pdf

    return build_tree_from_pdf(cfg, reforce=force)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(RAG_ROOT / "config/versionrag/versionrag_bookrag_config.yaml"),
        help="Benchmark config used for LLM/API settings.",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(RAG_ROOT / "datasets/VersionRAG/data/raw"),
        help="Directory containing raw VersionRAG files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RAG_ROOT / "ov_storage/VersionRAG/.official_pdf_tree_probe"),
        help="Directory for per-PDF official tree caches and report.json.",
    )
    parser.add_argument("--pdf", action="append", help="Specific PDF path to analyze.")
    parser.add_argument("--limit", type=int, default=None, help="Only analyze first N PDFs.")
    parser.add_argument("--force", action="store_true", help="Rebuild cached PDF trees.")
    parser.add_argument("--mineru-backend", default=None, help="Override cfg.mineru.backend.")
    parser.add_argument("--mineru-method", default=None, help="Override cfg.mineru.method.")
    parser.add_argument("--mineru-lang", default=None, help="Override cfg.mineru.lang.")
    parser.add_argument("--tokenizer", default="cl100k_base")
    args = parser.parse_args()

    load_dotenv(RAG_ROOT / ".env")
    config = _resolve_config_paths(_read_yaml(Path(args.config).resolve()))

    if args.pdf:
        pdfs = [Path(item).expanduser().resolve() for item in args.pdf]
    else:
        raw_dir = Path(args.raw_dir).expanduser().resolve()
        pdfs = sorted(raw_dir.rglob("*.pdf"))
    if args.limit is not None:
        pdfs = pdfs[: max(args.limit, 0)]
    if not pdfs:
        raise SystemExit("No PDF files found.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    started_all = time.time()
    for index, pdf_path in enumerate(pdfs, start=1):
        save_path = output_dir / _slug(pdf_path)
        if args.force and save_path.exists():
            shutil.rmtree(save_path)
        print(f"[{index}/{len(pdfs)}] {pdf_path}")
        started = time.time()
        item: dict[str, Any] = {
            "pdf_path": str(pdf_path),
            "save_path": str(save_path),
            "status": "ok",
        }
        try:
            tree = _build_pdf_tree(
                config,
                pdf_path,
                save_path,
                force=args.force,
                mineru_backend=args.mineru_backend,
                mineru_method=args.mineru_method,
                mineru_lang=args.mineru_lang,
            )
            item.update(_summarize_tree(tree, args.tokenizer))
        except Exception as error:
            item["status"] = "error"
            item["error"] = repr(error)
        item["seconds"] = round(time.time() - started, 2)
        results.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))

    ok_results = [item for item in results if item.get("status") == "ok"]
    report = {
        "config": str(Path(args.config).resolve()),
        "raw_dir": str(Path(args.raw_dir).expanduser().resolve()),
        "output_dir": str(output_dir),
        "pdf_count": len(pdfs),
        "ok_count": len(ok_results),
        "error_count": len(results) - len(ok_results),
        "total_seconds": round(time.time() - started_all, 2),
        "total_nodes": sum(int(item.get("nodes", 0)) for item in ok_results),
        "total_non_root_nodes": sum(int(item.get("non_root_nodes", 0)) for item in ok_results),
        "total_tokens": sum(int(item.get("tokens", 0)) for item in ok_results),
        "results": results,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to: {report_path}")
    print(
        "Totals: "
        f"ok={report['ok_count']}, errors={report['error_count']}, "
        f"nodes={report['total_nodes']}, tokens={report['total_tokens']}"
    )
    return 0 if report["error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
