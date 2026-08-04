#!/usr/bin/env python3
"""Reuse BookRAG per-PDF MinerU/tree caches across compatible configs.

This is useful when a split dataset, storage root, or experiment variant uses
documents that have already run BookRAG PDF ingestion.  The script materializes
target PDFs, finds matching source caches by raw/PDF content hashes and BookRAG
fingerprints, copies the cache into the target index directory, and rewrites the
manifest/provenance to the target paths.

By default, full trees are reused only when the target tree fingerprint matches.
When the LLM-dependent tree fingerprint differs, the script falls back to
reusing only MinerU content so the target config can rebuild its own tree.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

RAG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG_ROOT))
sys.path.insert(0, str(RAG_ROOT / "src"))
load_dotenv(RAG_ROOT / ".env")

from bookrag_core.Index.Tree import DocumentTree
from bookrag_core.checkpoint import atomic_write_json, file_sha256
from bookrag_core.pipelines.pdf_tree_builder import (
    _DOCUMENT_MANIFEST,
    _DOCUMENT_MANIFEST_VERSION,
    _apply_provenance,
    _parser_fingerprint,
    _safe_cache_name,
    _tree_fingerprint,
)
from bookrag_runner import build_bookrag_system_config
from core.pdf_materializer import PdfMaterializer


@dataclass(frozen=True)
class CacheRecord:
    cache_dir: Path
    manifest: dict[str, Any]
    has_tree: bool
    has_mineru: bool


@dataclass(frozen=True)
class MaterializedPdfRecord:
    sample_ids: tuple[str, ...]
    source_sha256: str
    output_pdf_path: Path
    output_pdf_sha256: str


def _resolve_path(path_str: str | os.PathLike[str], base_path: Path = RAG_ROOT) -> str:
    text = str(path_str)
    if os.path.isabs(text):
        return os.path.normpath(text)
    return os.path.normpath(str(base_path / text))


def load_resolved_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset_name = config.get("dataset_name", "UnknownDataset")
    retrieval_topk = config.get("execution", {}).get("retrieval_topk", 5)
    format_vars = {
        "dataset_name": dataset_name,
        "retrieval_topk": retrieval_topk,
        "search_limit": config.get("vikingbot", {}).get("search_limit", ""),
        "max_iterations": config.get("vikingbot", {}).get("max_iterations", ""),
    }

    for key in (
        "dataset_path",
        "output_dir",
        "vector_store",
        "log_file",
        "doc_output_dir",
    ):
        if key in config.get("paths", {}):
            rendered = str(config["paths"][key]).format(**format_vars)
            config["paths"][key] = _resolve_path(rendered)

    bookrag = config.get("bookrag") or {}
    for key in ("index_dir", "document_store_root", "qa_doc_mapping_path"):
        if bookrag.get(key):
            rendered = str(bookrag[key]).format(**format_vars)
            bookrag[key] = _resolve_path(rendered)

    config["_config_path"] = str(config_path)
    return config


def materialize_target_pdfs(config: dict[str, Any]) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    adapter_cfg = config.get("adapter") or {}
    module_path = adapter_cfg.get("module", "src.adapters.versionrag_adapter")
    class_name = adapter_cfg.get("class_name", "VersionRAGAdapter")
    module = importlib.import_module(module_path)
    adapter_class = getattr(module, class_name)
    adapter = adapter_class(raw_file_path=config["paths"]["dataset_path"])

    qa_doc_mapping_path = (config.get("bookrag") or {}).get("qa_doc_mapping_path")
    configure_mapping = getattr(adapter, "configure_qa_doc_mapping", None)
    if qa_doc_mapping_path and callable(configure_mapping):
        configure_mapping(qa_doc_mapping_path)

    doc_dir = config["paths"]["doc_output_dir"]
    source_documents = adapter.prepare_pdf_sources(doc_dir)
    pdf_preprocessing = (config.get("bookrag") or {}).get("pdf_preprocessing") or {}
    pdf_output_dir = pdf_preprocessing.get("output_dir") or os.path.join(doc_dir, "pdfs")
    materialized, stats = PdfMaterializer(pdf_output_dir).materialize(source_documents)
    return [
        (str(document.sample_id), Path(document.doc_path).expanduser().resolve())
        for document in materialized
    ], stats


def _pdf_output_dir(config: dict[str, Any]) -> Path:
    doc_dir = config["paths"]["doc_output_dir"]
    pdf_preprocessing = (config.get("bookrag") or {}).get("pdf_preprocessing") or {}
    return Path(pdf_preprocessing.get("output_dir") or os.path.join(doc_dir, "pdfs")).resolve()


def load_materialized_pdf_records(config: dict[str, Any]) -> tuple[
    dict[tuple[str, str], MaterializedPdfRecord],
    dict[str, MaterializedPdfRecord],
    Path,
]:
    """Load materialized-PDF records keyed by raw source hash and output path."""
    output_dir = _pdf_output_dir(config)
    manifest_path = output_dir / "_pdf_materialization_manifest.json"
    by_raw: dict[tuple[str, str], MaterializedPdfRecord] = {}
    by_output: dict[str, MaterializedPdfRecord] = {}
    if not manifest_path.is_file():
        return by_raw, by_output, manifest_path
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return by_raw, by_output, manifest_path
    for item in manifest.get("items") or []:
        if not isinstance(item, dict):
            continue
        source_sha = str(item.get("source_sha256") or "")
        output_path = item.get("output_pdf_path")
        output_sha = str(item.get("output_pdf_sha256") or "")
        sample_ids = tuple(str(value) for value in (item.get("sample_ids") or []))
        if not source_sha or not output_path or not output_sha or not sample_ids:
            continue
        record = MaterializedPdfRecord(
            sample_ids=sample_ids,
            source_sha256=source_sha,
            output_pdf_path=Path(output_path).expanduser().resolve(),
            output_pdf_sha256=output_sha,
        )
        for sample_id in sample_ids:
            by_raw[(sample_id, source_sha)] = record
        by_raw.setdefault(("*", source_sha), record)
        by_output[str(record.output_pdf_path)] = record
    return by_raw, by_output, manifest_path


def update_materialization_output_hash(
    manifest_path: Path,
    *,
    output_pdf_path: Path,
    output_pdf_sha256: str,
) -> None:
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    changed = False
    for item in manifest.get("items") or []:
        if not isinstance(item, dict):
            continue
        if item.get("output_pdf_path") != str(output_pdf_path):
            continue
        item["output_pdf_sha256"] = output_pdf_sha256
        changed = True
    if changed:
        atomic_write_json(manifest_path, manifest)


def _has_mineru_content(cache_dir: Path) -> bool:
    return any(cache_dir.rglob("*_merged_content.json"))


def build_source_cache_index(source_save_path: Path) -> tuple[
    dict[tuple[str, str, str, str], CacheRecord],
    dict[tuple[str, str, str], CacheRecord],
]:
    pdf_trees = source_save_path / "pdf_trees"
    if not pdf_trees.is_dir():
        raise FileNotFoundError(f"Source pdf_trees directory not found: {pdf_trees}")

    tree_records: dict[tuple[str, str, str, str], CacheRecord] = {}
    mineru_records: dict[tuple[str, str, str], CacheRecord] = {}
    for manifest_path in sorted(pdf_trees.glob(f"*/{_DOCUMENT_MANIFEST}")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("format_version") != _DOCUMENT_MANIFEST_VERSION:
            continue
        source_sha = str(manifest.get("source_sha256") or "")
        parser_fp = str(manifest.get("parser_fingerprint") or "")
        tree_fp = str(manifest.get("tree_fingerprint") or "")
        sample_id = str(manifest.get("sample_id") or "")
        if not source_sha or not parser_fp or not tree_fp or not sample_id:
            continue

        cache_dir = manifest_path.parent
        has_tree = (
            manifest.get("status") == "complete"
            and Path(DocumentTree.get_save_path(str(cache_dir))).is_file()
        )
        has_mineru = _has_mineru_content(cache_dir)
        if not has_tree and not has_mineru:
            continue

        record = CacheRecord(
            cache_dir=cache_dir,
            manifest=manifest,
            has_tree=has_tree,
            has_mineru=has_mineru,
        )
        # Prefer exact sample matches, but also allow same-content fallback by
        # storing a synthetic wildcard sample key.
        tree_records[(sample_id, source_sha, parser_fp, tree_fp)] = record
        tree_records.setdefault(("*", source_sha, parser_fp, tree_fp), record)
        if has_mineru:
            mineru_records[(sample_id, source_sha, parser_fp)] = record
            mineru_records.setdefault(("*", source_sha, parser_fp), record)

    return tree_records, mineru_records


def _target_manifest_valid(
    cache_dir: Path,
    *,
    sample_id: str,
    source_path: Path,
    source_sha: str,
    parser_fingerprint: str,
    tree_fingerprint: str,
) -> bool:
    manifest_path = cache_dir / _DOCUMENT_MANIFEST
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("format_version") == _DOCUMENT_MANIFEST_VERSION
        and manifest.get("sample_id") == sample_id
        and manifest.get("source_path") == str(source_path)
        and manifest.get("source_sha256") == source_sha
        and manifest.get("parser_fingerprint") == parser_fingerprint
        and manifest.get("tree_fingerprint") == tree_fingerprint
        and (Path(DocumentTree.get_save_path(str(cache_dir))).is_file() or _has_mineru_content(cache_dir))
    )


def transplant_cache(
    record: CacheRecord,
    *,
    target_cache_dir: Path,
    sample_id: str,
    target_pdf: Path,
    source_sha: str,
    parser_fingerprint: str,
    tree_fingerprint: str,
    copy_tree: bool,
    force: bool,
    dry_run: bool,
) -> str:
    if target_cache_dir.exists():
        if _target_manifest_valid(
            target_cache_dir,
            sample_id=sample_id,
            source_path=target_pdf,
            source_sha=source_sha,
            parser_fingerprint=parser_fingerprint,
            tree_fingerprint=tree_fingerprint,
        ):
            return "already-valid"
        if not force:
            return "exists-skip"

    if dry_run:
        return "would-copy-tree" if copy_tree and record.has_tree else "would-copy-mineru"

    target_cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target_cache_dir.name}.",
            dir=target_cache_dir.parent,
        )
    )
    try:
        shutil.rmtree(tmp_dir)
        shutil.copytree(record.cache_dir, tmp_dir)

        tree_path = Path(DocumentTree.get_save_path(str(tmp_dir)))
        node_count = record.manifest.get("node_count")
        if copy_tree and record.has_tree and tree_path.is_file():
            tree = DocumentTree.load_from_file(str(tree_path))
            _apply_provenance(tree, sample_id=sample_id, source_path=target_pdf)
            tree.save_dir = str(tmp_dir)
            tree.save_to_file()
            node_count = len(tree.nodes)
        else:
            if tree_path.exists():
                tree_path.unlink()
            node_count = None

        manifest = dict(record.manifest)
        manifest.update(
            {
                "format_version": _DOCUMENT_MANIFEST_VERSION,
                "status": "complete" if copy_tree and record.has_tree else "building",
                "sample_id": sample_id,
                "source_path": str(target_pdf),
                "source_sha256": source_sha,
                "parser_fingerprint": parser_fingerprint,
                "tree_fingerprint": tree_fingerprint,
            }
        )
        if node_count is not None:
            manifest["node_count"] = node_count
        else:
            manifest.pop("node_count", None)
        atomic_write_json(tmp_dir / _DOCUMENT_MANIFEST, manifest)

        if target_cache_dir.exists():
            shutil.rmtree(target_cache_dir)
        os.replace(tmp_dir, target_cache_dir)
        return "copied-tree" if copy_tree and record.has_tree else "copied-mineru"
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def reuse_for_target(
    *,
    source_tree_index: dict[tuple[str, str, str, str], CacheRecord],
    source_mineru_index: dict[tuple[str, str, str], CacheRecord],
    source_pdf_by_raw: dict[tuple[str, str], MaterializedPdfRecord],
    target_config_path: Path,
    reuse_level: str,
    force: bool,
    dry_run: bool,
) -> dict[str, int]:
    config = load_resolved_config(target_config_path)
    core_cfg = build_bookrag_system_config(config)
    target_save_path = Path(core_cfg.save_path).expanduser().resolve()
    target_docs, materialize_stats = materialize_target_pdfs(config)
    _, target_pdf_by_output, target_materialization_manifest = load_materialized_pdf_records(
        config
    )

    stats = {
        "documents": len(target_docs),
        "materialized": int(materialize_stats.get("pdf_count", 0) or 0),
        "matched": 0,
        "copied_tree": 0,
        "copied_mineru": 0,
        "copied_pdf": 0,
        "would_copy_pdf": 0,
        "already_valid": 0,
        "exists_skip": 0,
        "miss": 0,
    }

    for sample_id, target_pdf in sorted(target_docs, key=lambda item: item[0]):
        document_cfg = core_cfg.model_copy(deep=True)
        document_cfg.pdf_path = str(target_pdf)
        target_cache_dir = (
            target_save_path / "pdf_trees" / _safe_cache_name(sample_id, target_pdf)
        )
        document_cfg.save_path = str(target_cache_dir)
        source_sha = file_sha256(target_pdf)
        parser_fp = _parser_fingerprint(document_cfg)
        tree_fp = _tree_fingerprint(document_cfg)

        record = None
        copy_tree = False
        if reuse_level in {"auto", "tree"}:
            record = source_tree_index.get((sample_id, source_sha, parser_fp, tree_fp))
            if record is None:
                record = source_tree_index.get(("*", source_sha, parser_fp, tree_fp))
            copy_tree = record is not None and record.has_tree
        if record is None and reuse_level in {"auto", "mineru"}:
            record = source_mineru_index.get((sample_id, source_sha, parser_fp))
            if record is None:
                record = source_mineru_index.get(("*", source_sha, parser_fp))
            copy_tree = False
        if record is None:
            target_materialized = target_pdf_by_output.get(str(target_pdf))
            source_materialized = None
            if target_materialized is not None:
                source_materialized = source_pdf_by_raw.get(
                    (sample_id, target_materialized.source_sha256)
                ) or source_pdf_by_raw.get(("*", target_materialized.source_sha256))
            if source_materialized is not None:
                candidate_sha = source_materialized.output_pdf_sha256
                candidate = None
                if reuse_level in {"auto", "tree"}:
                    candidate = source_tree_index.get(
                        (sample_id, candidate_sha, parser_fp, tree_fp)
                    ) or source_tree_index.get(("*", candidate_sha, parser_fp, tree_fp))
                    copy_tree = candidate is not None and candidate.has_tree
                if candidate is None and reuse_level in {"auto", "mineru"}:
                    candidate = source_mineru_index.get(
                        (sample_id, candidate_sha, parser_fp)
                    ) or source_mineru_index.get(("*", candidate_sha, parser_fp))
                    copy_tree = False
                if candidate is not None:
                    record = candidate
                    source_sha = candidate_sha
                    if dry_run:
                        stats["would_copy_pdf"] += 1
                        print(
                            f"WOULD-COPY-PDF {sample_id}: "
                            f"{source_materialized.output_pdf_path.name}"
                        )
                    else:
                        shutil.copy2(source_materialized.output_pdf_path, target_pdf)
                        update_materialization_output_hash(
                            target_materialization_manifest,
                            output_pdf_path=target_pdf,
                            output_pdf_sha256=source_sha,
                        )
                        stats["copied_pdf"] += 1
        if record is None:
            stats["miss"] += 1
            print(f"MISS  {sample_id}: no matching source cache")
            continue

        stats["matched"] += 1
        action = transplant_cache(
            record,
            target_cache_dir=target_cache_dir,
            sample_id=sample_id,
            target_pdf=target_pdf,
            source_sha=source_sha,
            parser_fingerprint=parser_fp,
            tree_fingerprint=tree_fp,
            copy_tree=copy_tree,
            force=force,
            dry_run=dry_run,
        )
        if action in {"copied-tree", "would-copy-tree"}:
            stats["copied_tree"] += 1
        elif action in {"copied-mineru", "would-copy-mineru"}:
            stats["copied_mineru"] += 1
        elif action == "already-valid":
            stats["already_valid"] += 1
        elif action == "exists-skip":
            stats["exists_skip"] += 1
        print(f"{action.upper():13s} {sample_id}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reuse BookRAG MinerU/tree per-PDF caches from a full dataset index."
    )
    parser.add_argument(
        "--source-config",
        default="config/versionrag/versionrag_bookrag_config.yaml",
        help="Config whose BookRAG index already contains pdf_trees caches.",
    )
    parser.add_argument(
        "--target-config",
        action="append",
        required=True,
        help="Target split config. Repeat this option for multiple splits.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target per-PDF cache directories.",
    )
    parser.add_argument(
        "--reuse-level",
        choices=["auto", "tree", "mineru"],
        default="auto",
        help=(
            "Reuse strategy. auto reuses full trees only when tree fingerprints "
            "match and otherwise reuses MinerU content; tree requires matching "
            "tree fingerprints; mineru never copies tree.pkl."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be copied.",
    )
    args = parser.parse_args()

    source_config = load_resolved_config(Path(args.source_config))
    source_core_cfg = build_bookrag_system_config(source_config)
    source_save_path = Path(source_core_cfg.save_path).expanduser().resolve()
    source_tree_index, source_mineru_index = build_source_cache_index(source_save_path)
    source_pdf_by_raw, _, _ = load_materialized_pdf_records(source_config)
    print(
        f"Loaded {len(source_tree_index)} tree and {len(source_mineru_index)} "
        f"MinerU source cache lookup entries from "
        f"{source_save_path / 'pdf_trees'}"
    )

    totals: dict[str, dict[str, int]] = {}
    for target in args.target_config:
        target_path = Path(target)
        print(f"\n== Target: {target_path} ==")
        totals[str(target_path)] = reuse_for_target(
            source_tree_index=source_tree_index,
            source_mineru_index=source_mineru_index,
            source_pdf_by_raw=source_pdf_by_raw,
            target_config_path=target_path,
            reuse_level=args.reuse_level,
            force=args.force,
            dry_run=args.dry_run,
        )
        print(json.dumps(totals[str(target_path)], ensure_ascii=False, indent=2))

    print("\nSummary:")
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
