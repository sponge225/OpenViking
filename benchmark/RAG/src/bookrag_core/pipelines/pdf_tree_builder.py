"""Build a dataset tree by running BookRAG's original PDF flow per document."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from bookrag_core.checkpoint import atomic_write_json, canonical_sha256, file_sha256
from bookrag_core.Index.Tree import DocumentTree
from bookrag_core.pipelines.tree_aggregation import aggregate_document_trees
from bookrag_core.utils.ingest_timer import submit_document_task
from tqdm import tqdm

log = logging.getLogger(__name__)

_DOCUMENT_MANIFEST = "pdf_tree_manifest.json"
_DOCUMENT_MANIFEST_VERSION = 1


def _safe_cache_name(sample_id: str, source_path: Path) -> str:
    sample_part = re.sub(r"[^\w.\-]+", "_", str(sample_id), flags=re.UNICODE)
    sample_part = sample_part.strip("._-")[:80] or "sample"
    path_hash = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    return f"{sample_part}__{path_hash}"


def _parser_fingerprint(cfg) -> str:
    mineru = cfg.mineru
    return canonical_sha256(
        {
            "backend": mineru.backend,
            "method": mineru.method,
            "lang": mineru.lang,
            "server_url": mineru.server_url,
        }
    )


def _tree_fingerprint(cfg) -> str:
    llm = cfg.llm
    return canonical_sha256(
        {
            "parser": _parser_fingerprint(cfg),
            "tree": {
                "node_keywords": cfg.tree.node_keywords,
            },
            "llm": {
                "model": llm.model_name,
                "api_base": llm.api_base,
                "max_tokens": llm.max_tokens,
                "temperature": llm.temperature,
            },
        }
    )


def _load_manifest(cache_dir: Path) -> dict[str, Any] | None:
    path = cache_dir / _DOCUMENT_MANIFEST
    if not path.is_file():
        return None
    try:
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("format_version") != _DOCUMENT_MANIFEST_VERSION:
        return None
    return value


def _build_original_pdf_tree(cfg, *, reforce: bool) -> DocumentTree:
    # Lazy import keeps non-PDF commands usable until the MinerU extra is installed.
    # Dataset summaries are generated once after all document trees aggregate.
    from bookrag_core.pipelines.doc_tree_builder import build_tree_structure_from_pdf

    return build_tree_structure_from_pdf(cfg, reforce=reforce)


def _prepare_original_pdf_content(cfg, *, reforce: bool) -> None:
    # This hook is intentionally separate from tree construction: PDFium in
    # MinerU's local pipeline is not thread-safe, so dataset ingestion invokes
    # it serially before starting document post-processing workers.
    from bookrag_core.pipelines.doc_tree_builder import prepare_pdf_content

    prepare_pdf_content(cfg, reforce=reforce)


@dataclass(frozen=True)
class _DocumentContext:
    source_path: Path
    cache_dir: Path
    document_cfg: Any
    source_hash: str
    parser_fingerprint: str
    tree_fingerprint: str
    parser_matches: bool
    tree_matches: bool
    tree_path: Path


def _document_context(cfg, pdf_path: str | Path, sample_id: str) -> _DocumentContext:
    source_path = Path(pdf_path).expanduser().resolve()
    if source_path.suffix.lower() != ".pdf":
        raise ValueError(f"BookRAG MinerU input must be PDF: {source_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"BookRAG PDF source does not exist: {source_path}")

    cache_root = Path(cfg.save_path).resolve() / "pdf_trees"
    cache_dir = cache_root / _safe_cache_name(str(sample_id), source_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    document_cfg = cfg.model_copy(deep=True)
    document_cfg.pdf_path = str(source_path)
    document_cfg.save_path = str(cache_dir)
    # doc_workers is the only concurrency layer during per-document tree
    # construction.  Avoid multiplying it by another LLM worker pool inside
    # pdf_info_refiner.
    document_cfg.llm.max_workers = 1

    source_hash = file_sha256(source_path)
    parser_fingerprint = _parser_fingerprint(document_cfg)
    tree_fingerprint = _tree_fingerprint(document_cfg)
    manifest = _load_manifest(cache_dir)
    source_matches = bool(
        manifest
        and manifest.get("source_path") == str(source_path)
        and manifest.get("source_sha256") == source_hash
    )
    parser_matches = bool(
        source_matches and manifest.get("parser_fingerprint") == parser_fingerprint
    )
    tree_matches = bool(
        parser_matches
        and manifest.get("status") == "complete"
        and manifest.get("tree_fingerprint") == tree_fingerprint
    )
    return _DocumentContext(
        source_path=source_path,
        cache_dir=cache_dir,
        document_cfg=document_cfg,
        source_hash=source_hash,
        parser_fingerprint=parser_fingerprint,
        tree_fingerprint=tree_fingerprint,
        parser_matches=parser_matches,
        tree_matches=tree_matches,
        tree_path=Path(DocumentTree.get_save_path(str(cache_dir))),
    )


def _invalidate_stale_tree(context: _DocumentContext) -> None:
    if not context.tree_path.is_file() or context.tree_matches:
        return
    # Keep MinerU output when only the LLM/tree settings changed. Removing
    # tree.pkl makes the original builder reuse merged_content.json.
    try:
        context.tree_path.unlink()
    except OSError as error:
        raise RuntimeError(
            f"Cannot invalidate cached PDF tree: {context.tree_path}"
        ) from error


def _mark_document_building(context: _DocumentContext, sample_id: str) -> None:
    atomic_write_json(
        context.cache_dir / _DOCUMENT_MANIFEST,
        {
            "format_version": _DOCUMENT_MANIFEST_VERSION,
            "status": "building",
            "sample_id": str(sample_id),
            "source_path": str(context.source_path),
            "source_sha256": context.source_hash,
            "parser_fingerprint": context.parser_fingerprint,
            "tree_fingerprint": context.tree_fingerprint,
        },
    )


def prepare_pdf_document_content(
    cfg,
    pdf_path: str | Path,
    sample_id: str,
) -> None:
    """Serially prepare one document's durable MinerU merged-content cache."""
    context = _document_context(cfg, pdf_path, sample_id)
    if context.tree_matches and context.tree_path.is_file():
        log.info(
            "[BookRAG PDF] MinerU cache already covered by complete tree: sample=%s",
            sample_id,
        )
        return

    _invalidate_stale_tree(context)
    _mark_document_building(context, sample_id)
    log.info(
        "[BookRAG PDF] Serial MinerU preparation: sample=%s, source=%s",
        sample_id,
        context.source_path,
    )
    _prepare_original_pdf_content(
        context.document_cfg,
        reforce=not context.parser_matches,
    )


def _quiet_worker_logging() -> None:
    logging.getLogger("bookrag_core").setLevel(logging.WARNING)
    logging.getLogger("mineru").setLevel(logging.WARNING)
    try:
        from loguru import logger as loguru_logger

        loguru_logger.remove()
    except Exception:
        pass


def _prepare_pdf_document_content_task(args: tuple[Any, str, str]) -> dict[str, Any]:
    cfg, pdf_path, sample_id = args
    started = time.monotonic()
    context = _document_context(cfg, pdf_path, sample_id)
    worker_log_path = context.cache_dir / "mineru_prepare.log"
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    with worker_log_path.open("a", encoding="utf-8") as worker_log:
        worker_log.write(
            f"\n=== MinerU preparation started: sample={sample_id}, "
            f"source={pdf_path} ===\n"
        )
        worker_log.flush()
        with redirect_stdout(worker_log), redirect_stderr(worker_log):
            _quiet_worker_logging()
            prepare_pdf_document_content(cfg, pdf_path, sample_id)
        worker_log.write(
            f"=== MinerU preparation finished: sample={sample_id}, "
            f"elapsed={time.monotonic() - started:.2f}s ===\n"
        )
    return {
        "sample_id": str(sample_id),
        "source_path": str(pdf_path),
        "time": time.monotonic() - started,
        "log_path": str(worker_log_path),
    }


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def prepare_dataset_mineru_content(
    cfg,
    documents: Sequence[tuple[str, str | Path]],
    *,
    dataset_name: str = "dataset",
) -> dict[str, Any]:
    """Prepare durable MinerU merged-content caches for every PDF document."""
    ordered = sorted(
        ((str(sample_id), Path(path).resolve()) for sample_id, path in documents),
        key=lambda item: (item[0], os.path.normcase(str(item[1]))),
    )
    total = len(ordered)
    if not total:
        return {
            "dataset": dataset_name,
            "documents": 0,
            "mineru_workers": 0,
            "time": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    mineru_workers = min(
        total,
        max(1, int(getattr(cfg, "mineru_workers", 1))),
    )
    started = time.monotonic()
    log.info(
        "[BookRAG PDF] MinerU preparation: documents=%d, mineru_workers=%d.",
        total,
        mineru_workers,
    )

    per_document: list[dict[str, Any]] = []
    with tqdm(
        total=total,
        desc="BookRAG MinerU",
        unit="pdf",
        dynamic_ncols=True,
    ) as pbar:
        tasks = [(cfg, str(path), sample_id) for sample_id, path in ordered]
        with ProcessPoolExecutor(max_workers=mineru_workers) as executor:
            futures = {
                executor.submit(_prepare_pdf_document_content_task, task): task[2]
                for task in tasks
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                sample_id = futures[future]
                result = future.result()
                per_document.append(result)
                elapsed = time.monotonic() - started
                remaining = total - completed
                eta = (elapsed / completed) * remaining if completed else 0
                pbar.set_postfix_str(
                    f"last={sample_id}, eta={_format_eta(eta)}",
                    refresh=False,
                )
                pbar.update(1)

    elapsed = time.monotonic() - started
    task_time = sum(float(item.get("time", 0.0) or 0.0) for item in per_document)
    log.info(
        "[BookRAG PDF] MinerU preparation finished: documents=%d, wall=%.2fs, "
        "task_sum=%.2fs, mineru_workers=%d.",
        total,
        elapsed,
        task_time,
        mineru_workers,
    )
    return {
        "dataset": dataset_name,
        "documents": total,
        "mineru_workers": mineru_workers,
        "time": elapsed,
        "task_time": task_time,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _apply_provenance(
    tree: DocumentTree,
    *,
    sample_id: str,
    source_path: Path,
) -> None:
    tree.meta_info.sample_id = sample_id
    tree.meta_info.file_name = source_path.name
    tree.meta_info.file_path = str(source_path)
    for node in tree.nodes:
        node.meta_info.sample_id = sample_id
        node.meta_info.file_name = source_path.name
        node.meta_info.file_path = str(source_path)
        node.meta_info.local_index_id = node.index_id


def build_tree_from_pdf_document(
    cfg,
    pdf_path: str | Path,
    sample_id: str,
) -> DocumentTree:
    """Run or reuse the original BookRAG PDF tree pipeline for one document."""
    context = _document_context(cfg, pdf_path, sample_id)
    _invalidate_stale_tree(context)
    reforce = not context.parser_matches
    log.info(
        "[BookRAG PDF] %s document tree: sample=%s, source=%s, cache=%s",
        "Reusing" if context.tree_matches and context.tree_path.is_file() else "Building",
        sample_id,
        context.source_path,
        context.cache_dir,
    )
    # Persist parser identity before the expensive call. If the process stops
    # after MinerU has written merged_content.json but before tree.pkl is ready,
    # the next run enters the original builder with reforce=False and reuses
    # that completed MinerU output.
    _mark_document_building(context, sample_id)
    tree = _build_original_pdf_tree(context.document_cfg, reforce=reforce)
    if not isinstance(tree, DocumentTree) or tree.root_node is None:
        raise RuntimeError(
            f"BookRAG produced an invalid PDF tree for {context.source_path}"
        )
    _apply_provenance(
        tree,
        sample_id=str(sample_id),
        source_path=context.source_path,
    )
    tree.save_dir = str(context.cache_dir)
    tree.save_to_file()
    atomic_write_json(
        context.cache_dir / _DOCUMENT_MANIFEST,
        {
            "format_version": _DOCUMENT_MANIFEST_VERSION,
            "status": "complete",
            "sample_id": str(sample_id),
            "source_path": str(context.source_path),
            "source_sha256": context.source_hash,
            "parser_fingerprint": context.parser_fingerprint,
            "tree_fingerprint": context.tree_fingerprint,
            "node_count": len(tree.nodes),
        },
    )
    return tree


def build_dataset_tree_from_pdf(
    cfg,
    documents: Sequence[tuple[str, str | Path]],
    *,
    dataset_name: str = "dataset",
) -> DocumentTree:
    """Build every PDF independently, then add only one dataset root."""
    ordered = sorted(
        ((str(sample_id), Path(path).resolve()) for sample_id, path in documents),
        key=lambda item: (item[0], os.path.normcase(str(item[1]))),
    )
    if not ordered:
        return aggregate_document_trees([], cfg=cfg, dataset_name=dataset_name)

    doc_workers = min(
        len(ordered),
        max(1, int(getattr(cfg, "doc_workers", 1))),
    )
    log.info(
        "[BookRAG PDF] Phase 1/2: preparing MinerU content for %d documents.",
        len(ordered),
    )
    prepare_dataset_mineru_content(cfg, ordered, dataset_name=dataset_name)

    log.info(
        "[BookRAG PDF] Phase 2/2: building %d structural document trees "
        "with doc_workers=%d and no nested document pool.",
        len(ordered),
        doc_workers,
    )
    trees: list[DocumentTree | None] = [None] * len(ordered)
    with ThreadPoolExecutor(
        max_workers=doc_workers,
        thread_name_prefix="bookrag-document",
    ) as executor:
        futures = {
            submit_document_task(
                executor,
                build_tree_from_pdf_document,
                cfg,
                path,
                sample_id,
            ): position
            for position, (sample_id, path) in enumerate(ordered)
        }
        with tqdm(
            total=len(futures),
            desc="BookRAG trees",
            unit="pdf",
            dynamic_ncols=True,
        ) as pbar:
            for future in as_completed(futures):
                position = futures[future]
                trees[position] = future.result()
                sample_id = ordered[position][0]
                node_count = len(getattr(trees[position], "nodes", []) or [])
                pbar.set_postfix_str(
                    f"last={sample_id}, nodes={node_count}",
                    refresh=False,
                )
                pbar.update(1)

    # Completion order must not affect aggregate node IDs or provenance.
    completed_trees = [tree for tree in trees if tree is not None]
    if len(completed_trees) != len(ordered):
        raise RuntimeError("BookRAG did not produce a tree for every PDF document")
    return aggregate_document_trees(completed_trees, cfg=cfg, dataset_name=dataset_name)
