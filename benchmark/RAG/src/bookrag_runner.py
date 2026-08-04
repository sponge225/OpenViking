"""BookRAG Core adapter for the OpenViking RAG benchmark."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import quote

from adapters.base import StandardDoc
from bookrag_core.configs.embedding_config import EmbeddingConfig
from bookrag_core.configs.graph_config import GraphConfig
from bookrag_core.configs.llm_config import LLMConfig
from bookrag_core.configs.mineru_config import MinerU
from bookrag_core.configs.rag.gbc_config import GBCRAGConfig
from bookrag_core.configs.rag_config import RAGConfig
from bookrag_core.configs.rerank_config import RerankerConfig
from bookrag_core.configs.system_config import SystemConfig
from bookrag_core.configs.tree_config import TreeConfig
from bookrag_core.configs.vdb_config import VDBConfig
from bookrag_core.configs.vlm_config import VLMConfig
from bookrag_core.checkpoint import (
    STATE_FILENAME,
    STATE_FORMAT_VERSION,
    atomic_write_json,
)
from bookrag_core.construct_index import construct_GBC_index_from_pdfs
from bookrag_core.Index.GBCIndex import GBC
from bookrag_core.Index.Tree import NodeType
from bookrag_core.provider.TokenTracker import TokenTracker
from bookrag_core.rag import create_rag_agent
from bookrag_core.utils.ingest_timer import DocumentWorkTimer
from bookrag_core.utils.utils import num_tokens

log = logging.getLogger("Benchmark")

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SECRET_KEYS = {"api_key", "ak", "sk"}
_MANIFEST_NAME = "manifest.json"
_MANIFEST_BACKEND = "bookrag-gbc"


class _BuildFileLock:
    """Cross-process non-blocking lock released automatically on process exit."""

    def __init__(self, path: Path):
        self.path = path
        self._file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+b")
        self._file.seek(0, os.SEEK_END)
        if self._file.tell() == 0:
            self._file.write(b"0")
            self._file.flush()
        self._file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            self._file.close()
            self._file = None
            raise RuntimeError(
                f"Another BookRAG import is already using {self.path.parent}"
            ) from error

        self._file.seek(0)
        self._file.truncate()
        self._file.write(
            json.dumps({"pid": os.getpid(), "started_at": time.time()}).encode("utf-8")
        )
        self._file.flush()
        self._file.seek(0)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._file is None:
            return
        try:
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None


def _resolve_env_value(value: Any, field_name: str, *, required: bool = True) -> str:
    text = "" if value is None else str(value)

    def replace(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), match.group(0))

    resolved = os.path.expandvars(_ENV_PLACEHOLDER.sub(replace, text)).strip()
    if required and (not resolved or _ENV_PLACEHOLDER.search(resolved)):
        raise ValueError(f"BookRAG requires a resolved value for {field_name}")
    return resolved


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if str(key).lower() in _SECRET_KEYS else _redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def build_bookrag_system_config(config: dict[str, Any]) -> SystemConfig:
    """Translate benchmark YAML fields into the official Core configuration."""
    bookrag = config.get("bookrag") or {}
    tree = bookrag.get("tree") or {}
    graph = bookrag.get("graph") or {}
    gbc = bookrag.get("gbc") or {}
    mineru = bookrag.get("mineru") or {}
    reranker = bookrag.get("reranker") or {}
    execution = config.get("execution") or {}
    llm = config.get("llm") or {}
    embedding = config.get("embedding") or {}

    index_dir = Path(str(bookrag.get("index_dir") or "")).expanduser()
    if not str(index_dir):
        raise ValueError("bookrag.index_dir is required")

    llm_api_key = _resolve_env_value(llm.get("api_key"), "llm.api_key")
    embedding_api_key = _resolve_env_value(embedding.get("api_key"), "embedding.api_key")
    rerank_ak = _resolve_env_value(reranker.get("ak"), "bookrag.reranker.ak")
    rerank_sk = _resolve_env_value(reranker.get("sk"), "bookrag.reranker.sk")

    embedding_config = EmbeddingConfig(
        type="text",
        backend=str(embedding.get("backend", "volcengine")),
        api_key=embedding_api_key,
        api_base=str(embedding.get("base_url", "https://ark.cn-beijing.volces.com/api/v3")),
        model_name=str(embedding.get("model", "doubao-embedding-vision-250615")),
        max_length=int(embedding.get("max_length", 8192)),
        device=str(embedding.get("device", "auto")),
        max_workers=int(embedding.get("max_workers", 8)),
    )
    reranker_config = RerankerConfig(
        model_name=str(reranker.get("model", "doubao-seed-rerank")),
        model_version=str(reranker.get("model_version", "251028")),
        max_length=int(reranker.get("max_length", 8192)),
        device=str(reranker.get("device", "auto")),
        backend=str(reranker.get("backend", "vikingdb")),
        api_base=str(reranker.get("base_url", "")),
        ak=rerank_ak,
        sk=rerank_sk,
        host=str(reranker.get("host", "api-vikingdb.vikingdb.cn-beijing.volces.com")),
        threshold=float(reranker.get("threshold", 0.1)),
        batch_size=int(reranker.get("batch_size", 100)),
    )
    llm_config = LLMConfig(
        model_name=str(llm.get("model") or ""),
        api_key=llm_api_key,
        api_base=str(llm.get("base_url") or ""),
        temperature=float(llm.get("temperature", 0)),
        max_tokens=int(llm.get("max_tokens", 4096)),
        frequency_penalty=float(llm.get("frequency_penalty", 0)),
        presence_penalty=float(llm.get("presence_penalty", 0)),
        backend="openai",
        # Query-level concurrency belongs to execution.max_workers.  Do not
        # multiply it by an implicit per-query LLM pool.
        max_workers=max(1, int(llm.get("max_workers", 1))),
    )
    if not llm_config.model_name or not llm_config.api_base:
        raise ValueError("BookRAG requires llm.model and llm.base_url")

    tree_config = TreeConfig(
        node_keywords=bool(tree.get("node_keywords", True)),
        node_summary=bool(tree.get("node_summary", True)),
        use_vlm=bool(tree.get("use_vlm", False)),
    )
    graph_config = GraphConfig(
        extractor_type=str(graph.get("extractor_type", "llm")),
        image_description_force=False,
        max_gleaning=int(graph.get("max_gleaning", 0)),
        refine_type=str(graph.get("refine_type", "advanced")),
        g=float(graph.get("similarity_threshold", 0.6)),
        checkpoint_every=max(1, int(graph.get("checkpoint_every", 250))),
        embedding_config=copy.deepcopy(embedding_config),
        reranker_config=copy.deepcopy(reranker_config),
    )
    rag_strategy = GBCRAGConfig(
        strategy="gbc",
        variant=str(gbc.get("variant", "standard")),
        topk=int(gbc.get("topk", execution.get("retrieval_topk", 5))),
        sim_threshold_e=float(gbc.get("sim_threshold_e", 0.3)),
        select_depth=int(gbc.get("select_depth", 2)),
        x_percentile=float(gbc.get("x_percentile", 0.85)),
        alpha=float(gbc.get("alpha", 0.5)),
        topk_ent=int(gbc.get("topk_ent", 5)),
        max_retry=int(gbc.get("max_retry", 3)),
        reranker_config=copy.deepcopy(reranker_config),
    )

    return SystemConfig(
        llm=llm_config,
        # AnswerAgent and optional image summaries still expect a VLM object.
        # Reuse the same OpenAI-compatible online endpoint unless configured
        # separately in a later backend-specific extension.
        vlm=VLMConfig(
            backend="gpt",
            model_name=llm_config.model_name,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
            api_key=llm_config.api_key,
            api_base=llm_config.api_base,
        ),
        mineru=MinerU(
            backend=str(mineru.get("backend", "pipeline")),
            method=str(mineru.get("method", "auto")),
            # MinerU 2.1.11's current PDF-Extract-Kit snapshot ships the
            # ch_lite PyTorch OCR weights but not the en weights requested by
            # that release. ch_lite's dictionary also covers ASCII text.
            lang=str(mineru.get("lang", "ch_lite")),
            server_url=str(mineru.get("server_url", "http://127.0.0.1:30000")),
        ),
        tree=tree_config,
        graph=graph_config,
        vdb=VDBConfig(
            mm_embedding=False,
            vdb_dir_name=str(index_dir / "vdb"),
            collection_name="bookrag_collection",
            embedding_config=copy.deepcopy(embedding_config),
        ),
        index_type="gbc",
        mineru_workers=max(1, int(execution.get("mineru_workers", 1))),
        doc_workers=max(1, int(execution.get("doc_workers", 4))),
        ingest_workers=max(1, int(execution.get("ingest_workers", 4))),
        source_format="pdf",
        rag=RAGConfig(strategy_config=rag_strategy),
        pdf_path=None,
        save_path=str(index_dir),
    )


class BookRAGStoreWrapper:
    """Dataset-level adapter around the official BookRAG GBC Core."""

    def __init__(
        self,
        config: dict[str, Any],
        llm: Any | None = None,
        *,
        construct_index_fn: Callable[..., Any] = construct_GBC_index_from_pdfs,
        load_index_fn: Callable[[SystemConfig], Any] = GBC.load_gbc_index,
        agent_factory: Callable[..., Any] = create_rag_agent,
    ):
        del llm  # BookRAG owns its official LLM/AnswerAgent clients.
        self.benchmark_config = config
        self.core_config = build_bookrag_system_config(config)
        self.index_dir = Path(self.core_config.save_path).resolve()
        self.core_config.save_path = str(self.index_dir)
        self.resume_dir = self.index_dir.with_name(f"{self.index_dir.name}.resume")
        self.build_lock_path = self.index_dir.with_name(
            f"{self.index_dir.name}.build.lock"
        )
        index_config = ((config.get("bookrag") or {}).get("index") or {})
        # PDF/MinerU artifacts are durable index state, not disposable build
        # output. Build directly in the final directory by default so an
        # interrupted import and a later changed-source import reuse them.
        atomic_build = index_config.get("atomic_build", False)
        if not isinstance(atomic_build, bool):
            raise ValueError("bookrag.index.atomic_build must be true or false")
        self.atomic_build = atomic_build
        self._validate_index_dir(self.index_dir)
        self._construct_index = construct_index_fn
        self._load_index = load_index_fn
        self._agent_factory = agent_factory
        self._lock = threading.RLock()
        self._gbc_index: Any | None = None
        self._rag_agent: Any | None = None
        self._bridge_core_logging()

    @staticmethod
    def _bridge_core_logging() -> None:
        """Route official Core progress into the benchmark log and console."""
        benchmark_logger = logging.getLogger("Benchmark")
        if not benchmark_logger.handlers:
            return
        core_logger = logging.getLogger("bookrag_core")
        core_logger.setLevel(logging.INFO)
        for handler in benchmark_logger.handlers:
            if handler not in core_logger.handlers:
                core_logger.addHandler(handler)
        core_logger.propagate = False

    @staticmethod
    def _validate_index_dir(path: Path) -> None:
        path = path.resolve()
        anchor = Path(path.anchor)
        protected = {
            Path.cwd().resolve(),
            Path(__file__).resolve().parents[1],
            Path(__file__).resolve().parents[2],
            Path(__file__).resolve().parents[3],
        }
        if path == anchor or path in protected or not path.name:
            raise ValueError(f"Refusing to use broad BookRAG index directory: {path}")

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any] | None:
        manifest_path = path / _MANIFEST_NAME
        if not manifest_path.is_file():
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
        except (OSError, json.JSONDecodeError):
            return None
        return manifest if isinstance(manifest, dict) else None

    @classmethod
    def _is_bookrag_index(cls, path: Path) -> bool:
        manifest = cls._read_manifest(path)
        return bool(
            manifest
            and manifest.get("backend") == _MANIFEST_BACKEND
            and (path / "tree.pkl").is_file()
            and any(path.glob("graph_data*.json"))
        )

    @staticmethod
    def _is_resumable_build_dir(path: Path) -> bool:
        state_path = path / STATE_FILENAME
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = None
            if (
                isinstance(state, dict)
                and state.get("format_version") == STATE_FORMAT_VERSION
                and isinstance(state.get("sources"), list)
            ):
                return True

        # MinerU can fail or be interrupted before the aggregate tree exists.
        # A per-document manifest written by pdf_tree_builder is sufficient to
        # recognize this as our own resumable directory rather than arbitrary
        # user data.
        for manifest_path in path.glob("pdf_trees/*/pdf_tree_manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(manifest, dict)
                and manifest.get("format_version") == 1
                and manifest.get("status") in {"building", "complete"}
                and manifest.get("source_path")
                and manifest.get("source_sha256")
            ):
                return True
        return False

    @staticmethod
    def _content_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _config_hash(self) -> str:
        payload = json.dumps(
            _redact_secrets(self.benchmark_config),
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _validate_samples(samples: Sequence[StandardDoc]) -> list[StandardDoc]:
        ordered = sorted(
            samples,
            key=lambda sample: (
                str(sample.sample_id),
                str(Path(sample.doc_path).resolve()),
            ),
        )
        seen_sample_ids: set[str] = set()
        for sample in ordered:
            sample_id = str(sample.sample_id)
            if sample_id in seen_sample_ids:
                raise ValueError(
                    f"BookRAG requires unique sample_id values; duplicate: {sample_id}"
                )
            seen_sample_ids.add(sample_id)
            path = Path(sample.doc_path)
            if path.suffix.lower() != ".pdf":
                raise ValueError(f"BookRAG MinerU ingestion only accepts PDF: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"BookRAG PDF file not found: {path}")
        return ordered

    @staticmethod
    def _close_constructed_index(gbc_index: Any) -> None:
        entity_vdb = getattr(gbc_index, "entity_vdb", None)
        close_vdb = getattr(entity_vdb, "close", None)
        if callable(close_vdb):
            close_vdb()
        embedder = getattr(gbc_index, "embedder", None)
        close_embedder = getattr(embedder, "close", None)
        if callable(close_embedder):
            close_embedder()

    def _close_runtime(self) -> None:
        if self._rag_agent is not None:
            close_agent = getattr(self._rag_agent, "close", None)
            if callable(close_agent):
                close_agent()
        if self._gbc_index is not None:
            entity_vdb = getattr(self._gbc_index, "entity_vdb", None)
            close_vdb = getattr(entity_vdb, "close", None)
            if callable(close_vdb):
                close_vdb()
        self._rag_agent = None
        self._gbc_index = None

    def _atomic_install(self, temporary_dir: Path) -> None:
        backup = self.index_dir.with_name(f".{self.index_dir.name}.backup-{uuid.uuid4().hex}")
        had_existing = self.index_dir.exists()
        if had_existing:
            if not self._is_bookrag_index(self.index_dir):
                raise RuntimeError(f"Refusing to replace non-BookRAG directory: {self.index_dir}")
            os.replace(self.index_dir, backup)
        try:
            os.replace(temporary_dir, self.index_dir)
        except Exception:
            if had_existing and backup.exists():
                os.replace(backup, self.index_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)

    def _prepare_atomic_build_dir(self) -> Path:
        """Return one stable resume directory, adopting a legacy direct build."""
        if self.resume_dir.exists():
            if self.index_dir.exists() and not self._is_bookrag_index(self.index_dir):
                raise RuntimeError(
                    "Both the final BookRAG path and its resume path contain incomplete "
                    f"builds: {self.index_dir}, {self.resume_dir}"
                )
            log.info("[BookRAG] Resuming stable build directory: %s", self.resume_dir)
            return self.resume_dir

        if self.index_dir.exists() and not self._is_bookrag_index(self.index_dir):
            has_tree = (self.index_dir / "tree.pkl").is_file()
            if not has_tree:
                raise RuntimeError(
                    "Refusing to adopt a non-empty BookRAG directory without tree.pkl: "
                    f"{self.index_dir}"
                )
            try:
                os.replace(self.index_dir, self.resume_dir)
            except OSError as error:
                raise RuntimeError(
                    "Could not adopt the incomplete BookRAG directory. Ensure the previous "
                    f"import process is fully stopped: {self.index_dir}"
                ) from error
            log.info(
                "[BookRAG] Adopted legacy direct build as stable resume directory: %s",
                self.resume_dir,
            )

        self.resume_dir.mkdir(parents=True, exist_ok=True)
        return self.resume_dir

    def count_tokens(self, text: str) -> int:
        return num_tokens(str(text or ""))

    def prepare_mineru(
        self,
        samples: Sequence[StandardDoc],
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        """Prepare PDF/MinerU caches without building tree, KG, or GBC index."""
        execution = self.benchmark_config.get("execution") or {}
        mineru_workers = max(
            1,
            int(
                max_workers
                if max_workers is not None
                else execution.get("mineru_workers", 1)
            ),
        )
        started = time.time()
        with self._lock:
            with _BuildFileLock(self.build_lock_path):
                ordered = self._validate_samples(samples)
                if not ordered:
                    return {
                        "time": time.time() - started,
                        "task_time": 0,
                        "documents": 0,
                        "mineru_workers": mineru_workers,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }

                self._close_runtime()
                self.index_dir.parent.mkdir(parents=True, exist_ok=True)
                if self.atomic_build:
                    build_dir = self._prepare_atomic_build_dir()
                else:
                    if self.index_dir.exists():
                        has_tree = (self.index_dir / "tree.pkl").is_file()
                        is_resumable = self._is_resumable_build_dir(self.index_dir)
                        if (
                            any(self.index_dir.iterdir())
                            and not has_tree
                            and not is_resumable
                        ):
                            raise RuntimeError(
                                "Refusing to reuse a non-empty direct BookRAG directory "
                                "without tree.pkl or a valid build checkpoint: "
                                f"{self.index_dir}"
                            )
                    self.index_dir.mkdir(parents=True, exist_ok=True)
                    build_dir = self.index_dir

                build_config = self.core_config.model_copy(deep=True)
                build_config.save_path = str(build_dir)
                build_config.mineru_workers = mineru_workers
                documents = [
                    (str(sample.sample_id), Path(sample.doc_path).resolve())
                    for sample in ordered
                ]
                log.info(
                    "[BookRAG] MinerU-only preparation: documents=%d, "
                    "mineru_workers=%d, build_dir=%s",
                    len(documents),
                    mineru_workers,
                    build_dir,
                )
                from bookrag_core.pipelines.pdf_tree_builder import (
                    prepare_dataset_mineru_content,
                )

                stats = prepare_dataset_mineru_content(
                    build_config,
                    documents,
                    dataset_name=str(
                        self.benchmark_config.get("dataset_name", "dataset")
                    ),
                )
                stats["time"] = time.time() - started
                stats["input_tokens"] = 0
                stats["output_tokens"] = 0
                log.info(
                    "[BookRAG] MinerU-only preparation finished: time=%.2fs",
                    stats["time"],
                )
                return stats

    def ingest(
        self,
        samples: Sequence[StandardDoc],
        max_workers: int | None = None,
        monitor: Any | None = None,
        ingest_mode: str = "dataset",
    ) -> dict[str, Any]:
        del monitor
        execution = self.benchmark_config.get("execution") or {}
        ingest_workers = max(
            1,
            int(
                max_workers
                if max_workers is not None
                else execution.get("ingest_workers", 4)
            ),
        )
        mineru_workers = max(1, int(execution.get("mineru_workers", 1)))
        doc_workers = max(1, int(execution.get("doc_workers", 4)))
        started = time.time()
        with self._lock:
            with _BuildFileLock(self.build_lock_path):
                ordered = self._validate_samples(samples)
                if not ordered:
                    return {
                        "time": time.time() - started,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "embedding_tokens": 0,
                    }
                if ingest_mode not in {"dataset", "directory", "per_file"}:
                    raise ValueError(f"Unsupported BookRAG ingest mode: {ingest_mode}")

                self._close_runtime()
                self.index_dir.parent.mkdir(parents=True, exist_ok=True)
                if self.atomic_build:
                    build_dir = self._prepare_atomic_build_dir()
                else:
                    if self.index_dir.exists():
                        has_tree = (self.index_dir / "tree.pkl").is_file()
                        is_resumable = self._is_resumable_build_dir(self.index_dir)
                        if (
                            any(self.index_dir.iterdir())
                            and not has_tree
                            and not is_resumable
                        ):
                            raise RuntimeError(
                                "Refusing to reuse a non-empty direct BookRAG directory "
                                "without tree.pkl or a valid build checkpoint: "
                                f"{self.index_dir}"
                            )
                    self.index_dir.mkdir(parents=True, exist_ok=True)
                    build_dir = self.index_dir
                    log.info("[BookRAG] Building directly in final directory: %s", build_dir)
                build_config = self.core_config.model_copy(deep=True)
                build_config.save_path = str(build_dir)
                build_config.mineru_workers = mineru_workers
                build_config.doc_workers = doc_workers
                build_config.ingest_workers = ingest_workers
                # Aggregate-tree summary, KG, and embedding providers share the
                # ingestion worker budget. Per-document post-MinerU pipelines
                # explicitly avoid a nested provider pool.
                build_config.llm.max_workers = ingest_workers
                build_config.graph.embedding_config.max_workers = ingest_workers
                build_config.vdb.embedding_config.max_workers = ingest_workers
                documents = [
                    (str(sample.sample_id), Path(sample.doc_path).resolve()) for sample in ordered
                ]
                gbc_index = None
                try:
                    log.info(
                        "[BookRAG] Building one dataset-level GBC index from %d PDF documents.",
                        len(documents),
                    )
                    log.info(
                        "[BookRAG] Import concurrency: mineru_workers=%d, doc_workers=%d, "
                        "ingest_workers=%d (execution.max_workers is query-only).",
                        mineru_workers,
                        doc_workers,
                        ingest_workers,
                    )
                    log.info("[BookRAG] Phase 1/4: MinerU PDF trees and dataset aggregation")
                    tracker = TokenTracker.get_instance()
                    tracker.reset()
                    ingest_timer = DocumentWorkTimer()
                    with ingest_timer:
                        gbc_index = self._construct_index(
                            build_config,
                            documents,
                            dataset_name=str(
                                self.benchmark_config.get("dataset_name", "dataset")
                            ),
                        )
                    usage = tracker.get_usage()
                    tree_index = getattr(gbc_index, "TreeIndex", None)
                    graph_index = getattr(gbc_index, "GraphIndex", None)
                    node_count = len(getattr(tree_index, "nodes", []) or [])
                    graph = getattr(graph_index, "kg", None)
                    entity_count = (
                        int(graph.number_of_nodes()) if graph is not None else 0
                    )
                    self._close_constructed_index(gbc_index)
                    gbc_index = None

                    wall_time = time.time() - started
                    elapsed = ingest_timer.total_work_time(wall_time)
                    work_statistics = ingest_timer.statistics()
                    log.info(
                        "[BookRAG] Import timing: wall=%.2fs, task_sum=%.2fs, "
                        "parallel_union=%.2fs, reported_work_time=%.2fs, tasks=%d.",
                        wall_time,
                        work_statistics["task_time"],
                        work_statistics["task_wall_time"],
                        elapsed,
                        work_statistics["task_count"],
                    )
                    manifest = {
                        "format_version": 1,
                        "backend": _MANIFEST_BACKEND,
                        "dataset_name": str(
                            self.benchmark_config.get("dataset_name", "dataset")
                        ),
                        "sources": [
                            {
                                "sample_id": str(sample.sample_id),
                                "source_path": str(Path(sample.doc_path).resolve()),
                                "content_sha256": self._content_hash(
                                    Path(sample.doc_path)
                                ),
                            }
                            for sample in ordered
                        ],
                        "config_sha256": self._config_hash(),
                        "node_count": node_count,
                        "entity_count": entity_count,
                        "build_time_seconds": elapsed,
                        "token_usage": usage,
                    }
                    atomic_write_json(build_dir / _MANIFEST_NAME, manifest)

                    if self.atomic_build:
                        log.info(
                            "[BookRAG] Phase 4/4: atomically installing completed index"
                        )
                        self._atomic_install(build_dir)
                    else:
                        log.info(
                            "[BookRAG] Phase 4/4: direct build completed in final directory"
                        )
                    log.info(
                        "[BookRAG] Dataset GBC index ready: nodes=%d, entities=%d, "
                        "time=%.2fs",
                        node_count,
                        entity_count,
                        elapsed,
                    )
                    return {
                        "time": elapsed,
                        "input_tokens": int(usage.get("prompt_tokens", 0)),
                        "output_tokens": int(usage.get("completion_tokens", 0)),
                        "embedding_tokens": 0,
                    }
                except Exception:
                    if gbc_index is not None:
                        self._close_constructed_index(gbc_index)
                    log.warning(
                        "[BookRAG] Build failed; preserving resumable files at %s",
                        build_dir,
                    )
                    raise
                finally:
                    TokenTracker.get_instance().set_persistence_path(None)

    def _ensure_runtime(self) -> None:
        if self._rag_agent is not None:
            return
        if not self._is_bookrag_index(self.index_dir):
            raise RuntimeError(
                f"BookRAG index is missing or incomplete at {self.index_dir}; "
                "run --step import first"
            )
        log.info("[BookRAG] Loading dataset GBC index from %s", self.index_dir)
        self.core_config.save_path = str(self.index_dir)
        self._gbc_index = self._load_index(self.core_config)
        self._rag_agent = self._agent_factory(
            self.core_config.rag.strategy_config,
            self.core_config.llm,
            self.core_config.vlm,
            gbc_index=self._gbc_index,
        )

    @staticmethod
    def _node_uri(node: Any) -> str:
        sample_id = quote(str(node.meta_info.sample_id or "unknown"), safe="")
        return f"bookrag://resources/{sample_id}/nodes/{node.index_id}"

    @staticmethod
    def _context_block(node: Any) -> str:
        meta = node.meta_info
        title_path = " > ".join(meta.title_path) or "(untitled)"
        parts = [
            f"Source: {meta.file_path or meta.file_name or ''}",
            f"Sample ID: {meta.sample_id or ''}",
            f"Title path: {title_path}",
            f"BookRAG node: {node.index_id}",
            "",
            str(meta.content or ""),
        ]
        summary = str(getattr(node, "summary", "") or "").strip()
        if summary and summary != str(meta.content or "").strip():
            parts.extend(["", f"Summary: {summary}"])
        return "\n".join(parts).strip()

    def answer(
        self,
        query: str,
        topk: int,
        *,
        query_output_dir: str | Path,
        target_uri: str | None = None,
    ) -> dict[str, Any]:
        del target_uri  # Dataset-level GBC intentionally searches every document.
        if not str(query or "").strip():
            raise ValueError("BookRAG query must not be empty")
        if int(topk) < 1:
            raise ValueError("BookRAG topk must be positive")

        configured_topk = int(self.core_config.rag.strategy_config.topk)
        if int(topk) != configured_topk:
            raise ValueError(
                "BookRAG topk is initialized once from configuration; "
                f"requested {int(topk)}, configured {configured_topk}"
            )

        with self._lock:
            self._ensure_runtime()
            rag_agent = self._rag_agent
            gbc_index = self._gbc_index

        # Retrieval is read-only after initialization, so concurrent benchmark
        # workers must not hold the runtime lifecycle lock during generation.
        query_dir = Path(query_output_dir)
        query_dir.mkdir(parents=True, exist_ok=True)
        tracker = TokenTracker.get_instance()
        with tracker.query_scope() as query_usage:
            answer, retrieved_node_ids = rag_agent.generation(str(query), query_dir)
        usage = query_usage.get_usage()

        nodes = gbc_index.TreeIndex.get_nodes_by_ids(list(retrieved_node_ids or []))
        recall_texts: dict[str, str] = {}
        context_blocks: list[str] = []
        node_ids: list[int] = []
        for node in nodes:
            if node.type == NodeType.ROOT:
                continue
            uri = self._node_uri(node)
            block = self._context_block(node)
            if uri in recall_texts:
                continue
            recall_texts[uri] = block
            context_blocks.append(block)
            node_ids.append(int(node.index_id))

        return {
            "answer": str(answer or ""),
            "retrieved_node_ids": node_ids,
            "recall_texts": recall_texts,
            "context_blocks": context_blocks,
            "retrieved_uris": list(recall_texts),
            "retrieval_tokens": 0,
            "token_usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            },
            "trace_dir": str(query_dir),
        }

    def clear(self) -> None:
        with self._lock:
            self._validate_index_dir(self.index_dir)
            self._close_runtime()
            if not self.index_dir.exists():
                return
            if not self._is_bookrag_index(self.index_dir):
                raise RuntimeError(f"Refusing to delete non-BookRAG directory: {self.index_dir}")
            shutil.rmtree(self.index_dir)

    def close(self) -> None:
        with self._lock:
            self._close_runtime()
