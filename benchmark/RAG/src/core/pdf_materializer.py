"""Materialize benchmark source documents as stable, cached PDF files."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from adapters.base import StandardDoc
from tqdm import tqdm


_MANIFEST_NAME = "_pdf_materialization_manifest.json"
_MANIFEST_VERSION = 1
_RENDERER_VERSION = "markdown-it-pymupdf-story-v2-no-internal-links"
_COMPATIBLE_RENDERER_VERSIONS = {
    "markdown-it-pymupdf-story-v1",
    _RENDERER_VERSION,
}
_TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".text"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


class PdfMaterializer:
    """Convert normalized sources to PDFs before the measured ingest stage."""

    _CSS = """
        body { font-family: sans-serif; font-size: 10pt; line-height: 1.38; color: #111; }
        h1 { font-size: 22pt; margin: 18pt 0 10pt; }
        h2 { font-size: 18pt; margin: 16pt 0 8pt; }
        h3 { font-size: 15pt; margin: 14pt 0 7pt; }
        h4 { font-size: 13pt; margin: 12pt 0 6pt; }
        h5, h6 { font-size: 11pt; margin: 10pt 0 5pt; }
        p { margin: 0 0 8pt; }
        ul, ol { margin: 0 0 8pt 18pt; }
        pre { font-family: monospace; font-size: 8.5pt; white-space: pre-wrap; }
        code { font-family: monospace; font-size: 8.5pt; }
        table { border-collapse: collapse; width: 100%; margin: 8pt 0; }
        th, td { border: 0.5pt solid #777; padding: 3pt; vertical-align: top; }
        th { font-weight: bold; background-color: #eee; }
        blockquote { border-left: 2pt solid #999; margin-left: 8pt; padding-left: 8pt; }
        img { max-width: 100%; height: auto; }
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.manifest_path = self.output_dir / _MANIFEST_NAME

    @staticmethod
    def _resolve_path(raw_path: str | os.PathLike[str]) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path.resolve()

    @staticmethod
    def _safe_name(value: Any, *, fallback: str) -> str:
        text = re.sub(r"[^\w.\-]+", "_", str(value or "").strip(), flags=re.UNICODE)
        text = text.strip("._-")
        return text[:80] or fallback

    def _target_path(self, source_path: Path, sample_id: str) -> Path:
        sample_part = self._safe_name(sample_id, fallback="sample")
        stem_part = self._safe_name(source_path.stem, fallback="document")
        path_hash = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
        return self.output_dir / f"{sample_part}__{stem_part}__{path_hash}.pdf"

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        if not self.manifest_path.is_file():
            return {}
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(value, dict) or value.get("format_version") != _MANIFEST_VERSION:
            return {}
        items = value.get("items")
        if not isinstance(items, list):
            return {}
        return {
            str(item.get("source_path")): item
            for item in items
            if isinstance(item, dict) and item.get("source_path")
        }

    @staticmethod
    def _render_source_html(source_path: Path) -> str:
        source_text = source_path.read_text(encoding="utf-8", errors="replace")
        if source_path.suffix.lower() in {".md", ".markdown"}:
            from markdown_it import MarkdownIt

            renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
            tokens = renderer.parse(source_text)

            # Node.js-style Markdown uses project-specific fragment IDs such as
            # ``#assert_strict_mode``. CommonMark preserves those href values,
            # but does not add matching IDs to headings. PyMuPDF Story treats
            # every fragment href as a required PDF destination and aborts the
            # whole render when the destination is absent. Internal navigation
            # is irrelevant to MinerU, so keep the visible link text while
            # removing only fragment hrefs. External links remain untouched.
            pending = list(tokens)
            while pending:
                token = pending.pop()
                if token.children:
                    pending.extend(token.children)
                if token.type != "link_open":
                    continue
                href = token.attrGet("href")
                if isinstance(href, str) and href.startswith("#"):
                    token.attrs.pop("href", None)

            return renderer.renderer.render(tokens, renderer.options, {})
        return f'<pre class="plain-text">{html.escape(source_text)}</pre>'

    @classmethod
    def _render_text_pdf(cls, source_path: Path, target_path: Path) -> None:
        import fitz

        rendered_html = cls._render_source_html(source_path)
        archive = fitz.Archive(str(source_path.parent))
        story = fitz.Story(html=rendered_html, user_css=cls._CSS, archive=archive)
        media_box = fitz.paper_rect("a4")
        content_box = media_box + (54, 54, -54, -54)

        def page_rect(_rect_number, _filled):
            return media_box, content_box, fitz.Identity

        document = story.write_with_links(page_rect)
        temporary_path = target_path.with_name(
            f".{target_path.stem}.{uuid.uuid4().hex}.tmp.pdf"
        )
        try:
            metadata = document.metadata or {}
            metadata.update(
                {
                    "title": source_path.stem,
                    "creator": "OpenViking BookRAG PDF materializer",
                }
            )
            document.set_metadata(metadata)
            document.save(temporary_path, garbage=4, deflate=True)
            os.replace(temporary_path, target_path)
        finally:
            document.close()
            if temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _copy_pdf(source_path: Path, target_path: Path) -> None:
        temporary_path = target_path.with_name(
            f".{target_path.stem}.{uuid.uuid4().hex}.tmp.pdf"
        )
        try:
            shutil.copy2(source_path, temporary_path)
            os.replace(temporary_path, target_path)
        finally:
            if temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _is_reusable(
        previous: dict[str, Any] | None,
        *,
        source_hash: str,
        target_path: Path,
        method: str,
    ) -> bool:
        if not previous or not target_path.is_file() or target_path.stat().st_size == 0:
            return False
        if (
            previous.get("source_sha256") != source_hash
            or previous.get("output_pdf_path") != str(target_path)
            or previous.get("method") != method
            or previous.get("renderer_version") not in _COMPATIBLE_RENDERER_VERSIONS
        ):
            return False
        expected_output_hash = previous.get("output_pdf_sha256")
        return bool(expected_output_hash) and _file_sha256(target_path) == expected_output_hash

    def materialize(
        self,
        documents: Sequence[StandardDoc],
    ) -> tuple[list[StandardDoc], dict[str, Any]]:
        """Return PDF-only documents and preprocessing statistics."""
        started = time.monotonic()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        previous_manifest = self._load_manifest()

        source_records: dict[Path, dict[str, Any]] = {}
        for document in documents:
            source_path = self._resolve_path(document.doc_path)
            record = source_records.setdefault(
                source_path,
                {"sample_ids": set(), "first_sample_id": str(document.sample_id)},
            )
            record["sample_ids"].add(str(document.sample_id))

        stats: dict[str, Any] = {
            "enabled": True,
            "input_refs": len(documents),
            "unique_sources": len(source_records),
            "reused_pdf": 0,
            "copied_pdf": 0,
            "rendered_text": 0,
            "skipped_existing": 0,
        }
        source_to_pdf: dict[Path, Path] = {}
        manifest_items: list[dict[str, Any]] = []

        sorted_sources = sorted(source_records.items(), key=lambda item: str(item[0]))
        with tqdm(
            total=len(sorted_sources),
            desc="Preparing PDFs",
            unit="doc",
            dynamic_ncols=True,
        ) as pbar:
            for source_path, record in sorted_sources:
                pbar.set_postfix_str(source_path.name[:40], refresh=False)
                self._materialize_one_source(
                    source_path=source_path,
                    record=record,
                    previous_manifest=previous_manifest,
                    stats=stats,
                    source_to_pdf=source_to_pdf,
                    manifest_items=manifest_items,
                )
                pbar.update(1)

        materialized_documents = [
            StandardDoc(
                sample_id=str(document.sample_id),
                doc_path=str(source_to_pdf[self._resolve_path(document.doc_path)]),
            )
            for document in documents
        ]
        _atomic_write_json(
            self.manifest_path,
            {
                "format_version": _MANIFEST_VERSION,
                "renderer_version": _RENDERER_VERSION,
                "output_dir": str(self.output_dir),
                "items": manifest_items,
            },
        )
        stats.update(
            {
                "pdf_count": len(source_to_pdf),
                "manifest_path": str(self.manifest_path),
                "time": time.monotonic() - started,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )
        return materialized_documents, stats

    def _materialize_one_source(
        self,
        *,
        source_path: Path,
        record: dict[str, Any],
        previous_manifest: dict[str, dict[str, Any]],
        stats: dict[str, Any],
        source_to_pdf: dict[Path, Path],
        manifest_items: list[dict[str, Any]],
    ) -> None:
        if not source_path.is_file():
            raise FileNotFoundError(f"PDF materialization source not found: {source_path}")
        extension = source_path.suffix.lower()
        if extension not in _TEXT_EXTENSIONS and extension != ".pdf":
            raise ValueError(
                f"Unsupported PDF materialization source: {source_path} "
                f"(supported: .pdf, {', '.join(sorted(_TEXT_EXTENSIONS))})"
            )

        source_hash = _file_sha256(source_path)
        first_sample_id = str(record["first_sample_id"])
        if extension == ".pdf" and source_path.parent == self.output_dir:
            target_path = source_path
            method = "pdf_reused"
            stats["reused_pdf"] += 1
        else:
            target_path = self._target_path(source_path, first_sample_id)
            method = "pdf_copied" if extension == ".pdf" else "text_pdf_rendered"
            previous = previous_manifest.get(str(source_path))
            if self._is_reusable(
                previous,
                source_hash=source_hash,
                target_path=target_path,
                method=method,
            ):
                stats["skipped_existing"] += 1
            elif extension == ".pdf":
                self._copy_pdf(source_path, target_path)
                stats["copied_pdf"] += 1
            else:
                self._render_text_pdf(source_path, target_path)
                stats["rendered_text"] += 1

        target_path = target_path.resolve()
        source_to_pdf[source_path] = target_path
        manifest_items.append(
            {
                "source_path": str(source_path),
                "source_sha256": source_hash,
                "source_size_bytes": source_path.stat().st_size,
                "output_pdf_path": str(target_path),
                "output_pdf_sha256": _file_sha256(target_path),
                "method": method,
                "renderer_version": _RENDERER_VERSION,
                "sample_ids": sorted(record["sample_ids"]),
            }
        )
