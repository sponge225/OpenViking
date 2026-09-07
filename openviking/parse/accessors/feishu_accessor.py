# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Feishu/Lark Accessor.

Fetches Feishu/Lark cloud documents using the lark-oapi SDK.

Note: This accessor requires the `lark-oapi` package.
Included by default in `openviking[bot]` installation.
"""

import asyncio
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

from openviking_cli.utils.logger import get_logger

from .base import DataAccessor, LocalResource, SourceType
from .feishu.client import FeishuClientMixin, _raise_from_lark_response
from .feishu.documents import FeishuLegacyDocMixin, FeishuWikiMixin
from .feishu.docx import FeishuDocxMixin
from .feishu.drive import FeishuDriveMixin
from .feishu.media import _MAX_MEDIA_DOWNLOAD_CONTEXTS, FeishuMediaMixin, _MediaDownloadExtras
from .feishu.path_utils import (
    _markdown_file_name,
    _safe_path_segment,
    _title_as_filename,
    _unique_child_path,
)
from .feishu.tables import FeishuTablesMixin

logger = get_logger(__name__)

_FEISHU_DOC_PATH_TYPES = {"doc", "docs", "docx", "wiki", "sheets", "base"}


@dataclass(frozen=True)
class FeishuSourcePreflight:
    """Lightweight Feishu source identity resolved before enqueueing imports."""

    doc_type: str
    token: str
    source_name: Optional[str]
    source_format: str


@dataclass
class FeishuDocument:
    """Result from fetching a Feishu document."""

    doc_type: str
    token: str
    markdown_content: str
    title: str
    meta: Dict[str, Any]
    media_download_extras: _MediaDownloadExtras = field(default_factory=dict)


class FeishuAccessor(
    FeishuClientMixin,
    FeishuMediaMixin,
    FeishuDriveMixin,
    FeishuDocxMixin,
    FeishuWikiMixin,
    FeishuLegacyDocMixin,
    FeishuTablesMixin,
    DataAccessor,
):
    """
    Accessor for Feishu/Lark cloud documents.

    Supports:
    - Documents: https://*.feishu.cn/docx/{document_id}
    - Legacy documents: https://*.feishu.cn/docs/{doc_token}
    - Wiki pages: https://*.feishu.cn/wiki/{token}
    - Spreadsheets: https://*.feishu.cn/sheets/{token}
    - Bitable: https://*.feishu.cn/base/{app_token}
    - Drive files: https://*.feishu.cn/file/{file_token}
    - Drive folders: https://*.feishu.cn/drive/folder/{folder_token}

    Requires:
    - lark-oapi package
    - FEISHU_APP_ID and FEISHU_APP_SECRET environment variables, or
      configuration in ov.conf, for app-token imports. One-time user-token
      imports can pass feishu_access_token instead.
    """

    PRIORITY = 100  # Higher than Git/HTTP, very specific

    _DOC_TYPE_HANDLERS = {
        "doc": "_parse_legacy_doc",
        "docx": "_parse_docx",
        "sheets": "_parse_sheets",
        "base": "_parse_bitable",
    }

    def __init__(self):
        """Initialize Feishu accessor."""
        self._client = None
        self._user_token_client = None
        self._config = None

    @property
    def priority(self) -> int:
        return self.PRIORITY

    def can_handle(self, source: Union[str, Path], **kwargs) -> bool:
        """
        Check if this accessor can handle the source.

        Handles Feishu/Lark cloud document URLs.
        """
        source_str = str(source)

        # Only handle http/https URLs
        if not source_str.startswith(("http://", "https://")):
            return False

        return self._is_feishu_url(source_str)

    async def access(self, source: Union[str, Path], **kwargs) -> LocalResource:
        """
        Fetch a Feishu document and save to a temporary Markdown file.

        Args:
            source: Feishu document URL
            **kwargs: Additional arguments

        Returns:
            LocalResource pointing to the temporary Markdown file
        """
        source_str = str(source)
        feishu_access_token = kwargs.get("feishu_access_token")

        try:
            doc_type, token = self._parse_feishu_url(source_str)
            if doc_type == "file":
                content, content_type, filename = await asyncio.to_thread(
                    self._download_drive_file,
                    token,
                    feishu_access_token=feishu_access_token,
                )
                local_path = self._write_temp_drive_file(
                    token,
                    content,
                    content_type,
                    filename_hint=filename,
                )
                return LocalResource(
                    path=local_path,
                    source_type=SourceType.FEISHU,
                    original_source=source_str,
                    meta={
                        "feishu_doc_type": doc_type,
                        "feishu_token": token,
                        "original_filename": local_path.name,
                        "_cleanup_path": str(local_path.parent),
                    },
                    is_temporary=True,
                )

            if doc_type == "folder":
                temp_dir = Path(tempfile.mkdtemp(prefix="ov_feishu_folder_"))
                skipped_items: list[dict[str, Any]] = []
                folder_name = await asyncio.to_thread(
                    self._drive_folder_display_name,
                    token,
                    feishu_access_token=feishu_access_token,
                )
                try:
                    await self._materialize_drive_folder(
                        token,
                        temp_dir,
                        feishu_access_token=feishu_access_token,
                        skipped_items=skipped_items,
                        strict=bool(kwargs.get("strict", False)),
                    )
                except Exception:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    raise
                return LocalResource(
                    path=temp_dir,
                    source_type=SourceType.FEISHU,
                    original_source=source_str,
                    meta={
                        "feishu_doc_type": doc_type,
                        "feishu_token": token,
                        "original_filename": _safe_path_segment(folder_name, fallback=token),
                        "feishu_folder_skipped_items": skipped_items,
                    },
                    is_temporary=True,
                )

            # Fetch the document and convert to Markdown
            doc = await self._fetch_document(
                source_str,
                feishu_access_token=feishu_access_token,
            )

            # lark-oapi media downloads are synchronous; run them off the event
            # loop so a slow Feishu request cannot block unrelated async work.
            markdown_content, downloaded_images = await asyncio.to_thread(
                self._resolve_image_refs,
                doc.markdown_content,
                feishu_access_token=feishu_access_token,
                media_download_extras=doc.media_download_extras,
            )

            # Build metadata
            meta = {
                "feishu_doc_type": doc.doc_type,
                "feishu_token": doc.token,
                "feishu_title": doc.title,
                "original_filename": _title_as_filename(doc.title),
                **doc.meta,
            }

            if downloaded_images:
                temp_dir = Path(tempfile.mkdtemp(prefix="ov_feishu_"))
                markdown_path = temp_dir / "document.md"
                markdown_path.write_text(markdown_content, encoding="utf-8")
                for rel_path, image_bytes in downloaded_images.items():
                    image_path = temp_dir / rel_path
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(image_bytes)
                meta["_cleanup_path"] = str(temp_dir)
                local_path = markdown_path
            else:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".md",
                    prefix="ov_feishu_",
                    delete=False,
                    encoding="utf-8",
                )
                temp_file.write(markdown_content)
                temp_file.close()
                local_path = Path(temp_file.name)

            return LocalResource(
                path=local_path,
                source_type=SourceType.FEISHU,
                original_source=source_str,
                meta=meta,
                is_temporary=True,
            )

        except Exception as e:
            logger.error(f"[FeishuAccessor] Failed to access {source}: {e}", exc_info=True)
            raise

    async def preflight_source(
        self,
        source: Union[str, Path],
        *,
        feishu_access_token: Optional[str] = None,
    ) -> FeishuSourcePreflight:
        """Resolve lightweight source identity and root permission before enqueueing."""
        return await asyncio.to_thread(
            self._preflight_source_sync,
            str(source),
            feishu_access_token,
        )

    def _preflight_source_sync(
        self,
        url: str,
        feishu_access_token: Optional[str] = None,
    ) -> FeishuSourcePreflight:
        doc_type, token = self._parse_feishu_url(url)
        query = parse_qs(urlparse(url).query)
        table_id = (query.get("table") or [None])[0]
        view_id = (query.get("view") or [None])[0]

        if doc_type == "wiki":
            real_type, real_token, title = self._resolve_wiki_node(
                token,
                feishu_access_token,
            )
            if real_type != "base":
                table_id = view_id = None
            self._probe_document_permission(
                real_type,
                real_token,
                feishu_access_token=feishu_access_token,
                table_id=table_id,
                view_id=view_id,
            )
            source_name = None
            if title:
                scope = "/".join(value for value in (table_id, view_id) if value)
                source_name = _title_as_filename(f"{title} ({scope})" if scope else title)
            return FeishuSourcePreflight(
                doc_type=real_type,
                token=real_token,
                source_name=source_name,
                source_format="file",
            )

        if doc_type == "folder":
            name = self._get_drive_folder_name(
                token,
                feishu_access_token=feishu_access_token,
            )
            self._probe_drive_folder_children(
                token,
                feishu_access_token=feishu_access_token,
            )
            return FeishuSourcePreflight(
                doc_type=doc_type,
                token=token,
                source_name=_safe_path_segment(name or token, fallback=token),
                source_format="directory",
            )

        return FeishuSourcePreflight(
            doc_type=doc_type,
            token=token,
            source_name=self._preflight_document_source_name(
                doc_type,
                token,
                feishu_access_token=feishu_access_token,
                table_id=table_id,
                view_id=view_id,
            ),
            source_format="file",
        )

    def _preflight_document_source_name(
        self,
        doc_type: str,
        token: str,
        *,
        feishu_access_token: Optional[str],
        table_id: Optional[str],
        view_id: Optional[str],
    ) -> Optional[str]:
        if doc_type == "doc":
            metadata = self._fetch_legacy_doc_metadata(
                token,
                feishu_access_token=feishu_access_token,
            )
            title = metadata.get("title") or None
            return _title_as_filename(str(title)) if title else None
        if doc_type == "docx":
            self._probe_docx_document(token, feishu_access_token=feishu_access_token)
            return None
        if doc_type == "sheets":
            metadata = self._fetch_spreadsheet_metadata(
                token,
                feishu_access_token=feishu_access_token,
            )
            title = (metadata.get("properties") or {}).get("title") or "Spreadsheet"
            return _title_as_filename(title)
        if doc_type == "base":
            if view_id and not table_id:
                raise ValueError("Feishu Base URL with 'view' must also include 'table'")
            if table_id:
                self._probe_bitable_table(
                    token,
                    table_id,
                    view_id=view_id,
                    feishu_access_token=feishu_access_token,
                )
                return f"{table_id} ({view_id})" if view_id else table_id
            tables = self._list_bitable_tables(
                token,
                feishu_access_token=feishu_access_token,
            )
            return f"Bitable ({len(tables)} tables)"
        if doc_type == "file":
            return None
        raise ValueError(
            f"Unsupported Feishu document type: {doc_type}. "
            f"Supported: {list(self._DOC_TYPE_HANDLERS)}"
        )

    def _probe_document_permission(
        self,
        doc_type: str,
        token: str,
        *,
        feishu_access_token: Optional[str],
        table_id: Optional[str] = None,
        view_id: Optional[str] = None,
    ) -> None:
        self._preflight_document_source_name(
            doc_type,
            token,
            feishu_access_token=feishu_access_token,
            table_id=table_id,
            view_id=view_id,
        )

    async def _fetch_document(
        self,
        url: str,
        *,
        feishu_access_token: Optional[str] = None,
    ) -> FeishuDocument:
        """
        Fetch a Feishu document and convert to Markdown.

        The fetched document is materialized as Markdown for the standard parser chain.
        """
        doc_type, token = self._parse_feishu_url(url)
        query = parse_qs(urlparse(url).query)
        table_id = (query.get("table") or [None])[0]
        view_id = (query.get("view") or [None])[0]
        title = None
        meta = {}
        media_download_extras: _MediaDownloadExtras = {}

        if doc_type == "wiki":
            # Resolve wiki node to actual document type
            real_type, real_token, title = await asyncio.to_thread(
                self._resolve_wiki_node,
                token,
                feishu_access_token,
            )
            doc_type, token = real_type, real_token
            meta["wiki_resolved"] = True

        if doc_type != "base":
            table_id = view_id = None

        handler_name = self._DOC_TYPE_HANDLERS.get(doc_type)
        if handler_name is None:
            raise ValueError(
                f"Unsupported Feishu document type: {doc_type}. "
                f"Supported: {list(self._DOC_TYPE_HANDLERS)}"
            )

        handler_kwargs = {}
        if doc_type == "base":
            handler_kwargs = {
                "table_id": table_id,
                "view_id": view_id,
                "media_download_extras": media_download_extras,
            }
        elif doc_type == "sheets":
            handler_kwargs = {"media_download_extras": media_download_extras}

        # Feishu's SDK is synchronous; keep it off the event loop.
        markdown, doc_title = await asyncio.to_thread(
            getattr(self, handler_name),
            token,
            feishu_access_token,
            **handler_kwargs,
        )

        if title:
            scope = "/".join(value for value in (table_id, view_id) if value)
            doc_title = f"{title} ({scope})" if scope else title

        meta["original_url"] = url
        if table_id:
            meta["feishu_table_id"] = table_id
        if view_id:
            meta["feishu_view_id"] = view_id

        return FeishuDocument(
            doc_type=doc_type,
            token=token,
            markdown_content=markdown,
            title=doc_title,
            meta=meta,
            media_download_extras=media_download_extras,
        )

    @staticmethod
    def _is_feishu_url(url: str) -> bool:
        """Check if URL is a Feishu/Lark cloud document."""
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        path_parts = [p for p in parsed.path.split("/") if p]
        is_feishu_domain = any(
            host == allowed_host or host.endswith(f".{allowed_host}")
            for allowed_host in ("feishu.cn", "larksuite.com", "larkoffice.com")
        )
        if not is_feishu_domain or len(path_parts) < 2:
            return False
        has_doc_path = path_parts[0] in _FEISHU_DOC_PATH_TYPES
        has_drive_folder_path = len(path_parts) >= 3 and path_parts[:2] == ["drive", "folder"]
        has_file_path = path_parts[0] == "file"
        return is_feishu_domain and (has_doc_path or has_drive_folder_path or has_file_path)

    @staticmethod
    def _parse_feishu_url(url: str) -> Tuple[str, str]:
        """
        Extract doc_type and token from Feishu URL.

        Returns:
            (doc_type, token) e.g. ("docx", "doxcnABC123")
        """
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) < 2:
            raise ValueError(f"Cannot parse Feishu URL: {url}")
        if len(path_parts) >= 3 and path_parts[:2] == ["drive", "folder"]:
            return "folder", path_parts[2]
        if path_parts[0] == "file":
            return "file", path_parts[1]
        doc_type = "doc" if path_parts[0] in {"doc", "docs"} else path_parts[0]
        token = path_parts[1]
        return doc_type, token

    _unique_child_path = staticmethod(_unique_child_path)
    _markdown_file_name = staticmethod(_markdown_file_name)


__all__ = [
    "FeishuAccessor",
    "FeishuDocument",
    "FeishuSourcePreflight",
    "_MAX_MEDIA_DOWNLOAD_CONTEXTS",
    "_raise_from_lark_response",
]
