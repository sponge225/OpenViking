# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

from openviking.parse.accessors.mime_types import get_preferred_extension
from openviking_cli.exceptions import OpenVikingError
from openviking_cli.utils.logger import get_logger

from .client import _getattr_safe, _raise_from_lark_response
from .path_utils import _markdown_file_name, _safe_path_segment, _unique_child_path

logger = get_logger(__name__)

_FEISHU_DRIVE_DOC_TYPES = {
    "doc": "docs",
    "docx": "docx",
    "sheet": "sheets",
    "sheets": "sheets",
    "bitable": "base",
    "base": "base",
    "wiki": "wiki",
}


class FeishuDriveMixin:
    async def _materialize_drive_folder(
        self,
        folder_token: str,
        target_dir: Path,
        *,
        feishu_access_token: Optional[str] = None,
        _seen: Optional[set[str]] = None,
        skipped_items: Optional[list[dict[str, Any]]] = None,
        strict: bool = False,
    ) -> None:
        """Expand a Feishu Drive folder into a local directory tree."""
        target_dir.mkdir(parents=True, exist_ok=True)
        seen = _seen or set()
        if folder_token in seen:
            logger.warning("[FeishuAccessor] Skipping recursive Drive folder %s", folder_token)
            return
        seen.add(folder_token)

        try:
            children = await asyncio.to_thread(
                self._list_drive_folder_children,
                folder_token,
                feishu_access_token=feishu_access_token,
            )
            for item in children:
                item_type, item_token, item_name, item_url = self._normalize_drive_item(item)
                if not item_token:
                    logger.warning("[FeishuAccessor] Skipping Drive item without token: %s", item)
                    continue

                if item_type == "folder":
                    folder_name = _safe_path_segment(item_name or item_token, fallback=item_token)
                    child_dir = _unique_child_path(target_dir, folder_name)
                    try:
                        await self._materialize_drive_folder(
                            item_token,
                            child_dir,
                            feishu_access_token=feishu_access_token,
                            _seen=seen,
                            skipped_items=skipped_items,
                            strict=strict,
                        )
                    except Exception as exc:
                        shutil.rmtree(child_dir, ignore_errors=True)
                        self._record_skipped_drive_item(
                            skipped_items,
                            item_type=item_type,
                            token=item_token,
                            name=item_name,
                            target_dir=target_dir,
                            error=exc,
                        )
                        if strict:
                            raise
                    continue

                doc_path_type = _FEISHU_DRIVE_DOC_TYPES.get(item_type)
                if doc_path_type:
                    try:
                        url = item_url or self._build_feishu_doc_url(doc_path_type, item_token)
                        doc = await self._fetch_document(
                            url,
                            feishu_access_token=feishu_access_token,
                        )
                        markdown_content, downloaded_images = await asyncio.to_thread(
                            self._resolve_image_refs,
                            doc.markdown_content,
                            feishu_access_token=feishu_access_token,
                            media_download_extras=doc.media_download_extras,
                        )
                        doc_name = _safe_path_segment(
                            item_name or doc.title or item_token,
                            fallback=item_token,
                        )
                        markdown_path = _unique_child_path(
                            target_dir,
                            _markdown_file_name(doc_name),
                        )
                        markdown_path.write_text(markdown_content, encoding="utf-8")
                        for rel_path, image_bytes in downloaded_images.items():
                            image_path = markdown_path.parent / rel_path
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            image_path.write_bytes(image_bytes)
                    except Exception as exc:
                        self._record_skipped_drive_item(
                            skipped_items,
                            item_type=item_type,
                            token=item_token,
                            name=item_name,
                            target_dir=target_dir,
                            error=exc,
                        )
                        if strict:
                            raise
                    continue

                if item_type == "file":
                    try:
                        content, content_type, downloaded_name = await asyncio.to_thread(
                            self._download_drive_file,
                            item_token,
                            feishu_access_token=feishu_access_token,
                            filename_hint=item_name,
                        )
                        file_name = self._drive_file_name(
                            item_token,
                            content,
                            content_type,
                            filename_hint=downloaded_name or item_name,
                        )
                        file_path = _unique_child_path(target_dir, file_name)
                        file_path.write_bytes(content)
                    except Exception as exc:
                        self._record_skipped_drive_item(
                            skipped_items,
                            item_type=item_type,
                            token=item_token,
                            name=item_name,
                            target_dir=target_dir,
                            error=exc,
                        )
                        if strict:
                            raise
                    continue

                error = ValueError(f"Unsupported Feishu Drive item type: {item_type}")
                self._record_skipped_drive_item(
                    skipped_items,
                    item_type=item_type,
                    token=item_token,
                    name=item_name,
                    target_dir=target_dir,
                    error=error,
                )
                if strict:
                    raise error

        finally:
            seen.discard(folder_token)

    @staticmethod
    def _record_skipped_drive_item(
        skipped_items: Optional[list[dict[str, Any]]],
        *,
        item_type: str,
        token: str,
        name: str,
        target_dir: Path,
        error: Exception,
    ) -> None:
        message = str(error).replace("\n", " ")
        logger.warning(
            "[FeishuAccessor] Skipping Drive %s %s under %s: %s",
            item_type,
            token,
            target_dir,
            message,
        )
        if skipped_items is None:
            return
        skipped_items.append(
            {
                "path": str(target_dir / _safe_path_segment(name or token, fallback=token)),
                "name": name or token,
                "type": item_type,
                "token": token,
                "reason": message,
            }
        )

    def _fetch_drive_folder_children_page(
        self,
        folder_token: str,
        *,
        feishu_access_token: Optional[str] = None,
        page_token: Optional[str] = None,
        page_size: int = 200,
    ) -> tuple[List[Any], bool, Optional[str]]:
        import lark_oapi as lark

        client = self._get_client(use_user_token=bool(feishu_access_token))
        token_type = (
            lark.AccessTokenType.USER if feishu_access_token else lark.AccessTokenType.TENANT
        )
        raw_req = (
            lark.BaseRequest.builder()
            .http_method(lark.HttpMethod.GET)
            .uri("/open-apis/drive/v1/files")
            .token_types({token_type})
            .build()
        )
        raw_req.add_query("folder_token", folder_token)
        raw_req.add_query("page_size", page_size)
        if page_token:
            raw_req.add_query("page_token", page_token)

        response = self._call_api(client.request, raw_req, feishu_access_token)
        if not response.success():
            _raise_from_lark_response(
                response,
                operation=f"list Drive folder {folder_token}",
                resource=folder_token,
            )

        data = self._raw_response_data(response)
        items = _getattr_safe(data, "files", None) or _getattr_safe(data, "items", None) or []
        has_more = bool(_getattr_safe(data, "has_more", False))
        next_page_token = _getattr_safe(data, "next_page_token", None) or _getattr_safe(
            data,
            "page_token",
            None,
        )
        return list(items), has_more, next_page_token

    def _list_drive_folder_children(
        self,
        folder_token: str,
        *,
        feishu_access_token: Optional[str] = None,
    ) -> List[Any]:
        """List direct children under a Feishu Drive folder token."""
        all_children: List[Any] = []
        page_token = None
        while True:
            items, has_more, page_token = self._fetch_drive_folder_children_page(
                folder_token,
                feishu_access_token=feishu_access_token,
                page_token=page_token,
            )
            all_children.extend(items)

            if not has_more:
                break
            if not page_token:
                raise RuntimeError(
                    f"Feishu returned more Drive folder items for {folder_token} "
                    "without a page token"
                )

        return all_children

    def _probe_drive_folder_children(
        self,
        folder_token: str,
        *,
        feishu_access_token: Optional[str] = None,
    ) -> None:
        self._fetch_drive_folder_children_page(
            folder_token,
            feishu_access_token=feishu_access_token,
            page_size=1,
        )

    def _drive_folder_display_name(
        self,
        folder_token: str,
        *,
        feishu_access_token: Optional[str] = None,
    ) -> str:
        """Best-effort readable folder name for resource roots."""
        try:
            name = self._get_drive_folder_name(
                folder_token,
                feishu_access_token=feishu_access_token,
            )
        except Exception as exc:
            logger.warning(
                "[FeishuAccessor] Falling back to Drive folder token %s as name: %s",
                folder_token,
                exc,
            )
            return folder_token
        return name or folder_token

    def _get_drive_folder_name(
        self,
        folder_token: str,
        *,
        feishu_access_token: Optional[str] = None,
    ) -> Optional[str]:
        """Fetch a Feishu Drive folder display name by folder token."""
        import lark_oapi as lark

        client = self._get_client(use_user_token=bool(feishu_access_token))
        token_type = (
            lark.AccessTokenType.USER if feishu_access_token else lark.AccessTokenType.TENANT
        )
        raw_req = (
            lark.BaseRequest.builder()
            .http_method(lark.HttpMethod.GET)
            .uri(f"/open-apis/drive/explorer/v2/folder/{folder_token}/meta")
            .token_types({token_type})
            .build()
        )
        response = self._call_api(client.request, raw_req, feishu_access_token)
        if not response.success():
            _raise_from_lark_response(
                response,
                operation=f"fetch Drive folder metadata {folder_token}",
                resource=folder_token,
            )

        data = self._raw_response_data(response)
        folder_meta = _getattr_safe(data, "folder", None) or _getattr_safe(data, "meta", None)
        name = _getattr_safe(data, "name", None) or _getattr_safe(folder_meta, "name", None)
        return str(name) if name else None

    def _download_drive_file(
        self,
        file_token: str,
        *,
        feishu_access_token: Optional[str] = None,
        filename_hint: Optional[str] = None,
    ) -> Tuple[bytes, Optional[str], Optional[str]]:
        """Download a Feishu Drive binary file by file token."""
        import lark_oapi as lark

        client = self._get_client(use_user_token=bool(feishu_access_token))
        token_type = (
            lark.AccessTokenType.USER if feishu_access_token else lark.AccessTokenType.TENANT
        )
        raw_req = (
            lark.BaseRequest.builder()
            .http_method(lark.HttpMethod.GET)
            .uri(f"/open-apis/drive/v1/files/{file_token}/download")
            .token_types({token_type})
            .build()
        )
        response = self._call_api(client.request, raw_req, feishu_access_token)
        if not response.success():
            _raise_from_lark_response(
                response,
                operation=f"download Drive file {file_token}",
                resource=file_token,
            )

        raw = getattr(response, "raw", None)
        content = getattr(raw, "content", None)
        if content is None:
            raise OpenVikingError(
                f"Feishu Drive file download returned empty content: {file_token}",
                code="NOT_FOUND",
                details={"operation": "download Drive file", "resource": file_token},
            )
        if isinstance(content, str):
            content = content.encode("utf-8")

        content_type = self._response_content_type(raw)
        filename = self._filename_from_content_disposition(
            self._response_header(raw, "content-disposition")
        )
        return content, content_type, filename or filename_hint

    def _write_temp_drive_file(
        self,
        file_token: str,
        content: bytes,
        content_type: Optional[str],
        *,
        filename_hint: Optional[str] = None,
    ) -> Path:
        filename = self._drive_file_name(
            file_token,
            content,
            content_type,
            filename_hint=filename_hint,
        )
        temp_dir = Path(tempfile.mkdtemp(prefix="ov_feishu_file_"))
        path = temp_dir / filename
        path.write_bytes(content)
        return path

    @classmethod
    def _normalize_drive_item(cls, item: Any) -> Tuple[str, str, str, str]:
        item_type = str(_getattr_safe(item, "type", "") or "").lower()
        token = str(_getattr_safe(item, "token", "") or "")
        name = str(_getattr_safe(item, "name", "") or "")
        url = str(_getattr_safe(item, "url", "") or "")
        shortcut_info = _getattr_safe(item, "shortcut_info", None)
        if shortcut_info:
            item_type = str(
                _getattr_safe(shortcut_info, "target_type", item_type) or item_type
            ).lower()
            token = str(_getattr_safe(shortcut_info, "target_token", token) or token)
        return item_type, token, name, url

    @staticmethod
    def _build_feishu_doc_url(doc_type: str, token: str) -> str:
        if doc_type == "doc":
            doc_type = "docs"
        return f"https://open.feishu.cn/{doc_type}/{token}"

    @classmethod
    def _drive_file_name(
        cls,
        file_token: str,
        content: bytes,
        content_type: Optional[str],
        *,
        filename_hint: Optional[str] = None,
    ) -> str:
        raw_name = filename_hint or file_token
        if Path(raw_name).suffix:
            return _safe_path_segment(raw_name, fallback=file_token)
        ext = cls._guess_drive_file_ext(content, content_type)
        return _safe_path_segment(f"{raw_name}{ext}", fallback=f"{file_token}{ext}")

    @staticmethod
    def _guess_drive_file_ext(content: bytes, content_type: Optional[str]) -> str:
        if content.startswith(b"%PDF-"):
            return ".pdf"
        if (
            content.startswith(b"PK\x03\x04")
            or content.startswith(b"PK\x05\x06")
            or content.startswith(b"PK\x07\x08")
        ):
            return ".zip"
        if content.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            return ".doc"
        if content_type:
            ext = get_preferred_extension(content_type)
            if ext:
                return ext
        return ".bin"
