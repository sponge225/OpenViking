# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import re
from typing import Dict, List, Optional, Tuple

from openviking.parse.accessors.mime_types import get_preferred_extension
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

_MediaDownloadExtras = Dict[str, List[Optional[str]]]
_FEISHU_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(feishu://image/([^)]+)\)")
_MAX_MEDIA_DOWNLOAD_CONTEXTS = 8


class FeishuMediaMixin:
    # first, since the actual content is authoritative over a (possibly generic
    # or wrong) Content-Type header.
    _IMAGE_MAGIC = (
        (b"\x89PNG\r\n\x1a\n", ".png"),
        (b"\xff\xd8\xff", ".jpg"),
        (b"GIF87a", ".gif"),
        (b"GIF89a", ".gif"),
        (b"BM", ".bmp"),
    )

    @classmethod
    def _guess_image_ext(cls, content: bytes, content_type: Optional[str]) -> str:
        """Infer an image file extension from the bytes, then Content-Type.

        Feishu media are not guaranteed to be PNG, so we avoid a hardcoded
        extension that would misrepresent JPEG/WebP/GIF bytes to downstream
        consumers (e.g. emitting JPEG bytes as ``data:image/png``). Byte magic
        is checked first because the payload is authoritative; the response
        Content-Type is only a fallback for formats we do not sniff here.
        """
        # WebP: "RIFF....WEBP"
        if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
            return ".webp"
        for magic, ext in cls._IMAGE_MAGIC:
            if content.startswith(magic):
                return ext
        if content_type:
            ext = get_preferred_extension(content_type)
            if ext:
                return ext
        return ".png"

    @staticmethod
    def _image_filename(file_token: str, ext: str = ".png") -> str:
        """Return a conservative local filename for a Feishu media token."""
        safe_token = re.sub(r"[^A-Za-z0-9_.-]+", "_", file_token).strip("._")
        if not ext.startswith("."):
            ext = f".{ext}"
        return f"{safe_token or 'image'}{ext}"

    def _download_image(
        self,
        file_token: str,
        *,
        feishu_access_token: Optional[str] = None,
        extra: Optional[str] = None,
    ) -> Optional[Tuple[bytes, Optional[str]]]:
        """Download an image from Feishu Drive API by file token.

        Returns a ``(content, content_type)`` tuple, or ``None`` on failure.
        """
        import lark_oapi as lark

        client = self._get_client(use_user_token=bool(feishu_access_token))
        # Match the auth mode used to fetch the document: with a user access
        # token the request must advertise USER, otherwise lark-oapi never
        # injects it (see lark_oapi.core.token.auth.verify) and the download
        # silently fails — dropping images from user-token imports.
        token_type = (
            lark.AccessTokenType.USER if feishu_access_token else lark.AccessTokenType.TENANT
        )
        raw_req = (
            lark.BaseRequest.builder()
            .http_method(lark.HttpMethod.GET)
            .uri(f"/open-apis/drive/v1/medias/{file_token}/download")
            .token_types({token_type})
            .build()
        )
        if extra:
            raw_req.add_query("extra", extra)
        try:
            raw_resp = self._call_api(client.request, raw_req, feishu_access_token)
        except Exception as exc:
            logger.warning("[FeishuAccessor] Error downloading image %s: %s", file_token, exc)
            return None

        if not raw_resp.success():
            raw = getattr(raw_resp, "raw", None)
            http_status = getattr(raw, "status_code", None)
            detail = getattr(raw_resp, "msg", "") or f"HTTP {http_status}"
            if http_status == 403:
                detail = f"{detail} (missing Feishu permission docs:document.media:download)"
            logger.warning(
                "[FeishuAccessor] Failed to download image %s: code=%s, http=%s, msg=%s",
                file_token,
                getattr(raw_resp, "code", None),
                http_status,
                detail,
            )
            return None

        raw = getattr(raw_resp, "raw", None)
        content = getattr(raw, "content", None)
        if not content:
            logger.warning("[FeishuAccessor] Empty image response for %s", file_token)
            return None
        return content, self._response_content_type(raw)

    def _resolve_image_refs(
        self,
        markdown: str,
        *,
        feishu_access_token: Optional[str] = None,
        media_download_extras: Optional[_MediaDownloadExtras] = None,
    ) -> Tuple[str, Dict[str, bytes]]:
        """Download Feishu image refs and rewrite them to local relative paths."""
        config = self._get_config()
        if not getattr(config, "download_images", True):
            return markdown, {}

        matches = list(_FEISHU_IMAGE_RE.finditer(markdown))
        if not matches:
            return markdown, {}

        token_to_rel_path: Dict[str, str] = {}
        downloaded_images: Dict[str, bytes] = {}
        for match in matches:
            file_token = match.group(2)
            if file_token in token_to_rel_path:
                continue

            configured_extras = (media_download_extras or {}).get(file_token)
            if configured_extras:
                # Try protected contexts before the legacy token-only fallback.
                extras: List[Optional[str]] = list(
                    dict.fromkeys(extra for extra in configured_extras if extra)
                )[:_MAX_MEDIA_DOWNLOAD_CONTEXTS]
                extras.append(None)
            else:
                extras = [None]

            downloaded = None
            for extra in extras:
                downloaded = self._download_image(
                    file_token,
                    feishu_access_token=feishu_access_token,
                    extra=extra,
                )
                if downloaded is not None:
                    break
            if downloaded is None:
                continue
            image_bytes, content_type = downloaded

            ext = self._guess_image_ext(image_bytes, content_type)
            rel_path = f"images/{self._image_filename(file_token, ext)}"
            token_to_rel_path[file_token] = rel_path
            downloaded_images[rel_path] = image_bytes

        if not downloaded_images:
            return markdown, {}

        def _replace(match: re.Match[str]) -> str:
            alt_text = match.group(1)
            file_token = match.group(2)
            rel_path = token_to_rel_path.get(file_token)
            if not rel_path:
                return match.group(0)
            return f"![{alt_text}]({rel_path})"

        return _FEISHU_IMAGE_RE.sub(_replace, markdown), downloaded_images
