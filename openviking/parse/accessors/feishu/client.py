# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json
import os
import re
from typing import Any, NoReturn, Optional
from urllib.parse import unquote

from openviking.utils.exceptions import error_code_from_http_status
from openviking_cli.exceptions import OpenVikingError
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

_FEISHU_DOCUMENT_FORBIDDEN = 1770032
_FEISHU_WIKI_NODE_PERMISSION_DENIED = 131006
_FEISHU_BITABLE_PERMISSION_REQUIRED = 99991672
_FEISHU_LEGACY_DOC_LOGIN_REQUIRED = 91404
_FEISHU_PERMISSION_DENIED_CODES = {
    _FEISHU_DOCUMENT_FORBIDDEN,
    _FEISHU_WIKI_NODE_PERMISSION_DENIED,
    91403,
    95008,
    95009,
}
_FEISHU_NOT_FOUND_CODES = {91402, 95006, 95007}


def _getattr_safe(obj, key: str, default=None):
    """Get attribute from SDK object or dict, with safe fallback."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _response_http_status(response: Any) -> int | None:
    status = getattr(getattr(response, "raw", None), "status_code", None)
    return status if isinstance(status, int) else None


def _raise_from_lark_response(
    response: Any,
    *,
    operation: str,
    resource: str | None = None,
) -> NoReturn:
    code = getattr(response, "code", None)
    msg = getattr(response, "msg", None) or "Feishu API request failed"
    http_status = _response_http_status(response)
    details: dict[str, Any] = {
        "operation": operation,
        "feishu_code": code,
        "feishu_msg": msg,
        "http_status": http_status,
    }
    if resource:
        details["resource"] = resource

    logger.error(
        "[FeishuAPI] %s failed: code=%s msg=%s http=%s",
        operation,
        code,
        msg,
        http_status,
    )
    if code == _FEISHU_BITABLE_PERMISSION_REQUIRED:
        public_code = "FAILED_PRECONDITION"
        message = (
            f"Feishu application is missing required Bitable permissions: code={code}, msg={msg}"
        )
    else:
        if code in _FEISHU_PERMISSION_DENIED_CODES:
            public_code = "PERMISSION_DENIED"
        elif code in _FEISHU_NOT_FOUND_CODES:
            public_code = "NOT_FOUND"
        elif code == _FEISHU_LEGACY_DOC_LOGIN_REQUIRED:
            public_code = "UNAUTHENTICATED"
        else:
            public_code = error_code_from_http_status(http_status)
        message = f"Feishu {operation} failed: code={code}, msg={msg}"

    raise OpenVikingError(message, code=public_code, details=details)


class FeishuClientMixin:
    @staticmethod
    def _raw_response_data(response: Any) -> Any:
        data = getattr(response, "data", None)
        if data is not None:
            return data
        raw_content = getattr(getattr(response, "raw", None), "content", None)
        if not raw_content:
            return {}
        if isinstance(raw_content, bytes):
            raw_content = raw_content.decode("utf-8")
        return json.loads(raw_content).get("data", {})

    @staticmethod
    def _response_header(raw: Any, name: str) -> Optional[str]:
        headers = getattr(raw, "headers", None)
        if not headers:
            return None
        try:
            get = headers.get
        except AttributeError:
            return None
        return get(name) or get(name.title()) or get(name.lower())

    @staticmethod
    def _filename_from_content_disposition(content_disposition: Optional[str]) -> Optional[str]:
        if not content_disposition:
            return None
        utf8_match = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition, re.I)
        if utf8_match:
            return unquote(utf8_match.group(1))
        quoted_match = re.search(r'filename="([^"]+)"', content_disposition, re.I)
        if quoted_match:
            return quoted_match.group(1)
        simple_match = re.search(r"filename=([^;]+)", content_disposition, re.I)
        if simple_match:
            return simple_match.group(1).strip()
        return None

    # ========== Configuration & Client ==========

    def _get_config(self):
        """Get FeishuConfig from OpenViking config."""
        if self._config is None:
            from openviking_cli.utils.config import get_openviking_config

            self._config = get_openviking_config().feishu
        return self._config

    def _get_client(self, *, use_user_token: bool = False):
        """Lazy-init lark-oapi client."""
        cache_attr = "_user_token_client" if use_user_token else "_client"
        client = getattr(self, cache_attr)
        if client is None:
            try:
                import lark_oapi as lark
            except ImportError:
                raise ImportError(
                    "lark-oapi is required for Feishu document parsing. "
                    "Install it with: pip install lark-oapi>=1.0.0"
                )
            config = self._get_config()
            app_id = config.app_id or os.getenv("FEISHU_APP_ID", "")
            app_secret = config.app_secret or os.getenv("FEISHU_APP_SECRET", "")
            if (not app_id or not app_secret) and not use_user_token:
                raise ValueError(
                    "Feishu credentials not configured. Set FEISHU_APP_ID and "
                    "FEISHU_APP_SECRET environment variables, or configure in ov.conf."
                )
            domain = config.domain or "https://open.feishu.cn"
            builder = lark.Client.builder().domain(domain)
            if app_id and app_secret:
                builder = builder.app_id(app_id).app_secret(app_secret)
            if use_user_token:
                builder = builder.enable_set_token(True)
            client = builder.build()
            setattr(self, cache_attr, client)
        return client

    @staticmethod
    def _user_request_option(feishu_access_token: Optional[str]):
        if not feishu_access_token:
            return None
        from lark_oapi.core.model import RequestOption

        return RequestOption.builder().user_access_token(feishu_access_token).build()

    def _call_api(self, method, request, feishu_access_token: Optional[str] = None):
        option = self._user_request_option(feishu_access_token)
        return method(request) if option is None else method(request, option)

    @staticmethod
    def _response_content_type(raw) -> Optional[str]:
        """Best-effort extraction of the Content-Type header from a lark raw response."""
        headers = getattr(raw, "headers", None)
        if not headers:
            return None
        # lark's raw.headers may be a plain dict or a case-insensitive mapping.
        try:
            get = headers.get
        except AttributeError:
            return None
        return get("Content-Type") or get("content-type")
