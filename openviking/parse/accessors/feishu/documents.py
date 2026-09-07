# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from typing import Any, Dict, Optional, Tuple

from .client import _getattr_safe, _raise_from_lark_response


class FeishuWikiMixin:
    _WIKI_TYPE_MAP = {"doc": "doc", "sheet": "sheets", "bitable": "base"}

    def _resolve_wiki_node(
        self,
        token: str,
        feishu_access_token: Optional[str] = None,
    ) -> Tuple[str, str, Optional[str]]:
        """
        Resolve wiki token to actual document type, token, and title.

        Returns:
            (doc_type, obj_token, title)
        """
        from lark_oapi.api.wiki.v2 import GetNodeSpaceRequest

        client = self._get_client(use_user_token=bool(feishu_access_token))
        request = GetNodeSpaceRequest.builder().token(token).build()
        response = self._call_api(
            client.wiki.v2.space.get_node,
            request,
            feishu_access_token,
        )
        if not response.success():
            _raise_from_lark_response(
                response,
                operation=f"resolve wiki node {token}",
                resource=token,
            )
        node = response.data.node
        obj_type = node.obj_type or ""
        obj_token = node.obj_token or ""
        title = node.title

        # Normalize type names
        doc_type = self._WIKI_TYPE_MAP.get(obj_type, obj_type)

        return doc_type, obj_token, title


class FeishuLegacyDocMixin:
    def _parse_legacy_doc(
        self,
        doc_token: str,
        feishu_access_token: Optional[str] = None,
    ) -> Tuple[str, str]:
        metadata = self._fetch_legacy_doc_metadata(
            doc_token,
            feishu_access_token=feishu_access_token,
        )
        title = str(metadata.get("title") or "Untitled")
        content = self._fetch_legacy_doc_raw_content(
            doc_token,
            feishu_access_token=feishu_access_token,
        ).strip()

        if title and title != "Untitled":
            markdown = f"# {title}\n\n{content}" if content else f"# {title}"
        else:
            markdown = content
        return markdown, title

    def _fetch_legacy_doc_metadata(
        self,
        doc_token: str,
        *,
        feishu_access_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        import lark_oapi as lark

        client = self._get_client(use_user_token=bool(feishu_access_token))
        token_type = (
            lark.AccessTokenType.USER if feishu_access_token else lark.AccessTokenType.TENANT
        )
        request = (
            lark.BaseRequest.builder()
            .http_method(lark.HttpMethod.GET)
            .uri(f"/open-apis/doc/v2/meta/{doc_token}")
            .token_types({token_type})
            .build()
        )
        response = self._call_api(client.request, request, feishu_access_token)
        if not response.success():
            _raise_from_lark_response(
                response,
                operation=f"fetch legacy document metadata for {doc_token}",
                resource=doc_token,
            )
        data = self._raw_response_data(response)
        return data if isinstance(data, dict) else {}

    def _fetch_legacy_doc_raw_content(
        self,
        doc_token: str,
        *,
        feishu_access_token: Optional[str] = None,
    ) -> str:
        import lark_oapi as lark

        client = self._get_client(use_user_token=bool(feishu_access_token))
        token_type = (
            lark.AccessTokenType.USER if feishu_access_token else lark.AccessTokenType.TENANT
        )
        request = (
            lark.BaseRequest.builder()
            .http_method(lark.HttpMethod.GET)
            .uri(f"/open-apis/doc/v2/{doc_token}/raw_content")
            .token_types({token_type})
            .build()
        )
        response = self._call_api(client.request, request, feishu_access_token)
        if not response.success():
            _raise_from_lark_response(
                response,
                operation=f"fetch legacy document content for {doc_token}",
                resource=doc_token,
            )
        data = self._raw_response_data(response)
        content = _getattr_safe(data, "content", "")
        return str(content or "")
