# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json
from typing import Dict, Optional, Tuple

from openviking.parse.base import format_table_to_markdown
from openviking_cli.utils.logger import get_logger

from .client import _getattr_safe, _raise_from_lark_response

logger = get_logger(__name__)


class FeishuDocxMixin:
    # Attributes that skip processing (structural containers or metadata)
    _SKIP_ATTRS = {"page", "table_cell", "quote_container", "grid", "grid_column"}

    # Attribute → special handler method (non-text blocks)
    _SPECIAL_BLOCK_HANDLERS = {
        "divider": "_handle_divider",
        "image": "_handle_image",
        "table": "_table_block_to_markdown",
        "sheet": "_embedded_sheet_to_markdown",
    }

    # Attribute → markdown prefix template for text-bearing blocks.
    # "{text}" is replaced with extracted text content.
    # Headings are handled dynamically (heading1-heading9 → # through #########).
    _TEXT_FORMAT = {
        "bullet": "- {text}",
        "quote": "> {text}",
    }

    # Known block_type integer → SDK attribute name mapping.
    # Primary dispatch mechanism for reliable block detection.
    # Source: Feishu OpenAPI documentation + lark-oapi SDK Block class.
    _BLOCK_TYPE_TO_ATTR = {
        1: "page",
        2: "text",
        3: "heading1",
        4: "heading2",
        5: "heading3",
        6: "heading4",
        7: "heading5",
        8: "heading6",
        9: "heading7",
        10: "heading8",
        11: "heading9",
        12: "bullet",
        13: "ordered",
        14: "code",
        15: "quote",
        17: "todo",
        19: "callout",
        22: "divider",
        27: "image",
        30: "sheet",
        31: "table",
        32: "table_cell",
        34: "quote_container",
    }

    # All known content attribute names on SDK Block objects (for fallback detection).
    _KNOWN_CONTENT_ATTRS = frozenset(
        {
            "page",
            "text",
            "heading1",
            "heading2",
            "heading3",
            "heading4",
            "heading5",
            "heading6",
            "heading7",
            "heading8",
            "heading9",
            "bullet",
            "ordered",
            "code",
            "quote",
            "todo",
            "callout",
            "divider",
            "image",
            "sheet",
            "table",
            "table_cell",
            "quote_container",
            "equation",
            "task",
            "grid",
            "grid_column",
        }
    )

    def _parse_docx(
        self,
        document_id: str,
        feishu_access_token: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Fetch all blocks and convert to Markdown.

        Returns:
            (markdown_content, document_title)
        """
        blocks = self._fetch_all_blocks(
            document_id,
            feishu_access_token=feishu_access_token,
        )
        if not blocks:
            return "", "Untitled"

        # Build block lookup by block_id
        block_map = {b.block_id: b for b in blocks}

        # Find title from page block
        doc_title = "Untitled"
        for b in blocks:
            if b.page is not None:
                if b.page.elements:
                    doc_title = self._extract_text_from_elements(b.page.elements)
                break

        # Convert blocks to markdown
        markdown_lines = []
        ordered_counter: Dict[str, int] = {}

        for block in blocks:
            if block.page is not None:
                continue  # Skip page container

            line = self._block_to_markdown(
                block,
                block_map,
                ordered_counter,
                document_id=document_id,
                feishu_access_token=feishu_access_token,
            )
            if line is not None:
                markdown_lines.append(line)

        markdown = "\n\n".join(markdown_lines)

        if doc_title and doc_title != "Untitled":
            markdown = f"# {doc_title}\n\n{markdown}"

        return markdown, doc_title

    def _probe_docx_document(
        self,
        document_id: str,
        *,
        feishu_access_token: Optional[str] = None,
    ) -> None:
        """Check document block read permission without loading the whole document."""
        from lark_oapi.api.docx.v1 import ListDocumentBlockRequest

        client = self._get_client(use_user_token=bool(feishu_access_token))
        request = (
            ListDocumentBlockRequest.builder()
            .document_id(document_id)
            .page_size(1)
            .document_revision_id(-1)
            .build()
        )
        response = self._call_api(
            client.docx.v1.document_block.list,
            request,
            feishu_access_token,
        )
        if not response.success():
            _raise_from_lark_response(
                response,
                operation=f"probe document {document_id}",
                resource=document_id,
            )

    def _fetch_all_blocks(
        self,
        document_id: str,
        *,
        feishu_access_token: Optional[str] = None,
    ) -> list:
        """Fetch all blocks with pagination. Returns list of SDK block objects."""
        from lark_oapi.api.docx.v1 import ListDocumentBlockRequest

        client = self._get_client(use_user_token=bool(feishu_access_token))
        all_blocks = []
        page_token = None

        while True:
            builder = (
                ListDocumentBlockRequest.builder()
                .document_id(document_id)
                .page_size(500)
                .document_revision_id(-1)
            )
            if page_token:
                builder = builder.page_token(page_token)

            request = builder.build()
            response = self._call_api(
                client.docx.v1.document_block.list,
                request,
                feishu_access_token,
            )

            if not response.success():
                _raise_from_lark_response(
                    response,
                    operation=f"fetch blocks for {document_id}",
                    resource=document_id,
                )

            items = response.data.items or []
            all_blocks.extend(items)

            if not response.data.has_more:
                break
            page_token = response.data.page_token

        return all_blocks

    # ========== Block -> Markdown Conversion ==========

    def _detect_block_attr(self, block) -> Optional[str]:
        """Detect which content attribute is populated on a block object.

        Uses block_type integer as the primary dispatch (reliable), falling
        back to attribute inspection over a known whitelist for unknown types.
        """
        # Primary: lookup by block_type integer
        block_type = getattr(block, "block_type", None)
        if block_type is not None:
            attr = self._BLOCK_TYPE_TO_ATTR.get(block_type)
            if attr:
                return attr

        # Fallback: scan known content attributes for unknown block types
        for attr in self._KNOWN_CONTENT_ATTRS:
            if getattr(block, attr, None) is not None:
                return attr
        return None

    def _block_to_markdown(
        self,
        block,
        block_map: Dict,
        ordered_counter: Dict[str, int],
        document_id: str = "",
        feishu_access_token: Optional[str] = None,
    ) -> Optional[str]:
        """Convert a single SDK block object to markdown string.

        Uses block_type integer for primary dispatch, with attribute whitelist
        fallback for unknown types. Formatting is data-driven via _TEXT_FORMAT
        and _SPECIAL_BLOCK_HANDLERS tables.
        """
        attr = self._detect_block_attr(block)

        if attr is None:
            return None

        # Skip structural containers (processed via their children)
        if attr in self._SKIP_ATTRS:
            return None

        # Reset ordered list counter when any non-ordered block appears
        if attr != "ordered":
            parent_id = block.parent_id or ""
            if parent_id in ordered_counter:
                del ordered_counter[parent_id]

        # Special blocks (non-text: divider, image, table)
        special_handler = self._SPECIAL_BLOCK_HANDLERS.get(attr)
        if special_handler:
            return getattr(self, special_handler)(
                block,
                block_map,
                document_id=document_id,
                feishu_access_token=feishu_access_token,
            )

        # --- Text-bearing blocks: extract elements, apply formatting ---
        content_obj = getattr(block, attr, None)
        if not content_obj or not hasattr(content_obj, "elements") or not content_obj.elements:
            return None

        text = self._extract_text_from_elements(content_obj.elements)
        if not text:
            return None

        # Headings: heading1 -> #, heading2 -> ##, ...
        if attr.startswith("heading"):
            level = int(attr.replace("heading", "") or "1")
            return f"{'#' * level} {text}"

        # Ordered list (needs counter state)
        if attr == "ordered":
            parent_id = block.parent_id or ""
            counter = ordered_counter.get(parent_id, 0) + 1
            ordered_counter[parent_id] = counter
            return f"{counter}. {text}"

        # Code block (needs language from style)
        if attr == "code":
            lang = ""
            if hasattr(content_obj, "style") and content_obj.style:
                lang = str(getattr(content_obj.style, "language", "") or "")
            return f"```{lang}\n{text}\n```"

        # Todo (needs done state from style)
        if attr == "todo":
            done = False
            if hasattr(content_obj, "style") and content_obj.style:
                done = getattr(content_obj.style, "done", False)
            checkbox = "[x]" if done else "[ ]"
            return f"- {checkbox} {text}"

        # Simple template formatting (bullet, quote, etc.)
        fmt = self._TEXT_FORMAT.get(attr)
        if fmt:
            return fmt.format(text=text)

        # Default: return plain text (covers callout, equation, task, unknown, etc.)
        return text

    @staticmethod
    def _handle_divider(block, block_map: Dict = None, **_) -> str:
        """Convert divider block to markdown."""
        return "---"

    @staticmethod
    def _handle_image(block, block_map: Dict = None, **_) -> Optional[str]:
        """Convert image block to markdown."""
        image = block.image
        if not image:
            return None
        file_token = image.token or ""
        alt_text = getattr(image, "alt", "") or "image"
        return f"![{alt_text}](feishu://image/{file_token})"

    def _extract_block_text(self, block, attr_name: str) -> str:
        """Extract text from a block's named attribute (e.g. block.text, block.heading2)."""
        content_obj = getattr(block, attr_name, None)
        if content_obj and hasattr(content_obj, "elements") and content_obj.elements:
            return self._extract_text_from_elements(content_obj.elements)
        return ""

    def _extract_text_from_elements(self, elements) -> str:
        """Convert Feishu TextElement SDK objects to formatted text."""
        if not elements:
            return ""
        parts = []
        for element in elements:
            # TextRun
            text_run = element.text_run
            if text_run:
                content = text_run.content or ""
                style = text_run.text_element_style
                content = self._apply_text_style(content, style)
                parts.append(content)
                continue

            # MentionUser
            mention_user = element.mention_user
            if mention_user:
                user_id = _getattr_safe(mention_user, "user_id", "user")
                parts.append(f"@{user_id}")
                continue

            # MentionDoc
            mention_doc = element.mention_doc
            if mention_doc:
                title = _getattr_safe(mention_doc, "title", "document")
                url = _getattr_safe(mention_doc, "url", "")
                parts.append(f"[{title}]({url})" if url else str(title))
                continue

            # Equation
            equation = element.equation
            if equation:
                parts.append(f"${_getattr_safe(equation, 'content', '')}$")
                continue

        return "".join(parts)

    @staticmethod
    def _apply_text_style(text: str, style) -> str:
        """Apply markdown formatting based on TextElementStyle SDK object."""
        if not text or not style:
            return text
        # inline_code (SDK uses 'inline_code', not 'code_inline')
        if getattr(style, "inline_code", False):
            return f"`{text}`"
        # link
        link = getattr(style, "link", None)
        if link:
            url = _getattr_safe(link, "url", "")
            if url:
                text = f"[{text}]({url})"
        if getattr(style, "bold", False):
            text = f"**{text}**"
        if getattr(style, "italic", False):
            text = f"*{text}*"
        if getattr(style, "strikethrough", False):
            text = f"~~{text}~~"
        return text

    def _table_block_to_markdown(self, block, block_map: Dict, **_) -> Optional[str]:
        """Convert table block to markdown table."""
        table = block.table
        children = block.children
        if not table or not children:
            return None

        prop = table.property
        if not prop:
            return None
        row_size = prop.row_size or 0
        col_size = prop.column_size or 0
        if not row_size or not col_size:
            return None

        rows = []
        for row_idx in range(row_size):
            row = []
            for col_idx in range(col_size):
                cell_idx = row_idx * col_size + col_idx
                if cell_idx < len(children):
                    cell_block_id = children[cell_idx]
                    cell_block = block_map.get(cell_block_id)
                    cell_text = self._extract_cell_text(cell_block, block_map)
                    row.append(cell_text)
                else:
                    row.append("")
            rows.append(row)

        return format_table_to_markdown(rows, has_header=True) if rows else None

    def _extract_cell_text(self, cell_block, block_map: Dict) -> str:
        """Extract text from a table cell block by reading its children."""
        if not cell_block or not cell_block.children:
            return ""
        texts = []
        for child_id in cell_block.children:
            child = block_map.get(child_id)
            if not child:
                continue
            # Use attribute-driven detection to find text in any block type
            attr = self._detect_block_attr(child)
            if attr:
                text = self._extract_block_text(child, attr)
                if text:
                    texts.append(text)
        return " ".join(texts)

    def _embedded_sheet_to_markdown(
        self,
        block,
        block_map: Dict = None,
        *,
        document_id: str = "",
        feishu_access_token: Optional[str] = None,
        **_,
    ) -> Optional[str]:
        """Convert an embedded spreadsheet block in a docx document."""
        import lark_oapi as lark

        client = self._get_client(use_user_token=bool(feishu_access_token))
        token_type = (
            lark.AccessTokenType.USER if feishu_access_token else lark.AccessTokenType.TENANT
        )
        request = (
            lark.BaseRequest.builder()
            .http_method(lark.HttpMethod.GET)
            .uri(
                f"/open-apis/docx/v1/documents/{document_id or block.parent_id}"
                f"/blocks/{block.block_id}"
            )
            .token_types({token_type})
            .build()
        )
        response = self._call_api(client.request, request, feishu_access_token)
        if not response.success():
            logger.warning(
                "[FeishuAccessor] Failed to inspect embedded sheet %s: code=%s msg=%s",
                block.block_id,
                getattr(response, "code", None),
                getattr(response, "msg", None),
            )
            return None

        data = json.loads(response.raw.content)
        sheet_token = data.get("data", {}).get("block", {}).get("sheet", {}).get("token", "")
        parts = sheet_token.rsplit("_", 1)
        if len(parts) != 2:
            return None

        spreadsheet_token, sheet_id = parts
        try:
            rows = self._read_sheet_range(
                spreadsheet_token,
                sheet_id,
                max_rows=100,
                max_cols=26,
                feishu_access_token=feishu_access_token,
            )
        except Exception as exc:
            logger.warning(
                "[FeishuAccessor] Failed to read embedded sheet %s: %s",
                sheet_token,
                exc,
            )
            return None

        rows = self._trim_empty_columns(rows)
        return format_table_to_markdown(rows, has_header=True) if rows else None
