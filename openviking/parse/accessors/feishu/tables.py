# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json
import mimetypes
from typing import Any, Dict, List, Optional, Tuple

from openviking.parse.base import format_table_to_markdown

from .client import _raise_from_lark_response
from .media import _MAX_MEDIA_DOWNLOAD_CONTEXTS, _MediaDownloadExtras


class FeishuTablesMixin:
    @staticmethod
    def _trim_empty_columns(rows: List[List[str]]) -> List[List[str]]:
        """Remove trailing columns that are empty in every row."""
        if not rows:
            return rows
        last_col = 0
        for col in range(max(len(row) for row in rows)):
            if any(col < len(row) and row[col].strip() for row in rows):
                last_col = col + 1
        return [row[:last_col] for row in rows] if last_col else []

    def _parse_sheets(
        self,
        token: str,
        feishu_access_token: Optional[str] = None,
        *,
        media_download_extras: Optional[_MediaDownloadExtras] = None,
    ) -> Tuple[str, str]:
        """Fetch a Feishu spreadsheet and convert it to Markdown."""
        config = self._get_config()
        metadata = self._fetch_spreadsheet_metadata(
            token,
            feishu_access_token=feishu_access_token,
        )
        title = (metadata.get("properties") or {}).get("title") or "Spreadsheet"
        sheets = metadata.get("sheets") or []
        markdown_parts = [f"# {title}", f"**Sheets:** {len(sheets)}"]
        for sheet in sheets:
            sheet_id = sheet.get("sheetId") or ""
            sheet_title = sheet.get("title") or sheet_id
            parts = [f"## Sheet: {sheet_title}"]

            block_info = sheet.get("blockInfo")
            if block_info:
                block_type = block_info.get("blockType") or "unknown"
                if block_type != "BITABLE_BLOCK":
                    parts.append(f"*Unsupported sheet block: {block_type}*")
                else:
                    block_token = block_info.get("blockToken") or ""
                    tokens = block_token.rsplit("_", 1)
                    if len(tokens) != 2 or not all(tokens):
                        parts.append("*Invalid embedded bitable token*")
                    else:
                        bitable, _ = self._parse_bitable(
                            tokens[0],
                            feishu_access_token,
                            table_id=tokens[1],
                            table_name=sheet_title,
                            media_download_extras=media_download_extras,
                        )
                        parts.append(bitable or "*Empty bitable*")
                markdown_parts.append("\n\n".join(parts))
                continue

            row_count = int(sheet.get("rowCount") or 0)
            col_count = int(sheet.get("columnCount") or 0)
            if not row_count or not col_count:
                parts.append("*Empty sheet*")
                markdown_parts.append("\n\n".join(parts))
                continue

            parts.append(f"**Dimensions:** {row_count} rows x {col_count} columns")
            rows_to_read = min(row_count, config.max_rows_per_sheet)
            rows = self._read_sheet_range(
                token,
                sheet_id,
                rows_to_read,
                col_count,
                feishu_access_token=feishu_access_token,
            )
            if rows:
                parts.append(format_table_to_markdown(rows, has_header=True))
            if row_count > config.max_rows_per_sheet:
                parts.append(
                    f"\n*... {row_count - config.max_rows_per_sheet} more rows truncated ...*"
                )
            if col_count > 26:
                parts.append(f"\n*... {col_count - 26} columns after Z omitted ...*")
            markdown_parts.append("\n\n".join(parts))

        return "\n\n".join(markdown_parts), title

    def _fetch_spreadsheet_metadata(
        self,
        token: str,
        *,
        feishu_access_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        import lark_oapi as lark

        client = self._get_client(use_user_token=bool(feishu_access_token))
        token_type = (
            lark.AccessTokenType.USER if feishu_access_token else lark.AccessTokenType.TENANT
        )
        metadata_request = (
            lark.BaseRequest.builder()
            .http_method(lark.HttpMethod.GET)
            .uri(f"/open-apis/sheets/v2/spreadsheets/{token}/metainfo")
            .token_types({token_type})
            .build()
        )
        metadata_response = self._call_api(
            client.request,
            metadata_request,
            feishu_access_token,
        )
        if not metadata_response.success():
            _raise_from_lark_response(
                metadata_response,
                operation=f"fetch spreadsheet metadata for {token}",
                resource=token,
            )
        return json.loads(metadata_response.raw.content).get("data", {})

    def _read_sheet_range(
        self,
        token: str,
        sheet_id: str,
        max_rows: int,
        max_cols: int,
        feishu_access_token: Optional[str] = None,
    ) -> List[List[str]]:
        """Read a bounded cell range from a Feishu spreadsheet."""
        import lark_oapi as lark

        client = self._get_client(use_user_token=bool(feishu_access_token))
        # ponytail: the existing importer reads A:Z only; add chunked ranges if wider
        # spreadsheet imports become a real requirement.
        end_col = self._col_number_to_letter(min(max_cols, 26))
        cell_range = f"{sheet_id}!A1:{end_col}{max_rows}"
        token_type = (
            lark.AccessTokenType.USER if feishu_access_token else lark.AccessTokenType.TENANT
        )
        request = (
            lark.BaseRequest.builder()
            .http_method(lark.HttpMethod.GET)
            .uri(f"/open-apis/sheets/v2/spreadsheets/{token}/values/{cell_range}")
            .token_types({token_type})
            .build()
        )
        response = self._call_api(client.request, request, feishu_access_token)
        if not response.success():
            _raise_from_lark_response(
                response,
                operation=f"read spreadsheet range {cell_range}",
                resource=token,
            )

        data = json.loads(response.raw.content)
        values = data.get("data", {}).get("valueRange", {}).get("values", [])
        return [[str(cell) if cell is not None else "" for cell in row] for row in values]

    @staticmethod
    def _col_number_to_letter(number: int) -> str:
        return chr(ord("A") + number - 1) if 1 <= number <= 26 else "Z"

    def _parse_bitable(
        self,
        app_token: str,
        feishu_access_token: Optional[str] = None,
        *,
        table_id: Optional[str] = None,
        table_name: Optional[str] = None,
        view_id: Optional[str] = None,
        media_download_extras: Optional[_MediaDownloadExtras] = None,
    ) -> Tuple[str, str]:
        """Fetch a Feishu bitable app and convert it to Markdown."""
        if view_id and not table_id:
            raise ValueError("Feishu Base URL with 'view' must also include 'table'")

        from lark_oapi.api.bitable.v1 import (
            ListAppTableFieldRequest,
            ListAppTableRecordRequest,
        )

        client = self._get_client(use_user_token=bool(feishu_access_token))
        config = self._get_config()
        if table_id:
            tables = [(table_id, table_name or table_id)]
            title = table_name or table_id
            if view_id:
                title = f"{title} ({view_id})"
            markdown_parts = []
            heading = "###"
        else:
            table_models = self._list_bitable_tables(
                app_token,
                feishu_access_token=feishu_access_token,
            )
            tables = [(table.table_id, table.name or table.table_id) for table in table_models]
            title = f"Bitable ({len(tables)} tables)"
            markdown_parts = [f"# {title}"]
            heading = "##"

        for current_table_id, current_table_name in tables:
            fields = []
            page_token = None
            while True:
                builder = (
                    ListAppTableFieldRequest.builder()
                    .app_token(app_token)
                    .table_id(current_table_id)
                    .page_size(100)
                )
                if page_token:
                    builder = builder.page_token(page_token)
                fields_response = self._call_api(
                    client.bitable.v1.app_table_field.list,
                    builder.build(),
                    feishu_access_token,
                )
                if not fields_response.success():
                    _raise_from_lark_response(
                        fields_response,
                        operation=f"list fields for bitable table {current_table_id}",
                        resource=app_token,
                    )
                fields.extend(fields_response.data.items or [])
                if not getattr(fields_response.data, "has_more", False):
                    break
                page_token = getattr(fields_response.data, "page_token", None)
                if not page_token:
                    raise RuntimeError(
                        f"Feishu returned more fields for table {current_table_id} "
                        "without a page token"
                    )
            field_names = [field.field_name for field in fields]
            field_ids = {
                field.field_name: field.field_id
                for field in fields
                if getattr(field, "field_name", None) and getattr(field, "field_id", None)
            }

            records = []
            page_token = None
            records_truncated = False
            while len(records) < config.max_records_per_table:
                remaining = config.max_records_per_table - len(records)
                builder = (
                    ListAppTableRecordRequest.builder()
                    .app_token(app_token)
                    .table_id(current_table_id)
                    .page_size(min(remaining, 500))
                )
                if view_id:
                    builder = builder.view_id(view_id)
                if page_token:
                    builder = builder.page_token(page_token)
                records_response = self._call_api(
                    client.bitable.v1.app_table_record.list,
                    builder.build(),
                    feishu_access_token,
                )
                if not records_response.success():
                    _raise_from_lark_response(
                        records_response,
                        operation=f"list records for bitable table {current_table_id}",
                        resource=app_token,
                    )
                items = records_response.data.items or []
                records.extend(items[:remaining])
                has_more = bool(records_response.data.has_more)
                if len(items) > remaining:
                    records_truncated = True
                    break
                if not has_more:
                    break
                if len(records) >= config.max_records_per_table:
                    records_truncated = True
                    break
                page_token = records_response.data.page_token
                if not page_token:
                    raise RuntimeError(
                        f"Feishu returned more records for table {current_table_id} "
                        "without a page token"
                    )

            parts = [f"{heading} {current_table_name}", f"**Records:** {len(records)}"]
            if field_names and records:
                rows = [field_names]
                for record in records:
                    record_fields = record.fields or {}
                    row = []
                    for name in field_names:
                        value = record_fields.get(name, "")
                        row.append(self._format_bitable_field(value))
                        if media_download_extras is not None:
                            self._collect_bitable_media_extras(
                                value,
                                table_id=current_table_id,
                                field_id=field_ids.get(name),
                                record_id=getattr(record, "record_id", None),
                                media_download_extras=media_download_extras,
                            )
                    rows.append(row)
                parts.append(format_table_to_markdown(rows, has_header=True))
            if records_truncated:
                parts.append(f"\n*... records truncated at {config.max_records_per_table} ...*")
            markdown_parts.append("\n\n".join(parts))

        return "\n\n".join(markdown_parts), title

    def _list_bitable_tables(
        self,
        app_token: str,
        *,
        feishu_access_token: Optional[str] = None,
    ) -> List[Any]:
        from lark_oapi.api.bitable.v1 import ListAppTableRequest

        client = self._get_client(use_user_token=bool(feishu_access_token))
        table_models = []
        page_token = None
        while True:
            builder = ListAppTableRequest.builder().app_token(app_token).page_size(100)
            if page_token:
                builder = builder.page_token(page_token)
            tables_response = self._call_api(
                client.bitable.v1.app_table.list,
                builder.build(),
                feishu_access_token,
            )
            if not tables_response.success():
                _raise_from_lark_response(
                    tables_response,
                    operation=f"list bitable tables for {app_token}",
                    resource=app_token,
                )
            table_models.extend(tables_response.data.items or [])
            if not getattr(tables_response.data, "has_more", False):
                break
            page_token = getattr(tables_response.data, "page_token", None)
            if not page_token:
                raise RuntimeError("Feishu returned more bitable tables without a page token")
        return table_models

    def _probe_bitable_table(
        self,
        app_token: str,
        table_id: str,
        *,
        view_id: Optional[str] = None,
        feishu_access_token: Optional[str] = None,
    ) -> None:
        from lark_oapi.api.bitable.v1 import (
            ListAppTableFieldRequest,
            ListAppTableRecordRequest,
        )

        client = self._get_client(use_user_token=bool(feishu_access_token))
        fields_response = self._call_api(
            client.bitable.v1.app_table_field.list,
            ListAppTableFieldRequest.builder()
            .app_token(app_token)
            .table_id(table_id)
            .page_size(1)
            .build(),
            feishu_access_token,
        )
        if not fields_response.success():
            _raise_from_lark_response(
                fields_response,
                operation=f"probe bitable fields for table {table_id}",
                resource=app_token,
            )

        record_builder = (
            ListAppTableRecordRequest.builder().app_token(app_token).table_id(table_id).page_size(1)
        )
        if view_id:
            record_builder = record_builder.view_id(view_id)
        records_response = self._call_api(
            client.bitable.v1.app_table_record.list,
            record_builder.build(),
            feishu_access_token,
        )
        if not records_response.success():
            _raise_from_lark_response(
                records_response,
                operation=f"probe bitable records for table {table_id}",
                resource=app_token,
            )

    @classmethod
    def _collect_bitable_media_extras(
        cls,
        value: Any,
        *,
        table_id: str,
        field_id: Optional[str],
        record_id: Optional[str],
        media_download_extras: _MediaDownloadExtras,
    ) -> None:
        """Collect transient permission contexts for image refs emitted from a cell."""
        if isinstance(value, list):
            for item in value:
                cls._collect_bitable_media_extras(
                    item,
                    table_id=table_id,
                    field_id=field_id,
                    record_id=record_id,
                    media_download_extras=media_download_extras,
                )
            return
        if not isinstance(value, dict):
            return

        file_token = value.get("file_token")
        name = str(value.get("name") or "image")
        media_type = value.get("type") or mimetypes.guess_type(name)[0]
        if not file_token or not str(media_type).lower().startswith("image/"):
            return

        contexts = media_download_extras.setdefault(str(file_token), [])
        if field_id and record_id:
            extra = json.dumps(
                {
                    "bitablePerm": {
                        "tableId": table_id,
                        "attachments": {field_id: {record_id: [str(file_token)]}},
                    }
                },
                separators=(",", ":"),
            )
            context_count = sum(context is not None for context in contexts)
            if extra not in contexts and context_count < _MAX_MEDIA_DOWNLOAD_CONTEXTS:
                contexts.append(extra)
        elif None not in contexts:
            contexts.append(None)

    @classmethod
    def _format_bitable_field(cls, value: Any) -> str:
        """Render the common structured values returned by bitable fields."""
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(cls._format_bitable_field(item) for item in value)
        if isinstance(value, dict):
            file_token = value.get("file_token")
            name = str(value.get("name") or "image")
            media_type = value.get("type") or mimetypes.guess_type(name)[0]
            if file_token and str(media_type).lower().startswith("image/"):
                return f"![{name}](feishu://image/{file_token})"
            return str(value.get("text", value.get("name", value)))
        return str(value)
