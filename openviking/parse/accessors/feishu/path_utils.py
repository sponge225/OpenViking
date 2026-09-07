# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import re
from pathlib import Path

_MAX_PATH_SEGMENT_CHARS = 120
_MAX_PATH_SEGMENT_BYTES = 240


def _title_as_filename(title: str) -> str:
    """Keep a Feishu display title intact while making it one filename segment.

    Feishu titles may contain path separators.  ``original_filename`` is passed
    through filename-oriented helpers downstream, so leaving separators in that
    field makes ``Path(...).name`` silently discard the title prefix.
    """
    return title.replace("/", "_").replace("\\", "_")


def _truncate_text_for_path_segment(text: str, *, max_chars: int, max_bytes: int) -> str:
    """Trim text so its UTF-8 representation fits a single path segment budget."""
    if max_chars <= 0 or max_bytes <= 0:
        return ""
    result: list[str] = []
    used_bytes = 0
    for char in text[:max_chars]:
        char_bytes = len(char.encode("utf-8"))
        if used_bytes + char_bytes > max_bytes:
            break
        result.append(char)
        used_bytes += char_bytes
    return "".join(result)


def _fit_path_segment(text: str, *, max_chars: int, max_bytes: int) -> str:
    """Ensure a complete path segment fits the configured character and byte budgets."""
    if len(text) <= max_chars and len(text.encode("utf-8")) <= max_bytes:
        return text
    return _truncate_text_for_path_segment(
        text,
        max_chars=max_chars,
        max_bytes=max_bytes,
    ).rstrip(" ._")


def _safe_path_segment(
    name: str,
    *,
    fallback: str = "untitled",
    max_len: int = _MAX_PATH_SEGMENT_CHARS,
    max_bytes: int = _MAX_PATH_SEGMENT_BYTES,
) -> str:
    """Return one portable path segment while preserving readable names."""
    safe_name = re.sub(r"[\x00-\x1f/\\:*?\"<>|]+", "_", str(name or "")).strip(" ._")
    safe_name = re.sub(r"\s+", " ", safe_name)
    if not safe_name:
        safe_name = fallback
    if len(safe_name) <= max_len and len(safe_name.encode("utf-8")) <= max_bytes:
        return safe_name

    stem = Path(safe_name).stem
    suffix = Path(safe_name).suffix
    suffix_bytes = len(suffix.encode("utf-8"))
    if suffix_bytes >= max_bytes:
        suffix = ""
        suffix_bytes = 0

    stem = _truncate_text_for_path_segment(
        stem,
        max_chars=max_len - len(suffix),
        max_bytes=max_bytes - suffix_bytes,
    ).rstrip(" ._")
    if not stem:
        stem = _truncate_text_for_path_segment(
            fallback,
            max_chars=max_len - len(suffix),
            max_bytes=max_bytes - suffix_bytes,
        ).rstrip(" ._")
    return (
        _fit_path_segment(
            f"{stem or 'untitled'}{suffix}",
            max_chars=max_len,
            max_bytes=max_bytes,
        )
        or "untitled"
    )


def _numbered_path_segment(stem: str, suffix: str, index: int) -> str:
    marker = f" ({index})"
    marker_bytes = len(marker.encode("utf-8"))
    suffix_bytes = len(suffix.encode("utf-8"))
    max_stem_bytes = _MAX_PATH_SEGMENT_BYTES - marker_bytes - suffix_bytes
    max_stem_chars = _MAX_PATH_SEGMENT_CHARS - len(marker) - len(suffix)
    safe_stem = _truncate_text_for_path_segment(
        stem,
        max_chars=max_stem_chars,
        max_bytes=max_stem_bytes,
    ).rstrip(" ._")
    return (
        _fit_path_segment(
            f"{safe_stem or 'untitled'}{marker}{suffix}",
            max_chars=_MAX_PATH_SEGMENT_CHARS,
            max_bytes=_MAX_PATH_SEGMENT_BYTES,
        )
        or "untitled"
    )


def _unique_child_path(parent: Path, name: str) -> Path:
    path = parent / _safe_path_segment(name)
    if not path.exists():
        return path
    suffix = path.suffix
    stem = path.stem
    for index in range(2, 10000):
        candidate = parent / _numbered_path_segment(stem, suffix, index)
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to allocate unique path under {parent}")


def _markdown_file_name(name: str) -> str:
    if str(name).lower().endswith((".md", ".markdown")):
        return _safe_path_segment(name)
    return _safe_path_segment(f"{name}.md")
