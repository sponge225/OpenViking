"""OpenViking file system tools: read, write, list, search resources."""

import asyncio
import json
import os
import time
from abc import ABC
from pathlib import Path
from typing import Any, Optional, Union

import httpx
from loguru import logger

from vikingbot.agent.tools.base import Tool, ToolContext
from vikingbot.openviking_mount.ov_server import VikingClient, update_active_relation_groups


def _parse_relation_reason(reason: Any) -> dict[str, Any] | None:
    """Return structured relation reason only when it is valid JSON."""
    if isinstance(reason, dict):
        return reason
    if not isinstance(reason, str) or not reason.strip():
        return None
    try:
        parsed = json.loads(reason)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _short_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text[:limit]


class OVFileTool(Tool, ABC):
    def __init__(self):
        super().__init__()
        self._client = None

    async def _get_client(self, tool_context: ToolContext):
        if self._client is None:
            self._client = await VikingClient.create(tool_context.workspace_id)
        return self._client

class VikingListTool(OVFileTool):
    """Tool to list Viking resources."""

    @property
    def name(self) -> str:
        return "openviking_list"

    @property
    def description(self) -> str:
        return "List resources in a OpenViking folder path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": "The parent Viking uri to list (e.g., viking://resources/)",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to list recursively",
                    "default": False,
                },
            },
            "required": ["uri"],
        }

    async def execute(
        self, tool_context: "ToolContext", uri: str, recursive: bool = False, **kwargs: Any
    ) -> str:
        try:
            client = await self._get_client(tool_context)
            entries = await client.list_resources(path=uri, recursive=recursive)

            if not entries:
                return f"No resources found at {uri}"

            result = []
            for entry in entries:
                item = {
                    "name": entry["name"],
                    "size": entry["size"],
                    "uri": entry["uri"],
                    "isDir": entry["isDir"],
                }
                result.append(str(item))
            return "\n".join(result)
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            return f"Error listing Viking resources: {str(e)}"


class VikingSearchTool(OVFileTool):
    """Tool to search Viking resources."""

    @property
    def name(self) -> str:
        return "openviking_search"

    @property
    def description(self) -> str:
        return ("Using query to search for resources (knowledge, code, files, workflow, etc.) in OpenViking. "
                "This operation performs semantic retrieval, not full character matching. Please avoid repeated calls with similar queries as much as possible."
                "bad-case: after searching with ‘Nate Joanna dog playdate 3:00 pm', another search was performed using 'Nate Joanna dog playdate'.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "target_uri": {
                    "type": "string",
                    "description": "Optional target URI to limit search scope, if is None, then search the entire range.(e.g., viking://resources/)",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        tool_context: "ToolContext",
        query: str,
        target_uri: Optional[str] = "",
        **kwargs: Any,
    ) -> str:
        try:
            search_total_start = time.time()
            client = await self._get_client(tool_context)
            search_client = getattr(client, 'admin_user_client', client)
            # Get default limit from config, then request 3x to ensure enough L2 results after filtering
            try:
                from openviking_cli.utils.config import get_openviking_config
                default_limit = get_openviking_config().default_search_limit
            except Exception:
                default_limit = 10
            semantic_search_start = time.time()
            results = await search_client.search(query, target_uri=target_uri, limit=default_limit * 3)
            semantic_search_ms = (time.time() - semantic_search_start) * 1000
            logger.info(
                f"[Search][PROFILE] semantic_search_ms={semantic_search_ms:.0f}, "
                f"default_limit={default_limit}, target_uri={target_uri or '<all>'}, "
                f"query_len={len(query or '')}"
            )

            if not results:
                return f"No results found for query: {query}"

            if isinstance(results, dict):
                resources_list = results.get('resources', []) or []
            elif isinstance(results, list):
                resources_list = results
            else:
                resources_list = []

            if not resources_list:
                return str(results)

            raw_count = len(resources_list)
            filter_start = time.time()
            # Filter out L0 (abstract) and L1 (overview) documents - only keep L2 content
            resources_list = [
                r for r in resources_list
                if not r.get("uri", "").endswith(".abstract.md")
                and not r.get("uri", "").endswith(".overview.md")
            ]
            l2_count = len(resources_list)

            if not resources_list:
                return f"No L2 content results found for query: {query}"

            # Keep only top `default_limit` results by score
            resources_list = sorted(resources_list, key=lambda r: r.get("score", 0), reverse=True)[:default_limit]
            filter_ms = (time.time() - filter_start) * 1000

            use_relations = os.environ.get("VIKINGBOT_USE_RELATIONS", "0") == "1"
            relations_found = 0
            logger.info(
                f"[Search][PROFILE] filter_ms={filter_ms:.0f}, raw_count={raw_count}, "
                f"l2_count={l2_count}, kept={len(resources_list)}, use_relations={use_relations}"
            )
            if use_relations:
                link_strategy = os.environ.get("VIKINGBOT_LINK_STRATEGY", "llm_review")
                seen_uris = {r.get("uri", "") for r in resources_list}
                initial_search_uris = set(seen_uris)
                initial_count = len(resources_list)
                try:
                    relations_topk = max(0, int(os.environ.get("VIKINGBOT_RELATIONS_TOPK", "0")))
                except (TypeError, ValueError):
                    relations_topk = 0
                active_group_scores: dict[str, float] = {}
                pruned_relation_groups: set[str] = set()
                relation_total_start = time.time()
                relation_call_total_ms = 0.0
                abstract_total_ms = 0.0
                relation_errors = 0
                relation_depth = 0

                frontier = [r.get("uri", "") for r in resources_list if r.get("uri", "")]
                logger.info(
                    f"[Search] Relations expansion starting: "
                    f"frontier={len(frontier)} URIs, strategy={link_strategy}, "
                    f"relations_topk={relations_topk}"
                )

                while frontier:
                    relation_depth += 1
                    depth_start = time.time()
                    depth_frontier_count = len(frontier)
                    rel_call_start = time.time()
                    rel_tasks = [
                        client.relations(
                            uri,
                            query=query,
                            strategy=link_strategy,
                            include_match_meta=True,
                        )
                        for uri in frontier
                    ]
                    rel_results = await asyncio.gather(*rel_tasks, return_exceptions=True)
                    rel_call_ms = (time.time() - rel_call_start) * 1000
                    relation_call_total_ms += rel_call_ms

                    depth_candidates: list[tuple[str, dict[str, Any]]] = []
                    next_frontier = []
                    depth_records = 0
                    depth_new_docs = 0
                    depth_existing_hits = 0
                    depth_skipped = 0
                    depth_abstract_reads = 0
                    depth_abstract_ms = 0.0
                    for from_uri, rels in zip(frontier, rel_results):
                        if isinstance(rels, Exception):
                            relation_errors += 1
                            continue
                        depth_records += len(rels)
                        for rel in rels:
                            rel_uri = rel.get("uri", "")
                            if not rel_uri:
                                depth_skipped += 1
                                continue
                            # Skip L0 (abstract) and L1 (overview) documents
                            if rel_uri.endswith(".abstract.md") or rel_uri.endswith(".overview.md"):
                                depth_skipped += 1
                                continue
                            group_key = rel.get("group_key") or rel.get("question_id") or rel_uri
                            rel["group_key"] = group_key
                            if relations_topk > 0 and group_key in pruned_relation_groups:
                                depth_skipped += 1
                                continue
                            depth_candidates.append((from_uri, rel))

                    if relations_topk > 0:
                        selected_groups, newly_pruned = update_active_relation_groups(
                            active_group_scores,
                            [rel for _, rel in depth_candidates],
                            relations_topk,
                        )
                        pruned_relation_groups.update(newly_pruned)
                    else:
                        selected_groups = None

                    for from_uri, rel in depth_candidates:
                        rel_uri = rel.get("uri", "")
                        group_key = rel.get("group_key") or rel.get("question_id") or rel_uri
                        if selected_groups is not None and group_key not in selected_groups:
                            depth_skipped += 1
                            continue
                        if rel_uri in seen_uris:
                            for existing in resources_list:
                                if existing.get("uri") != rel_uri:
                                    continue
                                existing_match = str(existing.get("match_reason", ""))
                                should_update = not existing_match
                                if (
                                    not should_update
                                    and selected_groups is not None
                                    and existing_match.startswith("relation_from:")
                                ):
                                    existing_group = (
                                        existing.get("relation_group_key")
                                        or existing.get("relation_question_id")
                                        or ""
                                    )
                                    try:
                                        existing_similarity = float(existing.get("relation_similarity", 0.0) or 0.0)
                                    except (TypeError, ValueError):
                                        existing_similarity = 0.0
                                    should_update = (
                                        existing_group not in selected_groups
                                        or float(rel.get("similarity", 0.0) or 0.0) > existing_similarity
                                    )
                                if should_update:
                                    reason_json = _parse_relation_reason(rel.get("reason", ""))
                                    existing["match_reason"] = f"relation_from: {from_uri}"
                                    existing["relation_from"] = from_uri
                                    existing["relation_question_id"] = rel.get("question_id", "")
                                    existing["relation_group_key"] = group_key
                                    existing["relation_similarity"] = rel.get("similarity", 0.0)
                                    existing["is_priority"] = True
                                    existing["relation_reason"] = rel.get("reason", "") if reason_json else ""
                                    if reason_json:
                                        existing["relation_reason_json"] = reason_json
                                    else:
                                        existing.pop("relation_reason_json", None)
                                    if not existing_match:
                                        relations_found += 1
                                        depth_existing_hits += 1
                                break
                            continue
                        seen_uris.add(rel_uri)
                        reason_json = _parse_relation_reason(rel.get("reason", ""))
                        abstract_start = time.time()
                        try:
                            abstract = await client.read_content(rel_uri, level="abstract")
                        except Exception:
                            abstract = ""
                        abstract_ms = (time.time() - abstract_start) * 1000
                        abstract_total_ms += abstract_ms
                        depth_abstract_ms += abstract_ms
                        depth_abstract_reads += 1
                        item = {
                            "uri": rel_uri,
                            "context_type": "ContextType.RESOURCE",
                            "is_leaf": False,
                            "abstract": abstract or "",
                            "overview": None,
                            "category": "",
                            "score": 0,
                            "match_reason": f"relation_from: {from_uri}",
                            "relation_from": from_uri,
                            "relation_question_id": rel.get("question_id", ""),
                            "relation_group_key": group_key,
                            "relation_similarity": rel.get("similarity", 0.0),
                            "is_priority": True,
                            "relation_reason": rel.get("reason", "") if reason_json else "",
                            "relations": [],
                        }
                        if reason_json:
                            item["relation_reason_json"] = reason_json
                        resources_list.append(item)
                        relations_found += 1
                        depth_new_docs += 1
                        next_frontier.append(rel_uri)

                    depth_ms = (time.time() - depth_start) * 1000
                    logger.info(
                        f"[Search][PROFILE] relations_depth={relation_depth}, "
                        f"frontier={depth_frontier_count}, rel_call_ms={rel_call_ms:.0f}, "
                        f"rel_records={depth_records}, new_docs={depth_new_docs}, "
                        f"existing_hits={depth_existing_hits}, skipped={depth_skipped}, "
                        f"abstract_reads={depth_abstract_reads}, abstract_ms={depth_abstract_ms:.0f}, "
                        f"next_frontier={len(next_frontier)}, depth_ms={depth_ms:.0f}, "
                        f"active_groups={len(active_group_scores)}, total_results={len(resources_list)}"
                    )
                    frontier = next_frontier

                if relations_topk > 0:
                    active_groups = set(active_group_scores)
                    filtered_resources = []
                    removed_relation_docs = 0
                    demoted_search_hits = 0
                    for item in resources_list:
                        match_reason = str(item.get("match_reason", ""))
                        if match_reason.startswith("relation_from:"):
                            group_key = item.get("relation_group_key") or item.get("relation_question_id") or ""
                            if group_key not in active_groups:
                                if item.get("uri", "") in initial_search_uris:
                                    for key in (
                                        "relation_from",
                                        "relation_question_id",
                                        "relation_group_key",
                                        "relation_similarity",
                                        "is_priority",
                                        "relation_reason",
                                        "relation_reason_json",
                                        "priority_rank",
                                    ):
                                        item.pop(key, None)
                                    item["match_reason"] = ""
                                    filtered_resources.append(item)
                                    demoted_search_hits += 1
                                else:
                                    removed_relation_docs += 1
                                continue
                        filtered_resources.append(item)
                    resources_list = filtered_resources
                    if removed_relation_docs or demoted_search_hits:
                        logger.info(
                            f"[Search][PROFILE] relation_beam_final_filter "
                            f"active_groups={len(active_groups)}, removed_relation_docs={removed_relation_docs}, "
                            f"demoted_search_hits={demoted_search_hits}"
                        )

                relations_found = sum(
                    1
                    for item in resources_list
                    if str(item.get("match_reason", "")).startswith("relation_from:")
                )
                relation_total_ms = (time.time() - relation_total_start) * 1000
                logger.info(
                    f"[Search][PROFILE] relations_total_ms={relation_total_ms:.0f}, "
                    f"depth={relation_depth}, relation_call_ms={relation_call_total_ms:.0f}, "
                    f"abstract_ms={abstract_total_ms:.0f}, errors={relation_errors}, "
                    f"relations_found={relations_found}, active_groups={len(active_group_scores)}, "
                    f"total_results={len(resources_list)} "
                    f"(was {initial_count}), strategy={link_strategy}"
                )

            if tool_context:
                tool_context.structured_result = resources_list

            if use_relations and relations_found > 0:
                logger.info(f"[Search] Relations mode: {relations_found} related docs found, using priority format")
                relation_results = []
                search_results = []
                for r in resources_list:
                    if r.get("match_reason", "").startswith("relation_from:"):
                        r["is_priority"] = True
                        if "relation_from" not in r:
                            r["relation_from"] = r.get("match_reason", "").replace("relation_from:", "").strip()
                        reason_json = _parse_relation_reason(r.get("relation_reason", ""))
                        if reason_json:
                            r["relation_reason_json"] = reason_json
                        else:
                            r["relation_reason"] = ""
                            r.pop("relation_reason_json", None)
                        relation_results.append(r)
                    else:
                        search_results.append(r)
                for priority_rank, r in enumerate(relation_results, 1):
                    r["priority_rank"] = priority_rank

                result_strs = []
                idx = 1

                reason_groups: dict[str, dict[str, Any]] = {}
                for r in relation_results:
                    reason_json = r.get("relation_reason_json")
                    if not isinstance(reason_json, dict):
                        continue
                    key = r.get("relation_question_id") or reason_json.get("question_id")
                    if not key:
                        key = json.dumps(reason_json, ensure_ascii=False, sort_keys=True)
                    if key not in reason_groups:
                        reason_groups[key] = {"reason": reason_json, "uris": []}
                    reason_groups[key]["uris"].append(r.get("uri", ""))

                result_strs.append("=== PRIORITY (pre-explored results) ===")
                result_strs.append("")
                result_strs.append("[PRIORITY PROTOCOL]")
                result_strs.append("The PRIORITY documents below are the complete historical useful resource list for this matched question.")
                result_strs.append("Before using SEARCH RESULTS, read ALL PRIORITY documents in one openviking_multi_read call.")
                result_strs.append("After reading them, answer directly if the evidence is sufficient; only continue search/grep if the PRIORITY documents are insufficient.")
                result_strs.append("")
                result_strs.append("[STRUCTURED HISTORY SUMMARY]")

                if reason_groups:
                    for group in reason_groups.values():
                        reason = group["reason"]
                        result_strs.append("")
                        result_strs.append(f"Question: {_short_text(reason.get('question'), 320)}")
                        result_strs.append(f"Historical answer: {_short_text(reason.get('answer'), 500)}")
                        if reason.get("sufficient") is not None:
                            result_strs.append(f"Sufficient: {reason.get('sufficient')}")
                        tool_path = reason.get("tool_path")
                        if isinstance(tool_path, list) and tool_path:
                            path_items = []
                            for step in tool_path[:8]:
                                if not isinstance(step, dict):
                                    continue
                                tool_name = step.get("tool", "")
                                if step.get("query"):
                                    path_items.append(f"{tool_name}({str(step.get('query'))[:60]})")
                                elif step.get("uri"):
                                    path_items.append(f"{tool_name}({step.get('uri')})")
                                else:
                                    path_items.append(str(tool_name))
                            if path_items:
                                result_strs.append(f"Historical path: {' -> '.join(path_items)}")
                        evidence_summary = reason.get("evidence_summary")
                        if evidence_summary:
                            result_strs.append(f"Evidence summary: {_short_text(evidence_summary, 500)}")
                else:
                    result_strs.append("")
                    result_strs.append("No valid structured history summary is available for these priority documents.")

                result_strs.append("Documents selected as useful by that session:")
                for r in relation_results:
                    rel_uri = r.get("uri", "")
                    rel_abstract = r.get("abstract", "")[:200]
                    result_strs.append(f"  PRIORITY-{r.get('priority_rank', idx)}. [{rel_uri}] {rel_abstract}")
                    idx += 1
                result_strs.append("")
                result_strs.append("[YOUR STRATEGY]")
                result_strs.append("(1) Batch-read ALL PRIORITY documents above in a SINGLE openviking_multi_read call.")
                result_strs.append("(2) Answer immediately from the content. Only read SEARCH RESULTS if PRIORITY documents are insufficient.")
                result_strs.append("(3) Use the structured history summary only as guidance; verify the answer from the documents you read.")
                result_strs.append("")
                result_strs.append("=== SEARCH RESULTS ===")
                for r in search_results:
                    uri = r.get("uri", "")
                    abstract = r.get("abstract", "")[:200]
                    score = r.get("score", 0)
                    result_strs.append(f"{idx}. [{uri}] (score: {score:.2f}) {abstract}")
                    idx += 1

                output = "\n".join(result_strs)
                output += f"\n<!-- relations_found:{relations_found} searched:{len(resources_list)} -->"
            else:
                logger.info(f"[Search] Standard mode: {len(resources_list)} results")
                result_strs = [f"Search results for: {query}"]
                result_strs.append(f"\n{'=' * 40}")
                result_strs.append(f"SEARCH RESULTS ({len(resources_list)} items):")
                result_strs.append(f"{'=' * 40}")
                for i, item in enumerate(resources_list, 1):
                    uri = item.get("uri", "")
                    abstract = item.get("abstract", "")
                    score = item.get("score", 0.0)
                    result_strs.append(f"\n  [{i}] {uri} (score: {score:.3f})")
                    if abstract:
                        result_strs.append(f"      {abstract[:200]}")
                output = "\n".join(result_strs)

            search_total_ms = (time.time() - search_total_start) * 1000
            logger.info(
                f"[Search][PROFILE] total_ms={search_total_ms:.0f}, "
                f"returned={len(resources_list)}, use_relations={use_relations}, "
                f"relations_found={relations_found}"
            )
            return output
        except Exception as e:
            return f"Error searching Viking: {str(e)}"


class VikingAddResourceTool(OVFileTool):
    """Tool to add a resource to Viking."""

    @property
    def name(self) -> str:
        return "openviking_add_resource"

    @property
    def description(self) -> str:
        return "Add a resource (url like pic, git code or local file path) to OpenViking.This is a asynchronous operation."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Url or local file path"},
                "description": {"type": "string", "description": "Description of the resource"},
            },
            "required": ["path", "description"],
        }

    async def execute(
        self,
        tool_context: "ToolContext",
        path: str,
        description: str,
        **kwargs: Any,
    ) -> str:
        client = None
        try:
            if path and not path.startswith("http"):
                local_path = Path(path).expanduser().resolve()
                if not local_path.exists():
                    return f"Error: File not found: {path}"
                if not local_path.is_file():
                    return f"Error: Not a file: {path}"

            client = await VikingClient.create(tool_context.workspace_id)
            result = await client.add_resource(path, description)

            if result:
                root_uri = result.get("root_uri", "")
                return f"Successfully added resource: {root_uri}"
            else:
                return "Failed to add resource"
        except httpx.ReadTimeout:
            return f"Request timed out. The resource addition task may still be processing on the server side."
        except Exception as e:
            logger.warning(f"Error adding resource: {e}")
            return f"Error adding resource to Viking: {str(e)}"
        finally:
            if client:
                await client.close()


class VikingGrepTool(OVFileTool):
    """Tool to search Viking resources using regex patterns."""

    @property
    def name(self) -> str:
        return "openviking_grep"

    @property
    def description(self) -> str:
        return ("Search Viking resources using regex patterns (like grep). Supports multiple patterns to search concurrently."
                "Please avoid repeated calls with similar queries as much as possible.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": "The whole Viking URI to search within (e.g., viking://resources/)",
                },
                "pattern": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Regex pattern or array of regex patterns to search for",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search",
                    "default": False,
                },
            },
            "required": ["uri", "pattern"],
        }

    async def execute(
        self,
        tool_context: "ToolContext",
        uri: str,
        pattern: Union[str, list[str]],
        case_insensitive: bool = False,
        **kwargs: Any,
    ) -> str:
        try:
            client = await self._get_client(tool_context)
            patterns = [pattern] if isinstance(pattern, str) else pattern

            # Limit concurrent requests to avoid overwhelming the server and memory
            max_concurrent = 10
            semaphore = asyncio.Semaphore(max_concurrent)

            async def run_grep(p: str) -> tuple[str, list[Any]]:
                async with semaphore:
                    try:
                        result = await client.grep(uri, p, case_insensitive=case_insensitive)
                        if isinstance(result, dict):
                            matches = result.get("matches", [])
                        else:
                            matches = getattr(result, "matches", [])
                        return (p, matches)
                    except Exception as e:
                        logger.warning(f"Error searching for pattern '{p}': {e}")
                        return (p, [])

            tasks = [run_grep(p) for p in patterns]
            results = await asyncio.gather(*tasks)

            # Merge results by URI
            merged_results: dict[str, list[tuple[int, str, str]]] = {}
            total_matches = 0

            for p, matches in results:
                if not matches:
                    continue
                total_matches += len(matches)
                for match in matches:
                    if isinstance(match, dict):
                        match_uri = match.get("uri", "unknown")
                        line = match.get("line", "?")
                        content = match.get("content", "")
                    else:
                        match_uri = getattr(match, "uri", "unknown")
                        line = getattr(match, "line", "?")
                        content = getattr(match, "content", "")

                    if match_uri not in merged_results:
                        merged_results[match_uri] = []
                    merged_results[match_uri].append((line, content, p))

            if not merged_results:
                pattern_str = ", ".join(f"'{p}'" for p in patterns)
                return f"No matches found for patterns: {pattern_str}"

            result_lines = [f"Found {total_matches} match{'es' if total_matches != 1 else ''} across {len(patterns)} pattern{'s' if len(patterns) != 1 else ''}:"]

            struct_result: list[dict[str, Any]] = []

            for match_uri, matches in merged_results.items():
                matches.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)
                result_lines.append(f"\n[{match_uri}]")
                uri_struct = {"uri": match_uri, "lines": []}
                for line, content, pattern_name in matches:
                    result_lines.append(f"  L{line}: {content}")
                    uri_struct["lines"].append({"line": line, "content": content, "pattern": pattern_name})
                struct_result.append(uri_struct)


            return "\n".join(result_lines)
        except Exception as e:
            return f"Error searching Viking with grep: {str(e)}"


class VikingGlobTool(OVFileTool):
    """Tool to find Viking resources using glob patterns."""

    @property
    def name(self) -> str:
        return "openviking_glob"

    @property
    def description(self) -> str:
        return "Find Viking resources using glob patterns (like **/*.md, *.py)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g., **/*.md, *.py, src/**/*.js)",
                },
                "uri": {
                    "type": "string",
                    "description": "The whole Viking URI to search within (e.g., viking://resources/path/)",
                    "default": "",
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self, tool_context: "ToolContext", pattern: str, uri: str = "", **kwargs: Any
    ) -> str:
        try:
            client = await self._get_client(tool_context)
            result = await client.glob(pattern, uri=uri or None)

            if isinstance(result, dict):
                matches = result.get("matches", [])
                count = result.get("count", 0)
            else:
                matches = getattr(result, "matches", [])
                count = getattr(result, "count", 0)

            if not matches:
                return f"No files found for pattern: {pattern}"

            result_lines = [f"Found {count} file{'s' if count != 1 else ''}:"]
            for match_uri in matches:
                if isinstance(match_uri, dict):
                    match_uri = match_uri.get("uri", str(match_uri))
                result_lines.append(f"📄 {match_uri}")

            return "\n".join(result_lines)
        except Exception as e:
            return f"Error searching Viking with glob: {str(e)}"

class VikingMemoryCommitTool(OVFileTool):
    """Tool to commit messages to OpenViking session."""

    @property
    def name(self) -> str:
        return "openviking_memory_commit"

    @property
    def description(self) -> str:
        return "When user has personal information needs to be remembered, Commit messages to OpenViking."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "List of messages to commit, each with role, content",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string", "enum": ["user", "assistant"]},
                            "content": {"type": "string"},
                        },
                        "required": ["role", "content"],
                    },
                },
            },
            "required": ["messages"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        try:
            if not tool_context.sender_id:
                return "Error committed, sender_id is required."
            client = await self._get_client(tool_context)
            session_id = tool_context.session_key.safe_name()
            await client.commit(session_id, messages, tool_context.sender_id)
            return f"Successfully committed to session {session_id}"
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            return f"Error committing to Viking: {str(e)}"

class VikingMultiReadTool(OVFileTool):
    """Tool to read content from multiple Viking resources concurrently."""

    @property
    def name(self) -> str:
        return "openviking_multi_read"

    @property
    def description(self) -> str:
        return "Read full content from multiple OpenViking resources concurrently. Returns complete content for all URIs with no truncation."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "uris": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of Viking file URIs to read from (e.g., [\"viking://resources/path/123.md\", \"viking://resources/path/456.md\"])",
                },
            },
            "required": ["uris"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        uris: list[str],
        **kwargs: Any,
    ) -> str:
        level = "read"  # 默认获取完整内容
        try:
            if not uris:
                return "Error: No URIs provided."

            client = await self._get_client(tool_context)
            max_concurrent = 10
            semaphore = asyncio.Semaphore(max_concurrent)

            async def read_single_uri(uri: str) -> dict:
                async with semaphore:
                    try:
                        content = await client.read_content(uri, level=level)
                        return {
                            "uri": uri,
                            "content": content,
                            "success": True,
                        }
                    except Exception as e:
                        logger.warning(f"Error reading from {uri}: {e}")
                        return {
                            "uri": uri,
                            "content": f"Error reading from Viking: {str(e)}",
                            "success": False,
                        }

            # 并发读取所有URI
            read_tasks = [read_single_uri(uri) for uri in uris]
            results = await asyncio.gather(*read_tasks)

            # 构建结果
            result_lines = [f"Multi-read results for {len(uris)} resources (level: {level}):"]

            for i, result in enumerate(results, 1):
                uri = result["uri"]
                content = result["content"]
                success = result["success"]

                result_lines.append(f"\n--- START OF {uri} ---")
                if success:
                    result_lines.append(content)
                else:
                    result_lines.append(f"ERROR: {content}")
                result_lines.append(f"--- END OF {uri} ---")

            return "\n".join(result_lines)

        except Exception as e:
            logger.exception(f"Error in VikingMultiReadTool: {e}")
            return f"Error multi-reading Viking resources: {str(e)}"
