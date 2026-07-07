"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from vikingbot.agent.context import ContextBuilder
from vikingbot.agent.memory import MemoryStore
from vikingbot.heartbeat.service import HEARTBEAT_METADATA_KEY, is_heartbeat_noop_response
from vikingbot.agent.subagent import SubagentManager
from vikingbot.agent.tools import register_default_tools
from vikingbot.agent.tools.registry import ToolRegistry
from vikingbot.bus.events import InboundMessage, OutboundEventType, OutboundMessage
from vikingbot.bus.queue import MessageBus
from vikingbot.config import load_config
from vikingbot.config.schema import BotMode, Config, SessionKey
from vikingbot.hooks import HookContext
from vikingbot.hooks.manager import hook_manager
from vikingbot.providers.base import LLMProvider
from vikingbot.sandbox import SandboxManager
from vikingbot.session.manager import SessionManager
from vikingbot.utils.helpers import cal_str_tokens
from vikingbot.utils.tracing import trace

if TYPE_CHECKING:
    from vikingbot.config.schema import ExecToolConfig
    from vikingbot.cron.service import CronService


def _normalize_uri(uri: str) -> str:
    """Normalize a URI by removing whitespace."""
    return re.sub(r'\s+', '', uri)


def _extract_uris_from_args(args_raw) -> list[str]:
    """Extract viking:// URIs from tool call args."""
    if isinstance(args_raw, dict):
        args_str = json.dumps(args_raw)
    elif isinstance(args_raw, str):
        args_str = args_raw
    else:
        return []
    uris = re.findall(r'viking://[^\s"\\,\]\}]+', args_str)
    return [_normalize_uri(u) for u in uris]


_LINK_QUERY_MAX_CHARS = 2000
_LINK_QUERY_CONTAMINATION_MARKERS = (
    "Historical answer:",
    "=== SEARCH RESULTS ===",
    "Documents selected as useful",
    "[YOUR STRATEGY]",
    "<!-- relations_found:",
    "relation_reason",
)


def _extract_link_query(raw_query: str | None) -> str:
    """Return the clean user question to persist with LLM-reviewed relation edges."""
    if not isinstance(raw_query, str):
        return ""

    query = raw_query.strip()
    if not query:
        return ""

    for pattern in (
        r"(?ims)^\s*Here'?s the question:\s*(.+)",
        r"(?ims)^\s*Question:\s*(.+)",
    ):
        match = re.search(pattern, query)
        if not match:
            continue
        label_value = match.group(1).strip()
        if label_value:
            query = label_value
            break

    query_lower = query.lower()
    for marker in _LINK_QUERY_CONTAMINATION_MARKERS:
        marker_pos = query_lower.find(marker.lower())
        if marker_pos >= 0:
            query = query[:marker_pos].strip()
            query_lower = query.lower()

    if "\n" in query:
        query = next((line.strip() for line in query.splitlines() if line.strip()), "")

    if len(query) > _LINK_QUERY_MAX_CHARS:
        logger.warning(
            f"[LLMReview] skip linking because extracted query is too long: {len(query)} chars"
        )
        return ""

    return query


def _tool_args_to_dict(args_raw) -> dict:
    """Safely coerce tool-call args into a dict, tolerating malformed inputs."""
    if isinstance(args_raw, dict):
        return args_raw
    if isinstance(args_raw, str):
        try:
            parsed = json.loads(args_raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _build_edge_reason(original_query: str, tgt_uri: str, tools_used: list, final_content: str) -> str:
    """Build a structured relation reason for future relation-guided search."""
    tool_path: list[dict[str, str]] = []
    for tool in tools_used:
        tool_name = tool.get("tool_name", "")
        if not tool.get("execute_success"):
            continue

        args = tool.get("args", "")
        result = tool.get("result", "")

        involves_tgt = False

        if tool_name == "openviking_search":
            if isinstance(result, list):
                involves_tgt = any(
                    isinstance(item, dict) and item.get("uri", "") == tgt_uri
                    for item in result
                )
            elif isinstance(result, str):
                involves_tgt = tgt_uri in result

        elif tool_name in ("openviking_read", "openviking_multi_read"):
            args_str = json.dumps(args) if not isinstance(args, str) else args
            involves_tgt = tgt_uri in args_str

        elif tool_name == "openviking_grep":
            if isinstance(result, list):
                involves_tgt = any(
                    isinstance(item, dict) and item.get("uri", "") == tgt_uri
                    for item in result
                )
            elif isinstance(result, str):
                involves_tgt = tgt_uri in result

        if not involves_tgt:
            continue

        if tool_name == "openviking_search":
            args_obj = _tool_args_to_dict(args)
            query = args_obj.get("query", "")
            tool_path.append({"tool": "openviking_search", "query": query})

        elif tool_name in ("openviking_read", "openviking_multi_read"):
            tool_path.append({"tool": tool_name, "uri": tgt_uri})

        elif tool_name == "openviking_grep":
            args_obj = _tool_args_to_dict(args)
            pattern = args_obj.get("pattern", "")
            grep_hits = []
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict) and item.get("uri") == tgt_uri:
                        grep_hits.append(f"L{item.get('line', '?')}: {item.get('content', '')[:60]}")
            hit_str = "; ".join(grep_hits[:3])
            if hit_str:
                tool_path.append({"tool": "openviking_grep", "pattern": pattern, "hits": hit_str})
            else:
                tool_path.append({"tool": "openviking_grep", "pattern": pattern})

    reason = {
        "version": 1,
        "question": original_query,
        "answer": final_content,
        "target_uri": tgt_uri,
        "tool_path": tool_path,
        "evidence_summary": final_content[:800],
        "sufficient": True,
    }
    return json.dumps(reason, ensure_ascii=False)


def _collect_relation_derived_uris(tools_used: list) -> set[str]:
    """Return URIs that were surfaced by relation-guided search."""
    relation_uris: set[str] = set()
    for tool in tools_used or []:
        if not isinstance(tool, dict) or tool.get("tool_name") != "openviking_search":
            continue
        result = tool.get("result")
        if not isinstance(result, list):
            continue
        for item in result:
            if not isinstance(item, dict):
                continue
            if not (
                item.get("is_priority")
                or item.get("relation_from")
                or str(item.get("match_reason", "")).startswith("relation_from:")
            ):
                continue
            uri = item.get("uri", "")
            if uri:
                relation_uris.add(_normalize_uri(uri))
    return relation_uris


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 50,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exa_api_key: str | None = None,
        gen_image_model: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        session_manager: SessionManager | None = None,
        sandbox_manager: SandboxManager | None = None,
        config: Config = None,
        eval: bool = False,
        mcp_servers: dict | None = None,
    ):
        """
        Initialize the AgentLoop with all required dependencies and configuration.

        Args:
            bus: MessageBus instance for publishing and subscribing to messages.
            provider: LLMProvider instance for making LLM calls.
            workspace: Path to the workspace directory for file operations.
            model: Optional model identifier. If not provided, uses the provider's default.
            max_iterations: Maximum number of tool execution iterations per message (default: 50).
            memory_window: Maximum number of messages to keep in session memory (default: 50).
            brave_api_key: Optional API key for Brave search integration.
            exa_api_key: Optional API key for Exa search integration.
            gen_image_model: Optional model identifier for image generation (default: openai/doubao-seedream-4-5-251128).
            exec_config: Optional configuration for the exec tool (command execution).
            cron_service: Optional CronService for scheduled task management.
            session_manager: Optional SessionManager for session persistence. If not provided, a new one is created.
            sandbox_manager: Optional SandboxManager for sandboxed operations.
            config: Optional Config object with full configuration. Used if other parameters are not provided.

        Note:
            The AgentLoop creates its own ContextBuilder, SessionManager (if not provided),
            ToolRegistry, and SubagentManager during initialization.

        Example:
            >>> loop = AgentLoop(
            ...     bus=message_bus,
            ...     provider=llm_provider,
            ...     workspace=Path("/path/to/workspace"),
            ...     model="gpt-4",
            ...     max_iterations=30,
            ... )
        """
        from vikingbot.config.schema import ExecToolConfig  # noqa: F811

        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exa_api_key = exa_api_key
        self.gen_image_model = gen_image_model or "openai/doubao-seedream-4-5-251128"
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.sandbox_manager = sandbox_manager
        self.config = config

        self.context = ContextBuilder(workspace, sandbox_manager=sandbox_manager)

        self._register_builtin_hooks()
        self.sessions = session_manager or SessionManager(
            self.config.bot_data_path, sandbox_manager=sandbox_manager
        )
        self.tools = ToolRegistry()
        self._eval = eval
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            config=self.config,
            model=self.model,
            sandbox_manager=sandbox_manager,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._register_default_tools()

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy, retryable on failure).

        Ported from HKUDS/nanobot v0.1.5.
        """
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        try:
            from vikingbot.agent.tools.mcp import connect_mcp_servers

            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error(f"Failed to connect MCP servers (will retry next message): {e}")
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    async def close_mcp(self) -> None:
        """Close MCP server connections. Ported from HKUDS/nanobot v0.1.5."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None
        self._mcp_connected = False

    async def _publish_thinking_event(
        self, session_key: SessionKey, event_type: OutboundEventType, content: str
    ) -> None:
        """
        Publish a thinking event to the message bus.

        Thinking events are used to communicate the agent's internal processing
        state to the user, such as when the agent is executing a tool or
        processing a complex request.

        Args:
            session_key: The session key identifying the conversation.
            event_type: The type of thinking event (e.g., THINKING, TOOL_START).
            content: The message content to display to the user.

        Note:
            This is an internal method used by the agent loop to communicate
            progress to users during long-running operations.

        Example:
            >>> await self._publish_thinking_event(
            ...     session_key=SessionKey(channel="telegram", chat_id="123"),
            ...     event_type=OutboundEventType.TOOL_START,
            ...     content="Executing web search..."
            ... )
        """
        await self.bus.publish_outbound(
            OutboundMessage(
                session_key=session_key,
                content=content,
                event_type=event_type,
            )
        )

    def _register_builtin_hooks(self):
        """Register built-in hooks."""
        hook_manager.register_path(self.config.hooks)

    def _register_default_tools(self) -> None:
        """Register default set of tools."""
        register_default_tools(
            registry=self.tools,
            config=self.config,
            send_callback=self.bus.publish_outbound,
            subagent_manager=self.subagents,
            cron_service=self.cron_service,
            include_image_tool=not self._eval,
            include_cron_tool=not self._eval,
            include_spawn_tool=not self._eval,
            include_message_tool=not self._eval,
            include_web_tool=not self._eval,
            include_memory_tool=not self._eval,
            include_filesystem_tool=not self._eval,
        )

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)

                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.exception(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            session_key=msg.session_key,
                            content=f"Sorry, I encountered an error: {str(e)}",
                            metadata=msg.metadata,
                        )
                    )
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _run_agent_loop(
        self,
        messages: list[dict],
        session_key: SessionKey,
        publish_events: bool = True,
        sender_id: str | None = None,
        ov_tools_enable: bool = True,
        link_query: str | None = None,
    ) -> tuple[str | None, list[dict], dict[str, int], int, list[dict]]:
        """
        Run the core agent loop: call LLM, execute tools, repeat until done.

        Args:
            messages: Initial message list
            session_key: Session key for tool execution context
            publish_events: Whether to publish ITERATION/REASONING/TOOL_CALL events to the bus
            ov_tools_enable: Whether to enable OpenViking tools for this session
            link_query: Clean inbound user question to store on relation edges

        Returns:
            tuple of (final_content, tools_used, token_usage, iteration, messages)
        """
        iteration = 0
        final_content = None
        tools_used: list[dict] = []
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        while iteration < self.max_iterations:
            iteration += 1

            if publish_events:
                await self.bus.publish_outbound(
                    OutboundMessage(
                        session_key=session_key,
                        content=f"Iteration {iteration}/{self.max_iterations}",
                        event_type=OutboundEventType.ITERATION,
                    )
                )

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(ov_tools_enable=ov_tools_enable),
                model=self.model,
                session_id=session_key.safe_name(),
            )
            if response.usage:
                cur_token = response.usage
                token_usage["prompt_tokens"] += cur_token["prompt_tokens"]
                token_usage["completion_tokens"] += cur_token["completion_tokens"]
                token_usage["total_tokens"] += cur_token["total_tokens"]

            if publish_events and response.reasoning_content:
                await self.bus.publish_outbound(
                    OutboundMessage(
                        session_key=session_key,
                        content=response.reasoning_content,
                        event_type=OutboundEventType.REASONING,
                    )
                )

            if response.has_tool_calls:
                args_list = [tc.arguments for tc in response.tool_calls]
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(args),
                        },
                    }
                    for tc, args in zip(response.tool_calls, args_list)
                ]
                messages = self.context.add_assistant_message(
                    messages,
                    response.content,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                # Stage 2: Execute all tools in parallel
                async def execute_single_tool(idx: int, tool_call):
                    """Execute a single tool and track execution time."""
                    tool_execute_start_time = time.time()
                    result, tool_context = await self.tools.execute(
                        tool_call.name,
                        tool_call.arguments,
                        session_key=session_key,
                        sandbox_manager=self.sandbox_manager,
                        sender_id=sender_id,
                    )
                    tool_execute_duration = (time.time() - tool_execute_start_time) * 1000
                    return idx, tool_call, result, tool_execute_duration, tool_context

                # Run all tool executions in parallel
                tool_tasks = [
                    execute_single_tool(idx, tool_call)
                    for idx, tool_call in enumerate(response.tool_calls)
                ]
                results = await asyncio.gather(*tool_tasks)

                # Stage 3: Process results sequentially in original order
                for _idx, tool_call, result, tool_execute_duration, tool_context in results:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"[TOOL_CALL]: {tool_call.name}({args_str[:200]})")
                    logger.info(f"[RESULT]: {str(result)[:600]}")

                    if publish_events:
                        await self.bus.publish_outbound(
                            OutboundMessage(
                                session_key=session_key,
                                content=f"{tool_call.name}({args_str})",
                                event_type=OutboundEventType.TOOL_CALL,
                            )
                        )
                        await self.bus.publish_outbound(
                            OutboundMessage(
                                session_key=session_key,
                                content=str(result),
                                event_type=OutboundEventType.TOOL_RESULT,
                            )
                        )
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )

                    try:
                        args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except (json.JSONDecodeError, TypeError):
                        args_obj = args_str

                    display_result = getattr(tool_context, 'structured_result', None)
                    if display_result is None:
                        display_result = result

                    relations_found = 0
                    if tool_call.name == "openviking_search" and isinstance(display_result, list):
                        relations_found = sum(
                            1
                            for item in display_result
                            if isinstance(item, dict)
                            and str(item.get("match_reason", "")).startswith("relation_from:")
                        )

                    tool_used_dict = {
                        "tool_name": tool_call.name,
                        "args": args_obj,
                        "result": display_result,
                        "duration": tool_execute_duration,
                        "execute_success": True
                        if result and "Error executing" not in result
                        else False,
                        "input_token": tool_call.tokens,
                        "output_token": cal_str_tokens(result, text_type="mixed"),
                    }
                    if relations_found:
                        tool_used_dict["relations_found"] = relations_found
                    tools_used.append(tool_used_dict)

                messages.append(
                    {"role": "user", "content": "Reflect on the results and decide next steps."}
                )
            else:
                final_content = response.content
                break

        if final_content is None or (
            isinstance(final_content, str) and not final_content.strip()
        ):
            if iteration >= self.max_iterations:
                final_content = f"Reached {self.max_iterations} iterations without completion."
            else:
                final_content = "I've completed processing but have no response to give."

        # --- LLM Review 建边 ---
        _enable_linking = os.environ.get("VIKINGBOT_ENABLE_LINKING", "0")
        original_query = _extract_link_query(link_query)

        logger.info(
            f"[LLMReview] gate check: strategy=llm_review, enable={_enable_linking}, "
            f"tools_used={len(tools_used) if tools_used else 0}, "
            f"original_query={bool(original_query)}"
        )
        if _enable_linking == "1" and tools_used and original_query:
            _post_link_start = time.time()
            relation_derived_uris = _collect_relation_derived_uris(tools_used)
            if relation_derived_uris:
                logger.info(
                    f"[LLMReview] relation-derived search URIs detected: "
                    f"{len(relation_derived_uris)}; only matching to_uris will be skipped"
                )

            # Collect read_uris and uri_content
            _collect_start = time.time()
            read_uris: set[str] = set()
            uri_content: dict[str, str] = {}
            for tool in tools_used:
                if tool.get("tool_name") in ("openviking_read", "openviking_multi_read") and tool.get("execute_success"):
                    args = tool.get("args", "")
                    uris = _extract_uris_from_args(args)
                    result = tool.get("result", "")
                    for u in uris:
                        read_uris.add(u)
                        if u not in uri_content:
                            uri_content[u] = (result or "")
                if tool.get("tool_name") == "openviking_grep" and tool.get("execute_success"):
                    result = tool.get("result", "")
                    if isinstance(result, list):
                        for item in result:
                            if isinstance(item, dict):
                                uri = item.get("uri", "")
                                if uri:
                                    uri = _normalize_uri(uri)
                                    read_uris.add(uri)
                                    if uri not in uri_content:
                                        uri_content[uri] = ""
                                    uri_content[uri] += f"L{item.get('line', '?')}: {item.get('content', '')}\n"
                    elif isinstance(result, str):
                        current_uri = None
                        for line in result.split("\n"):
                            line_s = line.strip()
                            m = re.match(r'^\[(.+)\]$', line_s)
                            if m:
                                current_uri = _normalize_uri(m.group(1))
                                read_uris.add(current_uri)
                                if current_uri not in uri_content:
                                    uri_content[current_uri] = ""
                            elif current_uri and line_s.startswith("L"):
                                uri_content[current_uri] += line_s + "\n"

            # Collect search URIs and metadata
            search_uri_meta: dict[str, dict] = {}
            for tool in tools_used:
                if tool.get("tool_name") == "openviking_search" and tool.get("execute_success"):
                    result = tool.get("result", "")
                    if isinstance(result, list):
                        for item in result:
                            if not isinstance(item, dict):
                                continue
                            uri = item.get("uri", "")
                            if uri:
                                uri = _normalize_uri(uri)
                                if uri not in search_uri_meta:
                                    search_uri_meta[uri] = {
                                        "abstract": item.get("abstract", ""),
                                        "score": item.get("score", 0),
                                    }
                    elif isinstance(result, str):
                        for u in _extract_uris_from_args(result):
                            if u not in search_uri_meta:
                                search_uri_meta[u] = {"abstract": "", "score": 0}
            search_uris = set(search_uri_meta.keys())
            _collect_duration = (time.time() - _collect_start) * 1000

            logger.info(
                f"[LLMReview][PROFILE] collect_ms={_collect_duration:.0f}, "
                f"read_uris={len(read_uris)}, search_uris={len(search_uris)}, "
                f"search_uri_meta={len(search_uri_meta)}"
            )

            if read_uris or search_uris:
                # Build review prompt
                _prompt_build_start = time.time()
                if read_uris:
                    candidate_uris = set(read_uris)
                    uri_list_lines = []
                    for idx, u in enumerate(sorted(read_uris), 1):
                        preview = uri_content.get(u, "")[:300].replace("\n", " ")
                        uri_list_lines.append(f"{idx}. {u}\n   Content: {preview}")
                    uri_list = "\n".join(uri_list_lines)
                    review_prompt = (
                        f"Question: {original_query}\n"
                        f"Answer: {final_content[:800]}\n\n"
                        f"Documents read during research:\n{uri_list}\n\n"
                        f"Task: Which of the above documents were USEFUL for answering the question?\n"
                        f"The answer was generated using these documents, so at least one MUST be relevant.\n\n"
                        f"Rules:\n"
                        f"- You MUST return at least one URI. An empty array [] is NEVER valid.\n"
                        f"- If unsure, pick the document most likely related to the answer.\n\n"
                        f"Output ONLY a JSON array of URI strings, no other text:\n"
                        f'["viking://...", "viking://...", ...]\n'
                    )
                    mode = "read"
                else:
                    sorted_search = sorted(
                        search_uri_meta.items(),
                        key=lambda x: x[1].get("score", 0),
                        reverse=True,
                    )[:15]
                    candidate_uris = {uri for uri, _ in sorted_search}
                    uri_list_lines = []
                    for idx, (u, meta) in enumerate(sorted_search, 1):
                        abstract = (meta.get("abstract") or "")[:200].replace("\n", " ")
                        uri_list_lines.append(f"{idx}. {u}\n   Abstract: {abstract}")
                    uri_list = "\n".join(uri_list_lines)
                    review_prompt = (
                        f"Question: {original_query}\n"
                        f"Answer: {final_content[:800]}\n\n"
                        f"Documents found via search:\n{uri_list}\n\n"
                        f"Task: Based on the abstracts, which documents were USEFUL for answering the question?\n"
                        f"The answer was generated using these documents, so at least one MUST be relevant.\n\n"
                        f"Rules:\n"
                        f"- You MUST return at least one URI. An empty array [] is NEVER valid.\n"
                        f"- If unsure, pick the document most likely related to the answer.\n\n"
                        f"Output ONLY a JSON array of URI strings, no other text:\n"
                        f'["viking://...", "viking://...", ...]\n'
                    )
                    mode = "search-only"

                _prompt_build_duration = (time.time() - _prompt_build_start) * 1000
                logger.info(
                    f"[LLMReview][PROFILE] prompt_build_ms={_prompt_build_duration:.0f}, "
                    f"candidate_uris={len(candidate_uris)}, mode={mode}, "
                    f"search_uris={len(search_uris)}, prompt_chars={len(review_prompt)}"
                )

                try:
                    # Call LLM for review
                    _llm_review_start = time.time()
                    review_messages = [
                        {"role": "system", "content": "You are a document relation analyst. Output only valid JSON."},
                        {"role": "user", "content": review_prompt},
                    ]
                    review_response = await self.provider.chat(
                        messages=review_messages,
                        tools=[],
                        model=self.model,
                        session_id=session_key.safe_name(),
                    )
                    _llm_review_duration = (time.time() - _llm_review_start) * 1000
                    logger.info(f"[LLMReview][TIMING] LLM review call took {_llm_review_duration:.0f}ms")

                    if review_response.usage:
                        token_usage["prompt_tokens"] += review_response.usage.get("prompt_tokens", 0)
                        token_usage["completion_tokens"] += review_response.usage.get("completion_tokens", 0)
                        token_usage["total_tokens"] += review_response.usage.get("total_tokens", 0)

                    # Parse LLM output
                    _parse_start = time.time()
                    raw_output = review_response.content or ""
                    cleaned = re.sub(r'```(?:json)?\s*', '', raw_output).strip()
                    cleaned = re.sub(r'```\s*$', '', cleaned).strip()

                    json_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
                    useful_uris: list[str] = []
                    skipped_relation_to_uris: list[str] = []
                    parse_method = "none"

                    logger.info(f"[LLMReview] raw_output={raw_output[:300]}")

                    if json_match:
                        try:
                            parsed = json.loads(json_match.group(0))
                            if isinstance(parsed, list):
                                for u in parsed:
                                    if isinstance(u, str):
                                        u = _normalize_uri(u.rstrip('.'))
                                        if u in candidate_uris:
                                            useful_uris.append(u)
                            parse_method = "json"
                        except json.JSONDecodeError:
                            logger.warning(
                                f"[LLMReview] JSON parse FAILED. Raw output (first 300 chars): {raw_output[:300]}"
                            )

                    # Fallback: regex extract viking:// URIs
                    if not useful_uris:
                        found_uris = re.findall(r'viking://[^\s"\'\\,\]\}]+', raw_output)
                        for u in found_uris:
                            u = _normalize_uri(u.rstrip('.'))
                            if u in candidate_uris:
                                useful_uris.append(u)
                        if useful_uris:
                            parse_method = "regex-fallback"

                    # Deduplicate
                    useful_uris = list(dict.fromkeys(useful_uris))

                    # Forced fallback if empty
                    if not useful_uris:
                        if read_uris:
                            useful_uris = [sorted(candidate_uris)[0]]
                        elif search_uri_meta:
                            best_uri = max(search_uri_meta.keys(), key=lambda u: search_uri_meta[u].get("score", 0))
                            if best_uri in candidate_uris:
                                useful_uris = [best_uri]
                            else:
                                useful_uris = [sorted(candidate_uris)[0]]
                        else:
                            useful_uris = [sorted(candidate_uris)[0]]
                        parse_method = "forced-fallback"
                        logger.warning(
                            f"[LLMReview] LLM returned empty, forced fallback: {useful_uris}"
                        )

                    parsed_useful_count = len(useful_uris)
                    if useful_uris and relation_derived_uris:
                        skipped_relation_to_uris = [
                            u for u in useful_uris
                            if _normalize_uri(u) in relation_derived_uris
                        ]
                        if skipped_relation_to_uris:
                            useful_uris = [
                                u for u in useful_uris
                                if _normalize_uri(u) not in relation_derived_uris
                            ]
                            logger.info(
                                f"[LLMReview] Skipped {len(skipped_relation_to_uris)} "
                                f"relation-derived to_uri(s); remaining_to_uris={len(useful_uris)}"
                            )

                    if useful_uris:
                        logger.info(
                            f"[LLMReview] Parsed {parsed_useful_count} useful doc(s) via {parse_method}; "
                            f"linkable_to_uris={len(useful_uris)}"
                        )
                    elif skipped_relation_to_uris:
                        logger.info(
                            f"[LLMReview] All parsed useful docs were relation-derived to_uris; "
                            f"skipped_to_uris={len(skipped_relation_to_uris)}"
                        )
                    else:
                        logger.warning(
                            f"[LLMReview] 0 useful docs extracted. "
                            f"candidate_uris={len(candidate_uris)}, raw_output={raw_output[:300]}"
                        )
                    _parse_duration = (time.time() - _parse_start) * 1000
                    logger.info(
                        f"[LLMReview][PROFILE] parse_ms={_parse_duration:.0f}, "
                        f"parsed_useful_uris={parsed_useful_count}, "
                        f"linkable_to_uris={len(useful_uris)}, "
                        f"skipped_relation_to_uris={len(skipped_relation_to_uris)}, "
                        f"parse_method={parse_method}"
                    )

                    # Create edges
                    if useful_uris:
                        from vikingbot.openviking_mount.ov_server import VikingClient
                        workspace_id = self.sandbox_manager.to_workspace_id(session_key) if self.sandbox_manager else None
                        _client_create_start = time.time()
                        rv_client = await VikingClient.create(workspace_id)
                        _client_create_duration = (time.time() - _client_create_start) * 1000
                        logger.info(f"[LLMReview][TIMING] VikingClient.create took {_client_create_duration:.0f}ms")
                        try:
                            linked = 0
                            seen_pairs: set[tuple[str, str]] = set()

                            # Search top5 to expand from_uris
                            _top5_search_start = time.time()
                            try:
                                search_result = await rv_client.search(original_query)
                                top5_resources = search_result.get("resources", [])[:5]
                                top5_uris = set()
                                for r in top5_resources:
                                    uri = r.get("uri", "")
                                    if uri:
                                        top5_uris.add(_normalize_uri(uri))
                                all_search_uris = search_uris | top5_uris
                                _top5_search_duration = (time.time() - _top5_search_start) * 1000
                                logger.info(
                                    f"[LLMReview][TIMING] top5 search took {_top5_search_duration:.0f}ms, "
                                    f"added {len(top5_uris)} URIs to from_uris "
                                    f"(total search_uris: {len(search_uris)} -> {len(all_search_uris)})"
                                )
                            except Exception as e:
                                logger.warning(f"[LLMReview] top5 search failed, using original search_uris: {e}")
                                all_search_uris = search_uris

                            from_uris = all_search_uris - set(useful_uris)
                            if not from_uris:
                                from_uris = all_search_uris

                            _link_start = time.time()
                            _link_count = 0
                            _link_ms_values: list[float] = []
                            _edge_reason_total_ms = 0.0
                            for tgt in useful_uris:
                                _edge_reason_start = time.time()
                                edge_reason = _build_edge_reason(original_query, tgt, tools_used, final_content)
                                _edge_reason_total_ms += (time.time() - _edge_reason_start) * 1000
                                for src in from_uris:
                                    if src == tgt:
                                        continue
                                    pair = (min(src, tgt), max(src, tgt))
                                    if pair in seen_pairs:
                                        continue
                                    seen_pairs.add(pair)
                                    try:
                                        _single_link_start = time.time()
                                        await rv_client.link(src, [tgt], reason=edge_reason, query=original_query, strategy="llm_review", weight=1.0)
                                        _single_link_duration = (time.time() - _single_link_start) * 1000
                                        _link_ms_values.append(_single_link_duration)
                                        _link_count += 1
                                        linked += 1
                                        if _link_count <= 3:  # 只打印前3个避免刷屏
                                            logger.debug(f"[LLMReview][TIMING] single link #{_link_count} took {_single_link_duration:.0f}ms")
                                    except Exception as e:
                                        logger.warning(f"[LLMReview] Link failed: {e}")
                            _link_total_duration = (time.time() - _link_start) * 1000
                            _link_avg_ms = sum(_link_ms_values) / max(len(_link_ms_values), 1)
                            _link_max_ms = max(_link_ms_values) if _link_ms_values else 0.0
                            _link_min_ms = min(_link_ms_values) if _link_ms_values else 0.0
                            _post_link_duration = (time.time() - _post_link_start) * 1000
                            logger.info(
                                f"[LLMReview][PROFILE] link_pairs={linked}, from_uris={len(from_uris)}, "
                                f"to_uris={len(useful_uris)}, edge_reason_ms={_edge_reason_total_ms:.0f}, "
                                f"link_total_ms={_link_total_duration:.0f}, "
                                f"link_avg_ms={_link_avg_ms:.0f}, link_min_ms={_link_min_ms:.0f}, "
                                f"link_max_ms={_link_max_ms:.0f}, post_link_total_ms={_post_link_duration:.0f}"
                            )

                            iteration += 1
                            tools_used.append({
                                "tool_name": "openviking_link",
                                "args": {
                                    "from_uris": sorted(from_uris),
                                    "to_uris": useful_uris,
                                    "skipped_to_uris": skipped_relation_to_uris,
                                },
                                "reasoning": "(precise review step)",
                                "result": (
                                    f"Created {linked} relation(s) from {len(from_uris)} from_uris "
                                    f"to {len(useful_uris)} useful docs, "
                                    f"skipped_relation_to_uris={len(skipped_relation_to_uris)}, "
                                    f"parse={parse_method}"
                                ),
                                "duration": int(_post_link_duration),
                                "execute_success": True,
                                "input_token": 0,
                                "output_token": 0,
                                "iteration": iteration,
                                "relations_found": linked,
                                "skipped_relation_to_uris": skipped_relation_to_uris,
                                "profile": {
                                    "collect_ms": int(_collect_duration),
                                    "prompt_build_ms": int(_prompt_build_duration),
                                    "llm_review_ms": int(_llm_review_duration),
                                    "parse_ms": int(_parse_duration),
                                    "client_create_ms": int(_client_create_duration),
                                    "top5_search_ms": int(_top5_search_duration) if "_top5_search_duration" in locals() else None,
                                    "edge_reason_ms": int(_edge_reason_total_ms),
                                    "link_total_ms": int(_link_total_duration),
                                    "post_link_total_ms": int(_post_link_duration),
                                    "skipped_relation_to_uris": len(skipped_relation_to_uris),
                                },
                            })
                            logger.info(
                                f"[LLMReview] post_link DONE: created {linked} edge(s), "
                                f"from {len(from_uris)} from_uris to {len(useful_uris)} useful_docs, "
                                f"skipped_relation_to_uris={len(skipped_relation_to_uris)}, "
                                f"parse={parse_method}, total_ms={_post_link_duration:.0f}"
                            )
                        finally:
                            await rv_client.close()
                    elif skipped_relation_to_uris:
                        _post_link_duration = (time.time() - _post_link_start) * 1000
                        logger.info(
                            f"[LLMReview] post_link SKIPPED - all selected to_uris were "
                            f"relation-derived. skipped_to_uris={len(skipped_relation_to_uris)}, "
                            f"total_ms={_post_link_duration:.0f}"
                        )
                        iteration += 1
                        tools_used.append({
                            "tool_name": "openviking_link",
                            "args": {
                                "from_uris": sorted(search_uris),
                                "to_uris": [],
                                "skipped_to_uris": skipped_relation_to_uris,
                            },
                            "reasoning": "(skipped - relation-derived to_uris)",
                            "result": (
                                "Skipped link creation because all useful to_uris were "
                                "retrieved through existing relations"
                            ),
                            "duration": int(_post_link_duration),
                            "execute_success": True,
                            "input_token": 0,
                            "output_token": 0,
                            "iteration": iteration,
                            "relations_found": 0,
                            "parse_method": parse_method,
                            "skipped_reason": "relation_derived_to_uris_only",
                            "skipped_relation_to_uris": skipped_relation_to_uris,
                            "profile": {
                                "collect_ms": int(_collect_duration),
                                "prompt_build_ms": int(_prompt_build_duration),
                                "llm_review_ms": int(_llm_review_duration),
                                "parse_ms": int(_parse_duration),
                                "post_link_total_ms": int(_post_link_duration),
                                "skipped_relation_to_uris": len(skipped_relation_to_uris),
                            },
                        })
                    else:
                        logger.warning(
                            f"[LLMReview] post_link SKIPPED - no useful docs parsed. "
                            f"search_uris={len(search_uris)}, read_uris={len(read_uris)}, "
                            f"query={original_query[:100]}"
                        )
                        iteration += 1
                        tools_used.append({
                            "tool_name": "openviking_link",
                            "args": {"from_uris": sorted(search_uris), "to_uris": []},
                            "reasoning": "(precise review step - PARSE FAILED)",
                            "result": f"FAILED: parse_method={parse_method}",
                            "duration": 0,
                            "execute_success": False,
                            "input_token": 0,
                            "output_token": 0,
                            "iteration": iteration,
                            "relations_found": 0,
                            "parse_method": parse_method,
                        })
                except Exception:
                    logger.exception("[LLMReview] Review step failed, continuing")

        return final_content, tools_used, token_usage, iteration, messages

    @trace(
        name="process_message",
        extract_session_id=lambda msg: msg.session_key.safe_name(),
        extract_user_id=lambda msg: msg.sender_id,
    )
    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.

        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).

        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        start_time = time.time()
        long_running_notified = False

        # 监控处理时长，每50秒发送处理中提示事件
        async def check_long_running():
            nonlocal long_running_notified
            tick_count = 0
            # 最多发送7次提示
            max_ticks = 7

            while not long_running_notified and tick_count < max_ticks:
                await asyncio.sleep(60)
                if long_running_notified:
                    break
                if msg.metadata:
                    message_id = msg.metadata.get("message_id")
                    if message_id:
                        try:
                            # 发送处理中tick事件，对应channel会自行处理展示逻辑
                            await self.bus.publish_outbound(
                                OutboundMessage(
                                    session_key=msg.session_key,
                                    content="",
                                    metadata={
                                        "action": "processing_tick",
                                        "tick_count": tick_count,
                                        "message_id": message_id,
                                    },
                                )
                            )
                            tick_count += 1
                        except Exception as e:
                            logger.debug(f"Failed to send processing tick: {e}")

        monitor_task = asyncio.create_task(check_long_running())

        try:
            if msg.session_key.type == "system":
                return await self._process_system_message(msg)

            preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            logger.info(f"Processing message from {msg.session_key}:{msg.sender_id}: {preview}")

            session_key = msg.session_key
            # For CLI/direct sessions, skip heartbeat by default
            skip_heartbeat = session_key.type == "cli"
            session = self.sessions.get_or_create(session_key, skip_heartbeat=skip_heartbeat)

            ov_tools_enable = self._get_ov_tools_enable(session_key)
            # Get profile_user_list from channel config
            profile_user_list = []
            memory_user = ""
            channel_config = self._get_channel_config(session_key)

            if channel_config and ov_tools_enable:
                profile_user_list = getattr(channel_config, "profile_user_list", [])
                memory_user = getattr(channel_config, "memory_user", "")

            # Handle slash commands
            is_group_chat = msg.metadata.get("chat_type") == "group" if msg.metadata else False
            if is_group_chat:
                cmd = msg.content.replace(f"@{msg.sender_id}", "").strip().lower()
            else:
                cmd = msg.content.strip().lower()
            if cmd == "/new":
                # Clone session for async consolidation, then immediately clear original
                if not self._check_cmd_auth(msg):
                    return OutboundMessage(
                        session_key=msg.session_key, content="🐈 Sorry, you are not authorized to use this command.",
                        metadata=msg.metadata
                    )
                session.clear()
                await self.sessions.save(session)
                return OutboundMessage(
                    session_key=msg.session_key, content="🐈 New session started. Session history droped.", metadata=msg.metadata
                )
            elif cmd == "/compact":
                # Clone session for async consolidation, then immediately clear original
                if not self._check_cmd_auth(msg):
                    return OutboundMessage(
                        session_key=msg.session_key, content="🐈 Sorry, you are not authorized to use this command.",
                        metadata=msg.metadata
                    )
                session_clone = session.clone()
                session.clear()
                await self.sessions.save(session)
                # Run consolidation in background
                await self._safe_consolidate_memory(session_clone, archive_all=True)
                return OutboundMessage(
                    session_key=msg.session_key, content="🐈 New session started. Memory consolidated.", metadata=msg.metadata
                )
            if cmd == "/remember":
                if not self._check_cmd_auth(msg):
                    return OutboundMessage(
                        session_key=msg.session_key, content="🐈 Sorry, you are not authorized to use this command.",
                        metadata=msg.metadata
                    )
                if ov_tools_enable:
                    session_clone = session.clone()
                    await self._consolidate_viking_memory(session_clone)
                return OutboundMessage(
                    session_key=msg.session_key, content="This conversation has been submitted to memory storage.", metadata=msg.metadata
                )
            if cmd == "/help":
                return OutboundMessage(
                    session_key=msg.session_key,
                    content="🐈 vikingbot commands:\n/new — Start a new conversation\n/remember — Submit current session to memories and start new session\n/help — Show available commands",
                    metadata=msg.metadata
                )

            # Debug mode handling
            if self.config.mode == BotMode.DEBUG:
                # In debug mode, only record message to session, no processing or reply
                session.add_message("user", msg.content, sender_id=msg.sender_id)
                await self.sessions.save(session)
                return None

            if not msg.need_reply:
                session.add_message("user", msg.content, sender_id=msg.sender_id)
                await self.sessions.save(session)
                return OutboundMessage(
                    session_key=msg.session_key,
                    content="",
                    metadata=msg.metadata,
                    event_type=OutboundEventType.NO_REPLY,
                )

            # Consolidate memory before processing if session is too large
            if len(session.messages) > self.memory_window:
                # Clone session for async consolidation, then immediately trim original
                session_clone = session.clone()
                keep_count = min(10, max(2, self.memory_window // 2))
                session.messages = session.messages[-keep_count:] if keep_count else []
                await self.sessions.save(session)
                # Run consolidation in background
                await self._safe_consolidate_memory(session_clone, archive_all=False)

            if self.sandbox_manager:
                message_workspace = self.sandbox_manager.get_workspace_path(session_key)
            else:
                message_workspace = self.workspace

            from vikingbot.agent.context import ContextBuilder

            message_context = ContextBuilder(
                message_workspace,
                sandbox_manager=self.sandbox_manager,
                sender_id=msg.sender_id,
                sender_name=msg.sender_name,
                is_group_chat=is_group_chat,
                eval=self._eval,
            )

            # Build initial messages (use get_history for LLM-formatted messages)
            messages = await message_context.build_messages(
                history=session.get_history(),
                current_message=msg.content,
                media=msg.media if msg.media else None,
                session_key=msg.session_key,
                ov_tools_enable=ov_tools_enable,
                profile_user_list=profile_user_list,
                memory_user=memory_user,
            )
            # logger.info(f"New messages: {json.dumps(messages, indent=4)}")

            # Run agent loop
            final_content, tools_used, token_usage, iteration, messages = await self._run_agent_loop(
                messages=messages,
                session_key=session_key,
                publish_events=True,
                sender_id=msg.sender_id,
                ov_tools_enable=ov_tools_enable,
                link_query=msg.content,
            )

            # Log response preview
            preview = final_content[:300] + "..." if len(final_content) > 300 else final_content
            logger.info(f"Response to {msg.session_key}: {preview}")

            is_heartbeat = bool(msg.metadata.get(HEARTBEAT_METADATA_KEY))
            if not (is_heartbeat and is_heartbeat_noop_response(final_content)):
                session.add_message("user", msg.content, sender_id=msg.sender_id)
                session.add_message(
                    "assistant", final_content, tools_used=tools_used if tools_used else None, token_usage=token_usage,
                    sender_id=msg.sender_id,
                )
                await self.sessions.save(session)

            time_cost = round(time.time() - start_time, 2)
            if tools_used is not None:
                tools_used_names = [tool["tool_name"] for tool in tools_used]
            else:
                tools_used_names = []
            return OutboundMessage(
                session_key=msg.session_key,
                content=final_content,
                metadata=msg.metadata,
                token_usage=token_usage,
                time_cost=time_cost,
                iteration=iteration,
                tools_used_names=tools_used_names,
                tools_used=tools_used if tools_used else [],
                messages=messages if self._eval else None,
            )
        finally:
            long_running_notified = True
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    def _get_channel_config(self, session_key: SessionKey):
        """Get channel config for a session key.

        Args:
            session_key: Session key to get channel config for

        Returns:
            Channel config object if found, None otherwise
        """
        return self.config.channels_config.get_channel_by_key(session_key.channel_key())

    def _get_ov_tools_enable(self, session_key: SessionKey) -> bool:
        """Get ov_tools_enable setting from channel config.

        Args:
            session_key: Session key to get channel config for

        Returns:
            True if ov tools should be enabled, False otherwise
        """
        channel_config = self._get_channel_config(session_key)
        return getattr(channel_config, "ov_tools_enable", True) if channel_config else True

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).

        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")

        session = self.sessions.get_or_create(msg.session_key)

        # Get channel config
        ov_tools_enable = self._get_ov_tools_enable(msg.session_key)
        profile_user_list = []
        channel_config = self._get_channel_config(msg.session_key)
        if channel_config and ov_tools_enable:
            profile_user_list = getattr(channel_config, "profile_user_list", [])

        # Build messages with the announce content
        messages = await self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            session_key=msg.session_key,
            ov_tools_enable=ov_tools_enable,
            profile_user_list=profile_user_list,
        )

        # Run agent loop (no events published)
        final_content, tools_used, token_usage, iteration, messages = await self._run_agent_loop(
            messages=messages,
            session_key=msg.session_key,
            publish_events=False,
            ov_tools_enable=ov_tools_enable,
            link_query=None,
        )

        if final_content is None or (
            isinstance(final_content, str) and not final_content.strip()
        ):
            final_content = "Background task completed."

        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message(
            "assistant", final_content, tools_used=tools_used if tools_used else None
        )
        await self.sessions.save(session)

        return OutboundMessage(session_key=msg.session_key, content=final_content)

    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md. Works on a cloned session."""
        try:
            if not session.messages:
                return

            # use openviking tools to extract memory
            config = self.config
            if config.mode == BotMode.READONLY:
                if not config.channels_config or not config.channels_config.get_all_channels():
                    return
                allow_from = [config.ov_server.admin_user_id]
                for channel_config in config.channels_config.get_all_channels():
                    if channel_config and channel_config.type.value == session.key.type:
                        if hasattr(channel_config, "allow_from"):
                            allow_from.extend(channel_config.allow_from)
                messages = [msg for msg in session.messages if msg.get("sender_id") in allow_from]
                session.messages = messages
            await self._consolidate_viking_memory(session)

            if self.sandbox_manager:
                memory_workspace = self.sandbox_manager.get_workspace_path(session.key)
            else:
                memory_workspace = self.workspace

            memory = MemoryStore(memory_workspace)
            if archive_all:
                old_messages = session.messages
                keep_count = 0
            else:
                keep_count = min(10, max(2, self.memory_window // 2))
                old_messages = session.messages[:-keep_count]
            if not old_messages:
                return
            logger.info(
                f"Memory consolidation started: {len(session.messages)} messages, archiving {len(old_messages)}, keeping {keep_count}"
            )

            # Format messages for LLM (include tool names when available)
            lines = []
            for m in old_messages:
                if not m.get("content"):
                    continue
                tools_used = m.get("tools_used", [])
                if tools_used and isinstance(tools_used, list):
                    tool_names = [
                        tc.get("tool_name", "unknown") for tc in tools_used if isinstance(tc, dict)
                    ]
                    tools_str = f" [tools: {', '.join(tool_names)}]" if tool_names else ""
                else:
                    tools_str = ""
                lines.append(
                    f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools_str}: {m['content']}"
                )
            conversation = "\n".join(lines)
            current_memory = memory.read_long_term()

            prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

            response = await self.provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a memory consolidation agent. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                session_id=session.key.safe_name(),
            )
            text = (response.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(text)

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if load_config().use_local_memory and update != current_memory:
                    memory.write_long_term(update)

            # Session trimming and saving is handled by the caller before calling _consolidate_memory
            # This method works on a cloned session, so no need to save it
            logger.info("Memory consolidation done")
        except Exception as e:
            logger.exception(f"Memory consolidation failed: {e}")

    async def _consolidate_viking_memory(self, session) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md. Works on a cloned session."""
        try:
            if not session.messages:
                logger.info(f"No messages to commit openviking for session {session.key.safe_name()} (allow_from filter applied)")
                return

            # use openviking tools to extract memory
            await hook_manager.execute_hooks(
                context=HookContext(
                    event_type="message.compact",
                    session_id=session.key.safe_name(),
                    workspace_id=self.sandbox_manager.to_workspace_id(session.key),
                    session_key=session.key,
                ),
                session=session,
            )
        except Exception as e:
            logger.exception(f"Memory consolidation failed: {e}")

    async def _safe_consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Safe wrapper for _consolidate_memory that ensures all exceptions are caught."""
        try:
            await self._consolidate_memory(session, archive_all)
        except Exception as e:
            logger.exception(f"Background memory consolidation task failed: {e}")

    def _check_cmd_auth(self, msg: InboundMessage) -> bool:
        """Check if the session key is authorized for command execution.

        Returns:
            True if authorized, False otherwise.
        Args:
            session_key: Session key to check.
        """
        if self.config.mode == BotMode.NORMAL:
            return True
        allow_from = []
        if self.config.ov_server and self.config.ov_server.admin_user_id:
            allow_from.append(self.config.ov_server.admin_user_id)
        channel_config = self._get_channel_config(msg.session_key)
        if channel_config:
            allow_cmd = getattr(channel_config, 'allow_cmd_from', [])
            if allow_cmd:
                allow_from.extend(allow_cmd)

        # If channel not found or sender not in allow_from list, ignore message
        if msg.sender_id not in allow_from:
            logger.debug(f"Sender {msg.sender_id} not allowed in channel {msg.session_key.channel_key()}")
            return False
        return True

    async def process_direct(
        self,
        content: str,
        session_key: SessionKey = SessionKey(type="cli", channel_id="default", chat_id="direct"),
        metadata: dict[str, object] | None = None,
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).

        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).

        Returns:
            The agent's response.
        """
        await self._connect_mcp()
        msg = InboundMessage(
            session_key=session_key,
            sender_id="user",
            content=content,
            metadata=metadata or {},
        )

        response = await self._process_message(msg)
        return response.content if response else ""
