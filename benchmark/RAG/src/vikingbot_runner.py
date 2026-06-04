#!/usr/bin/env python3

import os
import re
import sys
import time
import json
import uuid
import random
import string
import hashlib
import subprocess
import atexit
import urllib.request
import urllib.error
import threading
import socket
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


sys.path.append(str(Path(__file__).parent))

from core.logger import get_logger

logger = get_logger()

_OV_CONF_PATH = str((Path(__file__).parent.parent / "ov.conf").resolve())
_OPENVIKING_SERVER_PROCESS: Optional[subprocess.Popen] = None
_CURRENT_OV_CONF_PATH: Optional[str] = None
_OPENVIKING_SERVER_LOG_FH: Optional[Any] = None
_SERVER_LOCK = threading.Lock()
_ACTIVE_BOT_PROCESSES: set = set()
_BOT_PROC_LOCK = threading.Lock()


def _generate_temp_ov_conf(original_conf_path: str, vector_store_path: str, search_limit=None, llm_config: dict | None = None, server_port: int | None = None) -> str:
    with open(original_conf_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if 'server' not in config or config.get('server') is None:
        config['server'] = {}
    if isinstance(config.get('server'), dict):
        if not config['server'].get('root_api_key'):
            config['server'].pop('root_api_key', None)

    if server_port is not None:
        if 'server' not in config or config.get('server') is None:
            config['server'] = {}
        config['server']['port'] = server_port

    if 'storage' not in config:
        config['storage'] = {}
    config['storage']['workspace'] = vector_store_path

    if search_limit is not None:
        config['default_search_limit'] = search_limit

    if llm_config:
        if 'vlm' not in config or config.get('vlm') is None:
            config['vlm'] = {}
        if 'model' in llm_config:
            config['vlm']['model'] = llm_config['model']
        if 'base_url' in llm_config:
            config['vlm']['api_base'] = llm_config['base_url']
        if 'temperature' in llm_config:
            config['vlm']['temperature'] = llm_config['temperature']

    temp_dir = Path(__file__).parent.parent / ".temp"
    temp_dir.mkdir(exist_ok=True)

    hash_input = vector_store_path.encode('utf-8')
    if search_limit is not None:
        hash_input += f"_sl{search_limit}".encode('utf-8')
    if llm_config:
        hash_input += json.dumps(llm_config, sort_keys=True).encode('utf-8')
    if server_port is not None:
        hash_input += f"_port{server_port}".encode('utf-8')
    path_hash = hashlib.md5(hash_input).hexdigest()
    temp_conf_path = str(temp_dir / f"ov_{path_hash}.conf")

    if os.path.exists(temp_conf_path):
        return temp_conf_path

    with open(temp_conf_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    return temp_conf_path


def _healthcheck(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _load_server_url_and_key(ov_conf_path: str) -> tuple[str, str]:
    with open(ov_conf_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    server = data.get("server", {}) if isinstance(data, dict) else {}
    host = server.get("host", "127.0.0.1")
    port = server.get("port", 1933)
    api_key = server.get("root_api_key", "") or ""
    return f"http://{host}:{port}", api_key


def _stop_openviking_server() -> None:
    global _OPENVIKING_SERVER_PROCESS, _CURRENT_OV_CONF_PATH, _OPENVIKING_SERVER_LOG_FH
    proc = _OPENVIKING_SERVER_PROCESS
    started_by_us = bool(proc) or bool(_CURRENT_OV_CONF_PATH)
    _OPENVIKING_SERVER_PROCESS = None
    _CURRENT_OV_CONF_PATH = None
    log_fh = _OPENVIKING_SERVER_LOG_FH
    _OPENVIKING_SERVER_LOG_FH = None
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    if log_fh:
        try:
            log_fh.close()
        except Exception:
            pass

    kill_all = os.environ.get("OPENVIKING_BENCH_KILL_ALL_SERVERS", "").strip() in ("1", "true", "True")
    if not (kill_all and started_by_us):
        return

    try:
        if sys.platform == "darwin" or sys.platform.startswith("linux"):
            result = subprocess.run(
                ["pgrep", "-f", "openviking-server"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"Found openviking-server processes: {result.stdout.strip()}")
                subprocess.run(["pkill", "-f", "openviking-server"], capture_output=True)
                time.sleep(1)
    except Exception as e:
        logger.debug(f"Failed to kill all openviking-server processes: {e}")


atexit.register(_stop_openviking_server)

import signal


def _kill_all_bot_processes():
    with _BOT_PROC_LOCK:
        procs = list(_ACTIVE_BOT_PROCESSES)
    for proc in procs:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass


def _sigint_handler(signum, frame):
    _stop_openviking_server()
    _kill_all_bot_processes()
    sys.exit(1)

signal.signal(signal.SIGINT, _sigint_handler)


def _wait_for_port_release(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return
            except OSError:
                time.sleep(0.3)
    logger.warning(f"Port {port} still in use after {timeout}s, proceeding anyway")


def _healthcheck_with_retry(url: str, retries: int = 3, interval: float = 2.0, timeout: float = 3.0) -> bool:
    for attempt in range(retries):
        if _healthcheck(url, timeout=timeout):
            return True
        if attempt < retries - 1:
            logger.debug(f"Healthcheck failed (attempt {attempt + 1}/{retries}), retrying in {interval}s...")
            time.sleep(interval)
    return False


def _kill_locking_pid(log_text: str) -> bool:
    """Parse DataDirectoryLocked PID from log and kill it. Returns True if killed."""
    match = re.search(r"Another OpenViking process \(PID (\d+)\)", log_text)
    if not match:
        return False
    pid = int(match.group(1))
    logger.warning(f"Killing stale OpenViking process PID {pid} that holds the data directory lock")
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            import signal
            os.kill(pid, signal.SIGKILL)
        time.sleep(1)
        return True
    except Exception as e:
        logger.warning(f"Failed to kill PID {pid}: {e}")
        return False


def _ensure_openviking_server(ov_conf_path: str) -> None:
    global _OPENVIKING_SERVER_PROCESS, _CURRENT_OV_CONF_PATH, _OPENVIKING_SERVER_LOG_FH

    with _SERVER_LOCK:
        server_url, api_key = _load_server_url_and_key(ov_conf_path)
        health_url = f"{server_url}/health"

        if (_CURRENT_OV_CONF_PATH == ov_conf_path and
            _OPENVIKING_SERVER_PROCESS and
            _OPENVIKING_SERVER_PROCESS.poll() is None and
            _healthcheck_with_retry(health_url)):
            return

        # Check if another process (e.g. VikingStoreWrapper) already has a healthy
        # server on this port. If so, reuse it without starting our own.
        if _healthcheck(health_url):
            logger.info(f"Reusing existing healthy server at {server_url}")
            _CURRENT_OV_CONF_PATH = ov_conf_path
            _OPENVIKING_SERVER_PROCESS = None
            return

        if _OPENVIKING_SERVER_PROCESS and _OPENVIKING_SERVER_PROCESS.poll() is None:
            logger.warning("OV server healthcheck failed after retries, restarting server")

        host = server_url.split("//")[1].split(":")[0]
        port = int(server_url.split(":")[-1].split("/")[0])
        _stop_openviking_server()
        _CURRENT_OV_CONF_PATH = None

        _wait_for_port_release(host, port)

        env = os.environ.copy()
        env["OPENVIKING_CONFIG_FILE"] = ov_conf_path
        
        temp_dir = Path(__file__).parent.parent / ".temp"
        temp_dir.mkdir(exist_ok=True)
        server_log_path = str(temp_dir / "openviking-server.log")
        try:
            _OPENVIKING_SERVER_LOG_FH = open(server_log_path, "a", encoding="utf-8")
        except Exception:
            _OPENVIKING_SERVER_LOG_FH = None

        _OPENVIKING_SERVER_PROCESS = subprocess.Popen(
            ["openviking-server", "--config", ov_conf_path],
            stdout=_OPENVIKING_SERVER_LOG_FH or subprocess.DEVNULL,
            stderr=_OPENVIKING_SERVER_LOG_FH or subprocess.DEVNULL,
            env=env,
        )
        _CURRENT_OV_CONF_PATH = ov_conf_path

        retried_lock = False
        deadline = time.time() + 60
        while time.time() < deadline:
            if _OPENVIKING_SERVER_PROCESS.poll() is not None:
                exit_code = _OPENVIKING_SERVER_PROCESS.returncode
                log_tail = ""
                try:
                    if _OPENVIKING_SERVER_LOG_FH:
                        _OPENVIKING_SERVER_LOG_FH.flush()
                    with open(server_log_path, "r", encoding="utf-8", errors="replace") as lf:
                        lines = lf.readlines()
                        log_tail = "".join(lines[-50:])
                except Exception:
                    pass
                logger.error(f"openviking-server exited with code {exit_code}. Config: {ov_conf_path}")
                if log_tail:
                    logger.error(f"Server log (last 50 lines):\n{log_tail}")

                if not retried_lock and "DataDirectoryLocked" in log_tail and _kill_locking_pid(log_tail):
                    retried_lock = True
                    logger.info("Retrying server start after killing stale process...")
                    _wait_for_port_release(host, port)
                    try:
                        _OPENVIKING_SERVER_LOG_FH = open(server_log_path, "a", encoding="utf-8")
                    except Exception:
                        _OPENVIKING_SERVER_LOG_FH = None
                    _OPENVIKING_SERVER_PROCESS = subprocess.Popen(
                        ["openviking-server", "--config", ov_conf_path],
                        stdout=_OPENVIKING_SERVER_LOG_FH or subprocess.DEVNULL,
                        stderr=_OPENVIKING_SERVER_LOG_FH or subprocess.DEVNULL,
                        env=env,
                    )
                    deadline = time.time() + 20
                    continue

                raise RuntimeError(f"openviking-server exited unexpectedly (code={exit_code})")
            if _healthcheck(health_url):
                return
            time.sleep(0.3)

        raise RuntimeError("openviking-server did not become healthy in time")


def _build_vikingbot_env(
    ov_conf_path: str,
    max_iterations: int,
    enable_linking: bool = False,
    use_relations: bool = False,
    embedding_config: dict = None,
    link_strategy: str = "llm_review",
    enable_reasoning: bool = True,
    relation_keyword_threshold: float = 0.7,
    relation_vector_threshold: float = 0.7,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["OPENVIKING_CONFIG_FILE"] = ov_conf_path
    env["NANOBOT_AGENTS__MAX_TOOL_ITERATIONS"] = str(int(max_iterations))
    env["VIKINGBOT_USE_RELATIONS"] = "1" if use_relations else "0"
    env["VIKINGBOT_ENABLE_LINKING"] = "1" if enable_linking else "0"
    env["VIKINGBOT_ENABLE_REASONING"] = "1" if enable_reasoning else "0"
    env["VIKINGBOT_LINK_STRATEGY"] = link_strategy
    env["VIKINGBOT_RELATION_KEYWORD_THRESHOLD"] = str(float(relation_keyword_threshold))
    env["VIKINGBOT_RELATION_VECTOR_THRESHOLD"] = str(float(relation_vector_threshold))
    if embedding_config:
        raw_key = embedding_config.get("api_key", "")
        resolved_key = os.path.expandvars(raw_key) if raw_key else ""
        env["VIKINGBOT_EMBEDDING_API_KEY"] = resolved_key
        env["VIKINGBOT_EMBEDDING_MODEL"] = embedding_config.get("model", "doubao-embedding-vision-250615")
        env["VIKINGBOT_EMBEDDING_BASE_URL"] = embedding_config.get("base_url", "https://ark.cn-beijing.volces.com/api/v3")

    original_ov_conf_dir = os.path.dirname(_OV_CONF_PATH)
    ovcli_conf_path = os.path.join(original_ov_conf_dir, "ovcli.conf")
    if os.path.exists(ovcli_conf_path):
        env["OPENVIKING_CLI_CONFIG_FILE"] = ovcli_conf_path
        logger.debug(f"Set OPENVIKING_CLI_CONFIG_FILE to: {ovcli_conf_path}")

    try:
        with open(ov_conf_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if 'vlm' in config and 'api_key' in config['vlm']:
            api_key = config['vlm']['api_key']
            env['OPENAI_API_KEY'] = api_key
            logger.debug("Set OPENAI_API_KEY from ov.conf")
    except Exception as e:
        logger.warning(f"Failed to read API key from ov.conf: {e}")

    return env


class VikingBotRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vikingbot_config = config.get('vikingbot', {})
        self.max_iterations = self.vikingbot_config.get('max_iterations', 50)
        self.log_tool_calls = self.vikingbot_config.get('log_tool_calls', True)
        self.search_limit = self.vikingbot_config.get('search_limit')
        self.use_relations = self.vikingbot_config.get('use_relations', False)
        self.enable_linking = self.vikingbot_config.get('enable_linking', False)
        self.link_strategy = self.vikingbot_config.get('link_strategy', 'llm_review')
        self.enable_reasoning = self.vikingbot_config.get('enable_reasoning', True)
        self.relation_keyword_threshold = self.vikingbot_config.get('relation_keyword_threshold', 0.7)
        self.relation_vector_threshold = self.vikingbot_config.get('relation_vector_threshold', 0.7)
        self.vector_store_path = config.get('paths', {}).get('vector_store')
        self.llm_config = config.get('llm', None)
        self.server_port = config.get('execution', {}).get('server_port', None)

    def generate_answer(
        self,
        question: str,
        session_id: Optional[str] = None,
        allowed_target_uris: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        session_id = session_id or f"eval_{uuid.uuid4().hex}"

        start_time = time.time()

        try:
            ov_conf_path = _OV_CONF_PATH
            temp_conf_path = None
            if self.vector_store_path:
                temp_conf_path = _generate_temp_ov_conf(_OV_CONF_PATH, self.vector_store_path, search_limit=self.search_limit, llm_config=self.llm_config, server_port=self.server_port)
                ov_conf_path = temp_conf_path
                logger.info(f"Using vector store: {self.vector_store_path}")

            _ensure_openviking_server(ov_conf_path)

            if allowed_target_uris:
                allowed_block = "\n".join(f"- {u}" for u in allowed_target_uris)
                scope_line = (
                    "Always search ONLY within the following directories:\n"
                    f"{allowed_block}\n"
                    "Do not access any other URI.\n"
                    "\n"
                    "Efficiency tip:\n"
                    "- Prefer openviking_search in the allowed directory. This is usually more efficient than layer-by-layer grep.\n"
                )
            else:
                scope_line = (
                    "Always search in viking://resources/ path.\n"
                )
            input_msg = (
                "Answer this question as briefly as possible. Use only the information available in the database. "
                "Do not use any external source. "
                "Always use OpenViking tools first. Search first, then read the results to answer. "
                + scope_line
                + f"\n\nQuestion: {question}"
            )
            env = _build_vikingbot_env(
                ov_conf_path, self.max_iterations, self.enable_linking, self.use_relations,
                embedding_config=self.config.get('embedding'),
                link_strategy=self.link_strategy,
                enable_reasoning=self.enable_reasoning,
                relation_keyword_threshold=self.relation_keyword_threshold,
                relation_vector_threshold=self.relation_vector_threshold,
            )

            # Write bot JSON output to a temp file (avoids stdout escape issues)
            safe_session = session_id.replace("/", "_").replace("\\", "_")
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "bot_json")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{safe_session}.json")

            cmd = [
                "vikingbot",
                "chat",
                "-m",
                input_msg,
                "-s",
                session_id,
                "-e",
                "--no-markdown",
                "-c",
                ov_conf_path,
                "-o",
                output_file,
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            with _BOT_PROC_LOCK:
                _ACTIVE_BOT_PROCESSES.add(proc)
            pid = proc.pid
            try:
                stdout, stderr = proc.communicate(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                raise subprocess.TimeoutExpired(cmd, 600)
            finally:
                with _BOT_PROC_LOCK:
                    _ACTIVE_BOT_PROCESSES.discard(proc)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd, stdout, stderr)

            # Read structured JSON from output file (guaranteed valid)
            resp_json = None
            try:
                if os.path.exists(output_file):
                    with open(output_file, "r", encoding="utf-8") as f:
                        resp_json = json.load(f)
                    logger.debug(f"Successfully read bot output from: {output_file}")
                else:
                    logger.warning(f"Output file not found: {output_file}, falling back to stdout parsing")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read output file: {e}, falling back to stdout parsing")

            # Fallback to stdout parsing if file-based approach failed
            stdout = (stdout or "").strip()
            trace = ""
            if resp_json is None:
                json_start = stdout.rfind('{"text"')
                if json_start == -1:
                    raise ValueError(f"No JSON output found in vikingbot stdout (len={len(stdout)})")
                trace = stdout[:json_start].strip() if json_start > 0 else ""
                raw_json = stdout[json_start:]
                raw_json = re.sub(r'[\x00-\x1f\x7f]', ' ', raw_json)
                raw_json = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw_json)
                try:
                    resp_json, _ = json.JSONDecoder().raw_decode(raw_json)
                except json.JSONDecodeError:
                    resp_json, _ = json.JSONDecoder(strict=False).raw_decode(raw_json)

            result_dict = {
                "answer": resp_json.get("text", "") or "",
                "total_time_sec": float(resp_json.get("time_cost", time.time() - start_time)),
                "token_usage": resp_json.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "tools_used_names": resp_json.get("tools_used_names") or [],
                "tools_used": resp_json.get("tools_used") or [],
                "iterations_used": int(resp_json.get("total_iterations", 0) or resp_json.get("iteration", 0) or 0),
                "debug_log": f"vikingbot.debug.{pid}.log",
                "session_id": session_id,
                "trace": trace,
                "stderr_output": (stderr or "").strip()[:10000],
            }

            # Print bot stderr so loguru output from the subprocess is visible
            if stderr and stderr.strip():
                for line in stderr.strip().splitlines():
                    logger.info(f"[bot-stderr] {line}")

            logger.info(f"VikingBot answer generated in {result_dict['total_time_sec']:.2f}s")
            return result_dict

        except Exception as e:
            logger.error(f"Error generating answer with VikingBot: {e}")
            return {
                "answer": f"[ERROR] {str(e)}",
                "total_time_sec": time.time() - start_time,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }


def stop_openviking_server() -> None:
    _stop_openviking_server()


def run_vikingbot_query(
    question: str,
    config: Dict[str, Any],
    session_id: Optional[str] = None,
    allowed_target_uris: Optional[List[str]] = None,
) -> Dict[str, Any]:
    runner = VikingBotRunner(config)
    return runner.generate_answer(question, session_id, allowed_target_uris=allowed_target_uris)
