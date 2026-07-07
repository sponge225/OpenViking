#!/usr/bin/env python3
import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
RAG_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(RAG_DIR))

from memory_utils import (  # noqa: E402
    ProcessMemorySampler,
    compact_sample,
    environment_info,
    samples_between,
    summarize_samples,
    write_json,
)
from src.core.llm_client import LLMClientWrapper  # noqa: E402
from src.core.logger import setup_logging  # noqa: E402
from src.core.vector_store import VikingStoreHTTPWrapper  # noqa: E402
from src.pipeline import BenchmarkPipeline  # noqa: E402
import src.vikingbot_runner as vikingbot_runner  # noqa: E402

try:
    import psutil
except ImportError:
    psutil = None


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> str:
    if not path_str or os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(str(RAG_DIR), path_str))


def resolve_config(path: str):
    config = load_config(path)
    dataset_name = config.get("dataset_name", "UnknownDataset")
    fmt = {
        "dataset_name": dataset_name,
        "retrieval_topk": config.get("execution", {}).get("retrieval_topk", 5),
        "search_limit": config.get("vikingbot", {}).get("search_limit", ""),
        "max_iterations": config.get("vikingbot", {}).get("max_iterations", ""),
    }
    for key in ["dataset_path", "output_dir", "vector_store", "log_file", "doc_output_dir"]:
        if key in config.get("paths", {}):
            config["paths"][key] = resolve_path(str(config["paths"][key]).format(**fmt))
    config.setdefault("execution", {})["mode"] = "ov_fallback_bot"
    return config


def build_adapter(config):
    adapter_cfg = config.get("adapter", {})
    mod = importlib.import_module(adapter_cfg.get("module", "src.adapters.locomo_adapter"))
    cls = getattr(mod, adapter_cfg.get("class_name", "LocomoAdapter"))
    return cls(raw_file_path=config["paths"]["dataset_path"])


def find_server_pid(config_path: str, server_url: str) -> int:
    if psutil is None:
        raise SystemExit("memory_profile requires psutil. Install it first: uv pip install psutil")

    candidates = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(p.info.get("cmdline") or [])
            name = p.info.get("name") or ""
            if "openviking-server" not in cmdline and "openviking-server" not in name:
                continue
            score = 0
            if config_path and config_path in cmdline:
                score += 10
            if server_url:
                score += 1
            candidates.append((score, p.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not candidates:
        proc = getattr(vikingbot_runner, "_OPENVIKING_SERVER_PROCESS", None)
        if proc is not None and proc.poll() is None:
            raise RuntimeError(
                "Cannot locate openviking-server host process for memory sampling; "
                f"subprocess PID {proc.pid} is likely a namespace PID"
            )
        raise RuntimeError("Cannot locate openviking-server process for memory sampling")
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile OV/VikingRAG server query-ready memory and first QA memory fluctuation")
    parser.add_argument("--config", default=str(RAG_DIR / "config" / "versionrag_config.yaml"))
    parser.add_argument("--query-count", type=int, default=3)
    parser.add_argument("--interval-sec", type=float, default=0.2)
    parser.add_argument("--idle-sec", type=float, default=3.0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    load_dotenv(RAG_DIR / ".env")
    config_path = os.path.abspath(args.config)
    config = resolve_config(config_path)
    output_dir = args.output_dir or os.path.join(config["paths"]["output_dir"], "memory_profile", "ov")
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(config["paths"]["log_file"])
    venv_bin = os.path.dirname(sys.executable)
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

    ov_config_path = RAG_DIR / "ov.conf"
    if not ov_config_path.exists():
        raise SystemExit(f"OpenViking config not found: {ov_config_path}")
    temp_conf = vikingbot_runner._generate_temp_ov_conf(
        str(ov_config_path),
        config["paths"].get("vector_store", ""),
        search_limit=config.get("vikingbot", {}).get("search_limit"),
        llm_config=config.get("llm"),
        server_port=config.get("execution", {}).get("server_port"),
    )
    os.environ["OPENVIKING_CONFIG_FILE"] = temp_conf
    vikingbot_runner._ensure_openviking_server(temp_conf)
    server_url, api_key = vikingbot_runner._load_server_url_and_key(temp_conf)
    server_pid = find_server_pid(temp_conf, server_url)

    adapter = build_adapter(config)
    api_key_llm = os.environ.get(config["llm"].get("api_key_env_var", ""), config["llm"].get("api_key", ""))
    api_key_llm = os.path.expandvars(api_key_llm) if api_key_llm else api_key_llm
    llm = LLMClientWrapper(config=config["llm"], api_key=api_key_llm)
    store = VikingStoreHTTPWrapper(server_url=server_url, api_key=api_key)
    pipeline = BenchmarkPipeline(config=config, adapter=adapter, vector_db=store, llm=llm, resume=False)

    sampler = ProcessMemorySampler(
        server_pid,
        os.path.join(output_dir, "memory_samples.jsonl"),
        args.interval_sec,
        {
            "method": "ov",
            "dataset_name": config.get("dataset_name"),
            "label": args.label,
            "driver_pid": os.getpid(),
            "server_url": server_url,
        },
    )
    per_query = []
    qa_start = qa_end = None
    ready_sample = None
    sampler.start()
    try:
        sampler.set_phase("query_ready_idle")
        idle_start = time.monotonic()
        time.sleep(max(args.idle_sec, 0.0))
        idle_end = time.monotonic()
        idle_samples = samples_between(sampler.samples, idle_start, idle_end)
        ready_sample = idle_samples[-1] if idle_samples else sampler.snapshot("query_ready_idle")

        tasks = pipeline._prepare_tasks(adapter.load_and_transform())[: args.query_count]
        qa_start = time.monotonic()
        for idx, task in enumerate(tasks):
            phase = f"qa_{idx}"
            sampler.set_phase(phase)
            before = sampler.snapshot(phase, {"query_index": idx, "event": "before"})
            start = time.monotonic()
            error = None
            result = None
            try:
                if hasattr(pipeline, "_process_ov_fallback_bot_task"):
                    result = pipeline._process_ov_fallback_bot_task(task)
                else:
                    result = pipeline._process_generation_task(task)
            except Exception as exc:
                error = str(exc)
            end = time.monotonic()
            after = sampler.snapshot(phase, {"query_index": idx, "event": "after"})
            window = summarize_samples(samples_between(sampler.samples, start, end))
            per_query.append({
                "index": idx,
                "global_index": task.get("id"),
                "sample_id": task.get("sample_id"),
                "question": getattr(task.get("qa"), "question", ""),
                "success": error is None,
                "error": error,
                "latency_sec": round(end - start, 3),
                "rss_before_mb": before.get("total_rss_mb"),
                "rss_after_mb": after.get("total_rss_mb"),
                "peak_rss_mb": window.get("rss_peak_mb"),
                "result_retrieval": (result or {}).get("retrieval", {}),
            })
        qa_end = time.monotonic()
    finally:
        sampler.snapshot("teardown")
        sampler.stop()

    qa_summary = summarize_samples(samples_between(sampler.samples, qa_start or 0.0, qa_end or time.monotonic()))
    ready_rss = ready_sample.get("total_rss_mb") if ready_sample else None
    peak_rss = qa_summary.get("rss_peak_mb")
    write_json(os.path.join(output_dir, "memory_summary.json"), {
        "method": "ov",
        "dataset_name": config.get("dataset_name"),
        "config_path": config_path,
        "temp_ov_config": temp_conf,
        "server_url": server_url,
        "server_pid": server_pid,
        "driver_pid": os.getpid(),
        "output_dir": output_dir,
        "query_ready": compact_sample(ready_sample),
        "ready_hooks": ["openviking-server /health"],
        "qa_window": {
            "query_count": len(per_query),
            **qa_summary,
            "rss_delta_from_ready_mb": round(peak_rss - ready_rss, 3) if peak_rss is not None and ready_rss is not None else None,
        },
        "per_query": per_query,
        "samples_path": os.path.join(output_dir, "memory_samples.jsonl"),
        "environment": environment_info(),
        "notes": ["Main memory metric samples the openviking-server process tree; driver process memory is not included."],
    })
    print(f"[memory-profile] wrote {os.path.join(output_dir, 'memory_summary.json')}")


if __name__ == "__main__":
    main()
