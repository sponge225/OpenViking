import json
import os
import platform
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

try:
    import psutil
except ImportError:
    psutil = None


def require_psutil():
    if psutil is None:
        raise SystemExit("memory_profile requires psutil. Install it first: uv pip install psutil")
    return psutil


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bytes_to_mb(value: Optional[int]) -> Optional[float]:
    return None if value is None else round(float(value) / 1024 / 1024, 3)


def resolve_host_pid(pid: int) -> int:
    """Map the current namespace PID to the host PID when running under bwrap."""
    pid = int(pid)
    if pid != os.getpid():
        return pid
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("NSpid:"):
                    values = [int(item) for item in line.split()[1:]]
                    if values:
                        return values[0]
    except OSError:
        pass
    return pid


def _mem(proc):
    try:
        info = proc.memory_info()
        pss = None
        try:
            pss = getattr(proc.memory_full_info(), "pss", None)
        except Exception:
            pass
        rss = int(getattr(info, "rss", 0) or 0)
        return {"pid": proc.pid, "name": proc.name(), "status": proc.status(), "rss_bytes": rss, "pss_bytes": int(pss) if pss is not None else None, "rss_mb": bytes_to_mb(rss), "pss_mb": bytes_to_mb(int(pss)) if pss is not None else None}
    except Exception:
        return None


def snapshot_process_tree(root_pid: int, phase: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    psutil_mod = require_psutil()
    root_pid = resolve_host_pid(root_pid)
    root = psutil_mod.Process(root_pid)
    root_info = _mem(root)
    children = [item for item in (_mem(child) for child in root.children(recursive=True)) if item]
    rss = int((root_info or {}).get("rss_bytes", 0) or 0)
    child_rss = sum(int(item.get("rss_bytes", 0) or 0) for item in children)
    pss_values = [(root_info or {}).get("pss_bytes")] + [item.get("pss_bytes") for item in children]
    total_pss = sum(int(value or 0) for value in pss_values) if any(value is not None for value in pss_values) else None
    sample = {"timestamp": utc_now(), "monotonic_sec": time.monotonic(), "phase": phase, "root_pid": root_pid, "root": root_info, "children": children, "children_rss_bytes": child_rss, "children_rss_mb": bytes_to_mb(child_rss), "total_rss_bytes": rss + child_rss, "total_rss_mb": bytes_to_mb(rss + child_rss), "total_pss_bytes": total_pss, "total_pss_mb": bytes_to_mb(total_pss)}
    if extra:
        sample.update(extra)
    return sample


class ProcessMemorySampler:
    def __init__(self, root_pid: int, output_path: str, interval_sec: float = 0.2, extra: Optional[Dict[str, Any]] = None):
        self.root_pid = int(root_pid)
        self.output_path = output_path
        self.interval_sec = max(float(interval_sec), 0.05)
        self.extra = dict(extra or {})
        self.samples: List[Dict[str, Any]] = []
        self._phase = "startup"
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._fh = None

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._fh = open(self.output_path, "a", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.interval_sec * 4))
        if self._fh:
            self._fh.close()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def snapshot(self, phase: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            current = phase or self._phase
        merged = dict(self.extra)
        if extra:
            merged.update(extra)
        sample = snapshot_process_tree(self.root_pid, current, merged)
        self.samples.append(sample)
        if self._fh:
            self._fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
            self._fh.flush()
        return sample

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.snapshot()
            except Exception as exc:
                self.samples.append({"timestamp": utc_now(), "monotonic_sec": time.monotonic(), "phase": "sampler_error", "error": str(exc)})
            self._stop.wait(self.interval_sec)


def samples_between(samples: Iterable[Dict[str, Any]], start_sec: float, end_sec: float) -> List[Dict[str, Any]]:
    return [s for s in samples if start_sec <= float(s.get("monotonic_sec", 0) or 0) <= end_sec]


def summarize_samples(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [s for s in samples if isinstance(s.get("total_rss_bytes"), int)]
    if not valid:
        return {}
    rss = [int(s["total_rss_bytes"]) for s in valid]
    pss = [s.get("total_pss_bytes") for s in valid if s.get("total_pss_bytes") is not None]
    peak = max(valid, key=lambda s: int(s.get("total_rss_bytes", 0) or 0))
    return {"sample_count": len(valid), "rss_min_mb": bytes_to_mb(min(rss)), "rss_avg_mb": bytes_to_mb(int(sum(rss) / len(rss))), "rss_peak_mb": bytes_to_mb(max(rss)), "pss_peak_mb": bytes_to_mb(max(pss)) if pss else None, "peak_timestamp": peak.get("timestamp"), "peak_phase": peak.get("phase")}


def compact_sample(sample: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {} if not sample else {"timestamp": sample.get("timestamp"), "rss_mb": sample.get("total_rss_mb"), "pss_mb": sample.get("total_pss_mb"), "children_rss_mb": sample.get("children_rss_mb"), "root_pid": sample.get("root_pid")}


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def environment_info() -> Dict[str, Any]:
    return {"python": platform.python_version(), "platform": platform.platform(), "pid": os.getpid(), "time_utc": utc_now()}
