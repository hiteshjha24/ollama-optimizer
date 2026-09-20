"""Statistics and resource measurement.

Two rules are enforced here:

1. Every aggregate carries its sample size, and confidence intervals are only
   produced when n >= 3 (below that the interval is reported as unavailable).
2. Resource metrics are only reported when a real source produced them
   (``psutil`` for CPU/RAM, ``nvidia-smi`` for NVIDIA GPUs). Otherwise the
   value is ``None`` and the UI/report prints "N/A - metric unavailable on this
   system". Nothing is estimated or filled in.
"""

from __future__ import annotations

import math
import platform
import shutil
import statistics
import subprocess
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .logging_setup import get_logger

log = get_logger("metrics")

try:  # optional
    import psutil  # type: ignore
    HAVE_PSUTIL = True
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
    HAVE_PSUTIL = False

# Student t critical values (95%) for small samples, df = n-1.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def _t_value(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    for key in keys:
        if df < key:
            return _T95[key]
    return 1.96


def summarize(values: Sequence[float], unit: str = "") -> Dict[str, Any]:
    """Mean/median/min/max/stdev/CV/95% CI for a numeric sample."""
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    n = len(clean)
    if n == 0:
        return {"n": 0, "unit": unit, "mean": None, "median": None, "min": None,
                "max": None, "stdev": None, "cv": None, "ci95": None,
                "note": "no successful samples"}
    mean = statistics.fmean(clean)
    median = statistics.median(clean)
    stdev = statistics.stdev(clean) if n >= 2 else 0.0
    cv = (stdev / mean) if mean not in (0, None) else None
    ci = None
    if n >= 3 and stdev > 0:
        margin = _t_value(n - 1) * stdev / math.sqrt(n)
        ci = [round(mean - margin, 6), round(mean + margin, 6)]
    elif n >= 3:
        ci = [round(mean, 6), round(mean, 6)]
    return {
        "n": n,
        "unit": unit,
        "mean": round(mean, 6),
        "median": round(median, 6),
        "min": round(min(clean), 6),
        "max": round(max(clean), 6),
        "stdev": round(stdev, 6),
        "cv": round(cv, 6) if cv is not None else None,
        "ci95": ci,
        "note": None if n >= 3 else f"sample size {n} is too small for a confidence interval",
    }


def percent_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    """Percentage change from ``old`` to ``new``; ``None`` if not computable."""
    if new is None or old in (None, 0):
        return None
    try:
        return round((new - old) / abs(old) * 100.0, 2)
    except ZeroDivisionError:
        return None


def welch_t(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Welch's t statistic between two small samples.

    Returns a coarse significance flag only; with n=5 per group this is a weak
    signal and the report labels it as such.
    """
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    if len(a) < 2 or len(b) < 2:
        return {"t": None, "df": None, "significant_95": None,
                "note": "needs at least 2 samples per group"}
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    denom = math.sqrt(va / na + vb / nb)
    if denom == 0:
        return {"t": None, "df": None, "significant_95": None,
                "note": "zero variance in both samples"}
    t = (statistics.fmean(a) - statistics.fmean(b)) / denom
    num = (va / na + vb / nb) ** 2
    den = ((va / na) ** 2 / (na - 1)) + ((vb / nb) ** 2 / (nb - 1))
    df = num / den if den else min(na, nb) - 1
    crit = _t_value(int(round(df)))
    return {"t": round(t, 4), "df": round(df, 2),
            "significant_95": bool(abs(t) > crit),
            "note": "Welch's t-test on a small sample; treat as indicative only"}


# --------------------------------------------------------------------------
# resource sampling
# --------------------------------------------------------------------------

def gpu_snapshot() -> Optional[List[Dict[str, Any]]]:
    """Query nvidia-smi if it exists. Returns ``None`` when unavailable."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append({
                "name": parts[0],
                "utilization_percent": float(parts[1]),
                "memory_used_mb": float(parts[2]),
                "memory_total_mb": float(parts[3]),
            })
        except ValueError:
            continue
    return gpus or None


class ResourceSampler:
    """Samples system CPU/RAM (and GPU when available) during one generation.

    Reports the *system-wide* load, not a per-process attribution: Ollama runs
    as a separate server process and the report states this explicitly.
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cpu: List[float] = []
        self._ram: List[float] = []
        self._gpu_util: List[float] = []
        self._gpu_mem: List[float] = []
        self._gpu_name: Optional[str] = None
        self._gpu_checked = False

    def _loop(self) -> None:
        while not self._stop.is_set():
            if HAVE_PSUTIL:
                try:
                    self._cpu.append(psutil.cpu_percent(interval=None))
                    self._ram.append(psutil.virtual_memory().percent)
                except Exception:  # pragma: no cover
                    pass
            if not self._gpu_checked:
                self._gpu_checked = True
                self._gpu_available = gpu_snapshot() is not None
            if getattr(self, "_gpu_available", False):
                gpus = gpu_snapshot()
                if gpus:
                    self._gpu_name = gpus[0]["name"]
                    self._gpu_util.append(gpus[0]["utilization_percent"])
                    self._gpu_mem.append(gpus[0]["memory_used_mb"])
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceSampler":
        if HAVE_PSUTIL:
            try:
                psutil.cpu_percent(interval=None)  # prime the counter
            except Exception:  # pragma: no cover
                pass
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="resource-sampler")
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def result(self) -> Dict[str, Any]:
        def agg(values: List[float]) -> Optional[Dict[str, float]]:
            if not values:
                return None
            return {"mean": round(statistics.fmean(values), 2),
                    "max": round(max(values), 2), "samples": len(values)}

        data: Dict[str, Any] = {
            "source": "psutil (system-wide)" if HAVE_PSUTIL else None,
            "cpu_percent": agg(self._cpu),
            "ram_percent": agg(self._ram),
            "gpu": None,
            "notes": [],
        }
        if not HAVE_PSUTIL:
            data["notes"].append("CPU/RAM unavailable: psutil is not installed.")
        if self._gpu_util:
            data["gpu"] = {
                "name": self._gpu_name,
                "utilization_percent": agg(self._gpu_util),
                "memory_used_mb": agg(self._gpu_mem),
                "source": "nvidia-smi",
            }
        else:
            data["notes"].append(
                "GPU metrics unavailable: nvidia-smi not found or returned no data."
            )
        return data


def host_info() -> Dict[str, Any]:
    """Hardware/OS snapshot stored with every experiment for reproducibility."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "cpu_count_logical": None,
        "cpu_count_physical": None,
        "ram_total_gb": None,
        "gpu": None,
        "captured_at": time.time(),
    }
    if HAVE_PSUTIL:
        try:
            info["cpu_count_logical"] = psutil.cpu_count(logical=True)
            info["cpu_count_physical"] = psutil.cpu_count(logical=False)
            info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:  # pragma: no cover
            pass
    gpus = gpu_snapshot()
    if gpus:
        info["gpu"] = [{"name": g["name"],
                        "memory_total_mb": g["memory_total_mb"]} for g in gpus]
    return info


def merge_resource_samples(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Average per-run resource samples into a per-configuration figure."""
    cpu, ram, gpu_util, gpu_mem = [], [], [], []
    gpu_name = None
    source = None
    for sample in samples:
        if not sample:
            continue
        source = source or sample.get("source")
        if sample.get("cpu_percent"):
            cpu.append(sample["cpu_percent"]["mean"])
        if sample.get("ram_percent"):
            ram.append(sample["ram_percent"]["mean"])
        gpu = sample.get("gpu")
        if gpu:
            gpu_name = gpu.get("name")
            if gpu.get("utilization_percent"):
                gpu_util.append(gpu["utilization_percent"]["mean"])
            if gpu.get("memory_used_mb"):
                gpu_mem.append(gpu["memory_used_mb"]["mean"])
    out: Dict[str, Any] = {
        "source": source,
        "cpu_percent_mean": round(statistics.fmean(cpu), 2) if cpu else None,
        "ram_percent_mean": round(statistics.fmean(ram), 2) if ram else None,
        "gpu_name": gpu_name,
        "gpu_utilization_mean": round(statistics.fmean(gpu_util), 2) if gpu_util else None,
        "gpu_memory_used_mb_mean": round(statistics.fmean(gpu_mem), 2) if gpu_mem else None,
    }
    return out
