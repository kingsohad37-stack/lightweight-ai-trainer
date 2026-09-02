"""
Real RAM monitoring and memory-budget estimation for low-memory (~6GB) machines.
No fake numbers: everything here reads actual process/system memory via /proc
(Linux) with a psutil fallback if available, and a resource-module fallback
if neither works.
"""
from __future__ import annotations
import os
import sys
import warnings

TARGET_RAM_GB = 6.0
SAFETY_MARGIN = 0.7  # only plan to use 70% of target RAM


def _read_proc_meminfo():
    """Parse /proc/meminfo for total/available memory in bytes. Linux only."""
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                val = parts[1].strip().split()[0]  # in kB
                info[key] = int(val) * 1024
    except FileNotFoundError:
        return None
    return info


def get_system_memory():
    """
    Returns dict: total_bytes, available_bytes, used_bytes.
    Real measurement, not a guess. Falls back gracefully across platforms.
    """
    info = _read_proc_meminfo()
    if info:
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", info.get("MemFree", 0))
        used = max(total - available, 0)
        return {"total_bytes": total, "available_bytes": available, "used_bytes": used, "source": "/proc/meminfo"}

    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return {"total_bytes": vm.total, "available_bytes": vm.available, "used_bytes": vm.used, "source": "psutil"}
    except ImportError:
        pass

    # Last resort: report current process RSS only, mark total as unknown.
    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss_kb * 1024 if sys.platform != "darwin" else rss_kb
        return {"total_bytes": 0, "available_bytes": 0, "used_bytes": rss_bytes, "source": "resource(process-only)"}
    except Exception:
        warnings.warn("Could not determine system memory on this platform.")
        return {"total_bytes": 0, "available_bytes": 0, "used_bytes": 0, "source": "unknown"}


def get_process_memory_bytes():
    """Real resident memory of THIS python process right now."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss_kb * 1024 if sys.platform != "darwin" else rss_kb
    except Exception:
        return 0


def estimate_dense_nn_bytes(layer_sizes, dtype_bytes=4, optimizer="adam"):
    """
    Estimate memory (bytes) required to hold parameters + optimizer state
    for a dense NN with the given layer sizes, e.g. [784, 128, 64, 10].
    Adam keeps 2 extra moment buffers per parameter; SGD/momentum keep 0-1.
    """
    n_params = 0
    for i in range(len(layer_sizes) - 1):
        n_params += layer_sizes[i] * layer_sizes[i + 1]  # weights
        n_params += layer_sizes[i + 1]  # biases

    multiplier = {"sgd": 1, "momentum": 2, "adam": 3}.get(optimizer, 3)
    return n_params * dtype_bytes * multiplier


def check_budget(estimated_bytes: int):
    """
    Compares an estimated requirement against the real available memory
    (bounded by TARGET_RAM_GB for portability across dev/deploy machines).
    Returns (ok: bool, message: str, safe_fraction: float)
    """
    mem = get_system_memory()
    available = mem["available_bytes"] or int(TARGET_RAM_GB * SAFETY_MARGIN * (1024 ** 3))
    budget = min(available, TARGET_RAM_GB * (1024 ** 3)) * SAFETY_MARGIN

    if estimated_bytes <= budget:
        return True, (
            f"OK: estimated {estimated_bytes/1e6:.2f} MB fits within "
            f"the {budget/1e6:.2f} MB safe budget."
        ), 1.0

    ratio = budget / estimated_bytes
    return False, (
        f"This configuration is likely to exceed available memory "
        f"(estimated {estimated_bytes/1e6:.2f} MB vs {budget/1e6:.2f} MB safe budget). "
        f"Recommend scaling parameters down by roughly {ratio:.2f}x."
    ), ratio
