"""Lightweight memory tracker for training loops."""

from __future__ import annotations

import gc
import logging
import resource
import time

logger = logging.getLogger(__name__)


def _rss_mb() -> float:
    """Current RSS in MB (not peak — actual current resident memory)."""
    import os
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    # Fallback: read /proc on Linux
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    # Last resort: maxrss (high-water mark, not current)
    import platform
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return ru / (1024 * 1024)
    return ru / 1024


def _device_mb() -> float:
    """PyTorch device memory in MB (MPS or CUDA). Returns 0 if unavailable."""
    try:
        import torch
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            if hasattr(torch.mps, "current_allocated_memory"):
                return torch.mps.current_allocated_memory() / 1e6
        elif torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e6
    except Exception:
        pass
    return 0.0


class MemoryTracker:
    """Track memory at named checkpoints within a batch."""

    def __init__(self):
        self._checkpoints: list[tuple[str, float, float, float]] = []

    def checkpoint(self, name: str):
        self._checkpoints.append((name, _rss_mb(), _device_mb(), time.time()))

    def cleanup(self):
        gc.collect()
        try:
            import torch
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def reset(self):
        self._checkpoints.clear()

    def to_dict(self) -> dict:
        result = {}
        for name, rss, dev, ts in self._checkpoints:
            result[name] = {"rss_mb": round(rss, 1), "device_mb": round(dev, 1)}
        if len(self._checkpoints) >= 2:
            first = self._checkpoints[0]
            last = self._checkpoints[-1]
            result["rss_delta_mb"] = round(last[1] - first[1], 1)
            result["device_delta_mb"] = round(last[2] - first[2], 1)
        return result

    def log_summary(self, log: logging.Logger | None = None):
        log = log or logger
        parts = [f"{name}={rss:.0f}" for name, rss, _, _ in self._checkpoints]
        summary = " → ".join(parts)
        dev_parts = [f"{name}={dev:.1f}" for name, _, dev, _ in self._checkpoints if dev > 0]
        if dev_parts:
            summary += f" | device: {' → '.join(dev_parts)}"
        log.info(f"  Memory RSS (MB): {summary}")

    @staticmethod
    def object_summary() -> dict[str, int]:
        """Count large object types in memory (for leak detection)."""
        import sys
        counts: dict[str, int] = {}
        for obj in gc.get_objects():
            try:
                t = type(obj).__name__
                if t in ("list", "dict", "ndarray", "Tensor", "Transition",
                         "Trajectory", "TrajectoryGroup"):
                    counts[t] = counts.get(t, 0) + 1
            except Exception:
                pass
        return counts
