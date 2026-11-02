
"""Power metering for AI workloads.

CPU sources:
  - RAPL: measured CPU package energy on supported Linux systems.
  - cpu_util_model: estimated CPU power based on utilisation.

GPU source:
  - NVML: measured NVIDIA GPU board power when the NVIDIA driver is available.

Important:
  - CPU and GPU readings are tracked separately.
  - Missing NVIDIA hardware or drivers do not prevent benchmarking.
  - RAPL measures CPU package energy, not whole-system energy.
  - NVML measures GPU board power, not whole-system energy.
  - The CPU utilisation model is an estimate, never a direct measurement.
"""

from __future__ import annotations

import glob
import os
import threading
import time
from datetime import datetime

import pandas as pd
import psutil

# Use the maintained NVIDIA bindings directly.
# NVIDIA Management Library (provided by nvidia-ml-py).
try:
    import pynvml as ml

    ml.nvmlInit()
    _NVML = ml
    _GPUS = [
        ml.nvmlDeviceGetHandleByIndex(i)
        for i in range(ml.nvmlDeviceGetCount())
    ]
except Exception:
    _NVML = None
    _GPUS = []


def _rapl_domains():
    """Find readable Intel RAPL energy counters."""
    domains = []

    for directory in sorted(
        glob.glob("/sys/class/powercap/intel-rapl:[0-9]*")
    ):
        energy_file = os.path.join(directory, "energy_uj")
        max_file = os.path.join(directory, "max_energy_range_uj")

        try:
            if os.access(energy_file, os.R_OK):
                with open(energy_file) as file:
                    current = int(file.read())

                with open(max_file) as file:
                    maximum = int(file.read())

                domains.append((energy_file, maximum))
        except (OSError, ValueError):
            continue

    return domains


_RAPL = _rapl_domains()


class PowerMeter:
    """Context manager for CPU/GPU power measurements.

    Backward-compatible attributes after exiting:
        seconds, joules, avg_w, method

    Additional attributes:
        cpu_joules, gpu_joules
        cpu_avg_w, gpu_avg_w
        cpu_method, gpu_method
        sample_coverage
    """

   
    def __init__(self, interval=0.25, cpu_tdp_w=15.0, cpu_idle_w=3.0):
    
        if interval <= 0:
            raise ValueError("interval must be greater than zero")

        if cpu_tdp_w < cpu_idle_w or cpu_idle_w < 0:
            raise ValueError("Invalid CPU power assumptions")

        self.interval = interval
        self.cpu_tdp_w = cpu_tdp_w
        self.cpu_idle_w = cpu_idle_w

        self.seconds = 0.0
        self.joules = 0.0
        self.avg_w = 0.0

        self.cpu_joules = 0.0
        self.gpu_joules = 0.0
        self.cpu_avg_w = 0.0
        self.gpu_avg_w = 0.0

        self._covered = 0.0
        self._gpu_covered = 0.0
        self.sample_coverage = 0.0

        self.cpu_method = (
            "rapl_pkg" if _RAPL else "cpu_util_model"
        )
        self.gpu_method = (
            "nvml_gpu" if _GPUS else "unavailable"
        )

        self.method = self.cpu_method
        if _GPUS:
            self.method += "+nvml_gpu"

    def _read_cpu_joules(self):
        """Read cumulative CPU package energy from RAPL."""
        values = []

        for energy_file, _ in _RAPL:
            with open(energy_file) as file:
                values.append(int(file.read()))

        return values

    def _cpu_power(self, dt):
        """Return CPU power in watts."""
        if _RAPL:
            current = self._read_cpu_joules()
            delta_uj = 0

            for now, previous, (_, maximum) in zip(
                current, self._previous_rapl, _RAPL
            ):
                if now >= previous:
                    delta_uj += now - previous
                else:
                    delta_uj += now + maximum - previous

            self._previous_rapl = current

            return delta_uj / 1_000_000 / dt

        utilisation = psutil.cpu_percent(interval=None) / 100.0

        return (
            self.cpu_idle_w
            + (self.cpu_tdp_w - self.cpu_idle_w) * utilisation
        )

    def _gpu_power(self):
        """Return total NVIDIA GPU board power and reading status."""
        if not _GPUS:
            return 0.0, False

        total = 0.0
        successful_readings = 0

        for handle in _GPUS:
            try:
                total += _NVML.nvmlDeviceGetPowerUsage(handle) / 1000.0
                successful_readings += 1
            except Exception:
                continue

        return total, successful_readings > 0

    def _loop(self):
        previous_time = time.perf_counter()

        while not self._stop.wait(self.interval):
            now = time.perf_counter()
            dt = now - previous_time
            previous_time = now

            if dt <= 0:
                continue

            cpu_ok = True

            try:
                cpu_w = self._cpu_power(dt)
            except (OSError, ValueError, ZeroDivisionError):
                cpu_w = 0.0
                cpu_ok = False

            gpu_w, gpu_ok = self._gpu_power()

            if gpu_ok:
                self.gpu_joules += gpu_w * dt
                self._gpu_covered += dt

            if cpu_ok:
                self.cpu_joules += cpu_w * dt
                self._covered += dt

    def __enter__(self):
        self.cpu_joules = 0.0
        self.gpu_joules = 0.0
        self._covered = 0.0
        self._gpu_covered = 0.0

        self._stop = threading.Event()

        self._previous_rapl = (
            self._read_cpu_joules() if _RAPL else []
        )

        psutil.cpu_percent(interval=None)

        self._t0 = time.perf_counter()

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )
        self._thread.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()

        self.seconds = time.perf_counter() - self._t0

        self.cpu_avg_w = (
            self.cpu_joules / self._covered
            if self._covered > 0 else 0.0
        )
        self.gpu_avg_w = (
            self.gpu_joules / self._gpu_covered
            if self._gpu_covered > 0 else 0.0
        )

        self.sample_coverage = (
            min(self._covered / self.seconds, 1.0)
            if self.seconds > 0 else 0.0
        )

        # Preserve the original total-power interface.
        self.joules = self.cpu_joules + self.gpu_joules

        self.avg_w = (
            self.joules / self._covered
            if self._covered > 0 else 0.0
        )

        # Keep historical method values for dashboard compatibility.
        self.method = self.cpu_method
        if _GPUS:
            self.method += "+nvml_gpu"

        return False


def idle_baseline(seconds: float = 20, **kwargs) -> float:
    """Measure average idle power using the configured telemetry sources."""
    with PowerMeter(**kwargs) as meter:
        time.sleep(seconds)

    return meter.avg_w


def codecarbon_tracker():
    """Create an optional CodeCarbon tracker."""
    try:
        from codecarbon import EmissionsTracker

        return EmissionsTracker(
            save_to_file=False,
            log_level="error",
            measure_power_secs=1,
        )
    except Exception:
        return None


def make_row(
    backend,
    machine,
    model,
    params_b,
    prompt_id,
    rep,
    prompt_tokens,
    out_tokens,
    wall_s,
    prefill_tok_s,
    gen_tok_s,
    m: PowerMeter,
    idle_w,
) -> dict:
    """Build a backward-compatible benchmark result row."""

    covered_s = min(m._covered, wall_s) if wall_s > 0 else 0.0
    active_j = max(m.joules - idle_w * covered_s, 0.0)

    return dict(
        ts=datetime.now().isoformat(timespec="seconds"),
        backend=backend,
        machine=machine,
        model=model,
        params_b=params_b,
        prompt_id=prompt_id,
        rep=rep,
        prompt_tokens=prompt_tokens,
        out_tokens=out_tokens,
        wall_s=round(wall_s, 3),
        prefill_tok_s=prefill_tok_s,
        gen_tok_s=gen_tok_s,
        joules=round(m.joules, 2),
        avg_w=round(m.avg_w, 2),
        idle_w=round(idle_w, 2),
        active_j=round(active_j, 2),
        j_per_out_token=(
            round(active_j / out_tokens, 4)
            if out_tokens else None
        ),
        method=m.method,
        cpu_method=m.cpu_method,
        gpu_method=m.gpu_method,
        cpu_joules=round(m.cpu_joules, 2),
        gpu_joules=round(m.gpu_joules, 2),
        cpu_avg_w=round(m.cpu_avg_w, 2),
        gpu_avg_w=round(m.gpu_avg_w, 2),
        sample_coverage=round(m.sample_coverage, 4),
        gpu_sample_coverage=round(
    m._gpu_covered / m.seconds if m.seconds > 0 else 0.0, 4),
        codecarbon_kg=None,
    )


def save_rows(
    rows: list[dict],
    backend: str,
    outdir: str = "data/results",
) -> str:
    os.makedirs(outdir, exist_ok=True)

    path = (
        f"{outdir}/{backend}_"
        f"{datetime.now():%Y%m%d_%H%M%S}.csv"
    )

    pd.DataFrame(rows).to_csv(path, index=False)

    return path
