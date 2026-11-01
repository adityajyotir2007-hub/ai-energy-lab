from __future__ import annotations

import glob
import os

import pandas as pd

from .calculator import LocalProfile


def load_results(d: str = "data/results") -> pd.DataFrame:
    files = glob.glob(f"{d}/*.csv")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()

def build_profiles(df: pd.DataFrame, quality_csv: str = "data/quality.csv") -> list[LocalProfile]:
    q = pd.read_csv(quality_csv).set_index("model")["score"].to_dict() if os.path.exists(quality_csv) else {}
    out = []

    for (backend, model), g in df.groupby(["backend", "model"]):
        methods = sorted(g["method"].dropna().astype(str).unique())
        method_text = ", ".join(methods) if methods else "unknown"

        if "cpu_util_model" in methods:
            source = f"Estimated power ({method_text})"
        elif "rapl_pkg" in methods or "nvml_gpu" in methods:
            source = f"Benchmark telemetry ({method_text})"
        else:
            source = f"Benchmark method unknown ({method_text})"

        out.append(LocalProfile(
            name=f"{model} ({backend})",
            params_b=g["params_b"].iloc[0],
            gen_tok_s=g["gen_tok_s"].median(),
            prefill_tok_s=g["prefill_tok_s"].median(),
            active_w=g["avg_w"].mean(),
            idle_w=g["idle_w"].mean(),
            quality=q.get(model),
            source=source
        ))

    return out
def placeholder_profiles() -> list[LocalProfile]:
    """ILLUSTRATIVE ONLY, NOT MEASURED. Exists so the app runs before you collect data."""
    s = "PLACEHOLDER (not measured)"
    return [LocalProfile("placeholder-1B", 1, 90, 900, 45, 15, None, s),
            LocalProfile("placeholder-3B", 3, 45, 500, 60, 15, None, s),
            LocalProfile("placeholder-8B", 8, 18, 200, 85, 15, None, s),
            LocalProfile("placeholder-70B", 70, 3, 30, 250, 20, None, s)]
