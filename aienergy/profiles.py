from __future__ import annotations

import glob
import os

import pandas as pd

from .calculator import LocalProfile


def load_results(d: str = "data/results") -> pd.DataFrame:
    files = glob.glob(f"{d}/*.csv")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()

def build_profiles(df: pd.DataFrame, quality_csv: str = "data/quality.csv") -> list[LocalProfile]:
    if df.empty or not {"model", "gen_tok_s"}.issubset(df.columns):
        return []

    clean = df.copy()
    clean["gen_tok_s"] = pd.to_numeric(clean["gen_tok_s"], errors="coerce")
    clean = clean[clean["model"].notna() & clean["gen_tok_s"].notna()]
    if clean.empty:
        return []

    if "backend" not in clean.columns:
        clean["backend"] = "local"
    else:
        clean["backend"] = clean["backend"].fillna("local")

    q: dict = {}
    if os.path.exists(quality_csv):
        try:
            q_df = pd.read_csv(quality_csv)
            score_col = next((c for c in ("score", "automatic_score") if c in q_df.columns), None)
            if score_col and "model" in q_df.columns:
                q = q_df.dropna(subset=["model", score_col]).groupby("model")[score_col].mean().to_dict()
        except Exception:
            q = {}
    elif os.path.exists("data/results/phase1_quality_manual_audit.csv"):
        try:
            audit_df = pd.read_csv("data/results/phase1_quality_manual_audit.csv")
            if "model" in audit_df.columns and "automatic_score" in audit_df.columns:
                q = audit_df.groupby("model")["automatic_score"].mean().to_dict()
        except Exception:
            q = {}

    out = []
    for (backend, model), g in clean.groupby(["backend", "model"]):
        methods = sorted(g["method"].dropna().astype(str).unique()) if "method" in g.columns else []
        method_text = ", ".join(methods) if methods else "unknown"

        if any("cpu_util_model" in m for m in methods):
            source = f"Estimated power ({method_text})"
        elif any("rapl_pkg" in m or "nvml_gpu" in m for m in methods):
            source = f"Benchmark telemetry ({method_text})"
        else:
            source = f"Benchmark method unknown ({method_text})"

        params_val = None
        if "params_b" in g.columns and not g["params_b"].dropna().empty:
            params_val = g["params_b"].dropna().iloc[0]

        prefill_val = (
            g["prefill_tok_s"].median()
            if "prefill_tok_s" in g.columns and not g["prefill_tok_s"].dropna().empty
            else g["gen_tok_s"].median() * 5
        )
        active_w_val = (
            g["avg_w"].mean()
            if "avg_w" in g.columns and not g["avg_w"].dropna().empty
            else 20.0
        )
        idle_w_val = (
            g["idle_w"].mean()
            if "idle_w" in g.columns and not g["idle_w"].dropna().empty
            else 5.0
        )

        out.append(LocalProfile(
            name=f"{model} ({backend})",
            params_b=params_val,
            gen_tok_s=float(g["gen_tok_s"].median()),
            prefill_tok_s=float(prefill_val),
            active_w=float(active_w_val),
            idle_w=float(idle_w_val),
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
