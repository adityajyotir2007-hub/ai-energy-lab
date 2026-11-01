"""AI Energy Lab - Project 74: Energy, Cost and Sustainability Analysis of AI Systems.

Streamlit dashboard. Reads benchmark CSVs from data/results, optional cloud options from
data/cloud_options.json, optional quality files, and uses the existing aienergy calculator
and scenarios.campus_chatbot workload. Nothing here invents measurements or prices.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from aienergy.model_arena import render_model_arena

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # backend modules use relative paths such as data/results
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="AI Energy Lab", page_icon="⚡", layout="wide")

try:
    from aienergy import calculator as C
    from aienergy.profiles import build_profiles, load_results, placeholder_profiles
    from scenarios.campus_chatbot import G as CG, W as CW
except ImportError as exc:  # pragma: no cover
    st.error(f"Could not import project modules: {exc}. Run Streamlit from the project folder.")
    st.stop()

NAVY, TEAL = "#12355B", "#0F766E"
# Streamlit 1.50+ replaced use_container_width with width="stretch"
_VER = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
STRETCH = {"width": "stretch"} if _VER >= (1, 50) else {"use_container_width": True}
PALETTE = [TEAL, "#1D4ED8", "#D97706", "#7C3AED", "#BE185D", "#475569", "#15803D"]
KIND_LABELS = {"local": "Local server", "cloud_api": "Cloud API", "cloud_gpu": "Cloud GPU rental"}
QUALITY_MANUAL = Path("data/results/phase1_quality_manual_audit.csv")
QUALITY_AUTO = Path("data/quality.csv")
CLOUD_PATH = Path("data/cloud_options.json")

# ---------------------------------------------------------------- styling
st.markdown(f"""
<style>
html, body, .stApp {{ font-size: 16px; }}
.block-container {{ max-width: 1400px; padding-top: 1.4rem; padding-bottom: 3rem; }}
.app-header {{ border-left: 6px solid {TEAL}; padding: 2px 0 2px 16px; margin-bottom: 0.6rem; }}
.app-header .kicker {{ color: {TEAL}; font-weight: 700; font-size: .85rem; letter-spacing: .08em; text-transform: uppercase; }}
.app-header h1 {{ margin: 0; padding: 0; font-size: 2.1rem; }}
.app-header p {{ margin: .3rem 0 0 0; opacity: .85; }}
[data-testid="stMetric"] {{ border: 1px solid rgba(128,128,128,.35); border-left: 4px solid {TEAL};
    border-radius: 8px; padding: 12px 14px; background: rgba(128,128,128,.06); }}
[data-testid="stMetricLabel"] p {{ font-size: .95rem; }}
[data-testid="stMetricValue"] {{ font-size: 1.7rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{ padding: 10px 16px; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 1rem; font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {TEAL} !important; }}
@media (max-width: 640px) {{
  [data-testid="stMetricValue"] {{ font-size: 1.35rem; }}
  .app-header h1 {{ font-size: 1.5rem; }}
}}
</style>
<div class="app-header">
  <div class="kicker">Project 74 · Sustainability of AI systems</div>
  <h1>AI Energy Lab</h1>
  <p>Energy, cost and carbon estimates for AI deployments, built on benchmarks from this project.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def fmt(value, unit: str = "", digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}{(' ' + unit) if unit else ''}"


def has_cols(frame: pd.DataFrame, cols) -> bool:
    return not frame.empty and set(cols).issubset(frame.columns)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()


def pretty_table(frame: pd.DataFrame, rename: dict, digits: int = 2) -> pd.DataFrame:
    """Keep only known columns, rename to readable headers and round numbers."""
    cols = [c for c in rename if c in frame.columns]
    out = frame[cols].rename(columns=rename).copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(digits)
    return out


def style_fig(fig, height: int = 420):
    fig.update_layout(
        font=dict(size=14), title_font_size=18, height=height, colorway=PALETTE,
        margin=dict(l=10, r=24, t=60, b=10), hoverlabel=dict(font_size=13),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, font=dict(size=13), title_text=""),
    )
    fig.update_xaxes(title_font_size=14, tickfont_size=13, automargin=True)
    fig.update_yaxes(title_font_size=14, tickfont_size=13, automargin=True)
    return fig


def show(fig, height: int = 420):
    st.plotly_chart(style_fig(fig, height), **STRETCH)


def hbar(frame, y, x, x_label, title, color=None, barmode="group", height=None):
    """Horizontal bar chart: long names stay readable on small screens."""
    data = frame.dropna(subset=[x])
    if data.empty:
        st.info(f"No data available for “{title}”.")
        return
    fig = px.bar(data, x=x, y=y, color=color, orientation="h", title=title, barmode=barmode,
                 labels={x: x_label, y: ""}, color_discrete_sequence=PALETTE)
    if color is None:
        fig.update_traces(marker_color=TEAL)
    fig.update_traces(texttemplate="%{x:,.3g}", textposition="outside", cliponaxis=False)
    fig.update_yaxes(categoryorder="total ascending")
    show(fig, height or max(300, 70 + 48 * len(data)))


def benchmark_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep real benchmark rows. data/results also holds the quality audit CSV, and the
    backend load_results() concatenates every CSV in that folder."""
    if frame.empty or not {"model", "gen_tok_s"}.issubset(frame.columns):
        return pd.DataFrame()
    out = frame.copy()
    out["gen_tok_s"] = pd.to_numeric(out["gen_tok_s"], errors="coerce")
    return out[out["model"].notna() & out["gen_tok_s"].notna()].reset_index(drop=True)


def power_basis(frame: pd.DataFrame):
    """Return (kind, message) describing how power figures were obtained."""
    if "method" not in frame.columns or frame["method"].dropna().empty:
        return "unknown", ("The benchmark files have no `method` column, so the power source is unknown. "
                           "Treat power and joules as unverified estimates.")
    methods = sorted(frame["method"].dropna().astype(str).unique())
    joined = ", ".join(methods)
    if any("cpu_util_model" in m for m in methods):
        return "estimate", (f"Power source: **{joined}**. At least part of the power data comes from a "
                            "CPU-utilisation model. These are **estimates, not wall-socket measurements**.")
    return "measured", (f"Power source: **{joined}**. RAPL covers the CPU package and NVML the NVIDIA GPU only; "
                        "RAM, disk, display and power-supply losses are not included.")


def metric_choice(key: str):
    options = {"Energy (kWh/year)": ("kwh", "kWh/year"),
               "Cost (USD/year)": ("cost_usd", "USD/year"),
               "Carbon (kg CO₂/year)": ("kg_co2", "kg CO₂/year")}
    label = st.radio("Metric to compare", list(options), horizontal=True, key=key)
    return (*options[label], label)


ESTIMATE_COLUMNS = {
    "option": "Deployment option", "kind_label": "Type", "servers": "Servers needed",
    "utilisation_pct": "Average utilisation (%)", "kwh": "Energy (kWh/year)",
    "cost_usd": "Cost (USD/year)", "kg_co2": "Carbon (kg CO₂/year)",
    "wh_per_query": "Energy per query (Wh)", "note": "Basis of figures",
}


# ---------------------------------------------------------------- data
raw = pd.DataFrame()
try:
    raw = load_results()
except Exception as exc:  # unreadable CSV etc.
    st.error(f"Could not read benchmark files in data/results: {exc}")
bench = benchmark_only(raw)
measured = not bench.empty

profiles = placeholder_profiles()
if measured:
    try:
        try:
            built = build_profiles(bench, quality_csv=str(QUALITY_AUTO))
        except TypeError:  # older build_profiles signature
            built = build_profiles(bench)
        if built:
            profiles = built
        else:
            measured = False
    except Exception as exc:
        measured = False
        st.warning(f"Benchmark rows were found but model profiles could not be built ({exc}). "
                   "Showing placeholder profiles.")

cloud: list = []
if CLOUD_PATH.exists():
    try:
        loaded = json.loads(CLOUD_PATH.read_text(encoding="utf-8"))
        cloud = [c for c in loaded if isinstance(c, dict)] if isinstance(loaded, list) else []
    except (json.JSONDecodeError, OSError):
        st.warning("data/cloud_options.json could not be read; cloud options are hidden.")

# ---------------------------------------------------------------- sidebar
DEFAULTS = dict(users=15000, active=0.20, qpd=5.0, days=240, hours=12, peak=8.0, tin=1500, tout=350,
                ci=0.71, price=0.10, tu=0.70, aon=True, bg=1.0, capex=1500)
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

with st.sidebar:
    st.header("Scenario settings")
    st.caption("Change a value and every estimate updates immediately.")
    st.button("Reset to defaults", on_click=lambda: st.session_state.update(DEFAULTS),
              **STRETCH)
    with st.expander("Workload", expanded=True):
        users = st.number_input("Users", 100, 1_000_000, step=500, key="users",
                                help="Total people who could use the service.")
        active = st.slider("Daily active share", 0.01, 1.0, step=0.01, key="active",
                           help="Fraction of users who send at least one query on a working day.")
        qpd = st.number_input("Queries per active user per day", 1.0, 100.0, step=1.0, key="qpd")
        days = st.number_input("Operating days per year", 1, 365, key="days",
                               help="Days the service is used, e.g. about 240 academic days.")
        hours = st.number_input("Service hours per day", 1, 24, key="hours",
                                help="Hours per day the load is spread over.")
        peak = st.number_input("Peak-to-average load", 1.0, 50.0, step=0.5, key="peak",
                               help="How much busier the busiest period is than average. "
                                    "Higher values need more servers.")
        tin = st.number_input("Input tokens per query", 10, 32000, step=100, key="tin",
                              help="Prompt size including any retrieved documents. A token is roughly ¾ of a word.")
        tout = st.number_input("Output tokens per query", 10, 4000, step=25, key="tout",
                               help="Length of the model's answer in tokens.")
    with st.expander("Electricity and carbon", expanded=True):
        ci = st.number_input("Grid carbon intensity (kg CO₂ per kWh)", 0.0, 1.5, step=0.01, key="ci",
                             help="CO₂ emitted per unit of electricity. Use a cited figure for your region and year.")
        price = st.number_input("Electricity price (USD per kWh)", 0.0, 1.0, step=0.01, key="price")
    with st.expander("Deployment", expanded=True):
        tu = st.slider("Target peak utilisation", 0.2, 0.95, step=0.05, key="tu",
                       help="Share of each server's capacity used at peak. Lower means more spare servers.")
        aon = st.checkbox("Servers powered 24/7", key="aon",
                          help="If ticked, idle power is counted for all 8,760 hours of the year, "
                               "not only service hours.")
        bg = st.number_input("Batching throughput gain", 1.0, 16.0, step=0.5, key="bg",
                             help="Extra requests served at once by batching. 1 = no batching (as benchmarked).")
        capex = st.number_input("Hardware cost per local server (USD)", 0, 100000, step=100, key="capex",
                                help="Spread over the hardware lifetime set in the calculator.")

W = C.Workload(users, active, qpd, days, tin, tout, hours, peak)
G = C.Grid(carbon_kg_per_kwh=ci, price_per_kwh=price)


def estimate(workload, grid, tu_, aon_, bg_, capex_, with_cloud=True):
    """Run the existing calculator for every profile and cloud option. Returns (frame, problems)."""
    rows, problems = [], []
    for p in profiles:
        try:
            rows.append(C.local_estimate(workload, p, grid, tu_, aon_, bg_, capex_))
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
            problems.append(f"{getattr(p, 'name', 'profile')}: {exc!r}")
    if with_cloud:
        for o in cloud:
            try:
                rows.append(C.cloud_estimate(workload, o, grid, tu_, aon_))
            except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
                problems.append(f"{o.get('name', 'cloud option')}: missing or invalid field {exc!r}")
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["kind_label"] = frame["kind"].map(KIND_LABELS).fillna(frame["kind"])
        if "utilisation" in frame.columns:
            frame["utilisation_pct"] = (frame["utilisation"] * 100).where(frame["servers"] > 0)
        frame["servers"] = frame["servers"].where(frame["servers"] > 0)
    return frame, problems


# ---------------------------------------------------------------- status line
n_models = bench["model"].nunique() if measured else 0
_basis_kind, _basis_msg = power_basis(bench) if measured else ("none", "")
st.caption(
    f"Benchmark runs loaded: **{len(bench):,}** · Models: **{n_models}** · "
    f"Cloud options configured: **{len(cloud)}** · Power data: "
    f"**{ {'estimate': 'estimated (CPU-utilisation model)', 'measured': 'RAPL/NVML counters', 'unknown': 'source unknown', 'none': 'none loaded'}[_basis_kind] }**"
)
if not measured:
    st.warning("No usable benchmark measurements were found in data/results. The calculator is using "
               "**placeholder profiles**, so results below are illustrations, not project findings.")

tab_over, tab_models, tab_quality, tab_campus, tab_method, tab_arena = st.tabs(
    ["Overview", "Model Measurements", "Quality Review",
     "Campus Chatbot", "Methodology", "Model Arena"])

# ================================================================ Overview
with tab_over:
    render_model_arena()
    st.subheader("Overview")
    st.caption("Scenario-based **estimates** from the calculator, driven by the sidebar settings.")
    res, problems = estimate(W, G, tu, aon, bg, capex)
    for msg in problems:
        st.warning(f"Skipped an option: {msg}")

    st.markdown("#### Workload summary")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Queries per year", f"{W.total_queries:,.0f}")
    w2.metric("Tokens per year", f"{W.total_queries * (tin + tout) / 1e6:,.1f} M",
              help="Input plus output tokens across all queries.")
    w3.metric("Average load", f"{W.avg_qps:,.2f} queries/s",
              help="Average during service hours.")
    w4.metric("Peak load", f"{W.peak_qps:,.2f} queries/s",
              help="Average load times the peak-to-average factor. Servers are sized for this.")

    if res.empty:
        st.error("No deployment options could be estimated. Check the benchmark files and cloud options.")
    else:
        st.markdown("#### Annual estimate for one option")
        names = res["option"].astype(str).tolist()
        pick_one = st.selectbox("Deployment option", names, index=int(res["kwh"].idxmin()),
                                help="Starts on the lowest-energy option. Lowest energy is not automatically the best choice.")
        row = res[res["option"].astype(str) == pick_one].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Energy", fmt(row["kwh"], "kWh/year", 0))
        m2.metric("Electricity + hardware or service cost", fmt(row["cost_usd"], "USD/year", 0),
                  help="Local: electricity plus amortised hardware. Cloud: configured rental or per-token prices.")
        m3.metric("Carbon emissions", fmt(row["kg_co2"], "kg CO₂/year", 0))
        m4.metric("Energy per query", fmt(row["wh_per_query"], "Wh", 2))
        st.caption(f"Type: {row['kind_label']} · Basis: {row.get('note', '—')}")
        if "placeholder" in (pick_one + str(row.get("note", ""))).lower():
            st.warning("This option uses placeholder values, not project data.")

        st.markdown("#### Compare all options")
        unit_key, unit_label, label = metric_choice("overview_metric")
        hbar(res.assign(option=res["option"].astype(str)), "option", unit_key, unit_label,
             f"Estimated {label}", color="kind_label")
        st.dataframe(pretty_table(res.sort_values(unit_key), ESTIMATE_COLUMNS, 2),
                     **STRETCH, hide_index=True)
        if not cloud:
            st.info("No cloud options are configured in data/cloud_options.json, so only local options are compared.")
        elif any("placeholder" in json.dumps(o).lower() for o in cloud):
            st.warning("Some cloud options contain placeholder prices or specifications. Replace them with cited values before presenting.")

        with st.expander("Sensitivity: how energy changes with the number of users"):
            chosen = st.multiselect("Options to include", names, default=names[:3])
            sweep = []
            for u in sorted({max(100, int(users * f)) for f in (0.25, 0.5, 1, 2, 4)}):
                frame, _ = estimate(replace(W, users=u), G, tu, aon, bg, capex)
                if has_cols(frame, ["option", "kwh"]):
                    sweep.append(frame[frame["option"].astype(str).isin(chosen)].assign(users=u))
            if sweep and chosen:
                fig = px.line(pd.concat(sweep, ignore_index=True), x="users", y="kwh", color="option",
                              markers=True, log_x=True, log_y=True, color_discrete_sequence=PALETTE,
                              labels={"users": "Users (log scale)", "kwh": "Energy, kWh/year (log scale)", "option": ""},
                              title="Estimated annual energy vs users")
                show(fig)
            else:
                st.info("Select at least one option.")

# ================================================================ Model Measurements
with tab_models:
    st.subheader("Model Measurements")
    st.caption("These figures come from **benchmark runs** on your machine, not from the scenario calculator.")
    if not measured:
        st.info("No benchmark measurements are loaded. Run a benchmark, then refresh this page.")
        st.code("python -m aienergy.bench_ollama --models <model names>", language="powershell")
    else:
        kind, message = power_basis(bench)
        (st.warning if kind in ("estimate", "unknown") else st.info)(message)

        for c in ["gen_tok_s", "j_per_out_token", "avg_w", "params_b", "prefill_tok_s", "idle_w"]:
            if c in bench.columns:
                bench[c] = pd.to_numeric(bench[c], errors="coerce")
        spec = {"runs": ("model", "size"), "tok_s": ("gen_tok_s", "median")}
        for out_col, (src, fn) in {"j_per_tok": ("j_per_out_token", "mean"), "watts": ("avg_w", "mean"),
                                   "params_b": ("params_b", "first"),
                                   "idle_w": ("idle_w", "mean")}.items():
            if src in bench.columns:
                spec[out_col] = (src, fn)
        agg = bench.groupby("model", as_index=False).agg(**spec)

        k1, k2, k3 = st.columns(3)
        k1.metric("Models benchmarked", f"{agg['model'].nunique()}")
        k2.metric("Benchmark runs", f"{len(bench):,}")
        k3.metric("Prompt types", f"{bench['prompt_id'].nunique()}" if "prompt_id" in bench.columns else "—")

        a, b = st.columns(2)
        with a:
            hbar(agg, "model", "tok_s", "Tokens per second", "Generation speed (median)")
        with b:
            if "j_per_tok" in agg.columns:
                hbar(agg, "model", "j_per_tok", "Joules per output token",
                     "Active energy per output token (mean)")
            else:
                st.info("Column `j_per_out_token` is missing, so energy per token cannot be shown.")
        c, d = st.columns(2)
        with c:
            if "watts" in agg.columns:
                hbar(agg, "model", "watts", "Watts", "Average power during generation (mean)")
            else:
                st.info("Column `avg_w` is missing, so power cannot be shown.")
        with d:
            if has_cols(agg, ["params_b", "j_per_tok"]) and agg["params_b"].gt(0).sum() >= 2:
                sc = agg[agg["params_b"] > 0].dropna(subset=["params_b", "j_per_tok"])
                fig = px.scatter(sc, x="params_b", y="j_per_tok", color="model", log_x=True, log_y=True,
                                 hover_data=["tok_s"], color_discrete_sequence=PALETTE,
                                 title="Energy per token vs model size",
                                 labels={"params_b": "Parameters, billions (log)",
                                         "j_per_tok": "Joules per output token (log)", "tok_s": "Tokens/s"})
                fig.update_traces(marker=dict(size=14))
                show(fig, 420)
            else:
                st.info("Model-size chart needs `params_b` for at least two models.")

        st.markdown("#### Summary table")
        st.dataframe(pretty_table(agg, {
            "model": "Model", "params_b": "Parameters (billions)", "runs": "Runs",
            "tok_s": "Generation speed (tokens/s, median)", "j_per_tok": "Active energy (J/output token)",
            "watts": "Mean power (W)", "idle_w": "Idle baseline (W)"}, 3),
            **STRETCH, hide_index=True)
        st.caption("Active energy = measured joules minus the idle baseline for the same duration. "
                   "Faster or lower-energy does not mean better: check the Quality Review tab before choosing a model.")
        with st.expander("All benchmark rows"):
            st.dataframe(bench, **STRETCH, hide_index=True)

# ================================================================ Quality Review
with tab_quality:
    st.subheader("Quality Review")
    st.info("Two **different** evaluation methods are shown separately. The automatic score is a heuristic "
            "check. The manual audit is a provisional human review. Neither is ground-truth accuracy, and "
            "they should not be averaged together.")
    manual = read_csv_safe(QUALITY_MANUAL)
    automatic = read_csv_safe(QUALITY_AUTO)

    st.markdown("#### Automatic quality scores (heuristic)")
    if automatic.empty:
        st.caption(f"No automatic scores found at `{QUALITY_AUTO.as_posix()}`.")
    else:
        score_col = next((c for c in ("score", "automatic_score") if c in automatic.columns), None)
        if score_col and "model" in automatic.columns:
            automatic[score_col] = pd.to_numeric(automatic[score_col], errors="coerce")
            avg_auto = automatic.groupby("model", as_index=False)[score_col].mean()
            hbar(avg_auto, "model", score_col, "Automatic score", "Automatic heuristic score by model")
        st.dataframe(automatic, **STRETCH, hide_index=True)

    st.markdown("#### Manual audit scores (provisional)")
    if not has_cols(manual, ["model", "prompt_id", "manual_score"]):
        st.caption(f"No manual audit found at `{QUALITY_MANUAL.as_posix()}` "
                   "(needs columns model, prompt_id, manual_score).")
    else:
        manual["manual_score"] = pd.to_numeric(manual["manual_score"], errors="coerce")
        st.warning("Provisional ratings by a reviewer. Subjective, not independently validated or calibrated.")
        pivot = manual.pivot_table(index="model", columns="prompt_id", values="manual_score", aggfunc="mean")
        if pivot.empty or pivot.isna().all().all():
            st.info("The audit file has no numeric manual scores yet.")
        else:
            fig = px.imshow(pivot, text_auto=".0f", aspect="auto", color_continuous_scale="Teal",
                            zmin=0, zmax=max(100, float(pivot.max().max())),
                            title="Manual score by model and prompt type",
                            labels={"x": "Prompt type", "y": "Model", "color": "Score"})
            show(fig, max(320, 90 + 60 * len(pivot)))
            avg_manual = manual.groupby("model", as_index=False).agg(mean_manual_score=("manual_score", "mean"))
            hbar(avg_manual, "model", "mean_manual_score", "Average manual score",
                 "Average provisional manual score by model")
        st.markdown("##### Audit details")
        st.dataframe(pretty_table(manual, {
            "model": "Model", "prompt_id": "Prompt type", "automatic_score": "Automatic score (heuristic)",
            "manual_score": "Manual score (provisional)", "audit_status": "Audit status",
            "manual_findings": "Reviewer findings"}, 1), **STRETCH, hide_index=True)

# ================================================================ Campus Chatbot
with tab_campus:
    st.subheader("Campus Chatbot")
    st.caption("A fixed, illustrative year-long scenario. These are **calculator estimates**, not a measured deployment.")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Users", f"{CW.users:,}")
    s2.metric("Daily active share", f"{CW.active_frac:.0%}")
    s3.metric("Queries per active user per day", f"{CW.queries_per_active_day:g}")
    s4.metric("Academic days", f"{CW.days}")
    st.caption(f"{CW.in_tokens:,} input and {CW.out_tokens:,} output tokens per query · {CW.hours_per_day:g} service hours/day "
               f"· peak load {CW.peak_factor:g}× average · grid {CG.carbon_kg_per_kwh:g} kg CO₂/kWh · "
               f"{CG.price_per_kwh:g} USD/kWh. Only the hardware cost in the sidebar affects this tab.")

    biz, p1 = estimate(CW, CG, 0.7, False, 1.0, capex)
    always, p2 = estimate(CW, CG, 0.7, True, 1.0, capex, with_cloud=False)
    for msg in p1 + p2:
        st.warning(f"Skipped an option: {msg}")
    if biz.empty:
        st.error("No scenario options could be estimated.")
    else:
        biz = biz.assign(mode=biz["kind"].map(lambda k: "Local · business hours" if k == "local" else "Cloud · as configured"))
        always = always.assign(mode="Local · always on (24/7)")
        camp = pd.concat([biz, always], ignore_index=True)
        camp["option"] = camp["option"].astype(str)

        unit_key, unit_label, label = metric_choice("campus_metric")
        hbar(camp, "option", unit_key, unit_label, f"Campus scenario: estimated {label}", color="mode")

        loc_b = biz[biz["kind"] == "local"][["option", "kwh"]].rename(columns={"kwh": "kwh_business"})
        loc_a = always[always["kind"] == "local"][["option", "kwh"]].rename(columns={"kwh": "kwh_always"}) if not always.empty else pd.DataFrame()
        if has_cols(loc_b, ["option"]) and has_cols(loc_a, ["option"]):
            cmp_ = loc_b.merge(loc_a, on="option")
            cmp_["ratio"] = cmp_["kwh_always"] / cmp_["kwh_business"].where(cmp_["kwh_business"] > 0)
            st.markdown("#### Business hours vs always-on (local models)")
            st.dataframe(pretty_table(cmp_, {
                "option": "Model", "kwh_business": "Business hours (kWh/year)",
                "kwh_always": "Always on (kWh/year)", "ratio": "Always-on ÷ business hours (×)"}, 2),
                **STRETCH, hide_index=True)
            st.caption("Always-on operation adds idle power for hours when nobody is using the chatbot.")
        st.markdown("#### Full scenario table")
        st.dataframe(pretty_table(camp, {"mode": "Operating mode", **ESTIMATE_COLUMNS}, 2),
                     **STRETCH, hide_index=True)
        if not cloud:
            st.info("No cloud options are configured, so the scenario compares local deployments only.")

# ================================================================ Methodology
with tab_method:
    st.subheader("Methodology")
    st.markdown("""
#### Two kinds of numbers
- **Benchmark measurements** (Model Measurements tab): speed, power and energy recorded while real models answered real prompts on this project's machine.
- **Scenario estimates** (Overview and Campus Chatbot tabs): the calculator scales the benchmark figures up to a planned deployment. They depend on the assumptions in the sidebar.

#### How the calculator works (from `aienergy/calculator.py`)
```
service time per query = output tokens / generation speed + input tokens / prompt-processing speed
servers needed         = ceil( peak queries per second x service time / (target utilisation x batching gain) )
busy energy            = busy seconds x (active watts - idle watts)
idle energy            = servers x powered hours x idle watts
energy (kWh)           = (busy energy + idle energy) x PUE / 3.6 million
carbon (kg CO2)        = energy (kWh) x grid carbon intensity
local cost (USD)       = energy x electricity price + servers x hardware cost / hardware lifetime
cloud API cost (USD)   = tokens / 1 million x configured price per million tokens
cloud GPU cost (USD)   = servers x rented hours x configured price per hour
```
**PUE** (power usage effectiveness) is data-centre overhead such as cooling. The calculator defaults are 1.0 for local and 1.2 for cloud; they are not sidebar controls.

#### Measurement limitations
- **Power source.** RAPL reads CPU-package energy on supported Linux systems and NVML reads NVIDIA GPU power. `cpu_util_model` estimates power from CPU utilisation. Windows runs normally use `cpu_util_model`, so those joule and watt values are **estimates, not wall-socket measurements**. A smart plug would be needed for wall power.
- **Idle baseline.** Active energy subtracts an idle baseline measured before the run.
- **Single-stream benchmarks.** One request at a time, so batching and concurrent load are not captured unless you set a batching gain.
- **Cloud API energy** is an assumed figure from `data/cloud_options.json`, not a measurement.
- **Grid carbon and prices** are inputs; use cited values for your region.

#### Quality caveats
Automatic scores are heuristic checks. Manual audit scores are provisional reviewer judgements. Neither is validated accuracy, and no model is declared best on energy alone.
""")
    st.markdown("#### Data files found")
    status = [
        ("Benchmark results (data/results/*.csv)", measured),
        ("Cloud options (data/cloud_options.json)", bool(cloud)),
        ("Automatic quality scores (data/quality.csv)", not automatic.empty),
        ("Manual quality audit (data/results/phase1_quality_manual_audit.csv)", QUALITY_MANUAL.exists()),
    ]
    st.dataframe(pd.DataFrame([{"File": n, "Status": "Found" if ok else "Missing or empty"} for n, ok in status]),
                 **STRETCH, hide_index=True)
    st.caption("AI Energy Lab · Project 74 · Results depend on assumptions and local benchmark conditions.")
