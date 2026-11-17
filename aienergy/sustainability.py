"""Sustainability and human-centric metrics for the AI Energy Lab.

Provides real-world environmental equivalents, Pareto efficiency scoring,
the AI Green Auditor recommendation engine, and executive report generation.
"""
from __future__ import annotations

import math
from typing import Any


def compute_equivalents(kwh: float, kg_co2: float) -> dict[str, Any]:
    """Convert raw energy and carbon values into human-relatable real-world analogies.

    Factors:
      - Smartphone recharge: ~0.012 kWh (approx 3,000 mAh at 3.7V + charger efficiency losses).
      - Gas passenger car: ~0.404 kg CO₂ per mile driven (US EPA standard baseline).
      - Urban tree seedlings grown for 10 years: ~21 kg CO₂ sequestered (EPA GHG Equivalencies).
      - Boiling a kettle of water: ~0.10 kWh (standard 1.5L electric kettle).
      - Home electricity day: ~20 kWh (global average residential household consumption).
      - 10W LED bulb illumination: 100 hours per 1 kWh.
    """
    kwh = max(float(kwh or 0.0), 0.0)
    kg_co2 = max(float(kg_co2 or 0.0), 0.0)

    return {
        "smartphones_charged": int(round(kwh / 0.012)),
        "gas_car_miles": round(kg_co2 / 0.404, 1),
        "tree_seedlings_10yr": round(kg_co2 / 21.0, 1),
        "kettles_boiled": int(round(kwh / 0.10)),
        "home_power_days": round(kwh / 20.0, 1),
        "led_bulb_hours": int(round(kwh * 100)),
    }


def compute_budget_equivalents(usd_amount: float) -> dict[str, Any]:
    """Translate dollar amounts into tangible educational/campus resources."""
    usd = max(float(usd_amount or 0.0), 0.0)
    return {
        "student_laptops": int(usd // 800),
        "textbook_stipends": int(usd // 100),
        "cafeteria_meals": int(usd // 6),
        "student_wifi_months": int(usd // 30),
    }


def compute_efficiency_index(quality_score: float | None, j_per_token: float) -> float | None:
    """Compute Energy Efficiency Index (EEI): Quality score delivered per Joule of active energy.

    Higher values indicate superior resource efficiency (delivering more quality per watt-second).
    """
    if quality_score is None or quality_score <= 0 or j_per_token <= 0:
        return None
    return round(float(quality_score) / float(j_per_token), 2)


def compute_pareto_frontier(model_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Determine Pareto-optimal models trading off Model Quality vs Active Energy per token.

    A model is Pareto-optimal (on the efficient frontier) if no other model achieves
    strictly higher quality with strictly lower or equal energy consumption.
    """
    valid = [
        m for m in model_stats
        if m.get("quality") is not None and m.get("j_per_tok") is not None and m["j_per_tok"] > 0
    ]
    if not valid:
        return model_stats

    frontier = []
    for candidate in valid:
        q_cand = float(candidate["quality"])
        e_cand = float(candidate["j_per_tok"])

        dominated = False
        for other in valid:
            q_other = float(other["quality"])
            e_other = float(other["j_per_tok"])

            # Dominated if another model has >= quality AND <= energy, with at least one strictly better
            if (q_other >= q_cand and e_other <= e_cand) and (q_other > q_cand or e_other < e_cand):
                dominated = True
                break

        cand_copy = dict(candidate)
        cand_copy["pareto_optimal"] = not dominated
        frontier.append(cand_copy)

    return frontier


def _profile_j_per_tok(p: Any, measured: dict[str, float] | None = None) -> float:
    """Active (above-idle) joules per output token: measured value if supplied, else derived."""
    name = getattr(p, "name", "")
    if measured and measured.get(name) and measured[name] > 0:
        return float(measured[name])
    extra_w = max(float(getattr(p, "active_w", 0) or 0) - float(getattr(p, "idle_w", 0) or 0), 0.0)
    return extra_w / max(float(getattr(p, "gen_tok_s", 0) or 0), 0.1)


def ai_green_auditor_recommendation(
    workload_type: str,
    query_volume_daily: int,
    data_privacy_level: str,
    operating_mode: str,
    profiles: list[Any],
    cloud_options: list[dict],
    j_per_tok: dict[str, float] | None = None,
    idle_tax: float | None = None,
) -> dict[str, Any]:
    """Rule-based deployment advisor. Every number it quotes comes from the supplied profiles.

    j_per_tok: optional {profile name: measured active J per output token}.
    idle_tax:  optional measured ratio of always-on energy to duty-cycled energy.
    """
    valid_profiles = [p for p in profiles if (getattr(p, "gen_tok_s", 0) or 0) > 0]

    def jt(p):
        return _profile_j_per_tok(p, j_per_tok)

    best_efficient_local = best_fastest_local = lowest_energy_local = heaviest_local = None
    if valid_profiles:
        # Efficiency = quality points per joule of active energy per token
        best_efficient_local = max(
            valid_profiles, key=lambda p: (getattr(p, "quality", None) or 50) / max(jt(p), 0.01)
        )
        best_fastest_local = max(valid_profiles, key=lambda p: getattr(p, "gen_tok_s", 0))
        lowest_energy_local = min(valid_profiles, key=jt)
        heaviest_local = max(valid_profiles, key=jt)

    saving_text = ""
    if best_efficient_local is not None and heaviest_local is not None and heaviest_local is not best_efficient_local:
        pct = (1 - jt(best_efficient_local) / max(jt(heaviest_local), 1e-9)) * 100
        if pct > 0:
            saving_text = (f" It uses **{pct:.0f}% less active energy per token** than "
                           f"{heaviest_local.name} in our benchmarks.")

    tax_text = f"{idle_tax:.1f}×" if idle_tax and idle_tax > 1 else "several times"
    cloud_is_placeholder = any("placeholder" in str(o.get("name", "")).lower() for o in (cloud_options or []))
    cloud_caveat = " Cloud figures in this tool use placeholder prices — verify with your provider." if cloud_is_placeholder else ""

    requires_local = "strict" in data_privacy_level.lower() or "local" in data_privacy_level.lower()
    always_on = "always" in operating_mode.lower() or "24/7" in operating_mode
    is_high_volume = query_volume_daily >= 20000
    is_low_volume = query_volume_daily <= 3000
    fallback_name = "Smallest benchmarked model"

    if requires_local:
        rec_kind = "Local Self-Hosted Server"
        chosen_model = best_efficient_local.name if best_efficient_local else fallback_name
        rationale = (f"Data must stay on-premise, so cloud is excluded. Run **{chosen_model}** — "
                     f"the best quality-per-joule model we benchmarked.{saving_text}")
    elif is_low_volume:
        rec_kind = "Cloud API (Pay-Per-Token)"
        chosen_model = "Pay-per-token cloud API"
        rationale = (f"At ~{query_volume_daily:,} queries/day a dedicated server would sit idle most of the time, "
                     f"and idle power is billed whether or not anyone asks a question. "
                     f"A pay-per-token API has no idle cost on your side.{cloud_caveat}")
    elif is_high_volume:
        rec_kind = "Local Server Fleet with Batching"
        chosen_model = best_efficient_local.name if best_efficient_local else fallback_name
        rationale = (f"At ~{query_volume_daily:,} queries/day servers stay busy, so idle waste shrinks. "
                     f"Run **{chosen_model}** with request batching enabled.{saving_text}{cloud_caveat}")
    else:
        rec_kind = "Compact Local Server"
        chosen_model = best_efficient_local.name if best_efficient_local else fallback_name
        rationale = (f"At ~{query_volume_daily:,} queries/day a single compact model balances privacy, "
                     f"cost and energy. Run **{chosen_model}**.{saving_text}{cloud_caveat}")

    if rec_kind.startswith("Cloud"):
        duty_cycle_advice = "No servers to schedule — you pay only for tokens processed."
    elif always_on:
        duty_cycle_advice = (f"⚠️ Switch to a sleep schedule. In our campus model, always-on uses "
                             f"**{tax_text}** the energy of a service-hours schedule.")
    else:
        duty_cycle_advice = "✅ Good choice — sleeping servers outside service hours avoids most idle waste."

    return {
        "recommended_architecture": rec_kind,
        "recommended_model": chosen_model,
        "primary_rationale": rationale,
        "duty_cycle_advice": duty_cycle_advice,
        "privacy_compliant": True,
        "fastest_alternative": best_fastest_local.name if best_fastest_local else "Fastest Local Model",
        "lowest_power_alternative": lowest_energy_local.name if lowest_energy_local else "Lowest Watt Model",
    }


def generate_executive_markdown_report(
    workload_summary: dict[str, Any],
    top_models: list[dict[str, Any]],
    campus_scenario: dict[str, Any],
    power_method_note: str,
    findings: list[str] | None = None,
    recommendations: list[str] | None = None,
) -> str:
    """Generate a printable sustainability report in Markdown.

    All headline numbers should be passed in (findings / campus_scenario) so the report
    never states a figure that was not computed from the project's own data.
    """
    queries = workload_summary.get("total_queries", 0)
    tokens_m = workload_summary.get("total_tokens_m", 0)
    users = workload_summary.get("users", 15000)

    lines = [
        "# ⚡ AI Systems Energy, Cost & Sustainability Audit",
        "**Project 74: Resource Cost of Local & Cloud AI Workloads**",
        "",
        "---",
        "",
        "## 1. Summary",
        "",
        "We benchmarked local language models on our own machine (runtime, tokens/s and power), "
        "then used a calculator to project energy, cost and carbon for a campus chatbot over a full academic year.",
        "",
        f"> **Power basis:** {power_method_note}",
        "",
        "### Key findings",
    ]
    if findings:
        lines += [f"{i}. {f}" for i, f in enumerate(findings, 1)]
    else:
        lines.append("- No benchmark data available yet — run the benchmark scripts first.")

    lines += [
        "",
        "---",
        "",
        "## 2. Workload (calculator settings)",
        f"- **Users**: {users:,}",
        f"- **Annual queries**: {queries:,.0f} ({tokens_m:,.1f} M tokens)",
        "",
        "---",
        "",
        "## 3. Benchmarked models",
        "",
        "| Model | Params | Speed (tok/s) | Active energy (J/tok) | Avg power (W) | Idle power (W) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m in top_models:
        params = f"{m.get('params_b')}B" if m.get("params_b") else "—"
        lines.append(
            f"| **{m.get('model', 'Model')}** | {params} | {m.get('tok_s', 0):.1f} | "
            f"{m.get('j_per_tok', 0):.2f} | {m.get('watts', 0):.1f} | {m.get('idle_w', 0):.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Campus Chatbot Case Study: Full Academic Year",
        "",
        f"- **Model**: {campus_scenario.get('model', '—')}",
        f"- **Calendar**: {campus_scenario.get('days', 240)} academic days, "
        f"{campus_scenario.get('hours', 12)} service hours/day",
        f"- **Load**: {campus_scenario.get('qpd', 5)} queries per active user per day",
        f"- **Service-hours energy**: {campus_scenario.get('biz_kwh', '—')} kWh/year",
        f"- **Always-on (24/7) energy**: {campus_scenario.get('always_kwh', '—')} kWh/year",
        f"- **Always-on multiplier**: **{campus_scenario.get('multiplier', '—')}**",
        "",
        "### What the extra 24/7 energy equals",
        f"- Charging **{campus_scenario.get('smartphones', '—')}** smartphones",
        f"- Driving **{campus_scenario.get('car_miles', '—')}** miles in a petrol car",
        f"- The CO₂ absorbed by **{campus_scenario.get('trees', '—')}** tree seedlings over 10 years",
        "",
        "---",
        "",
        "## 5. Recommendations",
        "",
    ]
    recs = recommendations or [
        "Pick the smallest model whose answers pass your quality check.",
        "Sleep servers outside service hours and during breaks.",
        "Confirm estimates with a wall-plug power meter before budgeting.",
    ]
    lines += [f"{i}. {r}" for i, r in enumerate(recs, 1)]
    lines += ["", "*Generated by AI Energy Lab (Project 74).*"]
    return "\n".join(lines)
