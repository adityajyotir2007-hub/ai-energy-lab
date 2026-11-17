from aienergy import sustainability as S


def test_compute_equivalents():
    eq = S.compute_equivalents(100.0, 71.0)
    assert eq["smartphones_charged"] > 0
    assert eq["gas_car_miles"] > 0
    assert eq["tree_seedlings_10yr"] > 0
    assert eq["kettles_boiled"] > 0
    assert eq["home_power_days"] > 0
    assert eq["led_bulb_hours"] > 0


def test_compute_budget_equivalents():
    b = S.compute_budget_equivalents(1600.0)
    assert b["student_laptops"] == 2
    assert b["textbook_stipends"] == 16
    assert b["cafeteria_meals"] > 200


def test_compute_efficiency_index():
    assert S.compute_efficiency_index(80.0, 2.0) == 40.0
    assert S.compute_efficiency_index(None, 2.0) is None
    assert S.compute_efficiency_index(80.0, 0.0) is None


def test_pareto_frontier_identifies_dominant_model():
    models = [
        {"model": "efficient", "quality": 90, "j_per_tok": 2.0},
        {"model": "wasteful", "quality": 85, "j_per_tok": 8.0},
        {"model": "balanced", "quality": 95, "j_per_tok": 5.0},
    ]
    frontier = S.compute_pareto_frontier(models)
    res = {m["model"]: m["pareto_optimal"] for m in frontier}
    # "efficient" has higher quality (90 > 85) and lower energy (2 < 8) than "wasteful", so "wasteful" is dominated
    assert res["efficient"] is True
    assert res["wasteful"] is False
    assert res["balanced"] is True


def test_ai_green_auditor_recommendation_strict_privacy():
    class DummyProfile:
        def __init__(self, name, speed, active_w, idle_w, quality):
            self.name = name
            self.gen_tok_s = speed
            self.active_w = active_w
            self.idle_w = idle_w
            self.quality = quality

    profiles = [
        DummyProfile("model-small", 25.0, 30.0, 10.0, 70.0),
        DummyProfile("model-large", 8.0, 50.0, 20.0, 65.0),
    ]

    rec = S.ai_green_auditor_recommendation(
        workload_type="Campus Assistant",
        query_volume_daily=10000,
        data_privacy_level="Strict local FERPA / GDPR",
        operating_mode="Business hours only",
        profiles=profiles,
        cloud_options=[],
    )

    assert "Local" in rec["recommended_architecture"]
    assert "model-small" in rec["recommended_model"]
    assert rec["privacy_compliant"] is True


def test_generate_executive_markdown_report():
    report = S.generate_executive_markdown_report(
        workload_summary={"total_queries": 3600000, "total_tokens_m": 6660, "users": 15000},
        top_models=[{"model": "qwen3:1.7b", "params_b": 1.7, "tok_s": 21.4, "j_per_tok": 3.0, "watts": 36.6, "idle_w": 10.7}],
        campus_scenario={"days": 240, "qpd": 5, "biz_kwh": "4,928", "always_kwh": "13,233", "multiplier": "2.69×",
                         "smartphones": "692,083", "car_miles": "14,642", "trees": "281"},
        power_method_note="CPU-Utilisation Model",
    )
    assert "# ⚡ AI Systems Energy, Cost & Sustainability Audit" in report
    assert "qwen3:1.7b" in report
    assert "Campus Chatbot Case Study" in report
