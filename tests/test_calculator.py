from aienergy import calculator as C

W = C.Workload(1000, 0.5, 4, 100, 200, 300)
G = C.Grid()
P = C.LocalProfile("m", 1, 50, 500, 60, 20)


def test_energy_positive_and_scales_with_users():
    a = C.local_estimate(W, P, G)
    b = C.local_estimate(C.Workload(2000, 0.5, 4, 100, 200, 300), P, G)
    assert a["kwh"] > 0 and b["kwh"] > a["kwh"]


def test_always_on_costs_more_energy():
    assert C.local_estimate(W, P, G, always_on=True)["kwh"] > C.local_estimate(W, P, G)["kwh"]


def test_bigger_model_uses_more_energy():
    big = C.LocalProfile("b", 70, 3, 30, 250, 20)
    assert C.local_estimate(W, big, G)["kwh"] > C.local_estimate(W, P, G)["kwh"]


def test_api_cost():
    o = {"name": "x", "type": "api", "usd_per_m_in": 1, "usd_per_m_out": 2, "wh_per_1k_tokens": 1}
    r = C.cloud_estimate(W, o, G)
    assert abs(r["cost_usd"] - (W.total_queries * 200 / 1e6 + W.total_queries * 300 * 2 / 1e6)) < 1e-9

def test_carbon_emissions_match_energy_and_grid_intensity():
    result = C.local_estimate(W, P, G)
    assert abs(result["kg_co2"] - result["kwh"] * G.carbon_kg_per_kwh) < 1e-9

def test_local_cost_includes_annualized_hardware_cost():
    capex = 1500
    life_years = 4

    result = C.local_estimate(
        W, P, G, capex_usd=capex, life_years=life_years
    )

    expected = (
    result["kwh"] * G.price_per_kwh
    + result["servers"] * capex / life_years
)
    assert abs(result["cost_usd"] - expected) < 1e-9