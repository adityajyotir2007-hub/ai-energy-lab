"""Energy, cost and carbon estimator for a planned AI deployment.

Model: each query occupies one replica for service_s = out/gen_tok_s + in/prefill_tok_s.
Replicas are sized so peak load stays under `target_util`. Energy = busy energy above idle
+ idle power of every powered replica for every powered hour (idle burn is what makes
oversized, always-on deployments wasteful).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Workload:
    users: int
    active_frac: float
    queries_per_active_day: float
    days: int
    in_tokens: int
    out_tokens: int
    hours_per_day: float = 12.0
    peak_factor: float = 8.0

    @property
    def total_queries(self) -> float:
        return self.users * self.active_frac * self.queries_per_active_day * self.days

    @property
    def avg_qps(self) -> float:
        return self.users * self.active_frac * self.queries_per_active_day / (self.hours_per_day * 3600)

    @property
    def peak_qps(self) -> float:
        return self.avg_qps * self.peak_factor


@dataclass
class Grid:
    carbon_kg_per_kwh: float = 0.71   # grid average; verify for your region/year
    price_per_kwh: float = 0.10       # USD
    local_pue: float = 1.0
    cloud_pue: float = 1.2
    cloud_carbon_kg_per_kwh: float | None = None  # None = same as local grid


@dataclass
class LocalProfile:
    name: str
    params_b: float | None
    gen_tok_s: float
    prefill_tok_s: float
    active_w: float       # measured total system watts while generating
    idle_w: float         # measured idle watts
    quality: float | None = None
    source: str = "measured"


def _hours(w: Workload, always_on: bool) -> float:
    return 365 * 24 if always_on else w.days * w.hours_per_day


def _result(option, kind, kwh, cost, carbon_ci, w, servers, util, note=""):
    return dict(option=option, kind=kind, kwh=kwh, cost_usd=cost, kg_co2=kwh * carbon_ci,
                wh_per_query=kwh * 1000 / w.total_queries if w.total_queries else 0,
                servers=servers, utilisation=util, note=note)


def _fleet(w, service_s, target_util, batch_gain, active_extra_w, idle_w, powered_h, pue):
    servers = max(1, math.ceil(w.peak_qps * service_s / (target_util * batch_gain)))
    busy_s = w.total_queries * service_s / batch_gain
    wh = (busy_s * active_extra_w + servers * powered_h * 3600 * idle_w) / 3600
    util = busy_s / (servers * powered_h * 3600)
    return servers, wh / 1000 * pue, util


def local_estimate(w: Workload, p: LocalProfile, g: Grid, target_util=0.7, always_on=False,
                   batch_gain=1.0, capex_usd=0.0, life_years=4.0) -> dict:
    service = w.out_tokens / p.gen_tok_s + w.in_tokens / p.prefill_tok_s
    servers, kwh, util = _fleet(w, service, target_util, batch_gain,
                                max(p.active_w - p.idle_w, 0), p.idle_w, _hours(w, always_on),
                                g.local_pue)
    cost = kwh * g.price_per_kwh + servers * capex_usd / life_years
    return _result(p.name, "local", kwh, cost, g.carbon_kg_per_kwh, w, servers, util, p.source)


def cloud_estimate(w: Workload, o: dict, g: Grid, target_util=0.7, always_on=False) -> dict:
    ci = g.cloud_carbon_kg_per_kwh or g.carbon_kg_per_kwh
    tin, tout = w.total_queries * w.in_tokens, w.total_queries * w.out_tokens
    if o["type"] == "api":
        kwh = (tin + tout) / 1000 * o["wh_per_1k_tokens"] / 1000 * g.cloud_pue
        cost = tin / 1e6 * o["usd_per_m_in"] + tout / 1e6 * o["usd_per_m_out"]
        return _result(o["name"], "cloud_api", kwh, cost, ci, w, 0, 0.0, "assumed energy")
    service = w.out_tokens / o["gen_tok_s"] + w.in_tokens / o["prefill_tok_s"]
    hours = _hours(w, always_on)
    servers, kwh, util = _fleet(w, service, target_util, o.get("batch_gain", 1.0),
                                o["active_w"] - o["idle_w"], o["idle_w"], hours, g.cloud_pue)
    cost = servers * hours * o["usd_per_hour"]
    return _result(o["name"], "cloud_gpu", kwh, cost, ci, w, servers, util, "rented, assumed specs")
