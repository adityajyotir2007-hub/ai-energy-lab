"""Full-academic-year campus chatbot. Run: python -m scenarios.campus_chatbot"""
import json

import pandas as pd

from aienergy import calculator as C
from aienergy.profiles import build_profiles, load_results, placeholder_profiles

W = C.Workload(users=15000, active_frac=0.20, queries_per_active_day=5, days=240,
               in_tokens=1500, out_tokens=350, hours_per_day=12, peak_factor=8)
G = C.Grid()


def run():
    df = load_results()
    profiles = build_profiles(df) if not df.empty else placeholder_profiles()
    cloud = json.load(open("data/cloud_options.json"))
    rows = [C.local_estimate(W, p, G) for p in profiles]
    rows += [C.local_estimate(W, p, G, always_on=True) | {"option": p.name + " [24/7]"} for p in profiles]
    rows += [C.cloud_estimate(W, o, G) for o in cloud]
    return pd.DataFrame(rows).sort_values("kwh")


if __name__ == "__main__":
    r = run()
    print(r[["option", "kind", "servers", "kwh", "cost_usd", "kg_co2", "wh_per_query"]].round(2).to_string(index=False))
    r.to_csv("data/campus_chatbot_results.csv", index=False)
