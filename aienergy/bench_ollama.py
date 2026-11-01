"""Measure local inference through Ollama.

  python -m aienergy.bench_ollama --models llama3.2:1b llama3.2:3b llama3.1:8b --repeats 3
Run with the machine otherwise idle, plugged in, same power profile for every model.
"""
from __future__ import annotations

import argparse
import platform
import time
import uuid

import requests

from .meter import PowerMeter, codecarbon_tracker, idle_baseline, make_row, save_rows
from .prompts import PROMPTS, parse_params_b


def call(host, model, prompt, num_predict, keep_alive="10m"):
    r = requests.post(f"{host}/api/generate", timeout=1200, json={
        "model": model, "prompt": prompt, "stream": False, "keep_alive": keep_alive,
        "options": {"num_predict": num_predict, "temperature": 0, "num_ctx": 4096}}).json()
    if "error" in r:
        raise RuntimeError(r["error"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=200)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--idle-seconds", type=float, default=20)
    ap.add_argument("--machine", default=platform.node())
    ap.add_argument("--codecarbon", action="store_true", help="cross-check each model run")
    a = ap.parse_args()

    print(f"measuring idle baseline for {a.idle_seconds}s ...")
    idle = idle_baseline(a.idle_seconds)
    print(f"idle: {idle:.1f} W")
    rows = []
    for model in a.models:
        call(a.host, model, "Hello", 8)  # warm-up: load weights, not measured
        tracker = codecarbon_tracker() if a.codecarbon else None
        if tracker:
            tracker.start()
        first = len(rows)
        for pid, prompt in PROMPTS.items():
            for rep in range(a.repeats):
                p = f"[{uuid.uuid4().hex[:8]}] {prompt}"  # nonce defeats prompt caching
                t0 = time.perf_counter()
                with PowerMeter() as m:
                    r = call(a.host, model, p, a.num_predict)
                wall = time.perf_counter() - t0
                pe, ee = r.get("prompt_eval_duration", 0), r.get("eval_duration", 0)
                rows.append(make_row(
                    "ollama", a.machine, model, parse_params_b(model), pid, rep,
                    r.get("prompt_eval_count", 0), r.get("eval_count", 0), wall,
                    round(r["prompt_eval_count"] / (pe / 1e9), 2) if pe else None,
                    round(r["eval_count"] / (ee / 1e9), 2) if ee else None, m, idle))
                print(model, pid, rep, rows[-1]["gen_tok_s"], "tok/s", rows[-1]["avg_w"], "W")
        if tracker:
            kg = tracker.stop()
            for row in rows[first:]:
                row["codecarbon_kg"] = kg
        call(a.host, model, "", 1, keep_alive=0)  # unload before next model
    print("saved", save_rows(rows, "ollama"))


if __name__ == "__main__":
    main()
