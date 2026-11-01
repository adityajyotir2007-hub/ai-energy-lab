"""Measure local inference with Hugging Face Transformers.

  python -m aienergy.bench_hf --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import platform
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .meter import PowerMeter, idle_baseline, make_row, save_rows
from .prompts import PROMPTS, parse_params_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--idle-seconds", type=float, default=20)
    ap.add_argument("--machine", default=platform.node())
    a = ap.parse_args()

    idle = idle_baseline(a.idle_seconds)
    print(f"idle: {idle:.1f} W")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for name in a.models:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

        def gen(inp, n):
            with torch.no_grad():
                return model.generate(**inp, max_new_tokens=n, min_new_tokens=n, do_sample=False)

        gen(tok("Hello", return_tensors="pt").to(dev), 4)  # warm-up
        for pid, prompt in PROMPTS.items():
            inp = tok(prompt, return_tensors="pt").to(dev)
            n_in = inp["input_ids"].shape[1]
            for rep in range(a.repeats):
                t = time.perf_counter(); gen(inp, 1); t_prefill = time.perf_counter() - t
                t0 = time.perf_counter()
                with PowerMeter() as m:
                    out = gen(inp, a.max_new_tokens)
                wall = time.perf_counter() - t0
                n_out = out.shape[1] - n_in
                gen_s = max(wall - t_prefill, 1e-6)
                rows.append(make_row("hf", a.machine, name, parse_params_b(name), pid, rep, n_in,
                                     n_out, wall, round(n_in / t_prefill, 2),
                                     round((n_out - 1) / gen_s, 2), m, idle))
                print(name, pid, rep, rows[-1]["gen_tok_s"], "tok/s", rows[-1]["avg_w"], "W")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    print("saved", save_rows(rows, "hf"))


if __name__ == "__main__":
    main()
