# AI Energy Lab

Measure real power and runtime of local LLM inference, compare with cloud options, and estimate
energy, cost and carbon of a planned deployment (default scenario: campus chatbot, one academic year).

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:1b llama3.2:3b llama3.1:8b   # pull each separately; pick sizes your machine can run
```

## 1. Measure
Close other apps, plug in, fix the power profile. Linux: allow RAPL reads with
`sudo chmod -R a+r /sys/class/powercap`.
```
python -m aienergy.bench_ollama --models llama3.2:1b llama3.2:3b llama3.1:8b --repeats 3 --codecarbon
python -m aienergy.bench_hf --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct
```
Results land in `data/results/*.csv`. Columns include `joules`, `idle_w`, `active_j`, `j_per_out_token`,
`gen_tok_s`, `prefill_tok_s`, and `method` (rapl_pkg, nvml_gpu, or cpu_util_model = estimate).

Strongly recommended: log the wall-plug draw with a smart plug / USB power meter for at least one model
and compare to RAPL. Report the gap. RAPL excludes RAM, SSD, display and PSU losses.

Optional quality axis ("small model run well vs large model run wastefully"): score each model on a fixed
question set and save `data/quality.csv` with columns `model,score`.

## 2. Calculator and app
```
streamlit run app.py
python -m scenarios.campus_chatbot     # table + data/campus_chatbot_results.csv
pytest
```

## 3. Before you submit
- Replace every PLACEHOLDER value in `data/cloud_options.json` with current, cited prices and specs.
- Set grid carbon intensity and electricity price for your country with a cited source.
- Repeat runs on at least 3 model sizes; report mean and spread, not a single run.
- Report limits: single-stream measurements ignore batching; cloud energy is assumed, not measured.

## Layout
`aienergy/meter.py` metering, `bench_*.py` benchmarks, `calculator.py` model, `profiles.py` measured to profile,
`app.py` Streamlit, `scenarios/` campus case, `tests/`.
