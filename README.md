# ⚡ AI Energy Lab
> **Empirical Resource Cost, Energy & Sustainability Analysis of Local & Cloud AI Systems**

Every AI response consumes physical electricity and money, yet almost no student or departmental project accounts for it. The **AI Energy Lab** measures the real resource cost of running AI workloads on local hardware vs. cloud alternatives, demonstrating the critical difference between **running a small model well** and **running a large model wastefully**.

Measurements are grounded in actual hardware instrumentation (RAPL processor registers, NVML GPU telemetry, and empirical CPU-utilization modeling) rather than estimated from articles. The lab includes an interactive estimation suite and applies it to a full-academic-year campus chatbot deployment scenario.

---

## 🛠️ Technology Stack
- **Languages & Frameworks:** Python 3.10+, Streamlit, Plotly Express
- **AI & Model Backends:** Ollama, Hugging Face Transformers, Accelerate
- **Instrumentation & Telemetry:** RAPL, NVML (via `pynvml`), `psutil`, CodeCarbon
- **Data & Testing:** Pandas, NumPy, Pytest

---

## 🚀 Quickstart & Setup

### 1. Environment Installation
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Pull Evaluation Models (Ollama)
```bash
ollama pull llama3.2:1b llama3.2:3b qwen3:1.7b gemma3:1b
```

### 3. Run Physical Power Benchmarks
Close background apps, fix power profile to Performance, and plug into AC mains.
```bash
# Ollama Benchmark across model sizes:
python -m aienergy.bench_ollama --models llama3.2:1b llama3.2:3b qwen3:1.7b --repeats 3

# Hugging Face benchmark:
python -m aienergy.bench_hf --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct
```
Benchmark outputs land in `data/results/*.csv` with hardware-measured metrics:
- `joules`: Total active energy consumed during generation
- `idle_w`: Baseline standby power drawn prior to inference
- `j_per_out_token`: Normalized active energy per output token
- `gen_tok_s` & `prefill_tok_s`: Generation throughput and prompt processing speed
- `method`: Measurement mechanism (`rapl_pkg`, `nvml_gpu`, or `cpu_util_model`)

---

## 🖥️ Launching the Interactive Dashboard

```bash
cd ai-energy-lab
streamlit run app.py
```

### Key Dashboard Modules (Human-Like UI):
1. **💡 Executive Verdict & Key Findings:** Direct side-by-side demonstration of *Small Model Well vs. Large Model Wastefully* (e.g. Qwen 1.7B beating Llama 3B on quality while consuming 76% less energy).
2. **🤖 The AI Green Auditor (Decision Wizard):** Interactive advisor that takes user constraints (privacy mandate, query volume, response SLA) and produces a personalized deployment blueprint.
3. **⚖️ The Pareto Efficiency Frontier:** 2D interactive scatter plot graphing **Quality Score vs. Active Joules per Token**, highlighting the optimal sweet spot models.
4. **🏫 Campus Chatbot Case Study:** 1-year academic simulation (15,000 students, 240 days) with an **interactive sleep-schedule simulator** showing the ~2.7× energy penalty of 24/7 unmonitored idling.
5. **🧮 Scenario Calculator:** Custom fleet sizing engine with server provisioning, utilization, and population scaling sensitivity curves.
6. **📊 Hardware Telemetry & Instrumentation:** Empirical sensor and CPU-utilization telemetry from RAPL, NVML, and `psutil` across local benchmark runs.
7. **🎯 Quality & Policy Review:** Heuristic keyword checking and provisional human ratings across prompt complexities.
8. **⚔️ Model Arena:** Live head-to-head inference battle between lightweight and heavy models with qualitative evaluation rubrics (requires local `torch`/`transformers`; gracefully disabled in lightweight web containers).
9. **📄 Executive Report Generator:** One-click publication-ready Markdown audit report ready for project submission or administrative review.

---

## 🧪 Running Automated Tests & Campus Case Study

```bash
# Execute unit test suite (16 passing tests):
python -m pytest

# Run the full-academic-year campus simulation script:
python -m scenarios.campus_chatbot
```

---

## ☁️ Deploying on Vercel (Container Runtime)

This repository includes Vercel Container Runtime configuration for deploying the Streamlit dashboard as an OCI container on Fluid Compute:

- `Dockerfile.vercel` — Builds a `python:3.12-slim` image, installs `requirements-web.txt`, exposes port `80`, and runs `streamlit run app.py` on `0.0.0.0:${PORT:-80}` in headless mode.
- `vercel.json` — Sets `"framework": "container"` and `"fluid": true`.
- `requirements-web.txt` — Lightweight runtime dependencies (`psutil`, `pandas`, `plotly`, `streamlit`) without local PyTorch/Transformers weights.
- `.dockerignore` — Excludes local virtualenvs, tests, docs, presentations, backups, and `.streamlit/credentials.toml` from the container build context.

### Deploy Steps
1. Ensure all changes and canonical benchmark/audit data files (`data/results/*.csv`, `data/cloud_options.json`) are committed and pushed to `main` on `https://github.com/adityajyotir2007-hub/ai-energy-lab`.
2. Import the repository at `https://vercel.com/new` (Root Directory: `.`) or deploy via Vercel CLI:
   ```bash
   npm i -g vercel
   vercel login
   vercel link
   vercel --prod
   ```

---

## 📁 Repository Structure
```
ai-energy-lab/
├── aienergy/
│   ├── calculator.py          # Analytical fleet sizing & energy consumption formulas
│   ├── meter.py               # Hardware telemetry sampling (RAPL, NVML, psutil)
│   ├── profiles.py            # Converts raw telemetry runs into model performance profiles
│   ├── sustainability.py      # Real-world equivalents, Pareto frontier, Green Auditor & reports
│   ├── model_arena.py         # Live interactive Hugging Face model shootout (local ML only)
│   ├── bench_ollama.py        # Ollama physical benchmarking CLI
│   ├── bench_hf.py            # Hugging Face transformers benchmarking CLI
│   ├── quality_eval.py        # Quality & policy term verification
│   └── prompts.py             # Standardized evaluation prompt suite
├── data/
│   ├── results/               # Canonical 5-model benchmark CSVs & phase1_quality_manual_audit.csv
│   └── cloud_options.json     # Cloud API and GPU rental baseline parameters (placeholder defaults)
├── docs/
│   └── local_measurement_results.md # Preliminary local benchmark notes
├── scenarios/
│   └── campus_chatbot.py      # Standardized 240-day campus deployment script
├── tests/
│   ├── test_calculator.py     # Fleet sizing and energy scaling tests
│   ├── test_meter.py          # Hardware power meter and attribute tests
│   └── test_sustainability.py # Equivalents, Pareto, and Green Auditor tests
├── .streamlit/
│   └── config.toml            # Default light theme (#F8FAFC) with in-app dark mode toggle & headless settings
├── .dockerignore              # Container build context exclusions
├── Dockerfile.vercel          # Vercel Container Runtime image definition
├── pytest.ini                 # Pytest configuration
├── requirements-web.txt       # Lightweight web container runtime dependencies
├── requirements.txt           # Full local development & ML benchmarking dependencies
├── vercel.json                # Vercel Container & Fluid Compute configuration
└── app.py                     # Streamlit web application
```
