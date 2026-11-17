import importlib.util
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from aienergy.meter import PowerMeter

_HAS_LOCAL_ML = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)

_VER = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
STRETCH = {"width": "stretch"} if _VER >= (1, 50) else {"use_container_width": True}


SMALL_MODELS = {
    "SmolLM2 — 360M": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen2.5 — 0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
}

HEAVY_MODELS = {
    "Qwen2.5 — 1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5 — 3B": "Qwen/Qwen2.5-3B-Instruct",
}

HISTORY_FILE = Path("data/results/model_arena_history.csv")


@st.cache_resource(show_spinner=False)
def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def prepare_input(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]

    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(encoded, return_tensors="pt")

    return tokenizer(prompt, return_tensors="pt")


def run_model(model_id, prompt, max_new_tokens):
    tokenizer, model = load_model(model_id)
    inputs = prepare_input(tokenizer, prompt)
    input_ids = inputs["input_ids"]
    input_tokens = len(input_ids[0])

    tracker = None
    emissions_kg = None
    meter = PowerMeter()
    output = None

    try:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(
            save_to_file=False,
            log_level="error",
            measure_power_secs=1,
        )
        tracker.start()
    except Exception:
        tracker = None

    try:
        with meter:
            import torch
            start_time = time.perf_counter()

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            elapsed = time.perf_counter() - start_time
    finally:
        if tracker is not None:
            try:
                emissions_kg = tracker.stop()
            except Exception:
                emissions_kg = None

    generated_ids = output[0][input_tokens:]
    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()
    output_tokens = len(generated_ids)

    # Token limit reached AND last token is not an end-of-sequence token.
    # If the model emitted EOS exactly at the limit, it ended naturally.
    hit_token_limit = False
    if output_tokens >= max_new_tokens:
        eos_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if eos_ids is None:
            eos_ids = tokenizer.eos_token_id
        if eos_ids is None:
            eos_ids = []
        elif isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        last_token = int(generated_ids[-1]) if output_tokens > 0 else None
        hit_token_limit = last_token is None or last_token not in set(eos_ids)

    return {
        "model": model_id,
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "seconds": elapsed,
        "tokens_per_second": output_tokens / elapsed if elapsed > 0 else 0,
        "energy_joules": getattr(meter, "joules", None),
        "power_method": getattr(meter, "method", "unknown"),
        "codecarbon_kg": emissions_kg,
        "hit_token_limit": hit_token_limit,
    }


def display_number(value, suffix="", digits=3):
    if value is None or pd.isna(value):
        return "Not available"
    return f"{value:.{digits}f}{suffix}"


RESPONSE_PANEL_HEIGHT = 400
TOKEN_LIMIT_WARNING = (
    "This response may be incomplete because the model reached its maximum "
    "generation-token limit. Increase the token limit and run the comparison again."
)


def build_single_response_text(result, title):
    return (
        f"{title}\n"
        f"Model name: {result.get('model_label', 'Not recorded')}\n"
        f"Model ID: {result['model']}\n"
        f"Date and time: {result.get('timestamp', 'Not recorded')}\n"
        f"Max generated tokens: {result.get('max_new_tokens', 'Not recorded')}\n"
        f"\nPrompt:\n{result.get('prompt', 'Not recorded')}\n"
        f"\nResponse:\n{result['answer'] or 'No response generated.'}\n"
    )


def build_combined_text(small, heavy):
    parts = [
        "AI Energy Lab - Model Arena comparison",
        f"Date and time: {small.get('timestamp', 'Not recorded')}",
        f"Max generated tokens (both models): {small.get('max_new_tokens', 'Not recorded')}",
        "",
        "PROMPT",
        small.get("prompt", "Not recorded"),
    ]
    for title, result in (("SMALL MODEL", small), ("HEAVY MODEL", heavy)):
        parts += [
            "",
            "=" * 60,
            title,
            f"Model name: {result.get('model_label', 'Not recorded')}",
            f"Model ID: {result['model']}",
            "=" * 60,
            result["answer"] or "No response generated.",
        ]
        if result.get("hit_token_limit"):
            parts += ["", "[Note: " + TOKEN_LIMIT_WARNING + "]"]
    return "\n".join(parts) + "\n"


def render_response_panel(result, title, key_prefix):
    st.markdown(f"#### {title}")
    st.caption(f"{result.get('model_label', 'Model')}  |  `{result['model']}`")

    # Fixed-height scrollable box. st.code keeps line breaks and spacing,
    # and its built-in copy icon copies the COMPLETE text, not just the visible part.
    with st.container(height=RESPONSE_PANEL_HEIGHT, border=True):
        text = result["answer"] or "No response generated."
        try:
            st.code(text, language=None, wrap_lines=True)
        except TypeError:  # older Streamlit without wrap_lines
            st.code(text, language=None)

    if result.get("hit_token_limit"):
        st.warning(TOKEN_LIMIT_WARNING)

    st.download_button(
        f"Download {title} response (.txt)",
        data=build_single_response_text(result, title).encode("utf-8"),
        file_name=f"{key_prefix}_model_response.txt",
        mime="text/plain",
        key=f"download_{key_prefix}_response",
        **STRETCH,
    )


def render_comparison(results):
    if len(results) != 2:
        return

    small, heavy = results

    st.subheader("1. Model responses")
    left, right = st.columns(2)
    with left:
        render_response_panel(small, "Small model", "small")
    with right:
        render_response_panel(heavy, "Heavy model", "heavy")

    st.download_button(
        "Download both responses (.txt)",
        data=build_combined_text(small, heavy).encode("utf-8"),
        file_name="model_arena_both_responses.txt",
        mime="text/plain",
        key="download_both_responses",
        **STRETCH,
    )
    st.caption("Use the copy icon at the top-right of a response box to copy the full response.")

    st.divider()
    st.subheader("2. Performance comparison")

    rows = [
        ("Generation time", small["seconds"], heavy["seconds"], "lower", " s", 2),
        ("Tokens per second", small["tokens_per_second"], heavy["tokens_per_second"], "higher", " tok/s", 2),
        ("Energy estimate", small["energy_joules"], heavy["energy_joules"], "lower", " J", 2),
        ("Estimated CO₂", small["codecarbon_kg"], heavy["codecarbon_kg"], "lower", " kg", 6),
        ("Output tokens", small["output_tokens"], heavy["output_tokens"], "none", "", 0),
    ]

    table = []
    for label, sv, hv, direction, suffix, digits in rows:
        if sv is None or hv is None or pd.isna(sv) or pd.isna(hv):
            winner = "Unavailable"
        elif direction == "lower":
            winner = "Small model" if sv < hv else "Heavy model" if hv < sv else "Tie"
        elif direction == "higher":
            winner = "Small model" if sv > hv else "Heavy model" if hv > sv else "Tie"
        else:
            winner = "Not ranked"

        table.append({
            "Metric": label,
            "Small model": display_number(sv, suffix, digits),
            "Heavy model": display_number(hv, suffix, digits),
            "Better result": winner,
        })

    st.dataframe(pd.DataFrame(table), hide_index=True, **STRETCH)

    st.subheader("3. Overall comparison")
    faster = (
        "Small model" if small["seconds"] < heavy["seconds"]
        else "Heavy model" if heavy["seconds"] < small["seconds"]
        else "Tie"
    )
    energy_winner = "Unavailable"
    if small["energy_joules"] is not None and heavy["energy_joules"] is not None:
        energy_winner = (
            "Small model" if small["energy_joules"] < heavy["energy_joules"]
            else "Heavy model" if heavy["energy_joules"] < small["energy_joules"]
            else "Tie"
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Faster model", faster)
    c2.metric("Lower energy estimate", energy_winner)
    c3.metric(
        "Speed difference",
        f'{abs(small["tokens_per_second"] - heavy["tokens_per_second"]):.2f} tok/s',
    )

    st.caption(
        "These results compare performance metrics, not answer correctness. "
        "Review both responses to judge quality. Repeat runs for more reliable averages."
    )

    st.markdown("##### 🗳️ Human Qualitative Evaluation")
    st.caption("Did the heavier model's response noticeably justify its higher energy draw?")
    vote_c1, vote_c2, vote_c3 = st.columns(3)
    if vote_c1.button("👍 Small Model Was Good Enough", key="vote_small", **STRETCH):
        st.success("Vote recorded! Compact models frequently satisfy user needs with a fraction of the energy.")
        st.session_state["arena_vote"] = "Small Model"
    if vote_c2.button("🌟 Heavy Model Was Noticeably Better", key="vote_heavy", **STRETCH):
        st.info("Vote recorded! Additional parameters provided superior nuance for this prompt.")
        st.session_state["arena_vote"] = "Heavy Model"
    if vote_c3.button("🤝 Indistinguishable Quality", key="vote_tie", **STRETCH):
        st.warning("Vote recorded! When quality is tied, deploying the larger model is pure resource waste.")
        st.session_state["arena_vote"] = "Tie"

    if "arena_vote" in st.session_state:
        st.markdown(f"**Your recorded judgment:** <span class='badge badge-teal'>{st.session_state['arena_vote']}</span>", unsafe_allow_html=True)

    st.divider()
    st.subheader("4. Detailed run data")
    details = pd.DataFrame([
        {
            "Model": result["model"],
            "Generation time (s)": result["seconds"],
            "Output tokens": result["output_tokens"],
            "Tokens/second": result["tokens_per_second"],
            "Energy (J)": result["energy_joules"],
            "Power method": result["power_method"],
            "Estimated CO₂ (kg)": result["codecarbon_kg"],
            "Reached token limit": result.get("hit_token_limit"),
        }
        for result in results
    ])
    st.dataframe(details, hide_index=True, **STRETCH)

    export_df = pd.DataFrame([
        {key: value for key, value in result.items() if key != "answer"}
        for result in results
    ])
    st.download_button(
        "Download this comparison as CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="model_arena_comparison.csv",
        mime="text/csv",
        **STRETCH,
    )


def append_history(results):
    """Append rows to the history CSV. Re-aligns columns if the file has an
    older header, so new columns never corrupt existing history."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(results)
    if HISTORY_FILE.exists():
        try:
            old_df = pd.read_csv(HISTORY_FILE)
            pd.concat([old_df, new_df], ignore_index=True).to_csv(HISTORY_FILE, index=False)
            return
        except Exception:
            pass  # unreadable file: fall through to plain append
    new_df.to_csv(
        HISTORY_FILE,
        mode="a",
        header=not HISTORY_FILE.exists(),
        index=False,
    )


def render_model_arena():
    st.subheader("Model Arena")
    st.write(
        "Compare a lightweight language model with a larger model using "
        "the same prompt and generation settings."
    )

    if not _HAS_LOCAL_ML:
        st.info(
            "Live Model Arena requires `torch` and `transformers` to run Hugging Face models on-device. "
            "In this lightweight web deployment, live inference is disabled — run locally with `.venv` to test models live."
        )
        return

    st.info(
        "Models run sequentially on your computer. Larger models can take "
        "considerably longer and may require more memory. Energy readings "
        "may be estimates, and CodeCarbon results from short runs can be noisy."
    )

    st.markdown("### Choose your models")
    small_col, heavy_col = st.columns(2)

    with small_col:
        st.markdown("#### Small model")
        small_label = st.selectbox(
            "Choose a lightweight model",
            list(SMALL_MODELS.keys()),
            key="arena_small_model",
        )
        st.caption("Lower resource use and faster local inference.")

    with heavy_col:
        st.markdown("#### Heavy model")
        heavy_label = st.selectbox(
            "Choose a larger model",
            list(HEAVY_MODELS.keys()),
            key="arena_heavy_model",
        )
        st.caption("Larger models may need more memory and processing time.")

    small_model_id = SMALL_MODELS[small_label]
    heavy_model_id = HEAVY_MODELS[heavy_label]

    st.divider()
    st.markdown("### Prompt laboratory")

    presets = [
        ("💡 Explain AI Simply", "Explain artificial intelligence in simple words for a beginner."),
        ("📜 Extension Policy", "A student was ill for three days. How can they request a coursework extension and what evidence is required?"),
        ("🐍 Python Fibonacci", "Write a concise, optimal Python function to compute the Nth Fibonacci number with docstring."),
        ("⚡ Energy Efficiency", "Explain why smaller language models are significantly more energy-efficient during inference than large models."),
    ]
    st.caption("Quick presets:")
    pr_cols = st.columns(len(presets))
    for col, (pr_label, pr_text) in zip(pr_cols, presets):
        if col.button(pr_label, key=f"btn_{pr_label}", **STRETCH):
            st.session_state["arena_prompt"] = pr_text

    prompt = st.text_area(
        "Enter the prompt to send to both models",
        value=st.session_state.get("arena_prompt", "Explain artificial intelligence in simple words."),
        height=130,
        placeholder="Type your prompt here...",
        key="arena_prompt_input",
    )

    max_new_tokens = st.slider(
        "Maximum generated tokens (same limit for both models)",
        min_value=20,
        max_value=512,
        value=200,
        step=20,
        key="arena_max_tokens",
    )
    st.caption(
        "Longer responses require more processing time and memory. If a response "
        "ends abruptly, increase the token limit and run the comparison again."
    )

    st.caption(f"Small: `{small_model_id}`  |  Heavy: `{heavy_model_id}`")

    if st.button(
        "Run small vs heavy comparison",
        type="primary",
        **STRETCH,
    ):
        if not prompt.strip():
            st.warning("Enter a prompt first.")
            return

        results = []
        progress = st.progress(0)
        status = st.empty()

        for index, model_id in enumerate([small_model_id, heavy_model_id]):
            category = "small" if index == 0 else "heavy"
            status.info(f"Running {category} model: {model_id}")
            try:
                with st.spinner(f"Loading and running {model_id}..."):
                    result = run_model(model_id, prompt.strip(), max_new_tokens)
                results.append(result)
            except Exception as exc:
                import traceback
                st.error(f"Could not run {model_id}: {repr(exc)}")
                st.code(traceback.format_exc())
                status.empty()
                progress.empty()
                return

            progress.progress((index + 1) / 2)

        status.success("Both models finished.")
        progress.empty()

        timestamp = datetime.now().isoformat(timespec="seconds")
        for index, result in enumerate(results):
            result["timestamp"] = timestamp
            result["prompt"] = prompt.strip()
            result["max_new_tokens"] = max_new_tokens
            result["category"] = "Small" if index == 0 else "Heavy"
            result["model_label"] = small_label if index == 0 else heavy_label

        append_history(results)
        st.session_state["arena_latest_results"] = results

    latest_results = st.session_state.get("arena_latest_results")
    if latest_results:
        render_comparison(latest_results)

    if HISTORY_FILE.exists():
        with st.expander("Previous comparison history"):
            try:
                history = pd.read_csv(HISTORY_FILE)
                st.dataframe(history, **STRETCH)
                st.download_button(
                    "Download full comparison history",
                    data=history.to_csv(index=False).encode("utf-8"),
                    file_name="model_arena_history.csv",
                    mime="text/csv",
                    **STRETCH,
                )
            except Exception as exc:
                st.warning(f"Could not read comparison history: {exc}")
