
"""Evaluate local Ollama model answers against basic quality and policy checks."""

import argparse
import os
from datetime import datetime

import pandas as pd
import requests

from .prompts import PROMPTS



POLICY_TERMS = {
    "short": [],
    "medium": [
        "course coordinator",
        "portal",
        "medical certificate",
        "before the deadline",
    ],
    "long_context": [
        "5%",
        "per day",
        "medical certificate",
        "course coordinator",
        "portal",
    ],
}



def call_model(host: str, model: str, prompt: str) -> dict:
    """Generate a response with thinking disabled and a larger output budget."""
    response = requests.post(
        f"{host}/api/generate",
        timeout=1200,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 500,
            },
        },
    )
    response.raise_for_status()
    data = response.json()

    if data.get("error"):
        raise RuntimeError(data["error"])

    return {
        "answer": data.get("response", "").strip(),
        "done_reason": data.get("done_reason", ""),
        "eval_count": data.get("eval_count", 0),
    }



def score_answer(prompt_id: str, answer: str, done_reason: str) -> dict:
    """Check response completion, required policy details, and suspicious claims."""
    import re

    terms = POLICY_TERMS[prompt_id]
    lower_answer = answer.lower()
    word_count = len(answer.split())

    if prompt_id == "short":
        length_ok = 10 <= word_count <= 100
    elif prompt_id == "medium":
        length_ok = 50 <= word_count <= 250
    else:
        length_ok = 15 <= word_count <= 250

    found = [term for term in terms if term.lower() in lower_answer]
    missing = [term for term in terms if term.lower() not in lower_answer]

    # Flag claims that need manual verification; this is not a definitive fact-check.
    warnings = []

    if prompt_id in ("medium", "long_context"):
        if re.search(r"\b(within 24 hours|within 24 hours of the deadline)\b", lower_answer):
            warnings.append("Unsupported 24-hour rule")
        if "instructor" in lower_answer and "course coordinator" not in lower_answer:
            warnings.append("Does not identify the course coordinator")
        if re.search(r"\b(automatically|will not lose any points|no penalty)\b", lower_answer):
            warnings.append("Possible unsupported penalty or waiver claim")
        
        if prompt_id == "long_context" and re.search(r"\b15\s*%", lower_answer):
            if "unless an extension is approved" not in lower_answer:
                warnings.append(
                    "Presents 15% penalty without adequately qualifying extension approval"
                )


    complete = bool(answer.strip()) and done_reason == "stop"
    checks_passed = int(length_ok) + len(found)
    total_checks = 1 + len(terms)
    score = round(100 * checks_passed / total_checks)

    # Penalize flagged policy risks, but retain warnings for manual review.
    score = max(0, score - 20 * len(warnings))

    if not complete:
        score = min(score, 20)

    return {
        "word_count": word_count,
        "length_ok": length_ok,
        "complete": complete,
        "done_reason": done_reason,
        "required_terms_found": "; ".join(found),
        "required_terms_missing": "; ".join(missing),
        "policy_warnings": "; ".join(warnings),
        "rule_score_percent": score,
        "manual_review_required": prompt_id != "short",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate local Ollama model answers."
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    rows = []

    for model in args.models:
        print(f"\nEvaluating {model}...")

        for prompt_id, prompt in PROMPTS.items():
            for repeat in range(1, args.repeats + 1):
                try:
                    result = call_model(args.host, model, prompt)
                    answer = result["answer"]
                    scores = score_answer(
                        prompt_id, answer, result["done_reason"]
                    )

                    rows.append({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "model": model,
                        "prompt_id": prompt_id,
                        "repeat": repeat,
                        **scores,
                        "eval_count": result["eval_count"],
                        "answer": answer,
                        "error": "",
                    })

                    print(
                        f"  {prompt_id} | run {repeat} | "
                        f"score: {scores['rule_score_percent']}% | "
                        f"complete: {scores['complete']}"
                    )
                    print(f"  Answer: {answer[:200].replace(chr(10), ' ')}")

                except Exception as exc:
                    print(f"  ERROR: {model} / {prompt_id}: {exc}")
                    rows.append({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "model": model,
                        "prompt_id": prompt_id,
                        "repeat": repeat,
                        "word_count": 0,
                        "length_ok": False,
                        "complete": False,
                        "done_reason": "",
                        "required_terms_found": "",
                        "required_terms_missing": "; ".join(
                            POLICY_TERMS[prompt_id]
                        ),
                        "rule_score_percent": 0,
                        "eval_count": 0,
                        "answer": "",
                        "error": str(exc),
                    })

    os.makedirs("data/results", exist_ok=True)
    output_path = (
        f"data/results/quality_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"\nSaved quality results to: {output_path}")
    print(
        "\nIMPORTANT: These are preliminary automated checks, "
        "not verified factual-accuracy scores. Review answers manually."
    )


if __name__ == "__main__":
    main()
