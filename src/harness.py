"""
Barebones eval harness. Single file. No framework.

Usage:
    python -m src.harness --model qwen3-0.8b --prompt configs/prompts/v1.md
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

_llm: LLM | None = None

load_dotenv()
# ---- Data structures -------------------------------------------------------


@dataclass
class EvalExample:
    id: str
    request: str
    label: str  # "relevant" or "junk"
    category: str


@dataclass
class EvalResult:
    example_id: str
    request: str
    expected: str
    predicted: str
    raw_output: str
    correct: bool
    latency_ms: float
    category: str


# ---- Corpus loading --------------------------------------------------------


def load_corpus(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(
                EvalExample(
                    id=obj["id"],
                    request=obj["request"],
                    label=obj["label"],
                    category=obj.get("category", "unknown"),
                )
            )
    return examples


# ---- Output parsing --------------------------------------------------------

# What we expect: the model returns "relevant" or "junk" (case-insensitive,
# possibly with whitespace or punctuation around it). We extract that.
_LABEL_PATTERN = re.compile(r"\b(relevant|junk)\b", re.IGNORECASE)


def parse_label(raw_output: str) -> str | None:
    """Extract the predicted label from the model's raw output.

    Returns "relevant", "junk", or None if unparseable.
    """
    match = _LABEL_PATTERN.search(raw_output)
    if match is None:
        return None
    return match.group(1).lower()


# ---- Model call ------------------------------------------------------------
# This is the one part you'll need to swap out per backend.
# The contract: take a system prompt and a user message, return a string.
# Implement this for vLLM, OpenAI, Anthropic, etc.


def call_model(system_prompt: str, user_message: str, model: str) -> str:
    global _llm
    if _llm is None:
        _llm = LLM(
            model=model, 
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=1024
        )
    sampling = SamplingParams(
        temperature=0.0,    # classification: greedy
        max_tokens=8,       # we only need one word
    )
    out = _llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        sampling_params=sampling,
    )
    return out[0].outputs[0].text


# ---- Scoring ---------------------------------------------------------------


def score_example(
    example: EvalExample,
    raw_output: str,
    latency_ms: float,
) -> EvalResult:
    predicted = parse_label(raw_output)
    return EvalResult(
        example_id=example.id,
        request=example.request,
        expected=example.label,
        # predicted=predicted if predicted is not None else "<unparseable>",
        predicted=raw_output,
        raw_output=raw_output,
        correct=(predicted == example.label),
        latency_ms=latency_ms,
        category=example.category,
    )


def summarize(results: list[EvalResult]) -> dict:
    if not results:
        return {}
    n = len(results)
    n_correct = sum(1 for r in results if r.correct)
    n_unparseable = sum(1 for r in results if r.predicted == "<unparseable>")

    # Per-class precision/recall on the binary classification
    # (skip the "unparseable" bucket in metric math, but report it)
    tp_relevant = sum(
        1 for r in results if r.expected == "relevant" and r.predicted == "relevant"
    )
    fp_relevant = sum(
        1 for r in results if r.expected == "junk" and r.predicted == "relevant"
    )
    fn_relevant = sum(
        1 for r in results if r.expected == "relevant" and r.predicted == "junk"
    )
    tp_junk = sum(
        1 for r in results if r.expected == "junk" and r.predicted == "junk"
    )
    fp_junk = sum(
        1 for r in results if r.expected == "relevant" and r.predicted == "junk"
    )
    fn_junk = sum(
        1 for r in results if r.expected == "junk" and r.predicted == "relevant"
    )

    precision_relevant = tp_relevant / (tp_relevant + fp_relevant) if (tp_relevant + fp_relevant) else 0.0
    recall_relevant = tp_relevant / (tp_relevant + fn_relevant) if (tp_relevant + fn_relevant) else 0.0
    precision_junk = tp_junk / (tp_junk + fp_junk) if (tp_junk + fp_junk) else 0.0
    recall_junk = tp_junk / (tp_junk + fn_junk) if (tp_junk + fn_junk) else 0.0

    avg_latency = sum(r.latency_ms for r in results) / n

    return {
        "total": n,
        "accuracy": n_correct / n,
        "n_correct": n_correct,
        "n_unparseable": n_unparseable,
        "precision_relevant": precision_relevant,
        "recall_relevant": recall_relevant,
        "precision_junk": precision_junk,
        "recall_junk": recall_junk,
        "avg_latency_ms": avg_latency,
    }


# ---- Runner ----------------------------------------------------------------


def run_eval(
    corpus: list[EvalExample],
    system_prompt: str,
    model: str,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for ex in corpus:
        t0 = time.perf_counter()
        raw = call_model(system_prompt, ex.request, model=model)
        latency_ms = (time.perf_counter() - t0) * 1000
        results.append(score_example(ex, raw, latency_ms))
    return results


def print_report(model: str, results: list[EvalResult], summary: dict) -> None:
    print(f"\n=== Eval: {model} ===")
    print(f"Accuracy: {summary['accuracy']:.1%}  ({summary['n_correct']}/{summary['total']})")
    print(f"Unparseable: {summary['n_unparseable']}")
    print(f"Precision/recall relevant: {summary['precision_relevant']:.2f} / {summary['recall_relevant']:.2f}")
    print(f"Precision/recall junk:     {summary['precision_junk']:.2f} / {summary['recall_junk']:.2f}")
    print(f"Avg latency: {summary['avg_latency_ms']:.0f} ms")
    print()
    print(f"{'id':<6} {'expected':<10} {'predicted':<14} {'correct':<8} {'category':<14} request")
    print("-" * 100)
    for r in results:
        flag = "OK" if r.correct else "FAIL"
        print(f"{r.example_id:<6} {r.expected:<10} {r.predicted:<14} {flag:<8} {r.category:<14} {r.request}")


# ---- CLI -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model identifier (passed to call_model)")
    parser.add_argument("--corpus", type=Path, default=Path("data/eval/corpus.jsonl"))
    parser.add_argument("--prompt", type=Path, default=Path("configs/prompts/v1.md"))
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    system_prompt = args.prompt.read_text()

    results = run_eval(corpus, system_prompt, model=args.model)
    summary = summarize(results)
    print_report(args.model, results, summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Persist raw results to a JSON file for later comparison
    out_path = Path(".output") / f"eval_{args.model.replace('/', '_')}_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "summary": summary,
                "results": [r.__dict__ for r in results],
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())