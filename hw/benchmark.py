from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output.

    Handles cases like:
        "A"
        "(B)"
        "Answer: C"
        "The correct answer is D."
    """
    if text is None:
        return None
    if not isinstance(text, str):
        return None

    upper_choices = tuple(c.upper() for c in choices)
    if not upper_choices:
        return None
    choice_class = "".join(upper_choices)

    candidate = text.strip()
    if not candidate:
        return None

    upper = candidate.upper()

    stripped = upper.strip(" .,!?:;()[]{}\"'`")
    if len(stripped) == 1 and stripped in upper_choices:
        return stripped

    patterns = [
        rf"(?:ОТВЕТ|ANSWER|ANS)\s*[:\-]?\s*\(?\s*([{choice_class}])\b",
        rf"(?:CORRECT\s+ANSWER\s+IS|ПРАВИЛЬНЫЙ\s+ОТВЕТ)\s*[:\-]?\s*\(?\s*([{choice_class}])\b",
        rf"^\s*\(?\s*([{choice_class}])\s*\)",
        rf"\b([{choice_class}])\s*[\.\)]",
    ]
    for pat in patterns:
        m = re.search(pat, upper)
        if m:
            return m.group(1)

    matches = re.findall(rf"(?<![A-Z])([{choice_class}])(?![A-Z])", upper)
    if matches:
        return matches[-1]
    return None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop."""
    import random as _random

    from hw.dataset import MathVQADataset

    data_cfg = config.get("data", {}) or {}
    manifest_path = data_cfg.get("eval_manifest") or data_cfg.get("manifest")
    split = data_cfg.get("split", "dev")
    max_samples = data_cfg.get("max_samples")
    if toy:
        max_samples = min(int(max_samples or 4), 4)

    if manifest_path is None:
        raise ValueError("config.data.eval_manifest must be set")

    dataset = MathVQADataset(
        manifest_path=manifest_path,
        split=split,
        max_samples=max_samples,
    )

    rng = _random.Random(int(config.get("seed", 0)))
    rows: list[dict[str, Any]] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        prompt = build_benchmark_prompt(sample.question, sample.options)
        fake_output = rng.choice(list(CHOICES[: max(1, len(sample.options))]))
        pred = parse_mc_answer(fake_output)
        rows.append(
            {
                "id": sample.id,
                "subject": sample.subject,
                "answer": sample.answer,
                "prediction": pred,
                "prompt": prompt,
            }
        )

    output_path = config.get("output_path")
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return compute_accuracy(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
