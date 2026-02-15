"""Evaluation harness for LongMemEval.

Implements the official GPT-4o judge with per-question-type prompts.
Binary scoring (0/1) following the paper's methodology.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Official judge prompts (from xiaowu0162/LongMemEval evaluate_qa.py)
# ---------------------------------------------------------------------------

JUDGE_PROMPT_DEFAULT = """You are an impartial judge evaluating whether a response correctly answers a question based on a reference answer.

Question: {question}
Reference Answer: {reference}
Response: {response}

Does the response contain the correct answer? Equivalent answers and reasonable intermediate steps count as correct. However, if the response only contains a subset of a multi-part answer, it should be marked incorrect.

Answer with ONLY "yes" or "no"."""

JUDGE_PROMPT_TEMPORAL = """You are an impartial judge evaluating whether a response correctly answers a temporal reasoning question based on a reference answer.

Question: {question}
Reference Answer: {reference}
Response: {response}

Does the response contain the correct answer? Equivalent answers count as correct. For questions involving counting days, weeks, or months, an off-by-one error is acceptable (e.g., if the answer is 5 days, both 4 and 6 are acceptable).

Answer with ONLY "yes" or "no"."""

JUDGE_PROMPT_KNOWLEDGE_UPDATE = """You are an impartial judge evaluating whether a response correctly answers a question about updated information based on a reference answer.

Question: {question}
Reference Answer: {reference}
Response: {response}

Does the response contain the correct updated answer? If the response mentions some previous/outdated information along with the updated answer, it is still correct as long as the updated answer is present.

Answer with ONLY "yes" or "no"."""

JUDGE_PROMPT_PREFERENCE = """You are an impartial judge evaluating whether a response correctly reflects user preferences based on a reference rubric.

Question: {question}
Reference Rubric: {reference}
Response: {response}

The response does not need to reflect ALL points in the rubric. It is correct as long as it recalls and utilizes the user's personal information/preferences correctly and addresses the question.

Answer with ONLY "yes" or "no"."""

JUDGE_PROMPT_ABSTENTION = """You are an impartial judge evaluating whether a response correctly identifies a question as unanswerable.

Question: {question}
Reference: This question is unanswerable based on the conversation history.
Response: {response}

Does the response correctly identify that the question cannot be answered? The response could say that the information is incomplete, unknown, or not available in the conversation history. A response that makes up an answer or guesses should be marked incorrect.

Answer with ONLY "yes" or "no"."""


def get_judge_prompt(question_type: str, is_abstention: bool) -> str:
    """Get the appropriate judge prompt for a question type."""
    if is_abstention:
        return JUDGE_PROMPT_ABSTENTION
    prompt_map = {
        "single-session-user": JUDGE_PROMPT_DEFAULT,
        "single-session-assistant": JUDGE_PROMPT_DEFAULT,
        "single-session-preference": JUDGE_PROMPT_PREFERENCE,
        "multi-session": JUDGE_PROMPT_DEFAULT,
        "temporal-reasoning": JUDGE_PROMPT_TEMPORAL,
        "knowledge-update": JUDGE_PROMPT_KNOWLEDGE_UPDATE,
    }
    return prompt_map.get(question_type, JUDGE_PROMPT_DEFAULT)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    """Result of judging a single question."""

    question_id: str
    question_type: str
    is_correct: bool
    judge_response: str = ""


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""

    overall_accuracy: float = 0.0
    task_averaged_accuracy: float = 0.0
    abstention_accuracy: float = 0.0
    per_type: dict[str, float] = field(default_factory=dict)
    per_type_counts: dict[str, int] = field(default_factory=dict)
    total_questions: int = 0
    total_correct: int = 0
    results: list[JudgeResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def judge_answer(
    question_id: str,
    question: str,
    reference_answer: str,
    hypothesis: str,
    question_type: str,
    openai_client: Any | None = None,
    model: str = "gpt-4o-2024-08-06",
) -> JudgeResult:
    """Judge a single answer using GPT-4o.

    Args:
        question_id: Unique question identifier.
        question: The benchmark question text.
        reference_answer: The gold-standard answer.
        hypothesis: The system's generated answer.
        question_type: One of the 6 LongMemEval question types.
        openai_client: OpenAI client instance. Created if None.
        model: Judge model ID.

    Returns:
        JudgeResult with binary correctness verdict.
    """
    if openai_client is None:
        try:
            from openai import OpenAI

            openai_client = OpenAI()
        except ImportError:
            raise ImportError(
                "openai package required for LongMemEval judge. "
                "Install with: pip install openai"
            )

    is_abstention = "_abs" in question_id
    prompt_template = get_judge_prompt(question_type, is_abstention)

    prompt = prompt_template.format(
        question=question,
        reference=reference_answer,
        response=hypothesis,
    )

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        judge_text = response.choices[0].message.content.strip().lower()
        is_correct = judge_text.startswith("yes")
    except Exception as e:
        logger.error(f"Judge failed for {question_id}: {e}")
        judge_text = f"error: {e}"
        is_correct = False

    return JudgeResult(
        question_id=question_id,
        question_type=question_type,
        is_correct=is_correct,
        judge_response=judge_text,
    )


def compute_metrics(results: list[JudgeResult]) -> EvalMetrics:
    """Compute aggregate metrics from judge results.

    Follows the official LongMemEval scoring:
    - Overall accuracy: mean across all questions
    - Task-averaged accuracy: mean of per-type averages
    - Abstention accuracy: accuracy on _abs questions only

    Args:
        results: List of JudgeResult from judge_answer().

    Returns:
        EvalMetrics with all computed scores.
    """
    if not results:
        return EvalMetrics()

    # Group by type
    type_correct: dict[str, list[int]] = {}
    abstention_correct: list[int] = []
    all_correct: list[int] = []

    for r in results:
        score = 1 if r.is_correct else 0
        all_correct.append(score)

        if r.question_type not in type_correct:
            type_correct[r.question_type] = []
        type_correct[r.question_type].append(score)

        if "_abs" in r.question_id:
            abstention_correct.append(score)

    # Per-type accuracy
    per_type: dict[str, float] = {}
    per_type_counts: dict[str, int] = {}
    type_means: list[float] = []
    for qtype, scores in type_correct.items():
        mean = sum(scores) / len(scores) if scores else 0.0
        per_type[qtype] = round(mean, 4)
        per_type_counts[qtype] = len(scores)
        type_means.append(mean)

    overall = sum(all_correct) / len(all_correct) if all_correct else 0.0
    task_avg = sum(type_means) / len(type_means) if type_means else 0.0
    abstention = (
        sum(abstention_correct) / len(abstention_correct)
        if abstention_correct
        else 0.0
    )

    return EvalMetrics(
        overall_accuracy=round(overall, 4),
        task_averaged_accuracy=round(task_avg, 4),
        abstention_accuracy=round(abstention, 4),
        per_type=per_type,
        per_type_counts=per_type_counts,
        total_questions=len(results),
        total_correct=sum(all_correct),
        results=results,
    )


def save_results(
    metrics: EvalMetrics,
    output_path: str | Path,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Save evaluation results to JSONL + summary JSON.

    Args:
        metrics: Computed metrics.
        output_path: Directory to write results into.
        extra_metadata: Additional metadata for the summary.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write per-question results (JSONL, compatible with official eval)
    results_path = output_dir / "results.jsonl"
    with open(results_path, "w") as f:
        for r in metrics.results:
            entry = {
                "question_id": r.question_id,
                "autoeval_label": {
                    "model": "gpt-4o-2024-08-06",
                    "label": r.is_correct,
                },
            }
            f.write(json.dumps(entry) + "\n")

    # Write summary
    summary = {
        "overall_accuracy": metrics.overall_accuracy,
        "task_averaged_accuracy": metrics.task_averaged_accuracy,
        "abstention_accuracy": metrics.abstention_accuracy,
        "per_type_accuracy": metrics.per_type,
        "per_type_counts": metrics.per_type_counts,
        "total_questions": metrics.total_questions,
        "total_correct": metrics.total_correct,
    }
    if extra_metadata:
        summary["metadata"] = extra_metadata

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


def print_metrics(metrics: EvalMetrics) -> None:
    """Print a formatted metrics table to stdout."""
    print("\n" + "=" * 60)
    print("LongMemEval Results")
    print("=" * 60)

    print(f"\n{'Question Type':<30} {'Accuracy':>10} {'Count':>8}")
    print("-" * 50)
    for qtype in [
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
    ]:
        if qtype in metrics.per_type:
            acc = metrics.per_type[qtype]
            count = metrics.per_type_counts[qtype]
            print(f"{qtype:<30} {acc:>9.1%} {count:>8}")

    print("-" * 50)
    print(f"{'Overall Accuracy':<30} {metrics.overall_accuracy:>9.1%} {metrics.total_questions:>8}")
    print(f"{'Task-Averaged Accuracy':<30} {metrics.task_averaged_accuracy:>9.1%}")
    if metrics.abstention_accuracy > 0:
        print(f"{'Abstention Accuracy':<30} {metrics.abstention_accuracy:>9.1%}")
    print("=" * 60 + "\n")
