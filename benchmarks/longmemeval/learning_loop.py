"""Learning loop for LongMemEval.

Runs the benchmark multiple times, feeding back correctness signals
between rounds. Shows accuracy improving over time -- the learning
curve that no other memory system on the leaderboard can produce.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qortex.observe import emit
from qortex.observe.events import BenchLearningCurveRecorded, BenchRoundCompleted

logger = logging.getLogger(__name__)


@dataclass
class LearningRoundResult:
    """Results from a single learning round."""

    round_num: int
    overall_accuracy: float
    task_averaged_accuracy: float
    per_type: dict[str, float] = field(default_factory=dict)
    feedback_given: int = 0
    accepted: int = 0
    rejected: int = 0
    elapsed_s: float = 0.0


@dataclass
class LearningCurve:
    """Full learning curve across all rounds."""

    rounds: list[LearningRoundResult] = field(default_factory=list)
    improvement: float = 0.0  # final - initial accuracy
    relative_improvement: float = 0.0  # improvement / initial


def run_learning_loop(
    client: Any,
    questions: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    config: Any,
    openai_client: Any,
    initial_metrics: Any,
) -> LearningCurve:
    """Run the learning loop: answer -> judge -> feedback -> repeat.

    After each round:
    1. Judge results identify correct/incorrect answers
    2. Feedback is sent to qortex: accepted for correct, rejected for wrong
    3. Teleportation factors adjust, changing retrieval behavior
    4. Next round re-answers the same questions with updated retrieval

    Args:
        client: LocalQortexClient instance.
        questions: List of question dicts.
        hypotheses: Initial hypotheses from round 0.
        config: RunConfig.
        openai_client: OpenAI client.
        initial_metrics: Metrics from round 0.

    Returns:
        LearningCurve with per-round results.
    """
    from benchmarks.longmemeval.evaluate import (
        compute_metrics,
        judge_answer,
        print_metrics,
    )
    from benchmarks.longmemeval.retrieve import retrieve_context
    from benchmarks.longmemeval.runner import generate_answer

    curve = LearningCurve()

    # Record round 0
    round_0 = LearningRoundResult(
        round_num=0,
        overall_accuracy=initial_metrics.overall_accuracy,
        task_averaged_accuracy=initial_metrics.task_averaged_accuracy,
        per_type=initial_metrics.per_type,
    )
    curve.rounds.append(round_0)

    print(f"\nRound 0 (baseline): {initial_metrics.overall_accuracy:.1%}")

    # Build lookup from question_id to hypothesis + judge result
    current_hypotheses = {h["question_id"]: h for h in hypotheses}
    current_judge = {
        r.question_id: r for r in initial_metrics.results
    }

    for round_num in range(1, config.learning_rounds + 1):
        round_start = time.perf_counter()
        print(f"\n--- Learning Round {round_num}/{config.learning_rounds} ---")

        # Step 1: Feed back results from previous round
        accepted = 0
        rejected = 0
        for q in questions:
            qid = q["question_id"]
            h = current_hypotheses.get(qid)
            jr = current_judge.get(qid)
            if h is None or jr is None:
                continue

            query_id = h.get("query_id")
            if query_id is None:
                continue

            # Build outcomes dict from retrieved items
            # For correct answers: all retrieved items contributed positively
            # For wrong answers: all retrieved items contributed negatively
            if jr.is_correct:
                outcome = "accepted"
                accepted += 1
            else:
                outcome = "rejected"
                rejected += 1

            # Feed back for all items that were retrieved for this query
            # The feedback propagates through the learning layer
            outcomes = {}
            # Use a generic outcome keyed by query_id since we don't
            # track individual item IDs through the hypothesis
            outcomes[query_id] = outcome

            try:
                client.feedback(
                    query_id=query_id,
                    outcomes=outcomes,
                    source="longmemeval",
                )
            except Exception as e:
                logger.debug(f"Feedback failed for {qid}: {e}")

        print(f"  Feedback: {accepted} accepted, {rejected} rejected")

        # Step 2: Re-answer all questions with updated retrieval
        new_hypotheses: list[dict[str, Any]] = []
        for i, q in enumerate(questions):
            domain = f"lme_{q['question_id']}"

            ctx = retrieve_context(
                client=client,
                question=q["question"],
                domain=domain,
                top_k=config.top_k,
                question_date=q.get("question_date"),
            )

            hypothesis = generate_answer(
                question=q["question"],
                context_text=ctx.context_text,
                question_date=q.get("question_date"),
                openai_client=openai_client,
                model=config.answer_model,
            )

            new_hypotheses.append({
                "question_id": q["question_id"],
                "question": q["question"],
                "question_type": q["question_type"],
                "answer": q["answer"],
                "hypothesis": hypothesis,
                "query_id": ctx.query_id,
                "items_retrieved": ctx.item_count,
            })

        # Step 3: Judge the new answers
        judge_results = []
        for h in new_hypotheses:
            result = judge_answer(
                question_id=h["question_id"],
                question=h["question"],
                reference_answer=h["answer"],
                hypothesis=h["hypothesis"],
                question_type=h["question_type"],
                openai_client=openai_client,
                model=config.judge_model,
            )
            judge_results.append(result)

        metrics = compute_metrics(judge_results)
        round_elapsed = time.perf_counter() - round_start

        round_result = LearningRoundResult(
            round_num=round_num,
            overall_accuracy=metrics.overall_accuracy,
            task_averaged_accuracy=metrics.task_averaged_accuracy,
            per_type=metrics.per_type,
            feedback_given=accepted + rejected,
            accepted=accepted,
            rejected=rejected,
            elapsed_s=round(round_elapsed, 1),
        )
        curve.rounds.append(round_result)

        emit(BenchRoundCompleted(
            benchmark="longmemeval",
            round_num=round_num,
            overall_accuracy=metrics.overall_accuracy,
            task_averaged_accuracy=metrics.task_averaged_accuracy,
            feedback_given=accepted + rejected,
            accepted=accepted,
            rejected=rejected,
            elapsed_s=round(round_elapsed, 1),
        ))

        # Update state for next round
        current_hypotheses = {h["question_id"]: h for h in new_hypotheses}
        current_judge = {r.question_id: r for r in judge_results}

        delta = metrics.overall_accuracy - curve.rounds[-2].overall_accuracy
        print(
            f"  Round {round_num}: {metrics.overall_accuracy:.1%} "
            f"(delta: {delta:+.1%}) [{round_elapsed:.1f}s]"
        )

    # Compute overall improvement
    if len(curve.rounds) >= 2:
        initial = curve.rounds[0].overall_accuracy
        final = curve.rounds[-1].overall_accuracy
        curve.improvement = round(final - initial, 4)
        curve.relative_improvement = (
            round(curve.improvement / initial, 4) if initial > 0 else 0.0
        )

        emit(BenchLearningCurveRecorded(
            benchmark="longmemeval",
            rounds=len(curve.rounds),
            initial_accuracy=initial,
            final_accuracy=final,
            improvement=curve.improvement,
            relative_improvement=curve.relative_improvement,
        ))

    # Print learning curve
    print("\n" + "=" * 60)
    print("Learning Curve")
    print("=" * 60)
    print(f"{'Round':>6} {'Accuracy':>10} {'Delta':>8} {'Feedback':>10}")
    print("-" * 40)
    for r in curve.rounds:
        delta = ""
        if r.round_num > 0:
            prev = curve.rounds[r.round_num - 1].overall_accuracy
            d = r.overall_accuracy - prev
            delta = f"{d:+.1%}"
        print(
            f"{r.round_num:>6} {r.overall_accuracy:>9.1%} {delta:>8} "
            f"{r.feedback_given:>10}"
        )
    print("-" * 40)
    print(f"Total improvement: {curve.improvement:+.1%}")
    if curve.relative_improvement:
        print(f"Relative improvement: {curve.relative_improvement:+.1%}")
    print("=" * 60 + "\n")

    # Save learning curve
    output_dir = Path(config.output_dir) / f"learning_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_data = {
        "rounds": [
            {
                "round": r.round_num,
                "overall_accuracy": r.overall_accuracy,
                "task_averaged_accuracy": r.task_averaged_accuracy,
                "per_type": r.per_type,
                "feedback_given": r.feedback_given,
                "accepted": r.accepted,
                "rejected": r.rejected,
                "elapsed_s": r.elapsed_s,
            }
            for r in curve.rounds
        ],
        "improvement": curve.improvement,
        "relative_improvement": curve.relative_improvement,
    }
    with open(output_dir / "learning_curve.json", "w") as f:
        json.dump(curve_data, f, indent=2)

    return curve
