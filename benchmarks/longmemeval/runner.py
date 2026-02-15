"""Main benchmark runner for LongMemEval.

Downloads the dataset from HuggingFace, ingests sessions into qortex,
answers questions via graph-enhanced retrieval, and evaluates with the
official GPT-4o judge.

Usage:
    # Quick test (10 questions, structured ingest, no judge)
    python -m benchmarks.longmemeval.runner --subset 10 --dry-run

    # Full run on small dataset
    python -m benchmarks.longmemeval.runner --dataset longmemeval_s

    # With learning loop
    python -m benchmarks.longmemeval.runner --dataset longmemeval_s --learning-rounds 5

    # Specific question type
    python -m benchmarks.longmemeval.runner --question-type temporal-reasoning --subset 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(
    dataset_name: str = "longmemeval_s",
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Load LongMemEval dataset from HuggingFace.

    Args:
        dataset_name: One of "longmemeval_s", "longmemeval_m", "longmemeval_oracle".
        cache_dir: Optional local cache directory.

    Returns:
        List of question dicts with keys: question_id, question_type,
        question, answer, question_date, haystack_sessions, haystack_dates,
        answer_session_ids.
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError(
            "datasets package required. Install with: pip install datasets"
        )

    # Map our names to HuggingFace split names
    split_map = {
        "longmemeval_s": "longmemeval_s",
        "longmemeval_m": "longmemeval_m",
        "longmemeval_oracle": "longmemeval_oracle",
    }
    split = split_map.get(dataset_name, dataset_name)

    logger.info(f"Loading dataset: xiaowu0162/longmemeval-cleaned ({split})")
    ds = hf_load(
        "xiaowu0162/longmemeval-cleaned",
        split=split,
        cache_dir=cache_dir,
    )

    questions = []
    for item in ds:
        questions.append({
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "question": item["question"],
            "answer": item["answer"],
            "question_date": item.get("question_date", ""),
            "haystack_sessions": item.get("haystack_sessions", []),
            "haystack_dates": item.get("haystack_dates", []),
            "answer_session_ids": item.get("answer_session_ids", []),
        })

    logger.info(f"Loaded {len(questions)} questions")
    return questions


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------


def create_client(
    mode: str = "auto",
    use_llm_ingest: bool = False,
) -> Any:
    """Create a LocalQortexClient for benchmarking.

    Args:
        mode: Retrieval mode ("auto", "graph", "vec").
        use_llm_ingest: If True, use real LLM for concept extraction.
            If False, use structured ingest (much faster, no LLM cost).

    Returns:
        LocalQortexClient instance.
    """
    from qortex.client import LocalQortexClient
    from qortex.core.memory import InMemoryBackend
    from qortex.vec.embeddings import SentenceTransformerEmbedding
    from qortex.vec.index import NumpyVectorIndex

    embedding = SentenceTransformerEmbedding()
    vec_index = NumpyVectorIndex(dimensions=embedding.dimensions)
    backend = InMemoryBackend(vector_index=vec_index)
    backend.connect()

    llm_backend = None
    if use_llm_ingest:
        try:
            from qortex.ingest.backends.anthropic import AnthropicBackend

            llm_backend = AnthropicBackend()
        except (ImportError, Exception):
            logger.warning("AnthropicBackend unavailable, falling back to StubLLMBackend")

    client = LocalQortexClient(
        vector_index=vec_index,
        backend=backend,
        embedding_model=embedding,
        llm_backend=llm_backend,
        mode=mode,
    )
    return client


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------


def generate_answer(
    question: str,
    context_text: str,
    question_date: str | None = None,
    openai_client: Any | None = None,
    model: str = "gpt-4o-2024-08-06",
) -> str:
    """Generate an answer using an LLM with retrieved context.

    Args:
        question: The benchmark question.
        context_text: Retrieved context from qortex.
        question_date: Optional date for temporal grounding.
        openai_client: OpenAI client. Created if None.
        model: Model to use for answer generation.

    Returns:
        Generated answer string.
    """
    if openai_client is None:
        try:
            from openai import OpenAI

            openai_client = OpenAI()
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    from benchmarks.longmemeval.retrieve import format_answer_prompt

    prompt = format_answer_prompt(question, context_text, question_date)

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    dataset: str = "longmemeval_s"
    subset: int | None = None
    question_type: str | None = None
    mode: str = "auto"
    use_llm_ingest: bool = False
    top_k: int = 20
    answer_model: str = "gpt-4o-2024-08-06"
    judge_model: str = "gpt-4o-2024-08-06"
    learning_rounds: int = 0
    output_dir: str = "benchmark_results/longmemeval"
    dry_run: bool = False
    cache_dir: str | None = None


def run_benchmark(config: RunConfig) -> dict[str, Any]:
    """Run the full LongMemEval benchmark.

    Args:
        config: RunConfig with all parameters.

    Returns:
        Dict with metrics and results.
    """
    from benchmarks.longmemeval.evaluate import (
        EvalMetrics,
        compute_metrics,
        judge_answer,
        print_metrics,
        save_results,
    )
    from benchmarks.longmemeval.ingest import ingest_sessions_structured
    from benchmarks.longmemeval.retrieve import retrieve_context

    # Load dataset
    print(f"Loading dataset: {config.dataset}")
    questions = load_dataset(config.dataset, cache_dir=config.cache_dir)

    # Filter by question type if specified
    if config.question_type:
        questions = [q for q in questions if q["question_type"] == config.question_type]
        print(f"Filtered to {len(questions)} {config.question_type} questions")

    # Apply subset
    if config.subset:
        questions = questions[: config.subset]
        print(f"Using subset of {len(questions)} questions")

    if not questions:
        print("No questions to process.")
        return {"metrics": None}

    # Create client
    print(f"Creating qortex client (mode={config.mode})")
    client = create_client(mode=config.mode, use_llm_ingest=config.use_llm_ingest)

    # Create OpenAI client for answer gen + judging
    openai_client = None
    if not config.dry_run:
        try:
            from openai import OpenAI

            openai_client = OpenAI()
        except ImportError:
            print("WARNING: openai not installed. Running in dry-run mode.")
            config.dry_run = True

    # Phase 1: Ingest sessions for each question
    print(f"\nPhase 1: Ingesting sessions for {len(questions)} questions...")
    ingest_start = time.perf_counter()

    for i, q in enumerate(questions):
        sessions = q.get("haystack_sessions", [])
        dates = q.get("haystack_dates", [])
        if sessions and dates:
            domain = f"lme_{q['question_id']}"
            stats = ingest_sessions_structured(
                client=client,
                sessions=sessions,
                session_dates=dates,
                domain=domain,
            )
            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  [{i + 1}/{len(questions)}] {q['question_id']}: "
                    f"{stats.concepts_created} concepts, {stats.edges_created} edges"
                )

    ingest_elapsed = time.perf_counter() - ingest_start
    print(f"Ingestion complete in {ingest_elapsed:.1f}s")

    # Phase 2: Retrieve + Answer
    print(f"\nPhase 2: Answering {len(questions)} questions...")
    answer_start = time.perf_counter()

    hypotheses: list[dict[str, Any]] = []
    for i, q in enumerate(questions):
        domain = f"lme_{q['question_id']}"

        # Retrieve context
        ctx = retrieve_context(
            client=client,
            question=q["question"],
            domain=domain,
            top_k=config.top_k,
            question_date=q.get("question_date"),
        )

        if config.dry_run:
            hypothesis = f"[DRY RUN] Retrieved {ctx.item_count} items for: {q['question'][:80]}"
        else:
            hypothesis = generate_answer(
                question=q["question"],
                context_text=ctx.context_text,
                question_date=q.get("question_date"),
                openai_client=openai_client,
                model=config.answer_model,
            )

        hypotheses.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "question_type": q["question_type"],
            "answer": q["answer"],
            "hypothesis": hypothesis,
            "query_id": ctx.query_id,
            "items_retrieved": ctx.item_count,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{len(questions)}] {q['question_id']}: {ctx.item_count} items retrieved")

    answer_elapsed = time.perf_counter() - answer_start
    print(f"Answering complete in {answer_elapsed:.1f}s")

    # Phase 3: Judge
    if config.dry_run:
        print("\nPhase 3: Skipped (dry run)")
        metrics = EvalMetrics()
    else:
        print(f"\nPhase 3: Judging {len(hypotheses)} answers...")
        judge_start = time.perf_counter()

        judge_results = []
        for i, h in enumerate(hypotheses):
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

            if (i + 1) % 10 == 0:
                correct_so_far = sum(1 for r in judge_results if r.is_correct)
                print(f"  [{i + 1}/{len(hypotheses)}] Running accuracy: {correct_so_far}/{i + 1}")

        judge_elapsed = time.perf_counter() - judge_start
        print(f"Judging complete in {judge_elapsed:.1f}s")

        metrics = compute_metrics(judge_results)
        print_metrics(metrics)

        # Save results
        run_id = f"run_{int(time.time())}"
        output_path = Path(config.output_dir) / run_id
        save_results(
            metrics,
            output_path,
            extra_metadata={
                "dataset": config.dataset,
                "subset": config.subset,
                "question_type": config.question_type,
                "mode": config.mode,
                "top_k": config.top_k,
                "answer_model": config.answer_model,
                "judge_model": config.judge_model,
                "ingest_elapsed_s": round(ingest_elapsed, 1),
                "answer_elapsed_s": round(answer_elapsed, 1),
                "judge_elapsed_s": round(judge_elapsed, 1),
            },
        )

    # Phase 4: Learning loop (if requested)
    if config.learning_rounds > 0 and not config.dry_run:
        from benchmarks.longmemeval.learning_loop import run_learning_loop

        print(f"\nPhase 4: Learning loop ({config.learning_rounds} rounds)...")
        learning_results = run_learning_loop(
            client=client,
            questions=questions,
            hypotheses=hypotheses,
            config=config,
            openai_client=openai_client,
            initial_metrics=metrics,
        )
        return {"metrics": metrics, "learning": learning_results}

    return {
        "metrics": {
            "overall_accuracy": metrics.overall_accuracy,
            "task_averaged_accuracy": metrics.task_averaged_accuracy,
            "per_type": metrics.per_type,
        },
        "hypotheses": hypotheses,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval benchmark for qortex")
    parser.add_argument(
        "--dataset",
        default="longmemeval_s",
        choices=["longmemeval_s", "longmemeval_m", "longmemeval_oracle"],
        help="Dataset variant",
    )
    parser.add_argument("--subset", type=int, help="Number of questions to process")
    parser.add_argument("--question-type", choices=QUESTION_TYPES, help="Filter by question type")
    parser.add_argument("--mode", default="auto", choices=["auto", "graph", "vec"], help="Retrieval mode")
    parser.add_argument("--use-llm-ingest", action="store_true", help="Use LLM for concept extraction")
    parser.add_argument("--top-k", type=int, default=20, help="Number of results to retrieve")
    parser.add_argument("--answer-model", default="gpt-4o-2024-08-06", help="Model for answer generation")
    parser.add_argument("--judge-model", default="gpt-4o-2024-08-06", help="Model for judging")
    parser.add_argument("--learning-rounds", type=int, default=0, help="Number of learning rounds")
    parser.add_argument("--output-dir", default="benchmark_results/longmemeval", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls (test pipeline only)")
    parser.add_argument("--cache-dir", help="HuggingFace cache directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = RunConfig(
        dataset=args.dataset,
        subset=args.subset,
        question_type=args.question_type,
        mode=args.mode,
        use_llm_ingest=args.use_llm_ingest,
        top_k=args.top_k,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        learning_rounds=args.learning_rounds,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        cache_dir=args.cache_dir,
    )

    run_benchmark(config)


if __name__ == "__main__":
    main()
