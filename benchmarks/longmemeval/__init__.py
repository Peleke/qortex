"""LongMemEval benchmark integration for qortex.

ICLR 2025 benchmark: 500 questions testing 5 long-term memory abilities.
Uses the official evaluation methodology (GPT-4o judge, binary scoring).

Usage:
    python -m benchmarks.longmemeval.runner --subset 10
    python -m benchmarks.longmemeval.runner --dataset longmemeval_s
    python -m benchmarks.longmemeval.runner --dataset longmemeval_s --learning-rounds 5
"""
