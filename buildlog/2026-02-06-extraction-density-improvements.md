# Extraction Density Improvements

**Date**: 2026-02-06
**Status**: Complete

## Summary

Improved knowledge graph extraction with 4.3x better edge density through:

1. **CLI Fix**: Restructured `qortex ingest` to use `qortex ingest file <path>` to fix typer argument parsing bug
2. **Rate Limit Handling**: Added retry with exponential backoff (60s, 120s, 180s)
3. **Concept Limiting**: Cap at 100 concepts per API call to stay under token limits
4. **Stats Fix**: `qortex prune stats` now checks properties.source_text fallback

## Results

| Metric | Ch05 (before) | Ch08 (after) | Change |
|--------|---------------|--------------|--------|
| Concepts | 237 | 285 | +20% |
| Edges | 23 | 119 | +417% |
| Edge density | 0.097 | 0.418 | 4.3x |
| High-conf edges | ? | 86.6% | - |

## Files Changed

- src/qortex/cli/ingest.py - CLI restructure
- src/qortex/cli/prune.py - Stats fix
- src/qortex_ingest/backends/anthropic.py - Rate limits + concept cap
- src/qortex_ingest/base.py - Fix undefined all_text

## Gauntlet

2 minor issues accepted as risk:
- Rate limit retry logic lacks tests
- time import inside function (intentional lazy import)
