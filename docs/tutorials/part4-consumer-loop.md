# Part 4: The Consumer Loop

You've projected 150 rules from your drug knowledge graph. They're sitting in a YAML file.

Now what?

Someone needs to *use* these rules. And you need to *measure* whether they actually help. Otherwise you're just generating text that nobody reads.

Enter **buildlog**.

## What buildlog does

buildlog is a system that:

1. **Ingests** rules from external sources (like qortex)
2. **Surfaces** them to AI agents as instructions
3. **Tracks** whether they reduce mistakes
4. **Learns** which rules work via Thompson Sampling

The loop closes: qortex generates rules → buildlog tests them → feedback updates confidence → better rules surface.

Rules aren't static knowledge dumps. They're **hypotheses** about what helps. The consumer loop is how you test them.

## The interop protocol

qortex and buildlog communicate through a simple protocol:

```
~/.qortex/seeds/pending/      ← qortex writes YAML files here
~/.qortex/seeds/processed/    ← buildlog moves files here on success
~/.qortex/seeds/failed/       ← buildlog moves files here on failure
~/.qortex/signals/projections.jsonl  ← event log
```

**Pull model**: buildlog scans `pending/` on its own schedule.

**Push model**: buildlog watches `projections.jsonl` for real-time events.

Either way, the systems are decoupled. qortex doesn't know buildlog's internals. buildlog doesn't know how qortex generated the rules.

## Publishing rules

```python
from qortex.interop import write_seed_to_pending

path = write_seed_to_pending(
    seed_data=result,           # From projection
    persona="drug_safety_rules",
    domain="pharmacology",
    emit_signal=True,           # Notify watchers
)
# Rules are now in ~/.qortex/seeds/pending/drug_safety_rules_2026-02-05T12-00-00.yaml
```

buildlog picks them up automatically. The next time you run `buildlog_gauntlet_rules()`, your drug safety rules are included.

## The feedback loop

```
qortex projects rules
    ↓
buildlog ingests seeds
    ↓
Agent uses rules in a session
    ↓
User logs mistakes/rewards
    ↓
Thompson Sampling updates posteriors
    ↓
High-confidence rules surface more often
    ↓
Repeated Mistake Rate (RMR) measured
```

This is how you know if your rules work. Not by reading them and nodding. By measuring whether they reduce mistakes in practice.

## Why decoupling matters

qortex could have been built as a buildlog plugin. It wasn't.

The decoupling means:
- Any consumer can use qortex projections (not just buildlog)
- qortex doesn't need to know how rules are used
- You can swap consumers without changing qortex
- Multiple consumers can ingest the same rules

The universal schema and interop protocol are the contract. Everything else is implementation detail.

## What you learned

- Rules are hypotheses; the consumer loop tests them
- buildlog ingests rules, surfaces them, tracks effectiveness
- The interop protocol decouples producer (qortex) from consumer (buildlog)
- Thompson Sampling learns which rules actually work
- Repeated Mistake Rate measures real impact

## Next

[Part 5: Pattern Completion](part5-pattern-completion.md): How do you retrieve rules by spreading activation through the graph?
