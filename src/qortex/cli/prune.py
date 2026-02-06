"""CLI command for edge pruning and analysis."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from qortex.cli._errors import handle_error

app = typer.Typer(help="Prune and analyze graph edges.")


@app.command("manifest")
def prune_manifest(
    manifest_path: Path = typer.Argument(..., help="Path to manifest JSON file"),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Show what would be pruned without modifying"
    ),
    min_confidence: float = typer.Option(
        0.55, "--min-confidence", "-c", help="Confidence floor (drop below this)"
    ),
    min_evidence: int = typer.Option(
        8, "--min-evidence", "-e", help="Minimum evidence tokens"
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Output path for pruned manifest"
    ),
    show_dropped: bool = typer.Option(
        False, "--show-dropped", help="Show details of dropped edges"
    ),
) -> None:
    """Prune edges in a saved manifest.

    Applies the 6-step pruning pipeline:
    1. Minimum evidence length
    2. Confidence floor
    3. Jaccard deduplication
    4. Competing relation resolution
    5. Isolated weak edge removal
    6. Structural/causal layer tagging

    Examples:
        qortex prune manifest ch05.manifest.json --dry-run
        qortex prune manifest ch05.manifest.json -c 0.6 -o pruned.json
    """
    from qortex.core.pruning import PruningConfig, prune_edges

    if not manifest_path.exists():
        handle_error(f"Manifest not found: {manifest_path}")

    try:
        data = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        handle_error(f"Invalid JSON: {e}")

    edges = data.get("edges", [])
    if not edges:
        typer.echo("No edges in manifest.")
        raise typer.Exit(0)

    config = PruningConfig(
        min_evidence_tokens=min_evidence,
        confidence_floor=min_confidence,
        enabled=not dry_run,  # If dry_run, we still compute stats
    )

    # Run pruning
    if dry_run:
        from qortex.core.pruning import prune_edges_dry_run
        result = prune_edges_dry_run(edges, PruningConfig(
            min_evidence_tokens=min_evidence,
            confidence_floor=min_confidence,
        ))
        typer.echo("\n[DRY RUN - no changes made]\n")
    else:
        result = prune_edges(edges, config)

    typer.echo(result.summary())

    if show_dropped and not dry_run:
        # Show what was dropped (compare input to output)
        output_ids = {(e["source_id"], e["target_id"], e["relation_type"]) for e in result.edges}
        dropped = [e for e in edges if (e["source_id"], e["target_id"], e["relation_type"]) not in output_ids]
        if dropped:
            typer.echo("\nDropped edges:")
            for e in dropped[:20]:  # Limit output
                typer.echo(f"  {e['source_id'].split(':')[-1]} -[{e['relation_type']}]-> {e['target_id'].split(':')[-1]} (conf={e.get('confidence', '?')})")
            if len(dropped) > 20:
                typer.echo(f"  ... and {len(dropped) - 20} more")

    if not dry_run and output:
        # Save pruned manifest
        data["edges"] = result.edges
        data["_pruning"] = {
            "input_count": result.input_count,
            "output_count": result.output_count,
            "config": {
                "min_evidence_tokens": config.min_evidence_tokens,
                "confidence_floor": config.confidence_floor,
            },
        }
        output.write_text(json.dumps(data, indent=2, default=str))
        typer.echo(f"\nPruned manifest saved to: {output}")


@app.command("stats")
def prune_stats(
    manifest_path: Path = typer.Argument(..., help="Path to manifest JSON file"),
) -> None:
    """Show edge statistics without pruning.

    Displays confidence distribution, layer breakdown, and quality metrics.
    """
    if not manifest_path.exists():
        handle_error(f"Manifest not found: {manifest_path}")

    try:
        data = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        handle_error(f"Invalid JSON: {e}")

    edges = data.get("edges", [])
    concepts = data.get("concepts", [])

    if not edges:
        typer.echo("No edges in manifest.")
        raise typer.Exit(0)

    # Confidence distribution
    confidences = [e.get("confidence", 0) for e in edges]
    buckets = {"<0.55": 0, "0.55-0.70": 0, "0.70-0.85": 0, ">=0.85": 0}
    for c in confidences:
        if c < 0.55:
            buckets["<0.55"] += 1
        elif c < 0.70:
            buckets["0.55-0.70"] += 1
        elif c < 0.85:
            buckets["0.70-0.85"] += 1
        else:
            buckets[">=0.85"] += 1

    typer.echo(f"Edge Statistics for: {manifest_path.name}")
    typer.echo(f"  Total edges: {len(edges)}")
    typer.echo(f"  Total concepts: {len(concepts)}")
    typer.echo(f"  Edge density: {len(edges) / len(concepts):.2f} edges/concept" if concepts else "")

    typer.echo("\nConfidence distribution:")
    for bucket, count in buckets.items():
        pct = count / len(edges) * 100 if edges else 0
        bar = "█" * int(pct / 5)
        typer.echo(f"  {bucket:>10}: {count:4} ({pct:5.1f}%) {bar}")

    # Relation type breakdown
    from collections import Counter
    rel_types = Counter(e.get("relation_type", "unknown") for e in edges)
    typer.echo("\nRelation types:")
    for rel, count in rel_types.most_common():
        typer.echo(f"  {rel}: {count}")

    # Layer breakdown (if tagged)
    layers = Counter(e.get("layer") for e in edges if e.get("layer"))
    if layers:
        typer.echo("\nLayers:")
        for layer, count in layers.items():
            typer.echo(f"  {layer}: {count}")

    # Evidence stats (check both top-level and properties.source_text)
    def get_source_text(e: dict) -> str:
        return e.get("source_text") or e.get("properties", {}).get("source_text", "")

    evidence_lengths = [len(get_source_text(e).split()) for e in edges]
    with_evidence = sum(1 for e in edges if get_source_text(e))
    typer.echo(f"\nEvidence:")
    typer.echo(f"  Edges with source_text: {with_evidence}/{len(edges)}")
    if evidence_lengths:
        typer.echo(f"  Avg evidence length: {sum(evidence_lengths)/len(evidence_lengths):.1f} words")
