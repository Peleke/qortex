#!/usr/bin/env python3
"""Demo: 7-step round-trip buildlog → qortex KG → derived rules → buildlog → posteriors shift.

Produces article artifacts:
- V1-V5: Cypher queries for Memgraph Lab screenshots
- C1-C4: Plotly charts (saved as HTML + PNG)
- Tables: Before/after posterior data

Requirements:
- Memgraph running: docker compose --profile local-graph up -d  (from qortex/docker/)
- buildlog v0.20.0+ installed with emissions data
- qortex installed in editable mode

Usage:
    python scripts/demo_round_trip.py [--dry-run] [--skip-ingest] [--output-dir ./demo_output]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 0: Setup + args
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="7-step round-trip demo")
    p.add_argument("--dry-run", action="store_true", help="Show what would happen, don't ingest")
    p.add_argument("--skip-ingest", action="store_true", help="Skip emission ingestion (reuse existing graph)")
    p.add_argument("--output-dir", type=Path, default=Path("demo_output"), help="Output directory for artifacts")
    p.add_argument("--embed", action="store_true", help="Embed concepts (needed for pattern 2)")
    return p.parse_args()


def banner(step: int, title: str):
    print(f"\n{'='*60}")
    print(f"  Step {step}: {title}")
    print(f"{'='*60}\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}  # Collect data for charts

    # ---------------------------------------------------------------------------
    # Step 1: Show buildlog emissions
    # ---------------------------------------------------------------------------
    banner(1, "buildlog emissions")

    emissions_dir = Path("~/.buildlog/emissions").expanduser()
    processed = emissions_dir / "processed"
    pending = emissions_dir / "pending"

    counts = {"mistake_manifest": 0, "reward_signal": 0, "session_summary": 0, "learned_rules": 0}
    if processed.exists():
        for f in processed.glob("*.json"):
            for prefix in counts:
                if f.name.startswith(prefix):
                    counts[prefix] += 1
                    break

    total = sum(counts.values())
    print(f"Emissions directory: {emissions_dir}")
    print(f"Total artifacts: {total}")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    results["step1"] = {"total": total, **counts}

    # ---------------------------------------------------------------------------
    # Step 2: Ingest into Memgraph
    # ---------------------------------------------------------------------------
    banner(2, "Ingest emissions into KG")

    from qortex.ingest_emissions import (
        aggregate_emissions,
        bridge_gauntlet_credits,
        bridge_gauntlet_rules,
        build_manifest,
        read_gauntlet_credits,
        resolve_historical_targets,
    )

    if not args.skip_ingest:
        result = aggregate_emissions(emissions_dir, include_pending=True, include_processed=True)
        print(f"Aggregated: {result.files_processed} files → {len(result.concepts)} concepts, {len(result.edges)} edges, {len(result.rules)} rules")

        # Resolve historical targets
        resolved = resolve_historical_targets(result)
        if resolved:
            print(f"Resolved {resolved} historical edge targets")

        manifest = build_manifest(result)

        # Bridge gauntlet rules (cross-domain)
        bridge_concepts, bridge_edges = bridge_gauntlet_rules()
        manifest.concepts.extend(bridge_concepts)
        manifest.edges.extend(bridge_edges)
        print(f"Gauntlet bridge: {len(bridge_concepts)} concepts, {len(bridge_edges)} edges")

        results["step2"] = {
            "concepts": len(manifest.concepts),
            "edges": len(manifest.edges),
            "rules": len(manifest.rules),
            "bridge_concepts": len(bridge_concepts),
            "bridge_edges": len(bridge_edges),
        }

        if not args.dry_run:
            from qortex.cli._config import get_config
            from qortex.core.backend import MemgraphBackend, MemgraphCredentials

            config = get_config()
            creds = MemgraphCredentials.from_tuple(config.memgraph_credentials.auth_tuple)
            graph = MemgraphBackend(uri=config.get_memgraph_uri(), credentials=creds)
            graph.connect()

            existing = graph.get_domain(manifest.domain)
            if not existing:
                graph.create_domain(manifest.domain, "buildlog emissions")
            graph.ingest_manifest(manifest)
            print(f"Loaded into Memgraph: {len(manifest.concepts)} concepts, {len(manifest.edges)} edges")

            # Embed if requested (needed for pattern 2)
            if args.embed:
                from qortex.cli.ingest import _embed_manifest_concepts
                embedded = _embed_manifest_concepts(manifest, graph)
                print(f"Embedded {embedded} concepts")
        else:
            graph = None
            print("[Dry run — skipped Memgraph load]")
    else:
        print("[Skipped — using existing graph]")
        graph = None
        results["step2"] = {"skipped": True}

    # ---------------------------------------------------------------------------
    # Step 2b: Bridge gauntlet credits
    # ---------------------------------------------------------------------------
    banner("2b", "Ingest gauntlet_credits (performance signal)")

    credit_stats = read_gauntlet_credits()
    print(f"Rules with credits: {len(credit_stats)}")
    for rid, s in credit_stats.items():
        print(f"  {rid}: {s.total_credits} credits, co-occurs with {len(s.co_occurring_rules)} rules")

    credit_nodes, credit_edges = bridge_gauntlet_credits()
    print(f"Enriched nodes: {len(credit_nodes)}")
    print(f"Co-occurrence edges: {len(credit_edges)}")

    if graph and not args.dry_run:
        from qortex.core.models import IngestionManifest, SourceMetadata
        credit_manifest = IngestionManifest(
            source=SourceMetadata(
                id="buildlog:credits", name="gauntlet_credits",
                source_type="structured", path_or_url="~/.buildlog/buildlog.db",
            ),
            domain="buildlog",
            concepts=credit_nodes,
            edges=credit_edges,
        )
        graph.ingest_manifest(credit_manifest)
        print("Credit data loaded into Memgraph")

    results["step2b"] = {
        "credited_rules": len(credit_stats),
        "enriched_nodes": len(credit_nodes),
        "cooccurrence_edges": len(credit_edges),
    }

    # ---------------------------------------------------------------------------
    # Step 3: Derive rules from KG
    # ---------------------------------------------------------------------------
    banner(3, "Cross-reference KG → derive new rules")

    from qortex.derive import derive_rules_from_graph, export_derived_rules

    if graph:
        derived = derive_rules_from_graph(
            backend=graph,
            credit_stats=credit_stats,
            min_cluster_size=2,
        )
    else:
        derived = []
        print("[No graph connection — skipping derivation]")

    print(f"Derived rules: {len(derived)}")
    for r in derived:
        print(f"  [{r.derivation_type}] {r.text[:80]}...")
        print(f"    confidence={r.confidence}, category={r.category}")
        if r.provenance.get("generation_method"):
            print(f"    method={r.provenance['generation_method']}")

    results["step3"] = {
        "derived_count": len(derived),
        "by_type": {},
    }
    for r in derived:
        t = r.derivation_type
        results["step3"]["by_type"][t] = results["step3"]["by_type"].get(t, 0) + 1

    # ---------------------------------------------------------------------------
    # Step 4: Export as buildlog seeds
    # ---------------------------------------------------------------------------
    banner(4, "Export rules in buildlog seed format")

    if derived:
        seed_data = export_derived_rules(derived)
        seed_path = args.output_dir / "qortex_derived_seed.yaml"

        import yaml
        seed_path.write_text(yaml.dump(seed_data, default_flow_style=False, sort_keys=False))
        print(f"Seed written to: {seed_path}")
        print(f"  Rules: {seed_data['metadata']['rule_count']}")
        print(f"  Patterns: {seed_data['metadata']['derivation_patterns']}")

        # Also write to qortex pending for real ingestion
        from qortex.interop import write_seed_to_pending
        pending_path = write_seed_to_pending(seed_data, persona="qortex_derived", domain="buildlog")
        print(f"  Also written to pending: {pending_path}")
    else:
        print("[No derived rules to export]")

    results["step4"] = {"seed_rules": len(derived)}

    # ---------------------------------------------------------------------------
    # Steps 5-7: buildlog side (instructions for manual execution)
    # ---------------------------------------------------------------------------
    banner(5, "Ingest seeds into buildlog (manual)")
    print("Run: buildlog ingest-seeds")
    print("This reads from ~/.qortex/seeds/pending/ and loads into gauntlet_rules")

    banner(6, "Gauntlet run with new rules (manual)")
    print("Run: buildlog gauntlet-loop target=src/")
    print("The derived rules should appear in the gauntlet prompt")
    print("If a reviewer cites a derived rule → it gets credited")

    banner(7, "Posteriors shift (manual)")
    print("Run: buildlog log-reward outcome=accepted")
    print("Check posteriors: the derived rule's alpha should increment")

    # ---------------------------------------------------------------------------
    # Cypher queries for Memgraph Lab screenshots
    # ---------------------------------------------------------------------------
    banner("V", "Cypher queries for Memgraph Lab")

    cypher_queries = {
        "V1_emission_subgraph": """
MATCH (m:Concept {domain: "buildlog"})-[r]->(t)
WHERE m.source_id STARTS WITH "buildlog:"
RETURN m, r, t LIMIT 100""",

        "V2_credit_cooccurrence": """
MATCH (a:Concept)-[r:CORRELATES_WITH]->(b:Concept)
RETURN a, r, b""",

        "V3_uncovered_mistakes": """
MATCH (m:Concept)
WHERE m.source_id STARTS WITH "buildlog:mistake"
AND NOT EXISTS {
  MATCH (m)-[:CHALLENGES]->(r:Concept)
  WHERE r.credit_count > 0
}
RETURN m LIMIT 50""",

        "V4_derived_provenance": """
MATCH (d:Concept {source_id: "qortex:derived"})-[r:DERIVED_FROM]->(source)
RETURN d, r, source""",

        "V5_cross_domain_bridge": """
MATCH (d:Concept)-[r:DERIVED_FROM]->(source)
WHERE d.domain <> source.domain
RETURN d, r, source""",
    }

    for name, query in cypher_queries.items():
        print(f"\n--- {name} ---")
        print(query.strip())

    # Save queries to file
    queries_path = args.output_dir / "cypher_queries.json"
    queries_path.write_text(json.dumps(cypher_queries, indent=2))
    print(f"\nQueries saved to: {queries_path}")

    # ---------------------------------------------------------------------------
    # Save results summary
    # ---------------------------------------------------------------------------
    summary_path = args.output_dir / "demo_results.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to: {summary_path}")

    # ---------------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------------
    if graph:
        graph.disconnect()

    print(f"\n{'='*60}")
    print("  Demo complete. Artifacts in: " + str(args.output_dir))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
