"""CLI command for ingesting content into the knowledge graph."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import typer

from qortex.cli._errors import handle_error

app = typer.Typer(help="Ingest content into the knowledge graph.", no_args_is_help=True)


BackendChoice = Literal["anthropic", "ollama", "auto"]


def _serialize_manifest(manifest) -> dict:
    """Serialize manifest to JSON-safe dict.

    Handles enum values and other non-serializable types.
    """
    data = asdict(manifest)

    # Convert enum values in edges
    for edge in data.get("edges", []):
        if hasattr(edge.get("relation_type"), "value"):
            edge["relation_type"] = edge["relation_type"].value
        elif isinstance(edge.get("relation_type"), str):
            pass  # Already a string
        else:
            # Handle RelationType enum stored as object
            rel = edge.get("relation_type")
            if rel is not None:
                edge["relation_type"] = str(rel)

    return data


def _save_manifest_to_file(manifest, output_path: Path) -> None:
    """Save manifest to JSON file."""
    data = _serialize_manifest(manifest)
    data["_saved_at"] = datetime.now(UTC).isoformat()
    output_path.write_text(json.dumps(data, indent=2, default=str))


@app.command("file")
def ingest_file(
    path: Path = typer.Argument(..., help="Path to file to ingest"),
    domain: str = typer.Option(
        None, "--domain", "-d", help="Domain name (default: auto-suggested)"
    ),
    backend: str = typer.Option(
        "auto",
        "--backend",
        "-b",
        help="Extraction backend: anthropic, ollama, or auto",
    ),
    model: str = typer.Option(None, "--model", "-m", help="Model override for extraction backend"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be extracted without saving"
    ),
    save_manifest: Path = typer.Option(
        None,
        "--save-manifest",
        "-o",
        help="Save extraction manifest to JSON file (always saves, even on graph failure)",
    ),
) -> None:
    """Ingest a file into the knowledge graph.

    Extracts concepts, relations, and rules from the file using LLM,
    then stores them in the graph backend.

    Use --save-manifest to persist extraction results to disk. This is useful
    for recovery if graph save fails, or for offline inspection.

    Examples:
        qortex ingest file chapter.txt --domain software_design
        qortex ingest file book.pdf --backend ollama --model llama3.2
        qortex ingest file notes.md --dry-run
        qortex ingest file chapter.txt -d my_domain -o manifest.json
    """
    if not path.exists():
        handle_error(f"File not found: {path}")

    # Get extraction backend
    from qortex.ingest.backends import get_extraction_backend

    try:
        prefer = None if backend == "auto" else backend
        llm = get_extraction_backend(prefer=prefer, model=model)
        backend_name = type(llm).__name__
    except ValueError as e:
        handle_error(str(e))
    except ImportError as e:
        handle_error(f"Backend dependency missing: {e}")

    typer.echo(f"Using extraction backend: {backend_name}")

    # Cost warning gate for paid backends
    if backend_name == "AnthropicExtractionBackend" and not dry_run:
        file_size_kb = path.stat().st_size / 1024
        est_cost = round(file_size_kb / 57 * 0.60, 2)  # ~$0.60 per 57KB chapter
        typer.echo(f"\nEstimated cost: ~${est_cost:.2f} (file: {file_size_kb:.1f}KB)")
        if not typer.confirm("Proceed with extraction?"):
            raise typer.Exit(0)

    # Get appropriate ingestor
    from qortex.ingest.base import Source
    from qortex.ingest.text import TextIngestor

    suffix = path.suffix.lower()
    if suffix in (".txt", ".text"):
        ingestor = TextIngestor(llm=llm)
        source_type = "text"
    elif suffix in (".md", ".markdown"):
        from qortex.ingest.markdown import MarkdownIngestor

        ingestor = MarkdownIngestor(llm=llm)
        source_type = "markdown"
    elif suffix == ".pdf":
        try:
            from qortex.ingest.pdf import PDFIngestor

            ingestor = PDFIngestor(llm=llm)
            source_type = "pdf"
        except ImportError:
            handle_error("PDF support requires pymupdf. Install with: pip install pymupdf")
    else:
        # Default to text
        ingestor = TextIngestor(llm=llm)
        source_type = "text"

    source = Source(
        path=path,
        source_type=source_type,
        name=path.stem,
    )

    typer.echo(f"Ingesting {path.name}...")

    # Run ingestion
    try:
        manifest = ingestor.ingest(source, domain=domain)
    except Exception as e:
        handle_error(f"Ingestion failed: {e}")

    # Report results
    typer.echo(f"\nDomain: {manifest.domain}")
    typer.echo(f"Concepts extracted: {len(manifest.concepts)}")
    typer.echo(f"Relations extracted: {len(manifest.edges)}")
    typer.echo(f"Rules extracted: {len(manifest.rules)}")
    typer.echo(f"Code examples extracted: {len(manifest.examples)}")

    # Always save manifest if path provided (before graph save attempt)
    if save_manifest:
        try:
            _save_manifest_to_file(manifest, save_manifest)
            typer.echo(f"\nManifest saved to: {save_manifest}")
        except Exception as e:
            typer.echo(f"Warning: Failed to save manifest: {e}", err=True)

    if dry_run:
        typer.echo("\n[Dry run - not saving to graph]")
        typer.echo("\nSample concepts:")
        for c in manifest.concepts[:5]:
            typer.echo(f"  - {c.name}: {c.description[:60]}...")
        if manifest.edges:
            typer.echo("\nSample relations:")
            for e in manifest.edges[:5]:
                rel_type = (
                    e.relation_type.value if hasattr(e.relation_type, "value") else e.relation_type
                )
                typer.echo(f"  - {e.source_id} --{rel_type}--> {e.target_id}")
        if manifest.rules:
            typer.echo("\nSample rules:")
            for r in manifest.rules[:3]:
                typer.echo(f"  - {r.text[:80]}...")
        if manifest.examples:
            typer.echo("\nSample code examples:")
            for ex in manifest.examples[:3]:
                code_preview = ex.code[:60].replace("\n", " ")
                typer.echo(f"  - [{ex.language}] {code_preview}...")
                if ex.concept_ids:
                    typer.echo(f"    Links: {', '.join(ex.concept_ids[:3])}")
        return

    # Save to graph - connect to Memgraph
    from qortex.cli._config import get_config
    from qortex.core.backend import MemgraphBackend, MemgraphCredentials

    config = get_config()
    try:
        creds = MemgraphCredentials.from_tuple(config.memgraph_credentials.auth_tuple)
        graph_backend = MemgraphBackend(
            uri=config.get_memgraph_uri(),
            credentials=creds,
        )
        graph_backend.connect()
    except Exception as e:
        uri = config.get_memgraph_uri()
        # Auto-save manifest on connection failure if not already saved
        if not save_manifest:
            fallback_path = path.with_suffix(".manifest.json")
            try:
                _save_manifest_to_file(manifest, fallback_path)
                typer.echo(f"\nManifest auto-saved to: {fallback_path}", err=True)
            except Exception:
                pass
        handle_error(
            f"Could not connect to Memgraph at {uri}.\n"
            f"Start it with: qortex infra up\n"
            f"Original error: {e}"
        )

    try:
        # Create domain if needed
        existing = graph_backend.get_domain(manifest.domain)
        if not existing:
            graph_backend.create_domain(
                manifest.domain,
                f"Domain for {path.name}",
            )

        # Ingest manifest
        graph_backend.ingest_manifest(manifest)

        typer.echo("\nSaved to graph backend.")
        typer.echo("View with: qortex inspect domains")
        typer.echo("Visualize with: qortex viz open")

    except Exception as e:
        # Auto-save manifest on graph failure if not already saved
        if not save_manifest:
            fallback_path = path.with_suffix(".manifest.json")
            try:
                _save_manifest_to_file(manifest, fallback_path)
                typer.echo(f"\nManifest auto-saved to: {fallback_path}", err=True)
            except Exception:
                pass
        handle_error(f"Failed to save to graph: {e}")
    finally:
        graph_backend.disconnect()


def _embed_manifest_concepts(
    manifest: "IngestionManifest",
    graph_backend: Any,
    *,
    embedding_model: Any = None,
    vector_index: Any = None,
) -> int:
    """Generate embeddings for manifest concepts and write to vec index + graph backend.

    Args:
        manifest: The ingestion manifest with concepts to embed.
        graph_backend: Graph backend for storing embeddings alongside nodes.
        embedding_model: Optional pre-built embedding model (for testing).
        vector_index: Optional pre-built vec index (for testing).

    Returns:
        Number of concepts embedded.
    """
    import os
    from pathlib import Path as _Path

    if embedding_model is None:
        try:
            from qortex.vec.embeddings import SentenceTransformerEmbedding

            embedding_model = SentenceTransformerEmbedding()
        except ImportError:
            typer.echo("  Skipping embed: sentence-transformers not installed (pip install qortex[vec])")
            return 0

    vec_backend = os.environ.get("QORTEX_VEC", "sqlite")
    if vector_index is None:
        if vec_backend == "sqlite":
            try:
                from qortex.vec.index import SqliteVecIndex

                vec_path = _Path("~/.qortex/vectors.db").expanduser()
                vector_index = SqliteVecIndex(db_path=str(vec_path), dimensions=embedding_model.dimensions)
            except ImportError:
                from qortex.vec.index import NumpyVectorIndex

                vector_index = NumpyVectorIndex(dimensions=embedding_model.dimensions)
        else:
            from qortex.vec.index import NumpyVectorIndex

            vector_index = NumpyVectorIndex(dimensions=embedding_model.dimensions)

    # Embed concept descriptions in batches
    concepts = [c for c in manifest.concepts if c.description]
    if not concepts:
        typer.echo("  No concepts with descriptions to embed.")
        return 0

    BATCH_SIZE = 64
    total_embedded = 0
    for i in range(0, len(concepts), BATCH_SIZE):
        batch = concepts[i : i + BATCH_SIZE]
        texts = [c.description for c in batch]
        ids = [c.id for c in batch]

        embeddings = embedding_model.embed(texts)
        vector_index.add(ids, embeddings)

        # Also write embeddings to graph backend for PPR node lookups
        for cid, emb in zip(ids, embeddings):
            graph_backend.add_embedding(cid, emb)

        total_embedded += len(batch)

    vector_index.persist()
    typer.echo(f"  Embedded {total_embedded} concepts into vec index ({vec_backend}).")
    return total_embedded


@app.command("load")
def load_manifest(
    manifest_path: Path = typer.Argument(..., help="Path to manifest JSON file"),
    embed: bool = typer.Option(
        False,
        "--embed",
        help="Generate and store vector embeddings for concepts (requires sentence-transformers).",
    ),
) -> None:
    """Load a previously saved manifest into the graph.

    Use this to retry graph ingestion from a saved manifest
    without re-running LLM extraction.

    With --embed, also generates vector embeddings for each concept
    and writes them to the vec index (sqlite-vec or numpy). This is
    required for vec search and graph-mode (PPR) retrieval to find
    seed nodes from these concepts.

    Example:
        qortex ingest load manifest.json
        qortex ingest load manifest.json --embed
    """
    if not manifest_path.exists():
        handle_error(f"Manifest file not found: {manifest_path}")

    try:
        data = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        handle_error(f"Invalid JSON in manifest: {e}")

    # Reconstruct manifest from JSON
    from qortex.core.models import (
        ConceptEdge,
        ConceptNode,
        ExplicitRule,
        IngestionManifest,
        RelationType,
        SourceMetadata,
    )

    try:
        source = SourceMetadata(**data["source"])
        concepts = [ConceptNode(**c) for c in data["concepts"]]

        edges = []
        for e in data.get("edges", []):
            # Convert string relation_type back to enum
            rel_type = e["relation_type"]
            if isinstance(rel_type, str):
                rel_type = RelationType(rel_type)
            edges.append(
                ConceptEdge(
                    source_id=e["source_id"],
                    target_id=e["target_id"],
                    relation_type=rel_type,
                    confidence=e.get("confidence", 1.0),
                    bidirectional=e.get("bidirectional", False),
                    properties=e.get("properties", {}),
                )
            )

        rules = [ExplicitRule(**r) for r in data.get("rules", [])]

        manifest = IngestionManifest(
            source=source,
            domain=data["domain"],
            concepts=concepts,
            edges=edges,
            rules=rules,
        )
    except (KeyError, TypeError, ValueError) as e:
        handle_error(f"Invalid manifest format: {e}")

    typer.echo(f"Loaded manifest: {manifest.domain}")
    typer.echo(f"  Concepts: {len(manifest.concepts)}")
    typer.echo(f"  Edges: {len(manifest.edges)}")
    typer.echo(f"  Rules: {len(manifest.rules)}")

    # Save to graph
    from qortex.cli._config import get_config
    from qortex.core.backend import MemgraphBackend, MemgraphCredentials

    config = get_config()
    try:
        creds = MemgraphCredentials.from_tuple(config.memgraph_credentials.auth_tuple)
        graph_backend = MemgraphBackend(
            uri=config.get_memgraph_uri(),
            credentials=creds,
        )
        graph_backend.connect()
    except Exception as e:
        uri = config.get_memgraph_uri()
        handle_error(
            f"Could not connect to Memgraph at {uri}.\n"
            f"Start it with: qortex infra up\n"
            f"Original error: {e}"
        )

    try:
        existing = graph_backend.get_domain(manifest.domain)
        if not existing:
            graph_backend.create_domain(
                manifest.domain,
                f"Domain loaded from {manifest_path.name}",
            )

        graph_backend.ingest_manifest(manifest)

        typer.echo("\nSaved to graph backend.")

        if embed:
            _embed_manifest_concepts(manifest, graph_backend)

        typer.echo("View with: qortex inspect domains")

    except Exception as e:
        handle_error(f"Failed to save to graph: {e}")
    finally:
        graph_backend.disconnect()


@app.command("emissions")
def ingest_emissions(
    emissions_dir: Path = typer.Option(
        "~/.buildlog/emissions",
        "--dir",
        "-d",
        help="Root emissions directory (contains pending/, processed/)",
    ),
    domain: str = typer.Option(
        "buildlog",
        "--domain",
        help="Domain name for the ingested data",
    ),
    include_pending: bool = typer.Option(
        True,
        "--pending/--no-pending",
        help="Include pending/ artifacts",
    ),
    include_processed: bool = typer.Option(
        True,
        "--processed/--no-processed",
        help="Include processed/ artifacts",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show aggregation stats without loading into graph",
    ),
    save_manifest: Path = typer.Option(
        None,
        "--save-manifest",
        "-o",
        help="Save aggregated manifest to JSON file",
    ),
    bridge: bool = typer.Option(
        True,
        "--bridge/--no-bridge",
        help="Bridge gauntlet rules to design pattern domains (cross-domain edges)",
    ),
    buildlog_db: Path = typer.Option(
        "~/.buildlog/buildlog.db",
        "--db",
        help="Path to buildlog SQLite database (for --bridge and --resolve-rules)",
    ),
    resolve_rules: bool = typer.Option(
        True,
        "--resolve-rules/--no-resolve-rules",
        help="Resolve historical skill ID edge targets to gauntlet_rule:{id} format",
    ),
) -> None:
    """Ingest buildlog emission artifacts into the knowledge graph.

    Reads mistake manifests, reward signals, session summaries, and learned
    rules from buildlog's emissions directory. Aggregates and deduplicates
    concepts and edges, then loads into Memgraph.

    With --bridge (default), also reads gauntlet rules from buildlog's DB and
    creates cross-domain edges linking experiential data to design pattern
    domains (observer_pattern, implementation_hiding, etc.).

    No LLM calls required — emission data is already structured.

    Examples:
        qortex ingest emissions
        qortex ingest emissions --dry-run
        qortex ingest emissions --no-bridge  # skip gauntlet bridging
        qortex ingest emissions --dir ~/.buildlog/emissions -o emissions_manifest.json
    """
    from qortex.ingest_emissions import (
        aggregate_emissions,
        bridge_gauntlet_rules,
        build_manifest,
        resolve_historical_targets,
    )

    expanded = Path(emissions_dir).expanduser()
    if not expanded.exists():
        handle_error(f"Emissions directory not found: {expanded}")

    typer.echo(f"Scanning emissions in {expanded}...")
    result = aggregate_emissions(
        emissions_dir=expanded,
        include_pending=include_pending,
        include_processed=include_processed,
    )

    typer.echo("\nAggregation complete:")
    typer.echo(f"  Files processed: {result.files_processed}")
    typer.echo(f"  Files failed:    {result.files_failed}")
    typer.echo(f"  Concepts:        {len(result.concepts)}")
    typer.echo(f"  Edges:           {len(result.edges)}")
    typer.echo(f"  Rules:           {len(result.rules)}")
    typer.echo("\n  By type:")
    for atype, count in sorted(result.by_type.items()):
        if count > 0:
            typer.echo(f"    {atype}: {count}")

    # Resolve historical skill IDs to gauntlet_rule:{id} targets
    if resolve_rules:
        db_expanded = Path(buildlog_db).expanduser()
        resolved_count = resolve_historical_targets(result, db_path=db_expanded)
        if resolved_count > 0:
            typer.echo(f"\n  Resolved {resolved_count} historical edge targets to gauntlet_rule:{{id}} format")

    if result.files_processed == 0:
        typer.echo("\nNo emission artifacts found.")
        return

    manifest = build_manifest(result, domain=domain)

    # Bridge gauntlet rules to design pattern domains
    if bridge:
        db_expanded = Path(buildlog_db).expanduser()
        bridge_concepts, bridge_edges = bridge_gauntlet_rules(db_path=db_expanded)
        if bridge_concepts:
            manifest.concepts.extend(bridge_concepts)
            manifest.edges.extend(bridge_edges)
            typer.echo("\n  Gauntlet bridge:")
            typer.echo(f"    Bridge concepts: {len(bridge_concepts)}")
            typer.echo(f"    Bridge edges:    {len(bridge_edges)}")

    # Save manifest if requested
    if save_manifest:
        try:
            _save_manifest_to_file(manifest, save_manifest)
            typer.echo(f"\nManifest saved to: {save_manifest}")
        except Exception as e:
            typer.echo(f"Warning: Failed to save manifest: {e}", err=True)

    if dry_run:
        typer.echo("\n[Dry run — not loading into graph]")
        typer.echo("\nSample concepts:")
        for c in list(result.concepts.values())[:8]:
            typer.echo(f"  - {c.name}: {c.description[:70]}")
        if result.edges:
            typer.echo("\nSample edges:")
            for e in result.edges[:8]:
                rel = e.relation_type.value if hasattr(e.relation_type, "value") else e.relation_type
                typer.echo(f"  - {e.source_id} --{rel}--> {e.target_id}")
        if result.rules:
            typer.echo("\nSample rules:")
            for r in result.rules[:5]:
                typer.echo(f"  - [{r.category}] {r.text[:70]}...")
        return

    # Load into Memgraph
    from qortex.cli._config import get_config
    from qortex.core.backend import MemgraphBackend, MemgraphCredentials

    config = get_config()
    try:
        creds = MemgraphCredentials.from_tuple(config.memgraph_credentials.auth_tuple)
        graph_backend = MemgraphBackend(
            uri=config.get_memgraph_uri(),
            credentials=creds,
        )
        graph_backend.connect()
    except Exception as e:
        uri = config.get_memgraph_uri()
        if not save_manifest:
            fallback = Path(f"emissions_manifest_{domain}.json")
            try:
                _save_manifest_to_file(manifest, fallback)
                typer.echo(f"\nManifest auto-saved to: {fallback}", err=True)
            except Exception:
                pass
        handle_error(
            f"Could not connect to Memgraph at {uri}.\n"
            f"Start it with: qortex infra up\n"
            f"Original error: {e}"
        )

    try:
        existing = graph_backend.get_domain(manifest.domain)
        if not existing:
            graph_backend.create_domain(
                manifest.domain,
                "Buildlog emission data — mistakes, rewards, sessions, rules",
            )

        graph_backend.ingest_manifest(manifest)

        typer.echo(f"\nLoaded into graph: {len(manifest.concepts)} concepts, {len(manifest.edges)} edges, {len(manifest.rules)} rules")
        typer.echo("View with: qortex inspect domains")
        typer.echo("Visualize: open http://localhost:3000 (Memgraph Lab)")

    except Exception as e:
        if not save_manifest:
            fallback = Path(f"emissions_manifest_{domain}.json")
            try:
                _save_manifest_to_file(manifest, fallback)
                typer.echo(f"\nManifest auto-saved to: {fallback}", err=True)
            except Exception:
                pass
        handle_error(f"Failed to save to graph: {e}")
    finally:
        graph_backend.disconnect()
