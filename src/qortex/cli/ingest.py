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


@app.command("load")
def load_manifest(
    manifest_path: Path = typer.Argument(..., help="Path to manifest JSON file"),
) -> None:
    """Load a previously saved manifest into the graph.

    Use this to retry graph ingestion from a saved manifest
    without re-running LLM extraction.

    Example:
        qortex ingest load manifest.json
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
        typer.echo("View with: qortex inspect domains")

    except Exception as e:
        handle_error(f"Failed to save to graph: {e}")
    finally:
        graph_backend.disconnect()
