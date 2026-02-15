"""AutoGen Task-Centric Memory adapter: QortexSimilarityMap.

Drop-in replacement for AutoGen's StringSimilarityMap (ChromaDB-backed).
Implements the same 4-method interface using qortex's graph-enhanced retrieval
instead of raw ChromaDB cosine search.

The key difference: after feedback signals, qortex's teleportation factors
shift and PPR seed weights change, improving retrieval over time.
ChromaDB is static after indexing.

Usage:
    from qortex.adapters.autogen_tcm import QortexSimilarityMap

    ssm = QortexSimilarityMap(reset=True, path_to_db_dir="/tmp/qortex_tcm")
    ssm.add_input_output_pair("OAuth2 for mobile apps", "use PKCE flow")
    results = ssm.get_related_string_pairs("mobile auth", n_results=5, threshold=1.7)
    # -> [("OAuth2 for mobile apps", "use PKCE flow", 0.42), ...]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from qortex.client import LocalQortexClient, QueryResult
from qortex.core.memory import InMemoryBackend
from qortex.hippocampus.interoception import InteroceptionConfig, LocalInteroceptionProvider
from qortex.vec.embeddings import SentenceTransformerEmbedding
from qortex.vec.index import NumpyVectorIndex

logger = logging.getLogger(__name__)

_DOMAIN = "autogen_tcm"


class QortexSimilarityMap:
    """Graph-enhanced replacement for AutoGen's StringSimilarityMap.

    Same interface as StringSimilarityMap but backed by qortex's knowledge
    graph instead of ChromaDB. Retrieval goes through the full GraphRAG
    pipeline: vec search -> online edge gen -> PPR -> combined scoring.

    Cosine similarity scores are converted to L2-like distances for
    compatibility with AutoGen's threshold semantics (lower = better).
    """

    def __init__(
        self,
        reset: bool,
        path_to_db_dir: str,
        logger: Any | None = None,
    ) -> None:
        self._path = Path(path_to_db_dir)
        self._path.mkdir(parents=True, exist_ok=True)
        self._pairs_path = self._path / "pairs.json"

        # Internal state: maps pair_id -> (input_text, output_text)
        self._pairs: dict[str, tuple[str, str]] = {}
        self._next_id = 0

        # Last query result for feedback routing
        self._last_result: QueryResult | None = None

        # Build qortex client with teleportation enabled for learning
        self._embedding = SentenceTransformerEmbedding()
        self._vec_index = NumpyVectorIndex(dimensions=self._embedding.dimensions)
        self._backend = InMemoryBackend(vector_index=self._vec_index)
        self._backend.connect()
        self._interoception = LocalInteroceptionProvider(
            InteroceptionConfig(teleportation_enabled=True, persist_on_update=False)
        )
        self._interoception.startup()
        self._client = LocalQortexClient(
            vector_index=self._vec_index,
            backend=self._backend,
            embedding_model=self._embedding,
            interoception=self._interoception,
            mode="graph",
        )

        if reset:
            self.reset_db()
        else:
            self._load_pairs()

    def reset_db(self) -> None:
        """Clear all stored pairs and rebuild the index."""
        self._pairs = {}
        self._next_id = 0
        self._last_result = None

        # Rebuild backend with teleportation enabled
        self._vec_index = NumpyVectorIndex(dimensions=self._embedding.dimensions)
        self._backend = InMemoryBackend(vector_index=self._vec_index)
        self._backend.connect()
        self._interoception = LocalInteroceptionProvider(
            InteroceptionConfig(teleportation_enabled=True, persist_on_update=False)
        )
        self._interoception.startup()
        self._client = LocalQortexClient(
            vector_index=self._vec_index,
            backend=self._backend,
            embedding_model=self._embedding,
            interoception=self._interoception,
            mode="graph",
        )
        self.save_string_pairs()

    def add_input_output_pair(self, input_text: str, output_text: str) -> None:
        """Store an input/output pair, embedding the input for retrieval."""
        pair_id = str(self._next_id)
        self._next_id += 1
        self._pairs[pair_id] = (input_text, output_text)

        # Ingest input_text as a concept node
        self._client.ingest_structured(
            domain=_DOMAIN,
            concepts=[{
                "id": f"{_DOMAIN}:pair_{pair_id}",
                "name": input_text[:80],
                "description": input_text,
                "properties": {
                    "pair_id": pair_id,
                    "output_text": output_text,
                },
            }],
        )

    def build_edges(self, sim_threshold: float = 0.5) -> int:
        """Build edges between semantically similar pairs.

        Must be called after all pairs are added. Without edges,
        the GraphRAG adapter stays as VecOnly and no PPR/learning occurs.

        Returns the number of edges created.
        """
        if len(self._pairs) < 2:
            return 0

        # Get embeddings for all pairs
        texts = [inp for inp, _ in self._pairs.values()]
        pair_ids = list(self._pairs.keys())
        embeddings = self._embedding.embed(texts)

        # Compute pairwise cosine similarity
        import numpy as np

        emb_matrix = np.array(embeddings)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = emb_matrix / norms
        sim_matrix = normalized @ normalized.T

        from qortex.core.models import ConceptEdge, RelationType

        edge_count = 0
        for i in range(len(pair_ids)):
            for j in range(i + 1, len(pair_ids)):
                if sim_matrix[i, j] >= sim_threshold:
                    self._backend.add_edge(ConceptEdge(
                        source_id=f"{_DOMAIN}:pair_{pair_ids[i]}",
                        target_id=f"{_DOMAIN}:pair_{pair_ids[j]}",
                        relation_type=RelationType.SIMILAR_TO,
                        confidence=float(sim_matrix[i, j]),
                    ))
                    edge_count += 1

        # No adapter upgrade needed -- initialized with mode="graph"

        logger.info(f"Built {edge_count} edges from {len(pair_ids)} pairs")
        return edge_count

    def get_related_string_pairs(
        self,
        query_text: str,
        n_results: int,
        threshold: float | int,
    ) -> list[tuple[str, str, float]]:
        """Find stored pairs similar to query_text.

        Returns list of (input_text, output_text, distance) tuples,
        where distance is L2-compatible (lower = more similar).
        Only results with distance < threshold are returned.
        """
        if not self._pairs:
            return []

        n_results = min(n_results, len(self._pairs))
        if n_results == 0:
            return []

        result = self._client.query(
            context=query_text,
            domains=[_DOMAIN],
            top_k=n_results,
        )
        self._last_result = result

        matches: list[tuple[str, str, float]] = []
        for item in result.items:
            # Convert cosine similarity [0,1] to L2-like distance [0,2]
            cosine_sim = max(0.0, min(1.0, item.score))
            distance = 2.0 * (1.0 - cosine_sim)

            if distance >= threshold:
                continue

            # Look up the pair by node properties
            pair_id = None
            if item.metadata:
                pair_id = item.metadata.get("pair_id")

            # Fallback: match by content against stored pairs
            if pair_id is None:
                for pid, (inp, _) in self._pairs.items():
                    if inp in item.content or item.content in inp:
                        pair_id = pid
                        break

            if pair_id is not None and pair_id in self._pairs:
                input_text, output_text = self._pairs[pair_id]
                matches.append((input_text, output_text, distance))

        return matches

    def save_string_pairs(self) -> None:
        """Persist pairs to disk."""
        data = {
            "next_id": self._next_id,
            "pairs": {k: list(v) for k, v in self._pairs.items()},
        }
        with open(self._pairs_path, "w") as f:
            json.dump(data, f)

    def _load_pairs(self) -> None:
        """Load pairs from disk if they exist."""
        if self._pairs_path.exists():
            with open(self._pairs_path) as f:
                data = json.load(f)
            self._next_id = data.get("next_id", 0)
            self._pairs = {
                k: tuple(v) for k, v in data.get("pairs", {}).items()
            }
            # Re-ingest all pairs
            for pair_id, (input_text, output_text) in self._pairs.items():
                self._client.ingest_structured(
                    domain=_DOMAIN,
                    concepts=[{
                        "id": f"{_DOMAIN}:pair_{pair_id}",
                        "name": input_text[:80],
                        "description": input_text,
                        "properties": {
                            "pair_id": pair_id,
                            "output_text": output_text,
                        },
                    }],
                )

    # -- Qortex-specific: feedback for learning loop --

    @property
    def client(self) -> LocalQortexClient:
        """Expose the client for feedback calls."""
        return self._client

    @property
    def last_result(self) -> QueryResult | None:
        """Full QueryResult from the last get_related_string_pairs call."""
        return self._last_result
