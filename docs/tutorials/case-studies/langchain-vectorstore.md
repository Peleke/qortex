# LangChain VectorStore Integration

`langchain-qortex` is a standalone package providing `QortexVectorStore`, a LangChain `VectorStore` backed by qortex's knowledge graph.

## Install

```bash
pip install langchain-qortex
```

## Quick start

```python
from langchain_qortex import QortexVectorStore

# Zero-config (like Chroma.from_texts)
vs = QortexVectorStore.from_texts(texts, embedding, domain="security")
docs = vs.similarity_search("authentication", k=5)
retriever = vs.as_retriever()

# qortex extras
explore = vs.explore(docs[0].metadata["node_id"])
rules = vs.rules(concept_ids=[d.metadata["node_id"] for d in docs])
vs.feedback({docs[0].id: "accepted"})
```

## Full documentation

See the [langchain-qortex repository](https://github.com/Peleke/langchain-qortex) for the full VectorStore guide, including:

- Standard VectorStore API (similarity_search, from_texts, as_retriever)
- Graph exploration from search results
- Rules auto-surfaced in query results
- Feedback-driven learning loop
- Evidence table of what's proven in E2E tests
