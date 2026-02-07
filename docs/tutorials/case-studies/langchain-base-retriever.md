# Case Study: Replacing Any LangChain Retriever

> **Status**: Compat proven — `tests/test_framework_compat.py`, `tests/test_dropin_dogfood.py`
>
> **The hook**: `QortexRetriever` IS a `BaseRetriever`. It works in any LCEL chain, any RAG pipeline, any place LangChain expects a retriever. isinstance check passes.

## What LangChain Has

LangChain's `BaseRetriever` is the universal retriever interface. Hundreds of integrations implement it. The entire LCEL composition system (pipes, parallel, etc.) works with any `BaseRetriever`.

| LangChain Interface | qortex Implementation |
|--------------------|----------------------|
| `retriever.invoke(query)` | `QortexRetriever.invoke()` → list[Document] |
| `retriever.ainvoke(query)` | Inherited from BaseRetriever |
| `retriever \| formatter` | Works — LCEL pipe operator |
| `RunnableParallel({"context": retriever})` | Works — parallel composition |
| `Document(page_content, metadata, id)` | Constructed from QueryItem |
| — (no equivalent) | `.feedback(outcomes)` |

## The Swap

### Any retriever → qortex

```python
# Before: any langchain retriever
from langchain_community.retrievers import SomeRetriever
retriever = SomeRetriever(...)

# After: qortex
from qortex.adapters.langchain import QortexRetriever
from qortex.client import LocalQortexClient

client = LocalQortexClient(vector_index, backend, embedding_model)
retriever = QortexRetriever(client=client, domains=["security"], top_k=5)

# Everything downstream stays the same
chain = retriever | format_docs | prompt | llm
```

### LCEL Chain (proven)

```python
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

# Retrieval + formatting chain
chain = retriever | RunnableLambda(lambda docs: "\n".join(d.page_content for d in docs))
result = chain.invoke("What is OAuth2?")  # → formatted string

# RAG context dict
rag = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
result = rag.invoke("What is RBAC?")
# → {"context": [Document(...)], "question": "What is RBAC?"}
```

## What We Proved

**Real BaseRetriever**: `isinstance(retriever, BaseRetriever)` is `True`. Not a duck type.

**LCEL pipe operator**: `retriever | RunnableLambda(format_docs)` — real chain execution, not mocked.

**RunnableParallel**: `{"context": retriever, "question": RunnablePassthrough()}` — real parallel composition.

**Document metadata convention**: Score goes in `metadata["score"]` per LangChain convention. Domain, node_id also in metadata.

**Cross-framework consistency**: Same IDs and scores as CrewAI, Mastra, Agno adapters on identical queries.

## Next Steps

<!-- TODO: Full RAG chain with an LLM (needs API key) -->
<!-- TODO: Show async retrieval (ainvoke) -->
<!-- TODO: Demo with LangGraph agent using qortex as retriever tool -->
<!-- TODO: Streaming support verification -->
