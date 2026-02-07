"""Framework adapters: drop-in replacements for agent memory systems.

Each adapter wraps QortexClient and presents the interface expected by
a specific framework. Consumers get qortex's vec + graph + feedback
learning without changing their code.

Adapters:
    crewai              → QortexKnowledgeStorage (BaseKnowledgeStorage)
    langchain           → QortexRetriever (BaseRetriever)
    langchain_vectorstore → QortexVectorStore (VectorStore — full integration)
    agno                → QortexKnowledge (knowledge protocol)
    mastra              → QortexVectorStore (MastraVector mapping)

Usage:
    from qortex.client import LocalQortexClient
    from qortex.adapters.langchain_vectorstore import QortexVectorStore

    # Zero-config (like Chroma.from_texts):
    vs = QortexVectorStore.from_texts(texts, embedding, domain="my-domain")
    docs = vs.similarity_search("query")

    # From existing client:
    vs = QortexVectorStore(client=client, domain="my-domain")
    docs = vs.similarity_search("query", k=5)
    retriever = vs.as_retriever()

    # qortex extras (graph + rules + feedback):
    explore = vs.explore(docs[0].metadata["node_id"])
    rules = vs.rules(concept_ids=[d.metadata["node_id"] for d in docs])
    vs.feedback({docs[0].id: "accepted"})
"""
