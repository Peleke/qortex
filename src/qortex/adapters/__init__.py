"""Framework adapters: drop-in replacements for agent memory systems.

Each adapter wraps QortexClient and presents the interface expected by
a specific framework. Consumers get qortex's vec + graph + feedback
learning without changing their code.

Adapters:
    crewai   → QortexKnowledgeStorage (BaseKnowledgeStorage)
    langchain → QortexRetriever (BaseRetriever)
    agno     → QortexKnowledge (knowledge protocol)
    mastra   → QortexVectorStore (MastraVector mapping)

Usage:
    from qortex.client import LocalQortexClient
    from qortex.adapters.langchain import QortexRetriever

    client = LocalQortexClient(vector_index, backend, embedding_model)
    retriever = QortexRetriever(client, domains=["my-domain"])

    # Use retriever anywhere langchain expects a BaseRetriever
    docs = retriever.invoke("What is authentication?")
"""
