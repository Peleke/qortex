# Mastra MastraVector Integration

`@peleke/mastra-qortex` is a standalone package providing `QortexVector`, a Mastra `MastraVector` backed by qortex's knowledge graph via MCP.

## Install

```bash
npm install @peleke/mastra-qortex
```

## Quick start

```typescript
import { QortexVector } from "@peleke/mastra-qortex";

const qortex = new QortexVector({
  id: "qortex",
  serverCommand: "uvx",
  serverArgs: ["qortex", "mcp-serve"],
});

// Standard MastraVector API — drop-in for Pinecone, Chroma, PG, etc.
await qortex.createIndex({ indexName: "docs", dimension: 384 });
await qortex.upsert({ indexName: "docs", vectors: embeddings, metadata });
const results = await qortex.query({ indexName: "docs", queryVector, topK: 10 });

// qortex extras
const explored = await qortex.explore(results[0].id);
const rules = await qortex.getRules({ domains: ["security"] });
await qortex.feedback(queryId, { [results[0].id]: "accepted" });
```

## Full documentation

See the [mastra-qortex repository](https://github.com/Peleke/mastra-qortex) for the full MastraVector guide, including:

- All 9 MastraVector abstract methods (create/list/describe/delete index, upsert, query, update, delete, deleteMany)
- Graph exploration from search results
- Rules projection and auto-surfacing
- Feedback-driven learning loop
- MongoDB-like metadata filters
- Architecture diagram (TypeScript → MCP → Python)
