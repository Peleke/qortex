# Part 1: The Multi-Hop Problem

There's a particular kind of frustration that visits you at 2am in a hospital when the computer is confidently, politely, *expensively* wrong.

You're a nurse. Patient in room 4 has chest pain. She's 67, arrived an hour ago, already on warfarin for her heart valve and ibuprofen for her knees. The attending wants to know if she can have aspirin.

Simple question. You type it into the clinical decision support system, the one the hospital paid two million dollars for, the one connected to every drug database ever compiled, the one that was supposed to make moments like this easy.

"Can this patient take aspirin for chest pain?"

The system thinks. The system returns: *"Aspirin (acetylsalicylic acid) is a non-steroidal anti-inflammatory drug that inhibits cyclooxygenase enzymes..."*

You stare at the screen.

This isn't an answer. This is a Wikipedia article with better formatting. You don't need to know how aspirin works. You need to know if it's going to kill Mrs. Chen when combined with the warfarin she's been taking for three years.

Here's what's actually in your head, the reasoning you do instantly without thinking:

- Aspirin thins blood
- Warfarin thins blood
- Two blood thinners = increased bleeding risk
- Mrs. Chen is already on warfarin
- Therefore: **no, or at least, be very careful**

The system has all of this information. It has the aspirin monograph. It has the warfarin monograph. It has Mrs. Chen's medication list right there in the EHR. But it couldn't connect the dots because nothing in its vector index says "aspirin" and "warfarin" are related. They're just two documents with no particular similarity score.

The two-million-dollar system failed a question a first-year pharmacy student could answer.

This is the **multi-hop problem**. The answer exists, scattered across three documents, linked only by concepts the embedding model never learned to represent. And vector similarity ("what documents look like this query?") is the wrong question to ask.

## The failure pattern

RAG systems embed documents as vectors. Retrieval finds documents with similar vectors to your query.

But "similar" isn't "connected."

When you ask about aspirin for a patient on warfarin, there's no single document about "aspirin-warfarin-bleeding-risk-for-this-specific-patient." The information exists in three separate places, linked only by concepts the embedding model doesn't explicitly represent.

Vector similarity asks: "What documents look like this query?"

What you need to ask: "What concepts connect to this query's concepts?"

The difference is **association** vs **similarity**. And similarity isn't enough.

## The fix (preview)

You need to store *relationships*, not just *embeddings*.

- Aspirin → IS_A → Anticoagulant
- Warfarin → IS_A → Anticoagulant
- Anticoagulant + Anticoagulant → INCREASES_RISK → Bleeding

Now when you query about aspirin for a warfarin patient, you can *traverse the graph* to discover the connection, even though no single document spelled it out.

That's a knowledge graph. That's what qortex builds.

## What you learned

- Vector similarity search fails when the answer requires connecting multiple facts
- Multi-hop questions need *association*, not *similarity*
- Knowledge graphs store relationships that embeddings don't capture
- The drug interaction example shows why this matters in high-stakes domains

## Next

[Part 2: Knowledge Graphs 101](part2-knowledge-graphs.md): What are concepts, edges, and semantic types?
