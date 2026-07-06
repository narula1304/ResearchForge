# Dataset Analysis

## Overview
- **Number of documents:** 104
- **Total pages:** 2131
- **Average pages per document:** 20.5
- **Average tokens per document:** 13257.8
- **Average references per paper:** 37.7
- **Average figures per paper:** 5.1
- **Average tables per paper:** 4.6

### So What?
The corpus consists of 104 papers with an average of 20.5 pages and 13257.8 tokens per document. A citation density of 37.7 references per paper indicates that multi-hop questions across cited papers will be a common occurrence, making robust citation tracking essential. The presence of figures and tables (5.1 figures and 4.6 tables per paper on average) suggests that multimodal parsing or table extraction capabilities will be beneficial in Phase 3.

## Chunking Estimates
Estimated total tokens: 1378806
- **300-token chunks:** ~4596 chunks
- **500-token chunks:** ~2757 chunks
- **800-token chunks:** ~1723 chunks

### So What?
These chunk counts provide a baseline for vector database sizing and retrieval latency tests. A chunk size of 500 tokens gives a manageable number of chunks overall, balancing context retention and retrieval speed for local or API-based embeddings.

## Topic Distribution
Rough clustering of titles and abstracts:
- Cluster 1: llms, models, language, tool, tasks
- Cluster 2: rag, retrieval, generation, augmented, knowledge
- Cluster 3: agentic, agents, agent, ai, self
- Cluster 4: reasoning, language, multimodal, code, llms
- Cluster 5: ai, university, cognitive, thinking, decision

### So What?
The topics reflect a varied distribution among the chosen domain keywords. Query classification (Phase 4) and metadata filtering (Phase 3) could use these topic clusters to route questions or narrow down context effectively.

## Section Distribution
Papers containing identifiable sections:
- Abstract: 97
- Introduction: 102
- Method/Methodology: 101
- Results: 97
- Related Work: 74

### So What?
Most papers consistently contain standard academic sections. Section-aware or semantic chunking (Phase 3) could be highly effective since core methodology and results are explicitly demarcated, helping avoid context-cutting across logical boundaries.
