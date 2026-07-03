# Build Plan: Agentic RAG Research Paper Assistant (v3)

## 0. Project Summary

Build a research assistant over a corpus of academic papers (~50-200 arXiv PDFs on a chosen topic, e.g. "RAG techniques" or "LLM agents") plus a few blog posts. The system answers research questions with grounded, cited, confidence-scored answers, using hybrid retrieval, query rewriting, reranking, context compression, a small reflection-based agent graph, memory, guardrails, and a real evaluation/observability/feedback discipline.

**Core principle: know your data and your components before you know your system.** You analyze the corpus before ingesting it, benchmark the embedding model before benchmarking retrieval, benchmark retrieval before adding agents, and instrument observability before adding branching control flow. Every layer is measured before the next one is built on top of it.

**Tech stack (defaults — an agent can substitute if unavailable):**
- Orchestration: LangGraph
- LLM: Claude API via Anthropic SDK
- Embeddings: candidates compared in Phase 2 — `BGE-base`, `BGE-large`, `E5-base`, OpenAI `text-embedding-3-small/large`
- Vector DB: Chroma (dev) / Qdrant (if scaling)
- Keyword search: `rank_bm25`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` or Cohere Rerank API
- PDF parsing: `pymupdf` + `unstructured` (tables/layout)
- Backend: FastAPI
- Frontend: Streamlit (with a trace panel — see final phase)
- Observability/cost: Langfuse (traces, spans, token/cost/latency tracking)
- Eval: custom scripts + `ragas`
- Experiment/failure logs: flat markdown + JSON files in-repo
- Prompt versioning: flat versioned folders (see Phase 4)
- Memory/feedback/logs DB: SQLite → Postgres if scaling

**Repo structure:**
```
research-assistant/
├── ingestion/            # loaders, chunking, metadata, table_extraction
├── retrieval/            # embeddings, vector_store, bm25, hybrid, rewrite,
│                         # multi_query, mmr, reranker, compression, metadata_filter
├── agents/               # graph.py, planner.py, retriever_node.py,
│                         # writer.py, citation_verifier.py, critic.py, state.py
├── memory/               # session_memory, episodic_memory, cache
├── guardrails/           # input, retrieval, output
├── eval/
│   ├── golden_set.json
│   ├── embedding_eval.py        # NEW
│   ├── retrieval_eval.py
│   ├── generation_eval.py
│   ├── latency_benchmark.py     # NEW
│   ├── regression_test.py
│   └── retrieval_failures/
├── experiments/                 # experiment_01_embeddings.md, 02_hybrid.md, ...
├── prompts/                     # NEW — versioned prompt folders
│   ├── v1/  v2/  v3/
├── observability/        # logging_config, tracing, cost_tracker.py
├── feedback/             # feedback_store.py
├── api/                  # FastAPI endpoints
├── ui/                   # Streamlit app + trace visualizer
├── data/raw/  data/processed/
├── tests/
├── .github/workflows/
│   └── ci.yml                   # NEW — regression tests gate on every push
├── Dockerfile                    # NEW
├── docker-compose.yml             # NEW — app + vector db (if self-hosted)
├── docs/
│   ├── dataset_analysis.md      # NEW
│   ├── architecture.md
│   ├── eval_report.md
│   └── adrs/                    # NEW — replaces decisions.md
│       ├── ADR-001-vector-db.md
│       ├── ADR-002-chunking.md
│       ├── ADR-003-embedding-model.md
│       ├── ADR-004-reranking.md
│       ├── ADR-005-memory.md
│       ├── ADR-006-observability.md
│       └── ADR-007-deployment.md
├── requirements.txt
└── README.md
```

---

## Phase 0 — Dataset Analysis (Week 1, before any ingestion pipeline is built)

**Goal:** understand the corpus before writing a single line of chunking/retrieval code. Chunking and retrieval decisions should be justified by what the data actually looks like, not guessed.

### Tasks
1. Collect the raw corpus (50-100 arXiv PDFs + a few HTML sources) but only run lightweight inspection scripts on it — no indexing yet.
2. Write `docs/dataset_analysis.md` covering:
   - Number of documents, total pages, average pages per document
   - Average tokens per document, estimated chunk count at a few candidate chunk sizes (e.g., 300/500/800 tokens) — this directly informs your Phase 3 chunking experiments
   - Topic distribution (rough clustering of paper titles/abstracts, even a simple keyword or embedding-cluster pass)
   - Citation density (average references per paper, useful context for why multi-hop questions will be common)
   - Section distribution (how consistently papers have Abstract/Intro/Method/Results/Related Work — informs whether section-aware chunking is worth the effort)
   - Table/figure density (how many papers have tables you'll need Phase 3's table extraction for)

### Deliverable
`docs/dataset_analysis.md` with real numbers and 2-3 sentences of "so what" per section (e.g., "high citation density → multi-hop questions are a first-class category, not an edge case"). This becomes the justification you cite in later ADRs.

---

## Phase 1 — Basic RAG Baseline (Week 1-2)

**Goal:** the simplest possible correct pipeline, single retrieval method, no intelligence layered on yet — a reference point to measure every later improvement against.

### Tasks
1. Ingestion: parse PDFs (`pymupdf`), extract metadata (title, authors, year, arXiv ID), parse HTML (`trafilatura`/`unstructured`).
2. Naive fixed-size/recursive chunking (~500 tokens, 50 overlap) — pick this size using the Phase 0 analysis, not arbitrarily.
3. Embed with a single default model (e.g., `bge-base`) just to get something working — this choice gets formally revisited in Phase 2.
4. Store in Chroma, single retrieval path: dense-only top-k → prompt → Claude generates an answer with a source list.
5. Minimal Streamlit UI: question in, answer + sources out.
6. Build a tiny golden eval set (10-15 Q&A pairs with known relevant chunks), informed by the query types you'll expect from Phase 0 (multi-hop given citation density, etc.).

### Deliverable
`v0-baseline` tag. Dense-only RAG, boring, working, and already measurable.

---

## Phase 2 — Embedding Evaluation (Week 2-3)

**Goal:** before touching hybrid search, chunking strategy, or reranking, settle which embedding model you're standing on — since every later retrieval experiment is only comparable if the embedding model is held constant.

### Tasks
1. Build `eval/embedding_eval.py`. For each candidate — `BGE-base`, `BGE-large`, `E5-base`, OpenAI `text-embedding-3-small` (and `-large` if budget allows) — embed the same chunk set and run the same golden-set queries.
2. Measure, per model:
   - Recall@5, Recall@10, MRR (against golden set)
   - Latency (embedding time per chunk, query embedding time)
   - Cost (API cost for OpenAI models; compute cost/local inference time for open-source models)
3. Log results in `experiments/experiment_01_embeddings.md` using the standard template (Hypothesis / Implementation / Metrics / Results / Lessons).
4. Write `docs/adrs/ADR-003-embedding-model.md` documenting the final choice with the actual comparison numbers and the tradeoff reasoning (e.g., "BGE-large gave +4% recall over BGE-base but 2.3x latency; chose BGE-base for the latency budget" or the reverse — whatever your numbers actually show).

### Deliverable
A table like:
```
Model              Recall@10   MRR    Latency(ms/query)   Cost/1K chunks
BGE-base           0.61        0.54   12                   $0 (local)
BGE-large          0.66        0.58   31                   $0 (local)
E5-base            0.63        0.55   14                   $0 (local)
OpenAI-3-small     0.68        0.60   180 (API roundtrip)  $0.02
```
Re-embed the full corpus once with the chosen model before moving to Phase 3, so all later retrieval experiments sit on a fixed embedding baseline.

---

## Phase 3 — Retrieval Benchmarking & Advanced Retrieval (Week 3-5)

**Goal:** with dataset understood and embeddings fixed, add each retrieval technique one at a time, measuring the delta after every addition.

### Step 0: Retrieval Benchmarking harness
`eval/retrieval_eval.py` computing Recall@5, Recall@10, MRR, hit rate. Run on BM25-only and dense-only (using the Phase 2 winning embedding model) as the new baselines.

### Step-by-step additions (each measured immediately, each logged as its own experiment file)
1. **Hybrid search** (BM25 + dense, RRF) → `experiments/experiment_02_hybrid.md`
2. **Semantic chunking** vs recursive, informed by Phase 0's chunk-size analysis → `experiments/experiment_03_chunking.md`
3. **Parent-child retrieval** → `experiments/experiment_04_parent_child.md`
4. **Query rewriting** (multi-turn eval subset) → `experiments/experiment_05_query_rewrite.md`
5. **Multi-query retrieval** → `experiments/experiment_06_multi_query.md`
6. **MMR** (diversity) → `experiments/experiment_07_mmr.md`
7. **Reranking** (cross-encoder) → `experiments/experiment_08_rerank.md`
8. **Context compression** (token savings vs faithfulness delta) → `experiments/experiment_09_compression.md`
9. **Metadata filtering** + **table extraction** (using Phase 0's table-density numbers to gauge how much effort this deserves)

### Deliverable
Full progression table in `docs/eval_report.md`, e.g.:
```
Dense-only (chosen embedding)   Recall@10 = 0.66
+Hybrid                          Recall@10 = 0.72
+Semantic chunking                Recall@10 = 0.79
+Parent-child                     Recall@10 = 0.85
+Rerank                            Recall@10 = 0.89
+Compression                      Recall unaffected, tokens/query -42%
```
Write `docs/adrs/ADR-001-vector-db.md` and `ADR-002-chunking.md` here, each citing these numbers.

---

## Phase 4 — Generation Evaluation, Query Classification & Prompt Versioning (Week 5-6)

### Tasks
1. **Generation metrics**: `ragas` (or LLM-as-judge) for faithfulness/groundedness and relevance on golden-set answers.
2. **Expand golden set** to 30-50 questions, categorized: `factual`, `comparative`, `summarization`, `multi-hop`, `research synthesis`, `out-of-scope`, `ambiguous`.
3. **Query classifier**: LLM-call labeling incoming queries into the categories above; evaluate standalone (accuracy against ~50 labeled example queries) before wiring into the planner later.
4. **Retrieval failure analysis**: for underperforming golden-set queries, log to `eval/retrieval_failures/`: `query, retrieved_docs, expected_docs, failure_reason, fix_applied`.
5. **Prompt versioning**: from this point on, every prompt (writer, query rewriter, classifier, critic later) lives in `prompts/v1/`, `prompts/v2/`, etc. When you materially change a prompt:
   - Copy to a new version folder
   - Note in a short changelog entry: what changed, why, and the before/after metric (faithfulness score, classifier accuracy, whatever's relevant)
   - Only promote the new version as default once its metrics are equal or better on the golden set

### Deliverable
Generation-quality baseline, a working query classifier with its own accuracy number, a failure log with 5-10 analyzed cases, and `prompts/v1/` populated with your current production prompts plus a changelog stub ready for future versions.

---

## Phase 5 — Observability, Cost & Latency Benchmarking (Week 6-7)

**Goal:** instrument the pipeline before adding agentic control flow.

### Tasks
1. **Tracing**: Langfuse (or structured JSON logs) around every stage (rewrite, multi-query, retrieval, rerank, compression, generation).
2. **Cost tracking**: `observability/cost_tracker.py` attributing cost per stage (embedding, retrieval, rerank if API-based, generation). Per-query breakdown, e.g. `Retrieval: $0.001 | Rerank: $0.003 | Generation: $0.020`.
3. **Latency benchmarking**: `eval/latency_benchmark.py` measuring, for each component in isolation and for the full pipeline:
   - BM25 lookup
   - Dense retrieval
   - Hybrid search
   - Reranking
   - Context compression
   - Full linear pipeline (pre-agent)
   Track **average latency** and **P95 latency** per component, plus token usage and cost alongside it, e.g.:
   ```
   Component            Avg (ms)   P95 (ms)   Tokens   Cost
   BM25                 8          15         -        $0
   Dense retrieval      45         80         -        $0
   Hybrid               52         95         -        $0
   Reranking            120        210        -        $0 (local) / $0.002 (API)
   Compression          800        1400       ~600     $0.004
   Full pipeline        1100       1900       ~1800    $0.011
   ```
   Re-run this benchmark again after Phase 6 to see what the agent loop (retries, critic calls) adds on top — this before/after is a strong latency-vs-quality tradeoff talking point.
4. **Metrics dashboard**: latency, cost, token usage, error rate over time in Streamlit/Langfuse.

### Deliverable
Full latency/cost table per component, plus the dashboard, before any agent node exists.

---

## Phase 6 — Agentic Layer (Week 7-9)

**Goal:** introduce branching/looping control flow on top of everything already measured.

### Agent state schema
`raw_query`, `chat_history`, `query_category`, `rewritten_query`, `sub_queries`, `retrieved_chunks`, `reranked_chunks`, `compressed_context`, `draft_answer`, `citation_verification_result`, `critic_verdict`, `retrieval_confidence`, `generation_confidence`, `overall_confidence`, `retry_count`, `final_answer`, `citations`, `cost_breakdown`, `latency_breakdown`.

### Nodes
1. **Planner**: routes on query category (as in v2) — `needs_retrieval` / `answerable_directly` / `out_of_scope` (abstain) / `needs_clarification`.
2. **Retriever node**: full Phase 3 pipeline, cost/latency-tagged.
3. **Writer node**: generates a draft answer with inline citations.
4. **Citation Verifier node (NEW)** — runs between Writer and Critic:
   - Confirms every cited source actually exists in the retrieved set (no invented citations)
   - Confirms the cited page/chunk is the correct one referenced (not a mismatched citation from a different chunk)
   - Confirms the cited passage actually supports the specific claim made (not just topically related)
   - Outputs a per-citation pass/fail list; any failing citation is either dropped, flagged, or triggers a re-write with the original context re-passed
5. **Critic node**: checks the (citation-verified) draft against retrieved evidence and outputs a **three-part confidence score** instead of one number:
   - **Retrieval Confidence** — derived from retrieval/rerank similarity scores of the chunks actually used
   - **Generation Confidence** — derived from groundedness/faithfulness of the generated claims against those chunks (this is where the citation verifier's pass/fail feeds in)
   - **Overall Confidence** — combined score (e.g., a weighted or minimum of the two, tunable)
   Verdict remains `supported` / `unsupported` / `insufficient_evidence`, now backed by the split scores rather than one opaque number.
6. **Abstain node**: honest "not enough information" response.

### Edges
```
START → planner
planner → [answerable_directly] → writer → END
planner → [out_of_scope] → abstain → END
planner → [needs_clarification] → ask user → END
planner → [needs_retrieval] → retriever → writer → citation_verifier → critic
critic → [supported] → END (attach retrieval/generation/overall confidence, citations, cost, latency)
critic → [unsupported] → retriever (retry, max 2, reformulated query) → writer → citation_verifier → critic
critic → [insufficient_evidence, retries exhausted] → abstain → END
```

### Deliverable
Trace a factual, comparative, multi-hop, and out-of-scope query through the graph. Final answers now show:
```
Retrieval Confidence: 92%
Generation Confidence: 85%
Overall Confidence: 88%
Citations verified: 6/6
```
Re-run the Phase 5 latency benchmark on the full agent pipeline (including citation verification and possible retries) and compare against the pre-agent numbers.

---

## Phase 7 — Memory (Week 9-10)

### Tasks
1. **Session memory**: chat history per session.
2. **Episodic memory**: LLM-generated per-session summaries stored in SQLite/vector store, retrieved on related future sessions.
3. **Semantic cache**: cache (query embedding → final answer); skip the full pipeline on high-similarity repeats; track hit rate, feeding the cost/latency dashboard.

### Deliverable
Cross-session context demo, measurable cache hit rate, and cost/latency savings shown against the Phase 5 benchmark.

Write `docs/adrs/ADR-005-memory.md` documenting why episodic memory was chosen over a fuller multi-tier system, with the actual cache-hit numbers.

---

## Phase 8 — Guardrails (Week 10-11)

### Tasks
1. **Input guardrails**: prompt-injection/toxicity screening before the planner.
2. **Retrieval guardrails**: scan ingested content for embedded instructions before indexing.
3. **Output guardrails**: enforced by the critic + citation verifier — no ungrounded or miscited answer reaches the user.
4. **HITL checkpoint**: the `needs_clarification` branch.

### Deliverable
Demonstrate a blocked prompt-injection attempt and a blocked ungrounded/miscited-answer attempt, each with a trace showing where it was caught (citation verifier vs critic vs input guardrail).

---

## Phase 9 — Feedback Loop & Regression Testing (Week 11-12)

### Tasks
1. **Feedback UI**: thumbs up/down in Streamlit; store (query, answer, chunks, confidence scores, rating) in `feedback/feedback_store.py`.
2. **Feedback → eval loop**: review thumbs-down entries, add hard cases to the golden set and to `eval/retrieval_failures/` where relevant.
3. **Regression testing**: `eval/regression_test.py` reruns the full golden set (retrieval + generation + latency) on every prompt/parameter change, flags regressions. This is where prompt versioning (Phase 4) and regression testing meet — a new prompt version only gets promoted if regression tests pass.

### Deliverable
Show a thumbs-down answer added to the golden set, and the regression suite catching a deliberately-introduced bad change.

---

## Phase 10 — Agent Trace UI & Polish (Week 12-13)

### Tasks
1. **Trace visualizer**: render the actual path taken per query, including the new citation verification step and split confidence scores:
   ```
   Question → Planner (category: comparative) → Query Rewrite → Multi-Query
   → Hybrid Search → Reranker → Compression → Writer → Citation Verifier (6/6 passed)
   → Critic (Retrieval: 92%, Generation: 85%, Overall: 88%) → Final Answer
   ```
2. Write `README.md`: problem, architecture diagram, dataset analysis summary, embedding + retrieval eval progression tables, latency/cost tables, split-confidence example, trace UI screenshot.
3. Finalize `docs/adrs/` — at minimum: vector DB, chunking, embedding model, reranking, memory, observability. Each ADR cites real numbers from the experiment logs, not general reasoning.
4. List explicit "Future Work": GraphRAG, multi-modal figure understanding, full HITL approval UI, production auth/multi-tenancy.
5. Record a 2-3 minute demo showing a query traveling through the trace UI end to end, including a case where citation verification catches a bad citation.

---

## Phase 11 — Deployment (Week 13-14)

**Goal:** a public, reproducible, cost-protected live demo — not just a repo that runs locally.

### Tasks
1. **Containerize**:
   - Write a single `Dockerfile` for the FastAPI backend + Streamlit frontend (or two stages/services if you prefer to keep them separate).
   - Write `docker-compose.yml` if running a self-hosted vector DB (e.g., Qdrant) alongside the app; skip this if using Chroma with a mounted volume or a managed vector DB.
   - Verify `docker build` + `docker run` works cleanly from a fresh clone — this is the actual test of "reproducible," not just "runs on my machine."

2. **Pick a host**:
   - **Hugging Face Spaces (Docker SDK)** or **Render/Railway free-hobby tier** — either supports a single Docker container with a public URL, no infra overkill needed.
   - Persist the vector store: mounted volume for Chroma, or switch to Qdrant Cloud free tier if you want a managed-DB story in your architecture doc.
   - If self-hosted embeddings (`sentence-transformers`) are too slow on the free CPU tier, switch the deployed build to an embedding API and document this as a deploy-time tradeoff in `docs/adrs/ADR-003-embedding-model.md` (dev = local/free, prod = API/fast).

3. **Protect against cost blowup** (public demo + LLM API calls is a real risk):
   - Rate limiting per IP/session (simple in-memory token bucket is enough for a demo; Redis if you want to look more production-grade).
   - A hard daily spend cap: read from your Phase 5 `cost_tracker.py` running total, disable further API calls past the threshold, and show a friendly "daily demo limit reached, try again tomorrow" message.
   - Log and surface this cap-hit event in your observability dashboard — it's evidence the guardrail actually works, not just code that exists.

4. **CI/CD**:
   - `.github/workflows/ci.yml`: on every push to `main`, run `eval/regression_test.py` (retrieval + generation + latency checks).
   - Block merge/deploy if regression tests fail.
   - Optional: auto-deploy to your chosen host on a successful merge to `main`.

5. **Secrets management**: API keys (Claude, embedding API, Langfuse) via the host's environment variable/secrets manager — never committed to the repo, and call this out explicitly in the README as a deliberate practice.

6. **Smoke test post-deploy**: a small script or manual checklist hitting the live URL with 2-3 golden-set questions to confirm the deployed system's answers/citations match what you'd expect from local testing.

### Deliverable
- A public URL an interviewer can open and query directly.
- A green CI badge in the README showing regression tests passing.
- A short "Deployment" section in `README.md` covering: host choice and why, cost-protection measures, and the one-command local run (`docker run ...`) for anyone who wants to reproduce it themselves.
- `docs/adrs/ADR-007-deployment.md` documenting host choice, vector DB persistence approach, and the cost-protection design, with any real numbers you gathered (e.g., observed latency on the free tier vs local, actual API cost per demo session).

---

## Notes for the AI coding agent building this

- **Order matters and should not be collapsed for speed**: dataset analysis → embedding choice → retrieval techniques → generation eval/classification → observability → agents → memory → guardrails → feedback → polish. Each phase's output is an input to the next phase's decisions (e.g., Phase 0's chunk-size analysis directly informs Phase 3's chunking experiments; Phase 2's embedding choice is held fixed through Phase 3 so those comparisons are valid).
- **Every retrieval/embedding/prompt-affecting change gets its own experiment file or prompt version and an eval run.** This is what turns the project into evidence, not a feature list.
- **The citation verifier and the critic are separate concerns**: the verifier checks factual/structural correctness of citations (does this citation exist, is it the right chunk, does it support the claim); the critic checks overall groundedness and produces the confidence scores. Keep them as distinct nodes even though they run back-to-back — it keeps failure diagnosis clean (a citation-verifier failure vs a critic "unsupported" verdict are different bugs).
- **Prompt versioning discipline**: never edit a prompt in place once it's in production — copy to a new version, log the change and the before/after metric, and only promote after regression tests pass.
- **ADRs get written as decisions are made**, not retroactively at the end — the numbers are freshest right after the relevant experiment.
- Prefer local/free tools for the dev loop (Chroma, local cross-encoder, SQLite); reach for paid APIs only if local quality is clearly insufficient, and note the tradeoff in the relevant ADR either way.
- **Do not deploy publicly before Phase 8's guardrails and Phase 9's regression tests exist.** A public URL calling the Claude API with no input guardrails, no cost cap, and no regression gate is a real financial and safety risk, not just a rough edge — deployment is intentionally the last phase for this reason.
