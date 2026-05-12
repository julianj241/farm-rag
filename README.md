# Farm RAG Assistant

A local, agentic RAG system over four years of organic farm field logs and structured records. Built in Python using LangChain, LangGraph, Chroma, SQLite, Streamlit, and Ollama — no cloud APIs.

## What it does

Answers natural-language questions about a small market-garden operation (arugula and watercress in El Cajon, CA) by routing each query to the right retrieval strategy:

- **Factual questions** ("Which bed produces the most arugula?", "How many times was JJ-4 fertilized in 2022?") → generates and executes SQL against a structured SQLite database of 8 tables.
- **Narrative questions** ("When do yellowing leaves tend to happen in JJ beds?") → semantic search over monthly-chunked field log entries in a Chroma vector store.

The routing decision is made by an LLM classifier node as the first step of a LangGraph state machine.

## Architecture

```
                     User query
                         │
                         ▼
                 ┌──────────────┐
                 │  classifier  │  → sql | semantic | recommend | conv.
                 └──────┬───────┘
                        │
       ┌──────┬─────────┴─────────┬──────────────┐
       ▼      ▼                   ▼              ▼
    ┌─────┐ ┌─────┐           ┌────────┐  ┌────────────┐
    │ sql │ │ sem │           │ conv.  │  │ recommend  │
    │     │ │     │           │ history│  │ multi-hop: │
    │     │ │     │           │  only  │  │ SQL +      │
    │     │ │     │           │        │  │ Chroma +   │
    │     │ │     │           │        │  │ Open-Meteo │
    └──┬──┘ └──┬──┘           └────┬───┘  └─────┬──────┘
       │      │                    │            │
       └──┬───┘                    │            │
          ▼                        │            │
     ┌──────────┐                  │            │
     │generator │                  │            │
     └────┬─────┘                  │            │
          │                        │            │
          └────────────┬───────────┴────────────┘
                       ▼
                      END
```

State is checkpointed with `MemorySaver` per session `thread_id`.

## Routing

The classifier routes queries through four paths:

- **SQL** — factual, quantitative questions run against the structured SQLite store. Example: *"Which bed produces the most arugula?"* → `SELECT Bed, SUM(Bundles) ...` → JJ-6 / 17,973 bundles.
- **Semantic** — narrative or analytical questions run against the monthly-chunked Chroma store.
- **Conversational** — meta-questions about the prior conversation itself (*"what bed were we just talking about?"*) are routed by keyword detection straight to a history-only answer, bypassing retrieval entirely. This path uses the `MemorySaver` checkpointer to preserve turn-over-turn context per session `thread_id`.
- **Recommend** — forward-looking advice questions (*"should I water JJ-7 tomorrow?"*) trigger a multi-hop node that pulls recent structured activity (last 5 irrigation + fertilizer events), narrative excerpts from comparable past situations, and the upcoming 3-day weather forecast via an **MCP (Model Context Protocol) client** that spawns and talks to `weather_mcp_server.py` over stdio. The LLM synthesizes a cited recommendation from all three sources.

Example of memory across turns:
> **User:** How many times was JJ-4 fertilized in 2022?
> **Assistant:** The farm's JJ-4 bed was fertilized 21 times in the year 2022.
> **User:** Which bed produces the most arugula?
> **Assistant:** The JJ-6 bed produces the most arugula, with a total of 17,973 bundles harvested.
> **User:** What bed were we just talking about?
> **Assistant:** We were discussing the JJ beds, specifically JJ-4 and JJ-6.

## MCP integration

The weather tool is exposed as a proper Model Context Protocol (MCP) server in `weather_mcp_server.py`. The recommend node connects to it as an MCP client over stdio, calling the `get_forecast` tool to retrieve weather data. This is the architecture the spec calls for: agent (MCP Client) → Weather MCP Server.

The MCP server runs as a subprocess spawned automatically by the graph; no separate process needs to be started at demo time. Same single `streamlit run app.py` command.

## Data

- **field_log.md** — 3,274 lines of daily farm log entries (Jan 2022 – Apr 2026), structured by month. Full data is kept private (gitignored); the schema and structure are documented above.
- **farm_records.xlsx** — 8 sheets of structured data: Beds (30), Plantings (977), Harvests (3,253), Fertilizer (2,469), Amendments (91), Weather (38), Irrigation (2,494), Summary.

## Pipeline

### Ingestion (`ingest.py`)

- **Markdown log → Chroma**: Split on `## Month Year` headings; one chunk per month (54 chunks total after auto-splitting 2 oversized months along `**Day M/D/YY**` boundaries). Each chunk embedded with `nomic-embed-text` via a custom Ollama model `nomic-embed-long` (context extended to 8192 tokens via Modelfile). Metadata per chunk: `year`, `month`, `part`, `total_parts`, `mentioned_beds` (normalized long-form bed names).
- **Excel workbook → SQLite**: Each sheet becomes a table. Dates stored as ISO strings. Re-runs are idempotent with a `--force` flag to re-ingest.

### Query pipeline (`graph.py`)

1. **Classifier node**: Prompts `llama3.1:8b` to return one word: `sql`, `semantic`, or `recommend`. Conversational meta-questions (e.g. *"what bed were we just discussing?"*) are detected by a keyword pre-filter before the LLM call. Recent conversation history is included for follow-up-aware classification.
2. **Semantic node**: Similarity search against Chroma, top 4 chunks.
3. **SQL node**: LLM generates SQLite from a hardcoded schema prompt that includes value conventions (e.g. Crop values are lowercase) and table-usage guidance. Query is sanity-checked to be SELECT-only, then executed; results formatted as a plain-text table.
4. **Generator node**: Takes retrieved context (chunks or SQL result) and writes the final natural-language answer.

### UI (`app.py`)

Streamlit chat interface with a sidebar that displays route-specific retrieval details — the actual SQL query and result table for SQL-routed queries, or the retrieved Chroma chunks for semantic queries.

## Example queries

| Query | Route | Answer |
|---|---|---|
| How many times was JJ-4 fertilized in 2022? | SQL | The farm's JJ-4 bed was fertilized 21 times in the year 2022. |
| Which bed produces the most arugula? | SQL | JJ-6, with 17,973 bundles harvested total. |
| When do yellowing leaves tend to happen in JJ beds? | Chroma | Grounded narrative answer citing specific log entries by month. |
| Should I water JJ-7 tomorrow? | Recommend | Multi-hop: pulls recent JJ-7 activity, comparable past entries, and 3-day forecast — produces a specific cited watering recommendation. |

## Results

The system was evaluated against the four representative queries above, each targeting a different retrieval strategy. All four returned correct, grounded answers:

- The two SQL-routed queries produced exact numbers verified against the underlying `harvests` and `fertilizer` tables (JJ-6 = 17,973 bundles is the actual top-producing bed across 3,253 harvest records; JJ-4 was indeed fertilized 21 times in 2022).
- The semantic-routed query about yellowing leaves cited actual log entries by date and bed, rather than inventing a generic horticulture answer (a failure mode seen during early development before prompt grounding was tightened).
- The recommend-routed query produced a forward-looking watering recommendation that referenced specific past activity for JJ-7, comparable past entries from the narrative log, and live weather forecast data fetched via the MCP weather server.

**Goal achievement:** The original goal was to build a hybrid RAG system that distinguishes factual vs. analytical queries and uses adaptive retrieval. The final system goes beyond this — it implements four distinct routes (SQL, semantic, conversational, recommend) and demonstrates predictive multi-source synthesis, which was originally listed as out of scope. All five spec components are functional end-to-end.

## Setup

Prerequisites: Python 3.11, [Ollama](https://ollama.com), Git.

```powershell
# Clone and install
git clone https://github.com/<username>/farm-rag.git
cd farm-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Pull models + build custom embedding model with extended context
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama create nomic-embed-long -f Modelfile

# Drop your farm docs into docs/, then ingest
python ingest.py

# Run
streamlit run app.py
```

## References and tools used

### AI tools

- **Claude Code (Anthropic)** - used as an interactive pair-programming assistant throughout development. Specific uses:

  - Architectural discussion and design decisions for the LangGraph state machine and multi-route classifier

  - Debugging assistance - particularly around the LangChain/Chroma context-length issue (resolved by building a custom Ollama embedding model via Modelfile), the MCP SDK API surface, and Streamlit chat rendering quirks

  - Assistance in documentation formatting for this README and PROJECT.md 

The system was end-to-end tested before submission.

## Hardware

Developed and tested on:

- **Primary development machine:** Windows 11 PC (specs: 32Gb Ram, AMD Ryzen 7 7800X3D, NVIDIA RTX 5070)

- **Secondary / demo machine:** MacBook Air M2, 16 GB unified memory (macOS)

Local inference runs `llama3.1:8b` (≈ 5 GB) and a custom `nomic-embed-long` embedding model (≈ 280 MB) via Ollama. Apple Silicon's Metal acceleration handles inference comfortably; the Windows machine relies on CPU (or GPU if available). No cloud APIs used for inference.

## Known limitations and future work

- **Comparison queries** (e.g., *"compare fertilizer patterns between JJ and CC beds"*) are ambiguously routed — they're neither pure SQL nor pure narrative, and the current classifier handles them inconsistently.
- **Query rewriting** for ambiguous or too-short queries is not yet implemented. The classifier handles pronoun resolution but doesn't reformulate terse queries before retrieval.
- **Streaming responses** were replaced with invoke-plus-spinner to eliminate an artifact in LangGraph's `stream_mode="messages"` that leaked intermediate node tokens into the UI. Re-adding token streaming would require proper event-level filtering via `astream_events`.

## Stack

- **LLM**: Ollama + `llama3.1:8b` (local inference, no cloud)
- **Embeddings**: Ollama + `nomic-embed-long` (custom Modelfile, 8192-token context)
- **Vector store**: Chroma (persistent on disk)
- **Structured store**: SQLite (8 tables, auto-generated from xlsx)
- **Weather**: Open-Meteo (free, no API key) for forecast and historical data, El Cajon CA coordinates
- **MCP**: Official `mcp` Python SDK for the weather server/client integration
- **Orchestration**: LangChain + LangGraph with `MemorySaver` checkpointer
- **UI**: Streamlit

## Files
farm-rag/
├── ingest.py           # Markdown → Chroma, xlsx → SQLite
├── bed_names.py        # Bed-name normalization between short/long forms
├── weather.py          # Open-Meteo client (forecast + historical)
├── weather_mcp_server.py  # MCP server exposing weather tools over stdio
├── graph.py            # LangGraph state machine: classifier + SQL/semantic routes + generator
├── app.py              # Streamlit UI
├── Modelfile           # Custom Ollama embedding model with extended context
├── PROJECT.md          # Project plan and scope
├── debug_graph.py      # Graph invocation sanity check
├── debug_memory.py     # Cross-turn memory validation
├── debug_recommend.py  # Recommend route validation
└── docs/               # Farm documents (private, not in repo)
