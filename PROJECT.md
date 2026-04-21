# Farm RAG Assistant

## Stack
- Python 3.11.9
- LangChain + LangGraph
- Chroma (persistent, ./chroma_db) for narrative retrieval
- SQLite (./farm.db) for structured queries
- Ollama: llama3.1:8b (LLM) + nomic-embed-text (embeddings)
- Streamlit

## Goal (today's milestone)
Working end-to-end hybrid RAG over farm docs in Streamlit, wrapped in a LangGraph with a classifier node that routes queries to either SQL (factual) or semantic search (analytical).

## In scope today
1. Ingest field_log.md -> Chroma (chunk by month, metadata: year/month/farm area)
2. Ingest farm_records.xlsx -> SQLite (one table per sheet: beds, plantings, harvests, fertilizer, amendments, weather, irrigation)
3. Streamlit chat UI with streaming responses
4. LangGraph with classifier + 2 retrieval nodes:
   - semantic_search (Chroma) for narrative/analytical queries
   - sql_query (SQLite) for quantitative/factual queries
5. MemorySaver checkpointer with per-session thread_id

## Out of scope today (Milestone 2)
- MCP weather server
- Query rewriter
- Full citation system with log dates

## Data in docs/
- field_log.md: narrative daily log, Jan 2022 - Apr 2026, ~3,274 lines, structured by month (## headings)
- farm_records.xlsx: 8 sheets — Summary, Beds, Plantings, Harvests, Fertilizer, Amendments, Weather, Irrigation

## Bed naming convention
- Three farm locations: Upstairs (U), JJ, ChickenCoop (CC)
- Beds: U1-U8, JJ-1 through JJ-12, CC1-CC10
- Field log uses short forms (U5, JJ-7, CC6); xlsx uses long forms (Upstairs-5, JJ-7, ChickenCoop-6)
- Ingestion must handle both