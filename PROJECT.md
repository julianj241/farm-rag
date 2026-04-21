# Farm RAG Assistant

## Stack
- Python 3.11.9
- LangChain + LangGraph
- Chroma (persistent, ./chroma_db)
- Ollama: llama3.1:8b (LLM) + nomic-embed-text (embeddings)
- Streamlit

## Goal (today's milestone)
Working end-to-end RAG over farm docs in Streamlit, wrapped in a minimal LangGraph with MemorySaver for short-term memory.

## In scope today
1. Ingest markdown log -> Chroma (chunked by month, metadata: year/month/farm)
2. Ingest xlsx sheets -> SQLite (beds, plantings, harvests, fertilizer, amendments, weather, irrigation)
3. Streamlit chat UI with streaming responses
4. LangGraph with classifier + 2 retrieval nodes:
   - semantic_search (Chroma) for narrative/analytical queries
   - sql_query (SQLite) for quantitative/factual queries
5. MemorySaver checkpointer with per-session thread_id

## Out of scope today (Milestone 2)
- MCP weather server (Weather sheet gives us historical weather for now)
- Query rewriter
- Citation back to specific log dates

## Doc formats in docs/
- field_log.md: narrative daily log, Jan 2022 - Apr 2026, ~3,274 lines, structured by month
- farm_records.xlsx: 8 sheets of structured records (Beds, Plantings, Harvests, Fertilizer, Amendments, Weather, Irrigation)