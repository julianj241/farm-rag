# TODO — Milestone 2

## Bugs to fix
- [ ] **Memory doesn't persist in Streamlit UI** — `debug_memory.py` proves MemorySaver works at graph level; issue is in the Streamlit ↔ graph integration. Suspected cause: `@st.cache_resource` + `MemorySaver` in-memory store interaction. Try: swap MemorySaver for `SqliteSaver` with a file path so state survives script reruns regardless of caching behavior.
- [ ] **Duplicate text rendering** was fixed by replacing `st.write_stream` with `st.write` + spinner. Re-add streaming using `graph.astream_events(version="v2")` with `event["event"] == "on_chat_model_stream"` and `event["metadata"]["langgraph_node"] == "generate_answer"` filtering.

## Classifier improvements
- [ ] **Meta-questions** ("what did we just discuss?", "summarize our conversation") are misrouted. Add a `conversational` route that skips retrieval and answers from message history alone.
- [ ] **Comparison queries** ("compare X and Y") route inconsistently. Needs either a multi-hop agent or a "comparison" classifier category that fires two sub-queries and merges.
- [ ] **Numerical follow-ups** ("what about 2023?") — the classifier sees the year but loses the bed context. Consider passing resolved entities forward in state.

## Spec components still to add
- [ ] **MCP weather server** — weather tool. Build as separate MCP server, add tool node to graph.
- [ ] **Query rewriter** — LLM node that expands/reformulates the user query before retrieval.
- [ ] **Citation system** — return specific log dates in answers, not just month+year.
- [ ] **Adaptive retrieval strategies** — different k, different filters based on query type.

## UX polish
- [ ] Sidebar filters by bed, date range, farm area
- [ ] Download conversation transcript
- [ ] Clean up Summary sheet ingestion (currently produces "Unnamed: 1" column)

## Code hygiene
- [ ] Extract shared helpers (source_label duplicated between `app.py` and `graph.py`)
- [ ] Add proper error handling around LLM calls
- [ ] Write actual tests for `bed_names.py` normalization