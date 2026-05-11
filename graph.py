import asyncio
import re
import sqlite3
import sys
from pathlib import Path
from typing import Annotated

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing_extensions import TypedDict

from bed_names import find_first_bed

ROOT = Path(__file__).parent
CHROMA_DIR = str(ROOT / "chroma_db")
SQLITE_DB = ROOT / "farm.db"
COLLECTION_NAME = "field_log"
EMBED_MODEL = "nomic-embed-long"
LLM_MODEL = "llama3.1:8b"
RETRIEVE_K = 4
SQL_ROW_LIMIT = 20

# Keywords that signal a meta-question about the conversation itself.
# If any of these appear in the user's question, we skip retrieval entirely
# and answer from conversation history.
_META_PATTERNS = [
    "we just", "we were just", "we discussed", "we talked",
    "previous", "earlier", "last question", "last bed",
    "just asked", "just talking about", "just discussing",
    "what did i ask", "which bed did i", "what bed did i",
    "what were we", "which one", "that bed",
]


def _is_conversational(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in _META_PATTERNS)


# ── Prompts ───────────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """Classify the question. Reply with ONE word: sql, semantic, or recommend.

Decide based on TIME ORIENTATION:

sql — asks about PAST/EXISTING facts and numbers from the database.
  Verb tense: produces, produced, was, has, had, last, ever
  Examples:
    "Which bed produces the most arugula?" → sql
    "How many times was JJ-4 fertilized in 2022?" → sql
    "What's the highest yield ever?" → sql
    "When was JJ-7 last fertilized?" → sql

semantic — asks about PAST narrative patterns or interpretation.
  Verb tense: typically descriptive past
  Examples:
    "When do yellowing leaves usually happen?" → semantic
    "Why did CC3 underperform last summer?" → semantic
    "How does the farmer handle heat waves?" → semantic

recommend — asks about FUTURE actions or decisions.
  Verb tense: should, will, going to, tomorrow, this week, next
  Examples:
    "Should I water JJ-7 tomorrow?" → recommend
    "What should I do this weekend?" → recommend
    "Is the heat wave going to affect JJ-5?" → recommend

If the question has no future-oriented words (should/tomorrow/will/next), it is NEVER recommend.

For pronoun-heavy follow-ups, resolve from conversation history first."""

DB_SCHEMA = """\
Tables in farm.db (SQLite). Use double-quoted identifiers for all column names.

summary
  "Farm Records Summary — Arugula & Watercress, El Cajon CA"  TEXT
  "Unnamed: 1"                                                 TEXT

beds
  "Bed ID"      TEXT     -- e.g. 'Upstairs-5', 'JJ-7', 'ChickenCoop-3'
  "Farm"        TEXT     -- 'Upstairs', 'JJ', or 'ChickenCoop'
  "Length (ft)" INTEGER
  "Width (ft)"  INTEGER
  "Square Ft"   INTEGER
  "Soil/Notes"  TEXT
  "Status"      TEXT
  "Current Use" TEXT

plantings
  "Plant Date"             TIMESTAMP
  "Bed"                    TEXT
  "Farm"                   TEXT
  "Crop"                   TEXT
  "Season"                 TEXT
  "Cycle #"                INTEGER
  "Expected Harvest-Ready" TIMESTAMP
  "Days to Harvest"        INTEGER
  "Expected Bundles"       INTEGER

harvests
  "Harvest Date" TIMESTAMP
  "Bed"          TEXT
  "Farm"         TEXT
  "Crop"         TEXT
  "Bundles"      INTEGER
  "Cycle #"      INTEGER
  "Note"         TEXT

fertilizer
  "Date"    TIMESTAMP
  "Bed"     TEXT
  "Farm"    TEXT
  "Crop"    TEXT
  "Product" TEXT
  "Reason"  TEXT
  "Cycle #" INTEGER

amendments
  "Date"     TIMESTAMP
  "Bed"      TEXT
  "Farm"     TEXT
  "Material" TEXT
  "Quantity" TEXT
  "Note"     TEXT

weather
  "Date"       TIMESTAMP
  "Event Type" TEXT
  "Detail"     TEXT

irrigation
  "Date"   TIMESTAMP
  "Action" TEXT
  "Reason" TEXT

VALUE CONVENTIONS (critical):
- "Crop" column: always lowercase — 'arugula' or 'watercress'. Never 'Arugula'.
- "Farm" column: three values — 'Upstairs', 'JJ', 'ChickenCoop' (exact capitalization)
- "Bed" / "Bed ID" columns: long form — 'Upstairs-5', 'JJ-7', 'ChickenCoop-6'

TABLE USAGE GUIDE:
- For production / yield / harvest questions ("most arugula", "top producing bed"): use "harvests" table, SUM("Bundles"). Do NOT count plantings.
- For planting activity ("when was X planted", "how many cycles"): use "plantings" table.
- For feeding/fertilizer questions: use "fertilizer" table.
- For irrigation or weather events: use "irrigation" or "weather" tables.
- For bed metadata (size, location, status): use "beds" table.
- "Harvest Date", "Plant Date", "Date" columns are ISO strings — filter with strftime('%Y', "Harvest Date") = '2024' for year, or "Harvest Date" LIKE '2024-%'."""

SQL_SYSTEM = """\
You are a SQL expert for a SQLite farm database. Write a single SELECT query to answer the question.

Rules:
- Output ONLY the raw SQL — no explanation, no markdown fences, no prose.
- Use double-quoted identifiers for every column name (e.g. "Bed ID", "Cycle #", "Harvest Date").
- Bed names use long form: Upstairs-N, JJ-N, ChickenCoop-N.
- Date columns are ISO-format strings; use string comparison or strftime() for filtering.
- Add LIMIT {limit} unless the question implies a smaller result set.

Database schema:
{schema}"""

SEMANTIC_SYSTEM = """You are an assistant for an organic market garden. Answer the user's question using the field log excerpts AND the prior conversation history.

Rules:
- For factual questions about the farm: answer using ONLY the field log excerpts. Cite specific dates and bed names. If the excerpts don't cover it, say exactly: "I don't see that in the logs."
- For meta-questions about the conversation itself ("what were we just discussing?", "which bed did I ask about?"): answer from the prior conversation history above. These are always valid.
- Do not make up numbers or estimate totals. If asked for counts or aggregates, say: "I can only reason over the narrative log, not compute totals. For precise figures, a structured-data query is needed."

Retrieved field log excerpts:
{context}"""

SQL_ANSWER_SYSTEM = """\
You are an assistant reporting farm data results. The user asked a question and a SQL query was run to answer it.
Present the answer clearly in 1–3 sentences of plain prose. Refer to specific numbers from the result.
If the result is empty, say so plainly. Do not add information beyond what the query result shows."""

CONVERSATIONAL_SYSTEM = """You are an assistant answering a question about the prior conversation. The conversation history is provided above in the message list. Answer the user's question directly and briefly based on that history. Do not apologize or claim you lack context — the history is there. Reference specific beds, numbers, or topics from earlier turns."""

RECOMMEND_SYSTEM = """You are an experienced market garden assistant. The user is asking for forward-looking advice about a specific bed or condition. You have been given:
- Recent activity for the relevant bed from structured records (irrigation + fertilizer)
- Narrative log excerpts from similar past situations
- Upcoming weather forecast for El Cajon, CA

Synthesize a specific, actionable recommendation. Cite which source informs each part of your answer (e.g., "based on last week's irrigation pattern...", "given tomorrow's forecast of...", "in past entries with similar conditions..."). Be concrete — name beds, dates, fertilizer products if relevant. If information is missing, say what you'd need to give a stronger recommendation.

Keep your response to 3 short paragraphs maximum. Be concise and concrete."""

# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    question: str
    route: str
    retrieved_docs: list          # list[Document]
    sql_query: str | None
    sql_result: str | None
    answer: str
    messages: Annotated[list[BaseMessage], add_messages]

# ── Shared resources (initialised once at import time) ────────────────────────

_llm = ChatOllama(model=LLM_MODEL, temperature=0)
_vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _source_label(doc: Document) -> str:
    meta = doc.metadata
    label = f"{meta.get('month', '?')} {meta.get('year', '?')}"
    if meta.get("total_parts", 1) > 1:
        label += f" (part {meta['part']}/{meta['total_parts']})"
    return label


def _extract_sql(text: str) -> str:
    """Strip markdown fences and take only the first SQL statement."""
    text = re.sub(r"```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    if ";" in text:
        text = text[: text.index(";") + 1]
    else:
        text = text.strip() + ";"
    return text.strip()


def _format_table(cols: list[str], rows: list) -> str:
    if not rows:
        return "(no rows returned)"
    widths = [
        max(len(str(c)), max((len(str(r[i] if r[i] is not None else "NULL")) for r in rows), default=0))
        for i, c in enumerate(cols)
    ]
    def fmt(row):
        return " | ".join(str(v if v is not None else "NULL").ljust(w) for v, w in zip(row, widths))
    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([fmt(cols), sep] + [fmt(r) for r in rows])


async def _call_weather_tool_async(tool_name: str, arguments: dict) -> str:
    """Spawn the weather MCP server as a stdio subprocess and call one tool."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "weather_mcp_server.py")],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            texts = [c.text for c in result.content if hasattr(c, "text")]
            return "\n".join(texts)


def _fetch_weather_via_mcp(days: int = 3) -> str:
    """Sync wrapper around the async MCP client call."""
    return asyncio.run(_call_weather_tool_async("get_forecast", {"days": days}))

# ── Nodes ─────────────────────────────────────────────────────────────────────

def classify_query(state: State) -> dict:

    # Fast path: meta-questions about the conversation itself.
    # Skip retrieval entirely and answer from history.
    if _is_conversational(state["question"]):
        return {"route": "conversational", "retrieved_docs": [], "sql_query": None, "sql_result": None, "answer": ""}

    # LLM classification for everything else
    recent = state.get("messages", [])[-4:]  # last 2 turns (user+ai pairs)
    response = _llm.invoke([
        SystemMessage(content=CLASSIFY_PROMPT),
        *recent,
        HumanMessage(content=state["question"]),
    ])
    first_word = response.content.strip().lower().split()[0] if response.content.strip() else ""
    if first_word == "sql":
        route = "sql"
    elif first_word == "recommend":
        route = "recommend"
    else:
        route = "semantic"
    return {"route": route, "retrieved_docs": [], "sql_query": None, "sql_result": None, "answer": ""}


def semantic_retrieve(state: State) -> dict:
    docs = _vectorstore.similarity_search(state["question"], k=RETRIEVE_K)
    return {"retrieved_docs": docs}


def sql_query(state: State) -> dict:
    response = _llm.invoke([
        SystemMessage(content=SQL_SYSTEM.format(schema=DB_SCHEMA, limit=SQL_ROW_LIMIT)),
        HumanMessage(content=state["question"]),
    ])
    sql = _extract_sql(response.content)

    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        return {"sql_query": sql, "sql_result": "ERROR: Only SELECT queries are permitted."}

    try:
        with sqlite3.connect(str(SQLITE_DB)) as conn:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchmany(SQL_ROW_LIMIT)
            cols = [d[0] for d in cur.description]
        result = _format_table(cols, rows)
    except Exception as exc:
        result = f"ERROR: {exc}"

    return {"sql_query": sql, "sql_result": result}


def generate_answer(state: State) -> dict:
    if state["route"] == "semantic":
        context = "\n\n---\n\n".join(
            f"[{_source_label(doc)}]\n{doc.page_content}"
            for doc in state["retrieved_docs"]
        )
        messages = [
            SystemMessage(content=SEMANTIC_SYSTEM.format(context=context)),
            *state["messages"][-4:],
            HumanMessage(content=state["question"]),
        ]
    else:
        body = (
            f"Question: {state['question']}\n\n"
            f"SQL executed:\n{state['sql_query']}\n\n"
            f"Result:\n{state['sql_result']}"
        )
        messages = [
            SystemMessage(content=SQL_ANSWER_SYSTEM),
            *state["messages"][-4:],
            HumanMessage(content=body),
        ]

    response_text = ""
    for chunk in _llm.stream(messages):
        response_text += chunk.content
    return {
        "answer": response_text,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=response_text),
        ],
    }


def conversational_answer(state: State) -> dict:
    """Answer meta-questions about the conversation itself, using only history."""
    recent = state.get("messages", [])[-10:]  # last 5 turns
    messages = [
        SystemMessage(content=CONVERSATIONAL_SYSTEM),
        *recent,
        HumanMessage(content=state["question"]),
    ]
    response_text = ""
    for chunk in _llm.stream(messages):
        response_text += chunk.content
    return {
        "answer": response_text,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=response_text),
        ],
    }


def recommend_answer(state: State) -> dict:
    question = state["question"]
    bed = find_first_bed(question)

    # Hop 1: structured recent activity
    sql_context = ""
    try:
        with sqlite3.connect(str(SQLITE_DB)) as conn:
            if bed:
                fert_rows = conn.execute(
                    'SELECT "Date", "Product", "Reason" FROM fertilizer '
                    'WHERE "Bed" = ? ORDER BY "Date" DESC LIMIT 5',
                    (bed,)
                ).fetchall()
            else:
                fert_rows = []
            irr_rows = conn.execute(
                'SELECT "Date", "Action", "Reason" FROM irrigation '
                'ORDER BY "Date" DESC LIMIT 5'
            ).fetchall()
        sql_context = (
            f"Recent irrigation events:\n"
            + "\n".join(f"  {d} | {a} | {r}" for d, a, r in irr_rows)
            + (f"\n\nRecent fertilizer for {bed}:\n"
               + "\n".join(f"  {d} | {p} | {r}" for d, p, r in fert_rows)
               if bed else "")
        )
    except Exception as exc:
        sql_context = f"(structured data lookup error: {exc})"

    # Hop 2: narrative context
    docs = _vectorstore.similarity_search(question, k=2)
    narrative_context = "\n\n---\n\n".join(
        f"[{_source_label(d)}]\n{d.page_content[:400]}"
        for d in docs
    )

    # Hop 3: weather (via MCP)
    try:
        forecast = _fetch_weather_via_mcp(days=3)
    except Exception as exc:
        forecast = f"(weather lookup error: {exc})"

    # Synthesis
    body = (
        f"User question: {question}\n\n"
        f"Bed identified: {bed or '(none — query is general)'}\n\n"
        f"=== Recent structured activity ===\n{sql_context}\n\n"
        f"=== Relevant narrative excerpts ===\n{narrative_context}\n\n"
        f"=== Upcoming weather (El Cajon, CA, next 5 days) ===\n{forecast}"
    )

    messages = [
        SystemMessage(content=RECOMMEND_SYSTEM),
        *state.get("messages", [])[-4:],
        HumanMessage(content=body),
    ]

    response_text = ""
    for chunk in _llm.stream(messages):
        response_text += chunk.content

    return {
        "answer": response_text,
        "retrieved_docs": list(docs),
        "sql_query": "(multi-hop synthesis)",
        "sql_result": sql_context[:1500] + "\n\n--- Weather ---\n" + forecast,
        "messages": [
            HumanMessage(content=question),
            AIMessage(content=response_text),
        ],
    }


def _route(state: State) -> str:
    return state["route"]

# ── Graph assembly ────────────────────────────────────────────────────────────

_builder = StateGraph(State)
_builder.add_node("classify_query", classify_query)
_builder.add_node("semantic_retrieve", semantic_retrieve)
_builder.add_node("sql_query", sql_query)
_builder.add_node("generate_answer", generate_answer)
_builder.add_node("conversational_answer", conversational_answer)
_builder.add_node("recommend_answer", recommend_answer)

_builder.set_entry_point("classify_query")
_builder.add_conditional_edges(
    "classify_query",
    _route,
    {
        "semantic": "semantic_retrieve",
        "sql": "sql_query",
        "conversational": "conversational_answer",
        "recommend": "recommend_answer",
    },
)
_builder.add_edge("semantic_retrieve", "generate_answer")
_builder.add_edge("sql_query", "generate_answer")
_builder.add_edge("generate_answer", END)
_builder.add_edge("conversational_answer", END)
_builder.add_edge("recommend_answer", END)

graph = _builder.compile(checkpointer=MemorySaver())