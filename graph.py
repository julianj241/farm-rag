import re
import sqlite3
from pathlib import Path
from typing import Annotated

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

ROOT = Path(__file__).parent
CHROMA_DIR = str(ROOT / "chroma_db")
SQLITE_DB = ROOT / "farm.db"
COLLECTION_NAME = "field_log"
EMBED_MODEL = "nomic-embed-long"
LLM_MODEL = "llama3.1:8b"
RETRIEVE_K = 4
SQL_ROW_LIMIT = 20

# ── Prompts ───────────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """You are a query classifier for a farm assistant. Classify the user's question as either "sql" or "semantic".

Reply with EXACTLY one word: sql  or  semantic

Use "sql" for questions about:
- Counts, totals, sums, averages, yields
- Specific records: when was X planted/harvested, how many bundles from bed Y
- Rankings: most/least/top/bottom, highest/lowest
- Scheduled or expected dates and durations

Use "semantic" for questions about:
- Narrative observations, trends, patterns over time
- Reasoning or analysis ("why", "how did", "what happened to")
- Open-ended questions about conditions, problems, decisions
- Anything requiring interpretation of log entries

Follow-up handling: If the current question is a pronoun-heavy follow-up ("what about it?", "which one?", "that bed?", "we just talked about"), use the conversation history above to understand what's actually being asked, then classify based on the resolved meaning."""

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

SEMANTIC_SYSTEM = """You are an assistant for an organic market garden. Answer questions using ONLY the field log excerpts provided below. Do not add general gardening knowledge, examples, or advice from outside the logs.

Rules:
- Cite specific dates and bed names from the excerpts.
- If the excerpts don't contain the answer, say exactly: "I don't see that in the logs." Do not speculate.
- Do not make up numbers. Do not estimate totals. If asked for counts or aggregates, say: "I can only reason over the narrative log, not compute totals. For precise figures, a structured-data query is needed."

Retrieved field log excerpts:
{context}"""

SQL_ANSWER_SYSTEM = """\
You are an assistant reporting farm data results. The user asked a question and a SQL query was run to answer it.
Present the answer clearly in 1–3 sentences of plain prose. Refer to specific numbers from the result.
If the result is empty, say so plainly. Do not add information beyond what the query result shows."""

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

# ── Nodes ─────────────────────────────────────────────────────────────────────

def classify_query(state: State) -> dict:
    recent = state.get("messages", [])[-4:]  # last 2 turns (user+ai pairs)
    response = _llm.invoke([
        SystemMessage(content=CLASSIFY_PROMPT),
        *recent,
        HumanMessage(content=state["question"]),
    ])
    first_word = response.content.strip().lower().split()[0] if response.content.strip() else ""
    route = "sql" if first_word == "sql" else "semantic"
    # Reset path-specific fields so stale values from a prior turn don't bleed through
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
            *state["messages"],
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
            *state["messages"],
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


def _route(state: State) -> str:
    return state["route"]

# ── Graph assembly ────────────────────────────────────────────────────────────

_builder = StateGraph(State)
_builder.add_node("classify_query", classify_query)
_builder.add_node("semantic_retrieve", semantic_retrieve)
_builder.add_node("sql_query", sql_query)
_builder.add_node("generate_answer", generate_answer)

_builder.set_entry_point("classify_query")
_builder.add_conditional_edges(
    "classify_query",
    _route,
    {"semantic": "semantic_retrieve", "sql": "sql_query"},
)
_builder.add_edge("semantic_retrieve", "generate_answer")
_builder.add_edge("sql_query", "generate_answer")
_builder.add_edge("generate_answer", END)

graph = _builder.compile(checkpointer=MemorySaver())
