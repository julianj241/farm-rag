from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

CHROMA_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "field_log"
EMBED_MODEL = "nomic-embed-long"
LLM_MODEL = "llama3.1:8b"
RETRIEVE_K = 4
SNIPPET_CHARS = 200

SYSTEM_TEMPLATE = """You are an assistant for an organic market garden. Answer questions using ONLY the field log excerpts provided below. Do not add general gardening knowledge, examples, or advice from outside the logs.

Rules:
- Cite specific dates and bed names from the excerpts.
- If the excerpts don't contain the answer, say exactly: "I don't see that in the logs." Do not speculate.
- Do not make up numbers. Do not estimate totals. If asked for counts or aggregates, say: "I can only reason over the narrative log, not compute totals. For precise figures, a structured-data query is needed."

Retrieved field log excerpts:
{context}
"""


@st.cache_resource
def load_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
    )


def retrieve(vs: Chroma, query: str) -> list:
    return vs.similarity_search(query, k=RETRIEVE_K)


def build_messages(docs: list, query: str) -> list:
    context = "\n\n---\n\n".join(
        f"[{source_label(doc)}]\n{doc.page_content}" for doc in docs
    )
    return [
        SystemMessage(content=SYSTEM_TEMPLATE.format(context=context)),
        HumanMessage(content=query),
    ]


def source_label(doc) -> str:
    meta = doc.metadata
    label = f"{meta.get('month', '?')} {meta.get('year', '?')}"
    if meta.get("total_parts", 1) > 1:
        label += f" (part {meta['part']}/{meta['total_parts']})"
    return label


def main() -> None:
    st.set_page_config(page_title="Farm Log Assistant", layout="wide")
    st.title("Farm Log Assistant")

    vs = load_vectorstore()
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = []

    # Sidebar: retrieved sources from the last query
    with st.sidebar:
        st.header("Retrieved Sources")
        if st.session_state.sources:
            for doc in st.session_state.sources:
                with st.expander(source_label(doc)):
                    snippet = doc.page_content[:SNIPPET_CHARS].strip()
                    if len(doc.page_content) > SNIPPET_CHARS:
                        snippet += "…"
                    st.caption(snippet)
        else:
            st.caption("Sources will appear here after your first query.")

    # Replay chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # New query
    if query := st.chat_input("Ask about the farm logs…"):
        with st.chat_message("user"):
            st.write(query)

        docs = retrieve(vs, query)
        st.session_state.sources = docs

        with st.chat_message("assistant"):
            response = st.write_stream(
                chunk.content for chunk in llm.stream(build_messages(docs, query))
            )

        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
