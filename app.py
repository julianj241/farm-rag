import uuid
import streamlit as st
from graph import graph as _graph

SNIPPET_CHARS = 200


@st.cache_resource
def get_graph():
    return _graph


def source_label(doc) -> str:
    meta = doc.metadata
    label = f"{meta.get('month', '?')} {meta.get('year', '?')}"
    if meta.get("total_parts", 1) > 1:
        label += f" (part {meta['part']}/{meta['total_parts']})"
    return label


def main() -> None:
    st.set_page_config(page_title="Farm Log Assistant", layout="wide")
    st.title("Farm Log Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "last_state" not in st.session_state:
        st.session_state.last_state = None

    st.caption(
        f"Session thread_id: {st.session_state.thread_id[:8]}... | "
        f"Messages in history: {len(st.session_state.messages)}"
    )

    thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Sidebar: route indicator + retrieval details from the last query
    with st.sidebar:
        st.header("Retrieval")
        s = st.session_state.last_state
        if s is None:
            st.caption("Retrieval details will appear here after your first query.")
        elif s.get("route") == "sql":
            st.caption("🗄 structured (SQL)")
            if s.get("sql_query"):
                st.code(s["sql_query"], language="sql")
            if s.get("sql_result"):
                st.text(s["sql_result"])
        elif s.get("route") == "conversational":
            st.caption("💬 conversational (history only)")
            st.caption("Answered from conversation context without retrieval.")
        else:
            st.caption("📚 narrative (Chroma)")
            for doc in s.get("retrieved_docs", []):
                with st.expander(source_label(doc)):
                    snippet = doc.page_content[:SNIPPET_CHARS].strip()
                    if len(doc.page_content) > SNIPPET_CHARS:
                        snippet += "…"
                    st.caption(snippet)

    # Replay chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # New query
    if query := st.chat_input("Ask about the farm logs…"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = get_graph().invoke(
                    {"question": query},
                    config=thread_config,
                )
            response = result["answer"]
            st.write(response)
        st.session_state.last_state = get_graph().get_state(thread_config).values
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()