from graph import graph

config = {"configurable": {"thread_id": "rec-test-1"}}
result = graph.invoke(
    {"question": "Should I water JJ-7 tomorrow?"},
    config=config,
)

print("=== ROUTE ===")
print(result.get("route"))
print("\n=== ANSWER ===")
print(result.get("answer"))
print("\n=== CONTEXT (truncated) ===")
print((result.get("sql_result") or "")[:800])