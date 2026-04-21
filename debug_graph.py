from graph import graph

config = {"configurable": {"thread_id": "debug-1"}}
result = graph.invoke(
    {"question": "Which bed produces the most arugula?"},
    config=config,
)

print("=== ROUTE ===")
print(result.get("route"))
print("\n=== SQL ===")
print(result.get("sql_query"))
print("\n=== SQL RESULT ===")
print(result.get("sql_result"))
print("\n=== DOCS ===")
for d in result.get("retrieved_docs") or []:
    print(" -", d.metadata)
print("\n=== ANSWER ===")
print(result.get("answer"))