from graph import graph

config = {"configurable": {"thread_id": "memtest-1"}}

# Turn 1: establish context
print("=== TURN 1 ===")
result1 = graph.invoke(
    {"question": "How many times was JJ-4 fertilized in 2022?"},
    config=config,
)
print("Route:", result1.get("route"))
print("Answer:", result1.get("answer"))
print("Messages in state:", len(result1.get("messages", [])))

print("\n=== TURN 2 (memory test) ===")
result2 = graph.invoke(
    {"question": "What bed were we just talking about?"},
    config=config,
)
print("Route:", result2.get("route"))
print("Answer:", result2.get("answer"))
print("Messages in state:", len(result2.get("messages", [])))