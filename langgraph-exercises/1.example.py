from langgraph.graph import StateGraph, END


def add_one(state: dict):
    return {"x": state["x"] + 1}

def multiply_two(state: dict):
    return {"x": state["x"] * 2}

workflow = StateGraph(dict)

workflow.add_node("add_node", add_one)

workflow.add_node("multiply_node", multiply_two)

workflow.set_entry_point("add_node")

workflow.add_edge("add_node", "multiply_node")

workflow.add_edge("multiply_node", END)

app = workflow.compile()

result = app.invoke({"x": 3 })

print(result)




# --- Visualization part ---
graph = app.get_graph()
graph.draw("workflow.png", prog="dot")  # Save as image
print("Graph saved as workflow.png")

