from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents import (
    GraphState,
    mode_selector_node,
    validator_node,
    scriptwriter_node,
    character_node,
    image_node,
    memory_commit_node
)

def route_mode(state: GraphState) -> str:
    if state["input_mode"] == "manual":
        return "validator"
    return "scriptwriter"

def route_after_validator(state: GraphState) -> str:
    if state.get("status") == "validation_failed":
        return END
    return "character"

def compile_workflow():
    builder = StateGraph(GraphState)
    
    # Define Nodes
    builder.add_node("mode_selector", mode_selector_node)
    builder.add_node("validator", validator_node)
    builder.add_node("scriptwriter", scriptwriter_node)
    builder.add_node("character", character_node)
    builder.add_node("image", image_node)
    builder.add_node("memory_commit", memory_commit_node)
    
    # Build Edges
    builder.add_edge(START, "mode_selector")
    
    # Mode dependent routing
    builder.add_conditional_edges(
        "mode_selector",
        route_mode,
        {"validator": "validator", "scriptwriter": "scriptwriter"}
    )
    
    # Validator to character (or END if validation failed)
    builder.add_conditional_edges(
        "validator",
        route_after_validator,
        {END: END, "character": "character"}
    )
    
    # Scriptwriter to character
    builder.add_edge("scriptwriter", "character")
    
    # The rest of the pipeline
    builder.add_edge("character", "image")
    builder.add_edge("image", "memory_commit")
    builder.add_edge("memory_commit", END)
    
    # Memory for HITL / Checkpointing
    memory = MemorySaver()
    
    # We interrupt before "character" to enforce Human-in-the-loop (HITL) 
    # to approve the generated/validated script before proceeding to heavy generation
    graph = builder.compile(checkpointer=memory, interrupt_before=["character"])
    return graph
