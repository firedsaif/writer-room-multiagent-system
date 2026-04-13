from typing import TypedDict, List, Dict, Any, Optional
from mcp_registry import mcp_registry

# Import tools so they register themselves in the registry
import tools

class GraphState(TypedDict):
    input_mode: str
    prompt: Optional[str]
    raw_script: Optional[str]
    script: Dict[str, Any]
    characters: List[Dict[str, Any]]
    images: List[str]
    status: str
    errors: List[str]

def mode_selector_node(state: GraphState) -> GraphState:
    print(f"--- MODE SELECTOR: {state['input_mode']} ---")
    return state

def validator_node(state: GraphState) -> GraphState:
    print("--- SCRIPT VALIDATOR AGENT ---")
    raw_script = state.get("raw_script")
    
    # Dynamically discovery tool
    # Wait, in the node we use the registry executor directly for simplicity
    if raw_script:
        result = mcp_registry.execute_tool("validate_script", {"raw_script": raw_script})
        if not result.get("is_valid", False):
            state["errors"] = result.get("errors", [])
            state["status"] = "validation_failed"
            print(f"Validation Failed: {state['errors']}")
        else:
            state["status"] = "processing"
            
        state["script"] = result.get("standardized_script", {})
    return state

def scriptwriter_node(state: GraphState) -> GraphState:
    print("--- SCRIPTWRITER AGENT ---")
    prompt = state.get("prompt", "Write a short scene.")
    
    # Query registry dynamically (implicitly done via executor)
    result = mcp_registry.execute_tool("generate_script_segment", {"prompt": prompt, "num_scenes": 2})
    state["script"] = result
    state["status"] = "processing"
    
    # Also we can assume commit_memory is called at the end, but the LangGraph workflow handles it at the end
    return state

def character_node(state: GraphState) -> GraphState:
    print("--- CHARACTER DESIGNER AGENT ---")
    script = state.get("script", {})
    if not script:
        return state
        
    result = mcp_registry.execute_tool("extract_characters", {"script_json": script})
    state["characters"] = result.get("characters", [])
    print(f"Extracted {len(state['characters'])} characters.")
    return state

def image_node(state: GraphState) -> GraphState:
    print("--- IMAGE SYNTHESIZER AGENT ---")
    characters = state.get("characters", [])
    images = []
    for char in characters:
        name = char.get("name", "Unknown")
        appearance = char.get("appearance_description", "")
        img_path = mcp_registry.execute_tool("generate_character_image", {
            "character_name": name,
            "appearance_description": appearance
        })
        images.append(img_path)
    state["images"] = images
    return state

def memory_commit_node(state: GraphState) -> GraphState:
    print("--- MEMORY COMMIT LAYER ---")
    mcp_registry.execute_tool("commit_memory", {
        "script": state.get("script", {}),
        "characters": state.get("characters", []),
        "images": state.get("images", [])
    })
    state["status"] = "completed"
    return state
