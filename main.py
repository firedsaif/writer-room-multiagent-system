import os
import json
from workflow import compile_workflow

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    print("Warning: GROQ_API_KEY not found in environment. Please ensure it's set in a .env file.")

def main():
    graph = compile_workflow()
    
    print("="*50)
    print("Welcome to PROJECT MONTAGE - The Writer's Room")
    print("="*50)
    print("1. Autonomous Mode (Prompt -> Script -> Characters -> Images)")
    print("2. Manual Mode (Raw Script -> Validate -> Characters -> Images)")
    
    choice = input("\nSelect mode (1/2): ")
    
    # Required for LangGraph MemorySaver
    config = {"configurable": {"thread_id": "session_1"}}
    
    if choice == "1":
        prompt = input("\nEnter your story prompt: ")
        initial_state = {
            "input_mode": "auto",
            "prompt": prompt,
            "status": "started"
        }
    else:
        print("\nYou selected Manual Mode.")
        print("Note: In a true pipeline, you'd paste a large script or provide a path.")
        raw_script = input("Enter your script snippet: ")
        
        initial_state = {
            "input_mode": "manual",
            "raw_script": raw_script,
            "status": "started"
        }
        
    print("\n--- STARTING EXECUTION ---")
    
    # Run the graph until it hits the interrupt (HITL) or END
    for event in graph.stream(initial_state, config=config):
        for key, value in event.items():
            pass
            
    # Check if graph paused at the interrupt
    state = graph.get_state(config)
    if state.next:
        print("\n" + "="*40)
        print("--- HUMAN IN THE LOOP CHECKPOINT ---")
        print("="*40)
        script = state.values.get("script", {})
        print("\nGenerated/Validated Script Manifest:")
        print(json.dumps(script, indent=2))
        
        approval = input("\nDo you approve this script to proceed to Character & Image synthesis? (y/n): ")
        if approval.lower() == 'y':
            print("\nScript approved. Resuming workflow...")
            for event in graph.stream(None, config=config):
                for key, value in event.items():
                    pass
            print("\nWorkflow completed successfully!")
            print("Check 'scene_manifest.json', 'character_db.json', and the 'image_assets/' folder.")
        else:
            print("\nScript rejected. Human intervention requested. Shutting down...")
    else:
        if state.values.get("status") == "validation_failed":
            print("\nWorkflow stopped: Script Validation Failed.")
            print("Errors:", state.values.get("errors"))
        else:
            print("\nWorkflow completed without interruption.")

if __name__ == "__main__":
    main()
