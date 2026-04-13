import os
import json
import uuid
import requests
import io
from PIL import Image, ImageDraw, ImageFont
import chromadb
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from mcp_registry import mcp_registry

# Setup VectorDB (Chroma)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
script_collection = chroma_client.get_or_create_collection("scripts")
character_collection = chroma_client.get_or_create_collection("characters")

def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# --- 1. generate_script_segment ---
class Scene(BaseModel):
    heading: str = Field(description="Scene heading (e.g., INT. COFFEE SHOP - DAY)")
    action: str = Field(description="Scene action description")
    dialogues: List[Dict[str, str]] = Field(description="List of dialogues, dict with 'speaker' and 'line'")
    visual_cues: str = Field(description="Visual cues for the scene")

class ScriptManifest(BaseModel):
    scenes: List[Scene]

def generate_script_segment(prompt: str, num_scenes: int = 1) -> dict:
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=ScriptManifest)
    template = """
    You are an expert scriptwriter. Generate a multi-scene screenplay based on the prompt.
    Prompt: {prompt}
    Generate exactly {num_scenes} scenes.
    Ensure there are clear scene headings, actions, dialogues, and visual cues.
    
    Format:
    {format_instructions}
    """
    prompt_obj = PromptTemplate(
        template=template,
        input_variables=["prompt", "num_scenes"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt_obj | llm | parser
    result = chain.invoke({"prompt": prompt, "num_scenes": num_scenes})
    return result

mcp_registry.register_tool(
    name="generate_script_segment",
    schema={
        "name": "generate_script_segment",
        "description": "Transforms a prompt into a structured, production-ready script segment.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "num_scenes": {"type": "integer", "default": 2}
            },
            "required": ["prompt"]
        }
    },
    func=generate_script_segment
)

# --- 2. validate_script ---
class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str]
    standardized_script: ScriptManifest

def validate_script(raw_script: str) -> dict:
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=ValidationResult)
    template = """
    You are a script validator. Check if the provided raw script has scene headings, dialogue labels, and actions.
    If it is valid, rewrite it into the standardized json format. If not valid, list the missing elements in errors and provide a best-effort standardized script anyway but set is_valid to false.
    
    Raw script:
    {raw_script}
    
    Format:
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["raw_script"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm | parser
    return chain.invoke({"raw_script": raw_script})

mcp_registry.register_tool(
    name="validate_script",
    schema={
        "name": "validate_script",
        "description": "Validates a raw script for required elements and converts it to standardized JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "raw_script": {"type": "string"}
            },
            "required": ["raw_script"]
        }
    },
    func=validate_script
)

# --- 3. extract_characters ---
class Character(BaseModel):
    id: str = Field(description="Unique character ID (e.g. char_1)")
    name: str
    personality_traits: List[str]
    appearance_description: str
    reference_style: str

class CharacterDB(BaseModel):
    characters: List[Character]

def extract_characters(script_json: dict) -> dict:
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=CharacterDB)
    template = """
    Extract all characters from the provided script JSON and formalize their identities.
    For each character, infer their personality and appearance based on the script.
    
    Script JSON:
    {script_json}
    
    Format:
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["script_json"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm | parser
    return chain.invoke({"script_json": json.dumps(script_json)})

mcp_registry.register_tool(
    name="extract_characters",
    schema={
        "name": "extract_characters",
        "description": "Extracts and formalizes character identities from a script.",
        "parameters": {
            "type": "object",
            "properties": {
                "script_json": {"type": "object"}
            },
            "required": ["script_json"]
        }
    },
    func=extract_characters
)

# --- 4. generate_character_image ---
def generate_character_image(character_name: str, appearance_description: str) -> str:
    os.makedirs("image_assets", exist_ok=True)
    img_path = f"image_assets/{character_name.replace(' ', '_')}.png"
    
    # Use FLUX.1-schnell via Hugging Face Free Inference API (latest supported high-quality model)
    API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers = {"Authorization": f"Bearer {hf_token}"}
        prompt = f"Concept art of {character_name}, {appearance_description}, high quality, beautiful, character design, clear face"
        
        try:
            print(f"Generating image for {character_name} via Hugging Face API...")
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=60)
            
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                image.save(img_path)
                return img_path
            else:
                print(f"Error generating image: {response.text}")
        except Exception as e:
            print(f"Failed to call Hugging Face API: {e}")
    else:
        print("HF_TOKEN not found in environment. Falling back to placeholder.")
        
    # Fallback to placeholder if the API fails
    img = Image.new('RGB', (512, 512), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    text = f"FAILED/NO TOKEN\n{character_name}\n\n{appearance_description}"
    d.text((20, 20), text, fill=(255, 0, 0))
    img.save(img_path)
    
    return img_path

mcp_registry.register_tool(
    name="generate_character_image",
    schema={
        "name": "generate_character_image",
        "description": "Generates a visual representation of a character.",
        "parameters": {
            "type": "object",
            "properties": {
                "character_name": {"type": "string"},
                "appearance_description": {"type": "string"}
            },
            "required": ["character_name", "appearance_description"]
        }
    },
    func=generate_character_image
)

# --- 5. commit_memory ---
def commit_memory(script: dict, characters: list, images: list) -> str:
    # Write to local JSON files
    with open("scene_manifest.json", "w") as f:
        json.dump(script, f, indent=4)
        
    with open("character_db.json", "w") as f:
        json.dump(characters, f, indent=4)
        
    print(f"Created scene_manifest.json and character_db.json. Images: {images}")
        
    # Write to VectorDB
    if script and "scenes" in script:
        for idx, scene in enumerate(script["scenes"]):
            doc_id = str(uuid.uuid4())
            script_collection.upsert(
                documents=[json.dumps(scene)],
                metadatas=[{"scene_idx": idx, "type": "scene"}],
                ids=[doc_id]
            )
            
    for char in characters:
        doc_id = str(uuid.uuid4())
        character_collection.upsert(
            documents=[json.dumps(char)],
            metadatas=[{"name": char.get("name"), "id": char.get("id")}],
            ids=[doc_id]
        )
        
    return "Memory successfully committed."

mcp_registry.register_tool(
    name="commit_memory",
    schema={
        "name": "commit_memory",
        "description": "Commits script, characters, and images to persistent memory (json and vector DB).",
        "parameters": {
            "type": "object",
            "properties": {
                "script": {"type": "object"},
                "characters": {"type": "array", "items": {"type": "object"}},
                "images": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["script", "characters", "images"]
        }
    },
    func=commit_memory
)
