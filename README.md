# PROJECT MONTAGE: The Writer's Room

**Phase 1: Autonomous Story and Image Generation Layer**

Project Montage is a LangGraph-based multi-agent creative system that transforms raw human intent into structured, machine-interpretable narrative representations. Unlike traditional pipelines, this module operates as an autonomous Writer's Room where specialized agents collaborate to create multi-scene scripts, extract persistent character identities, and synthesize consistent visual cues.

## 🌟 Key Features

* **Dual-Mode Script Ingestion**:
  * **Autonomous Mode**: Provide a raw prompt, and the agents expand it into a formatted multi-scene screenplay with dialogue and visual cues.
  * **Manual Mode**: Upload or paste an existing script for validation, formatting, and subsequent character/scene synthesis.
* **Persistent Memory Layer**: Uses ChromaDB to maintain agent continuity, storing script history, character metadata, and image references.
* **Dynamic Tool Discovery**: Strictly adheres to MCP constraints (Model Context Protocol). No hardcoded tool APIs—tools are dynamically queried at runtime via an MCP registry.
* **Human-in-the-Loop (HITL) Checkpoint**: Pauses the LangGraph workflow to ensure user alignment and intercept hallucinated scripts before proceeding to visual synthesis.

## 🏗️ Multi-Agent Architecture

The workflow routes stateful data between several specialized agents:

1. **Scriptwriter Agent**: Expands prompts into structured scenes, dialogues, and visual contexts.
2. **Script Validator Agent**: Parses manually injected scripts, ensuring correct formatting of headers, dialogue labels, and actions.
3. **Character Designer Agent**: Extracts character identities, maintaining consistency and generating robust persona metadata.
4. **Image Synthesizer Agent**: Generates AI-based reference character visuals using simulated localized image generation.

## 📦 Deliverables

The output of a successful run generates the following key files:
- `scene_manifest.json` : Structured screenplay split by scenes.
- `character_db.json`   : Extracted persistent character identity profiles.
- `image_assets/`       : Generated visual references.

## 🛠️ Setup & Installation

### Prerequisites
* Python 3.9+
* A valid **Groq API Key**

### 1. Clone the repository
```bash
git clone https://github.com/firedsaif/writer-room-multiagent-system.git
cd writer-room-multiagent-system
```

### 2. Install Dependencies
It's recommended to create a virtual environment first.
```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory and add your Groq API Key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## 🚀 Usage

Run the main application file to start the interactive terminal session:

```bash
python main.py
```

1. **Select Mode**: Choose between Mode 1 (Autonomous) or Mode 2 (Manual).
2. **Input**: Provide your prompt or paste your raw script.
3. **Approval**: Wait for the generated `scene_manifest.json` snippet to output, then approve the prompt (HITL checkpoint).
4. **Results**: Upon completion, you will find your generated manifest, database, and images in the project directory.

## 🔧 Technologies Used

* **LangChain / LangGraph**: Agent orchestration and workflow management.
* **ChromaDB**: Stateful semantic memory retrieval.
* **Groq**: High-speed LLM inference.
* **python-dotenv**: Environment configuration.
