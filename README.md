# UX Journey Storyboard Tool
This tool is designed for UX designers who are considering a multitude of situations of how their product is being used. It's intended to provide designers with a 6 panel storyboard, and then provide critiques backed by UX research with suggestions.

## Key Features
1. Journey Agent: Takes in the persona and situation to create a 6 panel storyboard
2. UX Critique Agent: Using UX research, it identifies pain points that need to be addressed from the storyboard
3. Design Agent: It takes the pain points identified in the previous stage to recommend design suggestions and ways to mitigate the problem

## Tech Stack
- **Python** — core language
- **Flask** — web framework and routing
- **LangChain** — LLM chain orchestration for all three agents
- **ChromaDB** — vector database for storing and retrieving UX research documents
- **Ollama** — local LLM inference (model: `qwen3.5:4b`)
- **Nomic Embed Text** — embeddings for RAG retrieval
- **Pydantic** — structured input/output schemas for agent pipelines

## Getting Started
### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- The `qwen3.5:4b` and `nomic-embed-text` models pulled:
```bash
  ollama pull qwen3.5:4b
  ollama pull nomic-embed-text
```

### Installation
1. Clone the repository
```bash
   git clone https://github.com/eruan22/UX-storyboard-project.git
   cd UX-storyboard-project
```
2. Install dependencies
```bash
   pip install -r requirements.txt
```
### Running the App
```bash
python app.py
```
Then open `http://127.0.0.1:5000/` in your browser.

> On first run, ChromaDB will chunk and index your PDFs. Subsequent runs load the existing vector store from `data/chroma_db/`.
