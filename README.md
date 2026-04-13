# WiQAS

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

WiQAS is a Retrieval-Augmented Generation (RAG) system for Filipino cultural question answering. It supports English and Filipino queries, retrieves grounded context from a curated knowledge base, and generates citation-aware responses.

## What WiQAS Provides

- End-to-end RAG pipeline: ingestion, retrieval, reranking, and generation
- Hybrid retrieval: semantic + keyword search with optional MMR diversification
- Multilingual support: Filipino/English query handling with translation options
- API and frontend: FastAPI backend and Svelte-based web interface
- Evaluation tooling: RAGAS and analysis notebooks under `WiQAS_Eval/`

## Tech Stack

- Backend: Python, FastAPI, Ollama
- Retrieval: ChromaDB, sentence-transformers, reranking
- Frontend: SvelteKit, TypeScript, Tailwind
- Evaluation: RAGAS, notebook workflows

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- Ollama running locally or on a reachable host

### 1) Install Python dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Validate environment

```bash
python set_dependencies.py --install base docs --test-imports --test-models --health-check
```

### 3) Ingest documents

```bash
python run.py ingest data/knowledge_base/
```

### 4) Ask questions from CLI

```bash
python run.py ask "What is bayanihan?"
python run.py search "Filipino hospitality"
python run.py status
```

### 5) Run backend API

```bash
python -m backend.app
```

API docs: `http://localhost:8000/docs`

### 6) Run frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:5173` (or the port shown by Vite)

## Project Layout

```text
WiQAS/
├── backend/            # FastAPI app, routers, websocket handlers
├── frontend/           # SvelteKit UI
├── src/                # Core RAG logic (ingest, retrieval, generation, utilities)
├── data/               # Knowledge base files and vector storage
├── test/               # Unit and integration tests
├── WiQAS_Eval/         # Evaluation notebooks and analysis assets
├── run.py              # Main CLI entry point
└── set_dependencies.py # Environment/setup checks
```

## Common Commands

```bash
# Ingestion
python run.py ingest <path>

# Retrieval only
python run.py search "<query>"

# Full RAG answer
python run.py ask "<query>"

# System checks
python run.py status
python run.py config
```

## Notes

- Configuration is environment-driven (`.env` supported in backend startup).
- For GPU or cloud deployment workflows, see scripts under `scripts/` and environment-specific docs in the repository.

## License

MIT License. See [LICENSE](LICENSE).
