# WiQAS

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

WiQAS is a Retrieval-Augmented Generation (RAG) system for Filipino cultural question answering. It supports English and Filipino queries, retrieves grounded context from a curated knowledge base, and generates citation-aware responses.

### Proponents

- Hernandez, Ralf Wadee — ralf_hernandez@dlsu.edu.ph
- Ortiz, China — ma_china_ortiz@dlsu.edu.ph
- Santos, Andrea Li — andrea_li_santos@dlsu.edu.ph

### Adviser

- Charibeth, Cheng — charibeth.cheng@dlsu.edu.ph

### Thesis Overview

WiQAS addresses a common problem in Filipino heritage learning: students can easily access information online, but often cannot verify whether the content is culturally accurate and trustworthy. To solve this, this thesis develops a RAG-based factoid QA system based on Filipino Culture, that grounds answers on retrieved evidence instead of free-form generation alone. The system uses hybrid retrieval (BM25 + BGE-M3) and LLM generation (Gemma and SEA-LION) via [Ollama](https://ollama.com/), then evaluates quality using RAGAS metrics and a Cultural LLM Judge. Overall, the work contributes a practical architecture and evaluation method for building reliable educational AI in low-resource cultural and linguistic settings.

## Deliverables

- [Main Paper](https://drive.google.com/drive/u/1/folders/13pxBXb8EgTOhCReKitAauYT4eTVu4HNV)
- [Technical Manual](https://drive.google.com/drive/u/1/folders/1R39qGq1logO4AEQT8Z4e-tHpV3eIkynE)
- [Conference Paper](https://drive.google.com/drive/u/1/folders/1CkiMJQ4-uBzapH98DaARAsxB9ZWh8G7o)
- [Others](https://drive.google.com/drive/u/1/folders/1qRhxj1TJBSu5rDmO1PtWC-8ZDgZEUUnu)
- [Presentations](https://drive.google.com/drive/u/1/folders/1qVNg6SfjfW6SVT2KmpOw0nghxhlPuXnl)
- [Source](https://drive.google.com/drive/u/1/folders/1zq-yszArNX_MO19JnelaKCQWgOXVVEGc)
- [Brochure](https://drive.google.com/drive/u/1/folders/1vEtSKaHRBrxlllkTtCTf8U2v-7JcOXjv)
- [Video Presentation](https://drive.google.com/drive/u/1/folders/1fFLjg2olhIvxy5gx0v3vepi7rgbqN5fV)
- [Endorsements](https://drive.google.com/drive/u/1/folders/1oUxf337b6np8vpIlihfoIUXHVsyrprrS)

## File Structure

```text
WiQAS/
├── backend/
│   ├── api/
│   ├── models/
│   ├── websockets/
│   ├── app.py
│   └── dependencies.py
├── data/
├── frontend/
│   └── src/
├── scripts/
├── src/
│   ├── core/
│   ├── evaluation/
│   ├── generation/
│   ├── retrieval/
│   └── utilities/
├── test/
├── pyproject.toml
├── requirements.txt
├── run.py
└── set_dependencies.py
```

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
