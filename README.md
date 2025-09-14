# WiQAS - Filipino Cultural Question Answering System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-work--in--progress-yellow.svg)](https://github.com/Ralf090102/WiQAS)

> A Retrieval-Augmented Generation (RAG) system designed to provide culturally relevant and contextually accurate answers about Filipino culture, traditions, and practices.

## Overview

WiQAS is an intelligent Question Answering system specifically designed to address the gap in culturally grounded Filipino content. Traditional search engines and general-purpose chatbots often struggle to provide accurate, culturally relevant answers about Filipino traditions, values, and practices due to their reliance on English-language or non-contextual data.

## Problem Statement

Filipino users face significant challenges when seeking culturally relevant information:

- **Limited Cultural Context**: General-purpose tools frequently miss important cultural nuances
- **Language Barriers**: Reduced performance in low-resource languages like Filipino
- **Inaccurate Interpretations**: Responses often lack proper sociocultural understanding
- **Scattered Resources**: Limited availability of centralized, culturally grounded Filipino content

## Solution

WiQAS implements a **Retrieval-Augmented Generation (RAG) architecture** that:

- **Understands Cultural Context**: Specialized in Filipino cultural topics and traditions
- **Provides Accurate Answers**: Delivers concise, factoid responses grounded in reliable sources
- **Handles Linguistic Nuances**: Processes both English and Filipino language queries
- **Maintains Cultural Sensitivity**: Ensures responses respect Filipino values and practices

## Features

- **Document Ingestion**: Process and index Filipino cultural documents
- **Intelligent Retrieval**: Find relevant cultural information using advanced search
- **Contextual Responses**: Generate culturally aware answers
- **Semantic Search**: Understand meaning beyond keyword matching
- **Multilingual Support**: Handle queries in English and Filipino
- **Configurable Pipeline**: Customize chunking, retrieval, and generation parameters

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ralf090102/WiQAS.git
   cd WiQAS
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run WiQAS:**
   ```bash
   python run.py
   ```

## Project Structure

```
WiQAS/
├── src/
│   ├── core/           # Core RAG components
│   │   ├── ingest.py   # Document ingestion
│   │   ├── llm.py      # Language model integration
│   │   └── query.py    # Query processing
│   ├── generation/     # Response generation
│   ├── retrieval/      # Information retrieval
│   └── utilities/      # Helper functions and config
├── data/               # Document storage
├── test/               # Unit tests
├── requirements.txt    # Python dependencies
├── pyproject.toml     # Project configuration
└── run.py             # Main application entry point
```

## Configuration

WiQAS supports flexible configuration through environment variables:

```bash
# LLM Configuration
export WIQAS_LLM_MODEL="mistral:latest"
export WIQAS_LLM_TEMPERATURE=0.7

# Chunking Configuration
export WIQAS_CHUNK_SIZE=128
export WIQAS_CHUNKING_STRATEGY="recursive"

# Vector Store Configuration
export WIQAS_VECTORSTORE_PERSIST_DIRECTORY="./chroma-data"

# Retrieval Configuration
export WIQAS_RETRIEVAL_DEFAULT_K=5
```

See [Configuration Guide](docs/configuration.md) for detailed options.

## Use Cases

- **Cultural Education**: Learn about Filipino traditions and customs
- **Research Support**: Academic research on Filipino culture
- **Content Creation**: Generate culturally accurate content
- **Language Learning**: Understand cultural context behind language usage
- **Tourism**: Cultural insights for visitors to the Philippines

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Filipino cultural experts and educators
- Open-source NLP community
- RAG research community
- Contributors and testers
