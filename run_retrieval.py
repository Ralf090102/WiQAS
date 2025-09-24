#!/usr/bin/env python3
"""
WiQAS Retrieval CLI Interface

A comprehensive command-line interface to test the entire WiQAS retrieval pipeline.
Supports ingestion, search, and system management operations.

Usage:
    python run_retrieval.py ingest <path>           # Ingest documents
    python run_retrieval.py search <query>          # Search documents
    python run_retrieval.py status                  # Show system status
    python run_retrieval.py clear                   # Clear knowledge base
    python run_retrieval.py config                  # Show configuration
"""

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Local imports
from src.core.ingest import DocumentIngestor, clear_knowledge_base, get_supported_formats
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.reranker import Document
from src.retrieval.search import MMRSearcher, SemanticSearcher
from src.retrieval.vector_store import ChromaVectorStore
from src.utilities.config import WiQASConfig

# Initialize CLI app and console
app = typer.Typer(
    name="WiQAS Retrieval",
    help="WiQAS Document Retrieval and Management CLI",
    add_completion=False,
)
console = Console()


# ========== UTILITY FUNCTIONS ==========
def print_header(title: str):
    """Print a formatted header"""
    console.print(Panel(f"[bold blue]{title}[/bold blue]", expand=False))


def print_success(message: str):
    """Print a success message"""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str):
    """Print an error message"""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str):
    """Print a warning message"""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str):
    """Print an info message"""
    console.print(f"[blue]ℹ[/blue] {message}")


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def load_config() -> WiQASConfig:
    """Load WiQAS configuration"""
    try:
        config = WiQASConfig.from_env()
        return config
    except Exception as e:
        print_error(f"Failed to load configuration: {e}")
        typer.Exit(1)


def check_system_health(config: WiQASConfig) -> dict[str, Any]:
    """Check system health and dependencies"""
    health = {"ollama": False, "vector_store": False, "embedding_model": False, "knowledge_base": False, "errors": []}

    # Check Ollama connection
    try:
        from src.core.llm import check_ollama_connection

        health["ollama"] = check_ollama_connection(config)
        if not health["ollama"]:
            health["errors"].append("Ollama service not available")
    except Exception as e:
        health["errors"].append(f"Ollama check failed: {e}")

    # Check vector store
    try:
        vector_store = ChromaVectorStore(config)
        stats = vector_store.get_collection_stats()
        health["vector_store"] = True
        health["document_count"] = stats.get("total_documents", 0)
    except Exception as e:
        health["errors"].append(f"Vector store check failed: {e}")

    # Check embedding model
    try:
        embedding_manager = EmbeddingManager(config)
        # Try to encode a test string
        embedding_manager.encode_single("test")
        health["embedding_model"] = True
    except Exception as e:
        health["errors"].append(f"Embedding model check failed: {e}")

    # Check knowledge base directory
    try:
        kb_path = Path(config.system.storage.knowledge_base_directory)
        health["knowledge_base"] = kb_path.exists()
        if health["knowledge_base"]:
            files = list(kb_path.rglob("*"))
            health["knowledge_base_files"] = len([f for f in files if f.is_file()])
        else:
            health["knowledge_base_files"] = 0
    except Exception as e:
        health["errors"].append(f"Knowledge base check failed: {e}")

    return health


# ========== INGESTION COMMANDS ==========
@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to file or directory to ingest"),
    clear: bool = typer.Option(False, "--clear", "-c", help="Clear existing data before ingesting"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r", help="Search subdirectories recursively"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
    config_env: bool = typer.Option(False, "--env", help="Load configuration from environment variables"),
):
    """Ingest documents into the knowledge base."""
    print_header("WiQAS Document Ingestion")

    # Load configuration
    config = WiQASConfig.from_env() if config_env else WiQASConfig()

    # Check if path exists
    source_path = Path(path)
    if not source_path.exists():
        print_error(f"Path not found: {path}")
        raise typer.Exit(1)

    print_info(f"Source: {source_path}")
    print_info(f"Clear existing: {clear}")
    print_info(f"Recursive: {recursive}")
    print_info(f"Workers: {workers}")

    # Show supported formats
    supported = get_supported_formats()
    print_info(f"Supported formats: {', '.join(supported.keys())}")

    try:
        # Initialize ingestor
        ingestor = DocumentIngestor(config)

        if clear:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Clearing existing data...", total=None)
                ingestor.vector_store.clear_collection()
                progress.remove_task(task)
            print_success("Cleared existing knowledge base")

        if source_path.is_file():
            stats = ingestor.ingest_knowledge_base(source_path, clear_existing=False)
        else:
            stats = ingestor.ingest_directory(source_path, recursive=recursive, max_workers=workers)

        # Display results
        console.print("\n[bold]Ingestion Results:[/bold]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Files", str(stats.total_files))
        table.add_row("Successful", str(stats.successful_files))
        table.add_row("Failed", str(stats.failed_files))
        table.add_row("Success Rate", f"{stats.success_rate:.1f}%")
        table.add_row("Total Chunks", str(stats.total_chunks))
        table.add_row("Processing Time", f"{stats.processing_time:.2f}s")

        console.print(table)

        if stats.errors:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in stats.errors:
                print_error(error)

        if stats.successful_files > 0:
            print_success(f"Successfully ingested {stats.successful_files} files!")
        else:
            print_warning("No files were successfully ingested")

    except Exception as e:
        print_error(f"Ingestion failed: {e}")
        raise typer.Exit(1)


@app.command()
def clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    config_env: bool = typer.Option(False, "--env", help="Load configuration from environment variables"),
):
    """Clear the entire knowledge base."""
    print_header("Clear Knowledge Base")

    if not confirm:
        confirm_clear = typer.confirm("This will delete all ingested documents. Continue?")
        if not confirm_clear:
            print_info("Operation cancelled")
            return

    config = WiQASConfig.from_env() if config_env else WiQASConfig()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Clearing knowledge base...", total=None)
            success = clear_knowledge_base(config)
            progress.remove_task(task)

        if success:
            print_success("Knowledge base cleared successfully")
        else:
            print_error("Failed to clear knowledge base")
            raise typer.Exit(1)

    except Exception as e:
        print_error(f"Clear operation failed: {e}")
        raise typer.Exit(1)


# ========== SEARCH COMMANDS ==========
@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(5, "--results", "-k", help="Number of results to return"),
    search_type: str = typer.Option("hybrid", "--type", "-t", help="Search type: semantic, hybrid, or keyword"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Enable reranking"),
    mmr: bool = typer.Option(True, "--mmr/--no-mmr", help="Enable MMR diversity search after reranking"),
    cultural_boost: bool = typer.Option(True, "--cultural/--no-cultural", help="Enable cultural content boosting"),
    llm_analysis: bool = typer.Option(True, "--llm-analysis/--no-llm-analysis", help="Enable LLM-based cultural analysis"),
    config_env: bool = typer.Option(False, "--env", help="Load configuration from environment variables"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
):
    """Search the knowledge base."""
    print_header("WiQAS Document Search")

    config = WiQASConfig.from_env() if config_env else WiQASConfig()

    print_info(f"Query: {query}")
    print_info(f"Search type: {search_type}")
    print_info(f"Results: {k}")
    print_info(f"Reranking: {rerank}")
    print_info(f"MMR diversity: {mmr}")
    print_info(f"Cultural boost: {cultural_boost}")
    print_info(f"LLM analysis: {llm_analysis}")

    try:
        # Initialize components
        vector_store = ChromaVectorStore(config)
        embedding_manager = EmbeddingManager(config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # Check if we have documents
            task = progress.add_task("Checking knowledge base...", total=None)
            stats = vector_store.get_collection_stats()
            progress.remove_task(task)

            doc_count = stats.get("document_count", 0)
            print_info(f"Found {doc_count} documents in knowledge base")

            if doc_count == 0:
                print_warning("No documents found in knowledge base. Run 'ingest' first.")
                raise typer.Exit(1)

            print_info(f"Searching {doc_count} documents...")

            # Perform search
            task = progress.add_task("Searching...", total=None)

            if search_type == "semantic":
                searcher = SemanticSearcher(embedding_manager, vector_store, config)
                results = searcher.search(query, k=k)
            elif search_type == "hybrid":
                # Create semantic and keyword searchers
                semantic_searcher = SemanticSearcher(embedding_manager, vector_store, config)
                from src.retrieval.search import HybridSearcher, KeywordSearcher

                keyword_searcher = KeywordSearcher(config)

                # Get documents from vector store for keyword indexing
                all_results = semantic_searcher.search(query, k=doc_count)  # Get all documents
                documents = [r.content for r in all_results]
                document_ids = [r.document_id for r in all_results]
                metadatas = [r.metadata for r in all_results]

                # Index documents for keyword search
                keyword_searcher.index_documents(documents, document_ids, metadatas)

                # Create hybrid searcher and search
                hybrid_searcher = HybridSearcher(semantic_searcher, keyword_searcher, config)
                results = hybrid_searcher.search(query, k=k)
            else:
                print_error(f"Unsupported search type: {search_type}")
                raise typer.Exit(1)

            progress.remove_task(task)

            # Apply reranking if enabled
            if rerank and results:
                task = progress.add_task("Reranking results...", total=None)
                from src.retrieval.reranker import RerankerManager

                reranker_config = config.rag.reranker

                # Override LLM analysis setting if specified
                if not llm_analysis:
                    # Create a copy of the config with LLM analysis disabled
                    from dataclasses import replace

                    reranker_config = replace(reranker_config, use_llm_cultural_analysis=False, score_threshold=0.0)

                reranker = RerankerManager(reranker_config)

                # Convert SearchResult objects to Document objects for reranking
                docs_to_rerank = []
                for result in results:
                    doc = Document(
                        content=result.content, metadata=result.metadata, score=result.score, doc_id=result.document_id
                    )
                    docs_to_rerank.append(doc)

                # Rerank documents
                reranked_docs = reranker.rerank_documents(query, docs_to_rerank, top_k=k)

                # Convert back to SearchResult objects
                from src.retrieval.search import SearchResult

                results = []
                for doc in reranked_docs:
                    result = SearchResult(
                        document_id=doc.doc_id,
                        content=doc.content,
                        metadata=doc.metadata,
                        score=doc.score,
                        search_type=f"{search_type}_reranked",
                    )
                    results.append(result)

                progress.remove_task(task)

            # Apply MMR diversity search after reranking if enabled
            if mmr and results and len(results) > 1:
                task = progress.add_task("Applying MMR diversity search...", total=None)

                # Initialize MMR searcher
                mmr_searcher = MMRSearcher(embedding_manager, config)

                # Apply MMR to get diverse subset
                mmr_results = mmr_searcher.search(query, candidate_results=results, k=k)

                # Update search_type to indicate MMR was applied
                for result in mmr_results:
                    current_search_type = f"{search_type}"
                    if rerank:
                        current_search_type += "_reranked"
                    current_search_type += "_mmr"
                    result.search_type = current_search_type

                results = mmr_results
                progress.remove_task(task)

        # Display results
        if json_output:
            output = {
                "query": query,
                "search_type": search_type,
                "reranked": rerank,
                "mmr_applied": mmr,
                "cultural_boost": cultural_boost,
                "llm_analysis": llm_analysis,
                "total_results": len(results),
                "results": [result.to_dict() for result in results],
            }
            console.print(json.dumps(output, indent=2))
        else:
            if not results:
                print_warning("No results found")
                return

            console.print(f"\n[bold]Found {len(results)} results:[/bold]")

            for i, result in enumerate(results, 1):
                console.print(f"\n[bold cyan]Result {i}[/bold cyan] (Score: {result.score:.4f})")

                # Create a table for result details
                table = Table(show_header=False, box=None, padding=(0, 1))
                table.add_column("Field", style="yellow", width=12)
                table.add_column("Value", style="white")

                table.add_row("Source", result.metadata.get("source_file", "Unknown"))
                table.add_row("Type", result.metadata.get("file_type", "Unknown"))

                if "chunk_index" in result.metadata:
                    table.add_row(
                        "Chunk", f"{result.metadata['chunk_index'] + 1}/{result.metadata.get('chunk_total', 'Unknown')}"
                    )

                table.add_row("Search", result.search_type)

                console.print(table)

                # Content preview
                content_preview = result.content[:500] + "..." if len(result.content) > 500 else result.content
                console.print(Panel(content_preview, title="Content", border_style="dim"))

        print_success(f"Search completed! Found {len(results)} results.")

    except Exception as e:
        print_error(f"Search failed: {e}")
        raise typer.Exit(1)


# ========== STATUS AND INFO COMMANDS ==========
@app.command()
def status(
    config_env: bool = typer.Option(False, "--env", help="Load configuration from environment variables"),
):
    """Show system status and health information."""
    print_header("WiQAS System Status")

    config = WiQASConfig.from_env() if config_env else WiQASConfig()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Checking system health...", total=None)
        health = check_system_health(config)
        progress.remove_task(task)

    # Health status table
    table = Table(title="System Health", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Details", style="dim")

    # Ollama
    status_icon = "✓" if health["ollama"] else "✗"
    status_color = "green" if health["ollama"] else "red"
    table.add_row(
        "Ollama Service",
        f"[{status_color}]{status_icon}[/{status_color}]",
        "Connected" if health["ollama"] else "Not available",
    )

    # Vector Store
    status_icon = "✓" if health["vector_store"] else "✗"
    status_color = "green" if health["vector_store"] else "red"
    doc_count = health.get("document_count", 0)
    table.add_row(
        "Vector Store",
        f"[{status_color}]{status_icon}[/{status_color}]",
        f"{doc_count} documents" if health["vector_store"] else "Not available",
    )

    # Embedding Model
    status_icon = "✓" if health["embedding_model"] else "✗"
    status_color = "green" if health["embedding_model"] else "red"
    table.add_row(
        "Embedding Model",
        f"[{status_color}]{status_icon}[/{status_color}]",
        config.rag.embedding.model if health["embedding_model"] else "Not available",
    )

    # Knowledge Base
    status_icon = "✓" if health["knowledge_base"] else "✗"
    status_color = "green" if health["knowledge_base"] else "red"
    kb_files = health.get("knowledge_base_files", 0)
    table.add_row(
        "Knowledge Base",
        f"[{status_color}]{status_icon}[/{status_color}]",
        f"{kb_files} files" if health["knowledge_base"] else "Directory not found",
    )

    console.print(table)

    # Show errors if any
    if health["errors"]:
        console.print("\n[bold red]Issues Found:[/bold red]")
        for error in health["errors"]:
            print_error(error)
    else:
        print_success("All systems operational!")

    # Configuration summary
    console.print("\n[bold]Configuration Summary:[/bold]")
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Setting", style="yellow", width=20)
    config_table.add_column("Value", style="white")

    config_table.add_row("LLM Model", config.rag.llm.model)
    config_table.add_row("Embedding Model", config.rag.embedding.model)
    config_table.add_row("Chunk Size", str(config.rag.chunking.chunk_size))
    config_table.add_row("Chunking Strategy", config.rag.chunking.strategy.value)
    config_table.add_row("Vector Store", config.rag.vectorstore.persist_directory)
    config_table.add_row("Reranking Enabled", str(config.rag.retrieval.enable_reranking))

    console.print(config_table)


@app.command()
def config(
    config_env: bool = typer.Option(False, "--env", help="Load configuration from environment variables"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format: table or json"),
):
    """Show current configuration."""
    print_header("WiQAS Configuration")

    config = WiQASConfig.from_env() if config_env else WiQASConfig()

    if output_format == "json":
        config_dict = config.model_dump()
        console.print(json.dumps(config_dict, indent=2))
    else:
        # Display as organized tables
        sections = [
            (
                "LLM Configuration",
                {
                    "Model": config.rag.llm.model,
                    "Base URL": config.rag.llm.base_url,
                    "Temperature": str(config.rag.llm.temperature),
                    "Max Tokens": str(config.rag.llm.max_tokens),
                    "Timeout": f"{config.rag.llm.timeout}s",
                },
            ),
            (
                "Embedding Configuration",
                {
                    "Model": config.rag.embedding.model,
                    "Batch Size": str(config.rag.embedding.batch_size),
                    "Cache Embeddings": str(config.rag.embedding.cache_embeddings),
                    "Timeout": f"{config.rag.embedding.timeout}s",
                },
            ),
            (
                "Preprocessing Configuration",
                {
                    "Text Normalization": str(config.rag.preprocessing.enable_normalization),
                    "Deduplication": str(config.rag.preprocessing.enable_deduplication),
                    "Similarity Threshold": str(config.rag.preprocessing.similarity_threshold),
                    "Min Text Length": str(config.rag.preprocessing.min_text_length),
                },
            ),
            (
                "Chunking Configuration",
                {
                    "Strategy": config.rag.chunking.strategy.value,
                    "Chunk Size": str(config.rag.chunking.chunk_size),
                    "Chunk Overlap": str(config.rag.chunking.chunk_overlap),
                    "Min Chunk Size": str(config.rag.chunking.min_chunk_size),
                    "Max Chunk Size": str(config.rag.chunking.max_chunk_size),
                },
            ),
            (
                "Retrieval Configuration",
                {
                    "Default K": str(config.rag.retrieval.default_k),
                    "Max K": str(config.rag.retrieval.max_k),
                    "Similarity Threshold": str(config.rag.retrieval.similarity_threshold),
                    "Enable Reranking": str(config.rag.retrieval.enable_reranking),
                    "Enable Hybrid Search": str(config.rag.retrieval.enable_hybrid_search),
                    "Semantic Weight": str(config.rag.retrieval.semantic_weight),
                    "Keyword Weight": str(config.rag.retrieval.keyword_weight),
                },
            ),
            (
                "Vector Store Configuration",
                {
                    "Type": config.rag.vectorstore.index_type,
                    "Persist Directory": config.rag.vectorstore.persist_directory,
                    "Distance Metric": config.rag.vectorstore.distance_metric,
                    "Batch Size": str(config.rag.vectorstore.batch_size),
                    "Use GPU": str(config.rag.vectorstore.use_gpu),
                },
            ),
        ]

        for title, settings in sections:
            table = Table(title=title, show_header=True, header_style="bold magenta")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")

            for key, value in settings.items():
                table.add_row(key, str(value))

            console.print(table)
            console.print()


# ========== MAIN ENTRY POINT ==========
def main():
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
