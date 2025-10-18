#!/usr/bin/env python3
"""
WiQAS Retrieval CLI Interface

A comprehensive command-line interface to test the entire WiQAS retrieval pipeline.
Supports ingestion, search, and system management operations.

Usage:
    python run_retrieval.py ingest <path>           # Ingest documents
    python run_retrieval.py search <query>          # Search documents
    python run_retrieval.py sources                 # List all sources
    python run_retrieval.py sources --file <path>   # Show chunks from specific source
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
from src.retrieval.evaluator import RetrievalEvaluator
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
    search_type: str = typer.Option("hybrid", "--type", "-t", help="Search type: semantic or hybrid"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Enable reranking"),
    mmr: bool = typer.Option(True, "--mmr/--no-mmr", help="Enable MMR diversity search"),
    llm_analysis: bool = typer.Option(True, "--llm-analysis/--no-llm-analysis", help="Enable LLM-based cultural analysis"),
    config_env: bool = typer.Option(False, "--env", help="Load configuration from environment variables"),
):
    """Search the knowledge base using the WiQAS retrieval pipeline."""
    print_header("WiQAS Document Search")

    print_info(f"Query: {query}")
    print_info(f"Search type: {search_type}")
    print_info(f"Results: {k}")
    print_info(f"Reranking: {rerank}")
    print_info(f"MMR diversity: {mmr}")
    print_info(f"LLM analysis: {llm_analysis}")

    try:
        # Import the convenience function
        from src.retrieval.retriever import query_knowledge_base

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Searching knowledge base...", total=None)

            # Use the convenience function with all the options
            result_text = query_knowledge_base(
                query=query,
                k=k,
                search_type=search_type,
                enable_reranking=rerank,
                enable_mmr=mmr,
                llm_analysis=llm_analysis,
                include_timing=True,
            )

            progress.remove_task(task)

        # Display the formatted results
        console.print(result_text)

        # Check if results were found (simple check for success message)
        if "No results found" not in result_text and "Error:" not in result_text:
            print_success("Search completed successfully!")
        elif "Error:" in result_text:
            print_error("Search encountered an error")
        else:
            print_warning("No results found for your query")

    except Exception as e:
        print_error(f"Search failed: {e}")
        raise typer.Exit(1)


# ========== EVALUATION COMMANDS ==========
@app.command()
def evaluate(
    output: str = typer.Option(None, "--output", "-o", help="Output file path for evaluation results (overrides config)"),
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of evaluation items (overrides config)"),
    no_cultural_llm: bool = typer.Option(False, "--no-cultural-llm", help="Disable cultural LLM analysis (overrides config)"),
    randomize: bool = typer.Option(False, "--randomize", "-r", help="Randomize the evaluation dataset order (overrides config)"),
    config_env: bool = typer.Option(False, "--env", help="Load configuration from environment variables"),
) -> None:
    """
    Evaluate retrieval performance using cosine similarity with ground truth.
    """

    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    console = Console()

    # Display header
    console.print(Panel.fit("🔍 WiQAS Retrieval Evaluation", style="bold blue"))

    try:
        # Load configuration
        if config_env:
            config = WiQASConfig.from_env()
            console.print("ℹ️  Using configuration from environment variables", style="blue")
        else:
            config = WiQASConfig()
            console.print("ℹ️  Using default configuration", style="blue")

        # Get evaluation config
        eval_config = config.rag.evaluation

        # Apply CLI overrides to config
        if limit is not None:
            eval_config.limit = limit
        if no_cultural_llm:
            eval_config.disable_cultural_llm_analysis = True
        if randomize:
            eval_config.randomize = True

        # Output
        if output:
            output_path = output
        else:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            output_path = f"./data/evaluation/retrieval/{timestamp}.json"

        output_file_path = Path(output_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Display configuration
        config_table = Table(title="📋 Evaluation Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Dataset", eval_config.dataset_path)
        config_table.add_row("Limit", str(eval_config.limit) if eval_config.limit else "None (all items)")
        config_table.add_row("Randomize", str(eval_config.randomize))
        config_table.add_row("Search Type", eval_config.search_type)
        config_table.add_row("Results per Query", str(eval_config.k_results))
        config_table.add_row("Reranking", str(eval_config.enable_reranking))
        config_table.add_row("MMR Diversity", str(eval_config.enable_mmr))
        config_table.add_row("Cultural LLM", str(not eval_config.disable_cultural_llm_analysis))
        config_table.add_row("Similarity Threshold", f"{eval_config.similarity_threshold:.2f}")
        config_table.add_row("Output File", output_path)

        console.print(config_table)
        console.print()

        # Run evaluation with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running evaluation...", total=None)

            evaluator = RetrievalEvaluator(config)
            results = evaluator.evaluate()

            progress.remove_task(task)

        if "error" in results:
            console.print(f"❌ Evaluation failed: {results['error']}", style="red")
            raise typer.Exit(code=1)

        # Display results summary
        console.print("📊 Evaluation Results", style="bold green")
        console.print()

        # Dataset info
        dataset_info = results["dataset_info"]
        info_table = Table(title="Dataset Information")
        info_table.add_column("Metric", style="cyan")
        info_table.add_column("Value", style="white")

        info_table.add_row("Total Items", str(dataset_info["total_items"]))
        info_table.add_row("Successful Evaluations", str(dataset_info["successful_evaluations"]))
        info_table.add_row("Errors", str(dataset_info["errors"]))
        info_table.add_row("Success Rate", dataset_info["success_rate"])

        console.print(info_table)
        console.print()

        # Similarity statistics
        similarity_stats = results["similarity_statistics"]
        stats_table = Table(title="Similarity Statistics")
        stats_table.add_column("Statistic", style="cyan")
        stats_table.add_column("Value", style="white")

        stats_table.add_row("Average", similarity_stats["average"])
        stats_table.add_row("Median", similarity_stats["median"])
        stats_table.add_row("Std Deviation", similarity_stats["std_deviation"])
        stats_table.add_row("Minimum", similarity_stats["min"])
        stats_table.add_row("Maximum", similarity_stats["max"])

        console.print(stats_table)
        console.print()

        # Classification metrics
        if "classification_metrics" in results:
            classification_metrics = results["classification_metrics"]
            classification_table = Table(title="Classification Metrics")
            classification_table.add_column("Metric", style="cyan")
            classification_table.add_column("Value", style="white")

            classification_table.add_row("Accuracy", classification_metrics["accuracy"])
            classification_table.add_row("Precision", classification_metrics["precision"])
            classification_table.add_row("Recall", classification_metrics["recall"])
            classification_table.add_row("F1-Score", classification_metrics["f1_score"])

            console.print(classification_table)
            console.print()

            # Confusion matrix
            confusion_matrix = classification_metrics["confusion_matrix"]
            confusion_table = Table(title="Confusion Matrix")
            confusion_table.add_column("Metric", style="cyan")
            confusion_table.add_column("Count", style="white")

            confusion_table.add_row("True Positives", str(confusion_matrix["true_positives"]))
            confusion_table.add_row("False Positives", str(confusion_matrix["false_positives"]))
            confusion_table.add_row("True Negatives", str(confusion_matrix["true_negatives"]))
            confusion_table.add_row("False Negatives", str(confusion_matrix["false_negatives"]))

            console.print(confusion_table)
            console.print()

        # Retrieval metrics at K
        if "retrieval_metrics_at_k" in results and results["retrieval_metrics_at_k"]:
            retrieval_at_k = results["retrieval_metrics_at_k"]
            retrieval_table = Table(title="Retrieval Metrics at Different K Values")
            retrieval_table.add_column("K", style="cyan")
            retrieval_table.add_column("Precision@K", style="white")
            retrieval_table.add_column("Recall@K", style="white")
            retrieval_table.add_column("Relevant Found", style="white")

            for k_key, metrics in retrieval_at_k.items():
                k_value = k_key.replace("k_", "")
                retrieval_table.add_row(
                    k_value,
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    str(metrics['relevant_found'])
                )

            console.print(retrieval_table)
            console.print()

        # Threshold analysis
        threshold_info = results["threshold_analysis"]
        threshold_table = Table(title="Threshold Analysis")
        threshold_table.add_column("Metric", style="cyan")
        threshold_table.add_column("Value", style="white")

        threshold_table.add_row("Threshold", str(threshold_info["threshold"]))
        threshold_table.add_row("Above Threshold", str(threshold_info["above_threshold"]))
        threshold_table.add_row("Above Threshold Rate", threshold_info["above_threshold_rate"])

        console.print(threshold_table)
        console.print()

        # Save results
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Saving results...", total=None)
            evaluator.save_results(results, output_path)
            progress.remove_task(task)

        console.print(f"✅ Evaluation completed! Results saved to: {output_path}", style="green")

        # Show top and bottom performing examples
        detailed_results = results["detailed_results"]
        valid_results = [r for r in detailed_results if "error" not in r]

        if valid_results:
            # Sort by similarity score
            sorted_results = sorted(valid_results, key=lambda x: x["similarity_score"], reverse=True)

            console.print("\n🔝 Top 3 Performing Queries:", style="bold green")
            for i, result in enumerate(sorted_results[:3]):
                console.print(f"{i+1}. Similarity: {result['similarity_score']:.4f}")
                console.print(f"   Question: {result['question'][:80]}{'...' if len(result['question']) > 80 else ''}")
                console.print(f"   Ground Truth Context: {result['ground_truth_context'][:60]}{'...' if len(result['ground_truth_context']) > 60 else ''}")
                console.print(f"   Retrieved Content: {result['retrieved_content'][:60]}{'...' if len(result['retrieved_content']) > 60 else ''}")
                console.print()

            console.print("🔻 Bottom 3 Performing Queries:", style="bold red")
            for i, result in enumerate(sorted_results[-3:]):
                console.print(f"{i+1}. Similarity: {result['similarity_score']:.4f}")
                console.print(f"   Question: {result['question'][:80]}{'...' if len(result['question']) > 80 else ''}")
                console.print(f"   Ground Truth Context: {result['ground_truth_context'][:60]}{'...' if len(result['ground_truth_context']) > 60 else ''}")
                console.print(f"   Retrieved Content: {result['retrieved_content'][:60]}{'...' if len(result['retrieved_content']) > 60 else ''}")
                console.print()

    except KeyboardInterrupt:
        console.print("\n⚠️ Evaluation interrupted by user", style="yellow")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}", style="red")
        raise typer.Exit(code=1)


# ========== STATUS AND INFO COMMANDS ==========
@app.command()
def sources(
    source_file: str = typer.Option(None, "--file", "-f", help="Show chunks from specific source file"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    config_env: bool = typer.Option(False, "--env", help="Load configuration from environment variables"),
):
    """List all sources in the knowledge base or show chunks from a specific source."""
    print_header("Knowledge Base Sources")

    config = WiQASConfig.from_env() if config_env else WiQASConfig()

    try:
        vector_store = ChromaVectorStore(config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            if source_file:
                # Show chunks from specific source
                task = progress.add_task(f"Loading chunks from {source_file}...", total=None)
                chunks = vector_store.get_chunks_by_source(source_file)
                progress.remove_task(task)

                if json_output:
                    output = {"source_file": source_file, "chunk_count": len(chunks), "chunks": chunks}
                    console.print(json.dumps(output, indent=2))
                else:
                    if not chunks:
                        print_warning(f"No chunks found for source: {source_file}")
                        return

                    console.print(f"\n[bold]Found {len(chunks)} chunks from: {source_file}[/bold]\n")

                    for i, chunk in enumerate(chunks, 1):
                        metadata = chunk["metadata"]
                        chunk_idx = metadata.get("chunk_index", i - 1) + 1
                        chunk_total = metadata.get("chunk_total", len(chunks))

                        console.print(f"[bold cyan]Chunk {chunk_idx}/{chunk_total}[/bold cyan] (ID: {chunk['id']})")

                        # Content preview
                        content = chunk["content"][:300] + "..." if len(chunk["content"]) > 300 else chunk["content"]
                        console.print(Panel(content, border_style="dim"))
                        console.print()

            else:
                # List all sources
                task = progress.add_task("Loading sources...", total=None)
                sources_list = vector_store.list_all_sources()
                progress.remove_task(task)

                if json_output:
                    output = {"total_sources": len(sources_list), "sources": sources_list}
                    console.print(json.dumps(output, indent=2))
                else:
                    if not sources_list:
                        print_warning("No sources found in knowledge base")
                        return

                    console.print(f"\n[bold]Found {len(sources_list)} sources in knowledge base:[/bold]\n")

                    # Create table for sources
                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("File Name", style="cyan", min_width=20)
                    table.add_column("Title", style="bright_blue", min_width=25)
                    table.add_column("Type", style="yellow", width=10)
                    table.add_column("Chunks", style="green", width=8, justify="right")
                    table.add_column("Source Path", style="dim", no_wrap=False)

                    total_chunks = 0
                    for source in sources_list:
                        title = source.get("title", "")
                        if not title or title == source["file_name"]:
                            title = "[dim]No title[/dim]"

                        table.add_row(source["file_name"], title, source["file_type"], str(source["chunk_count"]), source["source_file"])
                        total_chunks += source["chunk_count"]

                    console.print(table)

                    # Summary
                    console.print("\n[bold]Summary:[/bold]")
                    console.print(f"  • Total sources: {len(sources_list)}")
                    console.print(f"  • Total chunks: {total_chunks}")
                    console.print("\n[dim]Use --file <source_path> to view chunks from a specific source[/dim]")

        print_success("Sources listed successfully!")

    except Exception as e:
        print_error(f"Failed to list sources: {e}")
        raise typer.Exit(1)


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
