#!/usr/bin/env python3
"""
WiQAS Main CLI Interface

A comprehensive command-line interface for the WiQAS RAG system.
Supports document ingestion, retrieval, answer generation, and system management.

Usage:

    **python run.py --help**
    **python run.py [COMMAND] --help**

    # Document Management
    python run.py ingest <path>                              # Ingest documents
        --clear                                              # Clear existing data before ingesting
        --recursive/--no-recursive                           # Search subdirectories (default: True)
        --workers <n>                                        # Number of parallel workers (default: 4)

    python run.py clear --yes                                # Clear knowledge base (skip confirmation)
    python run.py sources                                    # List all sources
    python run.py sources --file <path>                      # Show chunks from specific source

    # Retrieval
    python run.py search "<query>"                           # Search documents
        --results <k>                                        # Number of results (default: 5)
        --type <semantic|hybrid>                             # Search type (default: hybrid)
        --rerank/--no-rerank                                 # Enable/disable reranking (default: enabled)
        --mmr/--no-mmr                                       # Enable/disable MMR diversity (default: enabled)

    python run.py evaluate                                   # Evaluate retrieval performance
        --limit <n>                                          # Limit evaluation items
        --randomize                                          # Randomize dataset order
        --output <file>                                      # Output file path

    # Answer Generation
    python run.py ask "<question>"                           # Ask a question (retrieval + LLM)
        --results <k>                                        # Number of retrieval results (default: 5)
        --type <Factual|Analytical|Procedural|...>          # Query type (auto-detected if not specified)
        --language <fil|en>                                  # Response language (auto-detected if not specified)
        --json                                               # Output as JSON
        --show-contexts/--hide-contexts                      # Show/hide retrieved contexts (default: show)
        --timing/--no-timing                                 # Show/hide timing breakdown (default: show)
        --classification/--no-classification                 # Show/hide query classification (default: show)

    python run.py batch-ask <input_file>                     # Batch question answering
        --output <file>                                      # Output JSON file (default: batch_output.json)
        --delimiter <char>                                   # Question delimiter (default: '?')
        --timing/--classification                            # Include timing/classification info

    # System Management
    python run.py status                                     # Show system status
    python run.py config                                     # Show configuration
        --format json                                        # Output as JSON
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
from src.generation.generator import WiQASGenerator
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.evaluator import RetrievalEvaluator
from src.retrieval.vector_store import ChromaVectorStore
from src.utilities.config import WiQASConfig

# Initialize CLI app and console
app = typer.Typer(
    name="WiQAS",
    help="WiQAS - Your Intelligent Question Answering System",
    add_completion=False,
)
console = Console()


# ========== UTILITY FUNCTIONS ==========
def print_banner():
    """Print welcome banner"""
    banner = """
╔═════════════════════════════════════════════════════════════════╗
║                                                                 ║
║        ░██       ░██ ░██  ░██████      ░███      ░██████        ║
║        ░██       ░██     ░██   ░██    ░██░██    ░██   ░██       ║
║        ░██  ░██  ░██ ░██░██     ░██  ░██  ░██  ░██              ║
║        ░██ ░████ ░██ ░██░██     ░██ ░█████████  ░████████       ║
║        ░██░██ ░██░██ ░██░██     ░██ ░██    ░██         ░██      ║
║        ░████   ░████ ░██ ░██   ░██  ░██    ░██  ░██   ░██       ║
║        ░███     ░███ ░██  ░██████   ░██    ░██   ░██████        ║
║                                ░██                              ║
║                                ░██                              ║
╚═════════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="cyan")
    console.print("🚀 [bold green]Welcome to WiQAS - A RAG-Driven AI Assistant![/bold green]")
    console.print("   [blue]Your Intelligent Question Answering System[/blue]\n")


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


# ========== DOCUMENT MANAGEMENT COMMANDS ==========
@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to file or directory to ingest"),
    clear: bool = typer.Option(False, "--clear", "-c", help="Clear existing data before ingesting"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r", help="Search subdirectories recursively (default: True)"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers for processing (default: 4, recommended: 4-8)"),
    config_env: bool = typer.Option(False, "--env", help="Load configuration from environment variables"),
):
    """
    Ingest documents into the knowledge base.

    Processes documents (PDF, DOCX, TXT, etc.) and stores them as vector embeddings.
    Supports both single files and entire directories with recursive scanning.

    Examples:
        python run.py ingest ./data/documents
        python run.py ingest document.pdf --clear
        python run.py ingest ./data --workers 8 --no-recursive
    """
    print_header("WiQAS Document Ingestion")

    config = WiQASConfig.from_env() if config_env else WiQASConfig()

    source_path = Path(path)
    if not source_path.exists():
        print_error(f"Path not found: {path}")
        raise typer.Exit(1)

    print_info(f"Source: {source_path}")
    print_info(f"Clear existing: {clear}")
    print_info(f"Recursive: {recursive}")
    print_info(f"Workers: {workers}")

    supported = get_supported_formats()
    print_info(f"Supported formats: {', '.join(supported.keys())}")

    try:
        ingestor = DocumentIngestor(config)

        if clear:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                task = progress.add_task("Clearing existing data...", total=None)
                ingestor.vector_store.clear_collection()
                progress.remove_task(task)
            print_success("Cleared existing knowledge base")

        if source_path.is_file():
            stats = ingestor.ingest_knowledge_base(source_path, clear_existing=False)
        else:
            stats = ingestor.ingest_directory(source_path, recursive=recursive, max_workers=workers)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Files", str(stats.total_files))
        table.add_row("Successful", str(stats.successful_files))
        table.add_row("Failed", str(stats.failed_files))
        table.add_row("Success Rate", f"{stats.success_rate:.1f}%")
        table.add_row("Total Chunks", str(stats.total_chunks))
        table.add_row("Processing Time", f"{stats.processing_time:.2f}s")

        console.print("\n[bold]Ingestion Results:[/bold]")
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
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
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
        results = vector_store.get_all_documents()

        if not results:
            print_warning("No documents found in knowledge base")
            return

        if source_file:
            # Filter by specific source file
            filtered = [r for r in results if r.metadata.get("source_file") == source_file]

            if not filtered:
                print_warning(f"No chunks found for source: {source_file}")
                return

            if json_output:
                output = [{"content": r.content, "metadata": r.metadata} for r in filtered]
                console.print(json.dumps(output, indent=2, ensure_ascii=False))
            else:
                print_success(f"Found {len(filtered)} chunks from {source_file}")
                for i, result in enumerate(filtered, 1):
                    console.print(f"\n[bold cyan]Chunk {i}:[/bold cyan]")
                    console.print(f"Content: {result.content[:200]}..." if len(result.content) > 200 else result.content)
                    console.print(f"Metadata: {result.metadata}")
        else:
            # List all unique sources
            sources = {}
            for r in results:
                src = r.metadata.get("source_file", "Unknown")
                sources[src] = sources.get(src, 0) + 1

            if json_output:
                console.print(json.dumps(sources, indent=2, ensure_ascii=False))
            else:
                table = Table(title=f"Knowledge Base Sources ({len(sources)} files)", show_header=True, header_style="bold magenta")
                table.add_column("Source File", style="cyan")
                table.add_column("Chunks", style="green", justify="right")

                for src, count in sorted(sources.items()):
                    table.add_row(src, str(count))

                console.print(table)
                print_success(f"Total: {len(sources)} source files, {len(results)} chunks")

    except Exception as e:
        print_error(f"Failed to retrieve sources: {e}")
        raise typer.Exit(1)


# ========== RETRIEVAL COMMANDS ==========
@app.command()
def search(
    query: str = typer.Argument(..., help="Search query string"),
    k: int = typer.Option(5, "--results", "-k", help="Number of results to return (default: 5, recommended: 3-10)"),
    search_type: str = typer.Option("hybrid", "--type", "-t", help="Search type: 'semantic' (vector similarity) or 'hybrid' (semantic + keyword)"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Enable cross-encoder reranking for better relevance (default: enabled)"),
    mmr: bool = typer.Option(True, "--mmr/--no-mmr", help="Enable MMR (Maximal Marginal Relevance) for diverse results (default: enabled)"),
    config_env: bool = typer.Option(False, "--env", help="Load configuration from environment variables"),
):
    """
    Search the knowledge base using the WiQAS retrieval pipeline.

    Performs semantic search with optional reranking and diversity optimization.
    Supports both English and Filipino queries with automatic language detection.

    Examples:
        python run.py search "What is bayanihan?"
        python run.py search "Filipino culture" --results 10 --type semantic
        python run.py search "Ano ang pakikisama?" --no-mmr
    """
    print_header("WiQAS Document Search")

    print_info(f"Query: {query}")
    print_info(f"Search type: {search_type}")
    print_info(f"Results: {k}")
    print_info(f"Reranking: {rerank}")
    print_info(f"MMR diversity: {mmr}")

    try:
        from src.retrieval.retriever import query_knowledge_base

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Searching knowledge base...", total=None)

            result_text = query_knowledge_base(
                query,
                k=k,
                search_type=search_type,
                enable_reranking=rerank,
                enable_mmr=mmr,
                formatted=True,
            )

            progress.remove_task(task)

        console.print(result_text)

        if "No results found" not in result_text and "Error:" not in result_text:
            print_success("Search completed successfully")
        elif "Error:" in result_text:
            print_error("Search encountered an error")
        else:
            print_warning("No results found")

    except Exception as e:
        print_error(f"Search failed: {e}")
        raise typer.Exit(1)


@app.command()
def evaluate(
    output: str = typer.Option(None, "--output", "-o", help="Output file path for evaluation results"),
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of evaluation items"),
    randomize: bool = typer.Option(False, "--randomize", "-r", help="Randomize the evaluation dataset order"),
    config_env: bool = typer.Option(False, "--env", help="Load configuration from environment variables"),
):
    """Evaluate retrieval performance using cosine similarity with ground truth."""
    console.print(Panel.fit("🔍 WiQAS Retrieval Evaluation", style="bold blue"))

    try:
        config = WiQASConfig.from_env() if config_env else WiQASConfig()
        eval_config = config.rag.evaluation

        if limit is not None:
            eval_config.limit = limit
        if randomize:
            eval_config.randomize = True

        if output:
            output_path = output
        else:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            output_path = f"./data/evaluation/retrieval/{timestamp}.json"

        output_file_path = Path(output_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

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
        config_table.add_row("Similarity Threshold", f"{eval_config.similarity_threshold:.2f}")
        config_table.add_row("Output File", output_path)

        console.print(config_table)
        console.print()

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Running evaluation...", total=None)
            evaluator = RetrievalEvaluator(config)
            results = evaluator.evaluate()
            progress.remove_task(task)

        if "error" in results:
            console.print(f"❌ Evaluation failed: {results['error']}", style="red")
            raise typer.Exit(code=1)

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

        # Save results
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print_success(f"Evaluation results saved to: {output_path}")

    except KeyboardInterrupt:
        console.print("\n⚠️  Evaluation interrupted by user", style="yellow")
        raise typer.Exit(code=130)
    except Exception as e:
        print_error(f"Evaluation failed: {e}")
        raise typer.Exit(1)


# ========== ANSWER GENERATION COMMANDS ==========
@app.command()
def ask(
    query: str = typer.Argument(..., help="The question to ask"),
    k: int = typer.Option(5, "--results", "-k", help="Number of retrieval results to use as context (default: 5, recommended: 3-7)"),
    query_type: str = typer.Option(None, "--type", "-t", help="Query type: Factual, Analytical, Procedural, Comparative, Exploratory (auto-detected if not specified)"),
    language: str = typer.Option(None, "--language", "-l", help="Response language: 'fil' (Filipino) or 'en' (English) - auto-detected if not specified"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON with full metadata"),
    show_contexts: bool = typer.Option(True, "--show-contexts/--hide-contexts", help="Show retrieved context documents with scores (default: show)"),
    show_timing: bool = typer.Option(True, "--timing/--no-timing", help="Show performance timing breakdown (default: show)"),
    show_classification: bool = typer.Option(True, "--classification/--no-classification", help="Show query classification details (default: show)"),
):
    """
    Ask a question and get an AI-generated answer using retrieval + LLM.

    Performs full RAG (Retrieval-Augmented Generation) pipeline:
    1. Retrieves relevant context from knowledge base
    2. Classifies query type and detects language
    3. Generates answer using LLM with retrieved context

    Supports both English and Filipino questions with automatic detection.

    Examples:
        python run.py ask "What is bayanihan?"
        python run.py ask "Ano ang pakikisama?" --language fil
        python run.py ask "Explain Filipino hospitality" --results 7 --type Analytical
        python run.py ask "How to show respect?" --json --no-timing
    """
    console.print(Panel("[bold blue]WiQAS Answer Generation[/bold blue]"))

    generator = WiQASGenerator(WiQASConfig.from_env(), use_query_classifier=True)

    result = generator.generate(
        query=query,
        k=k,
        query_type=query_type,
        language=language,
        show_contexts=True,
        include_timing=show_timing,
        include_classification=show_classification,
    )

    answer = result["answer"]
    contexts = result.get("contexts", [])
    timing = result.get("timing")
    classification = result.get("classification")
    detected_type = result.get("query_type")
    detected_language = result.get("language")

    if show_classification and classification:
        class_table = Table(title="Query Classification", show_header=True, header_style="bold magenta")
        class_table.add_column("Property", style="cyan")
        class_table.add_column("Value", style="green")

        class_table.add_row("Detected Type", classification.get("detected_type", "N/A"))
        class_table.add_row("Used Type", classification.get("used_type", "N/A"))
        class_table.add_row("Detected Language", classification.get("detected_language", "N/A"))
        class_table.add_row("Used Language", classification.get("used_language", "N/A"))
        class_table.add_row("Confidence", f"{classification.get('confidence', 0.0):.2%}")

        console.print(class_table)
    elif not show_classification:
        console.print(f"[bold cyan]Query Type:[/bold cyan] {detected_type} | [bold cyan]Language:[/bold cyan] {detected_language.upper()}")

    if show_contexts and contexts:
        context_table = Table(title=f"Retrieved Contexts (top {len(contexts)})", show_header=True, header_style="bold blue")
        context_table.add_column("#", style="cyan", width=3)
        context_table.add_column("Text", style="white", width=60)
        context_table.add_column("Score", style="green", width=6)
        context_table.add_column("Source", style="yellow", width=30)

        for idx, c in enumerate(contexts, 1):
            if isinstance(c, dict):
                text = c.get("text", "")[:200] + "..." if len(c.get("text", "")) > 200 else c.get("text", "")
                final_score = c.get("final_score", 0.0)
                citation_text = c.get("citation_text", "")
                source_file = c.get("source_file", "Unknown")

                source_display = citation_text if citation_text else source_file

                context_table.add_row(str(idx), text, f"{final_score:.3f}", source_display[:30] + "..." if len(source_display) > 30 else source_display)

        console.print(context_table)

    if show_timing and timing:
        timing_table = Table(title="Performance Timing", show_header=True, header_style="bold yellow")
        timing_table.add_column("Component", style="cyan")
        timing_table.add_column("Time (s)", style="green", justify="right")
        timing_table.add_column("% of Total", style="magenta", justify="right")

        total_time = timing.total_time if timing.total_time > 0 else 1.0

        components = [
            ("Embedding", timing.embedding_time),
            ("Search", timing.search_time),
            ("Reranking", timing.reranking_time),
            ("MMR", timing.mmr_time),
            ("Language Detection", timing.language_detection_time),
            ("Translation", timing.translation_time),
            ("Context Preparation", timing.context_preparation_time),
            ("Query Classification", getattr(timing, "classification_time", 0.0)),
            ("Prompt Building", timing.prompt_building_time),
            ("LLM Generation", timing.llm_generation_time),
        ]

        for name, time_val in components:
            if time_val > 0:
                percentage = (time_val / total_time) * 100
                timing_table.add_row(name, f"{time_val:.4f}", f"{percentage:.1f}%")

        timing_table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_time:.4f}[/bold]", "[bold]100.0%[/bold]")

        console.print(timing_table)

    if json_output:
        output = {
            "query": query,
            "answer": answer,
            "query_type": detected_type,
            "language": detected_language,
            "contexts": contexts,
        }

        if show_classification and classification:
            output["classification"] = classification

        if show_timing and timing:
            output["timing"] = {
                "classification_time": getattr(timing, "classification_time", 0.0),
                "embedding_time": timing.embedding_time,
                "search_time": timing.search_time,
                "reranking_time": timing.reranking_time,
                "mmr_time": timing.mmr_time,
                "context_preparation_time": timing.context_preparation_time,
                "prompt_building_time": timing.prompt_building_time,
                "llm_generation_time": timing.llm_generation_time,
                "translation_time": timing.translation_time,
                "language_detection_time": timing.language_detection_time,
                "total_time": timing.total_time,
            }
        console.print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        console.print(Panel(answer, title="[bold green]Answer[/bold green]", border_style="green"))


@app.command(name="batch-ask")
def batch_ask(
    input_file: str = typer.Argument(..., help="Path to input text file with questions"),
    output_file: str = typer.Option("batch_output.json", "--output", "-o", help="Path to output JSON file"),
    k: int = typer.Option(5, "--results", "-k", help="Number of retrieval results per question"),
    query_type: str = typer.Option(None, "--type", "-t", help="Query type for all questions (auto-detected if not specified)"),
    language: str = typer.Option(None, "--language", "-l", help="Response language for all questions (auto-detected if not specified)"),
    include_timing: bool = typer.Option(True, "--timing", help="Include timing information in output"),
    include_classification: bool = typer.Option(True, "--classification", help="Include classification info in output"),
    delimiter: str = typer.Option("?", "--delimiter", "-d", help="Question delimiter (default: '?')"),
):
    """Run batch question answering from a text file."""
    console.print(Panel("[bold blue]WiQAS Batch Answer Generation[/bold blue]"))

    generator = WiQASGenerator(WiQASConfig.from_env())

    with open(input_file, encoding="utf-8") as f:
        content = f.read()

    if delimiter == "\\n":
        raw_questions = [q.strip() for q in content.split("\n") if q.strip()]
    else:
        raw_questions = [q.strip() for q in content.split(delimiter) if q.strip()]

    questions = []
    for q in raw_questions:
        if not q.endswith("?") and delimiter == "?":
            q = q + "?"
        questions.append(q)

    console.print(f"[bold]Found {len(questions)} questions to process[/bold]")

    results = []

    for i, query in enumerate(questions, 1):
        console.print(f"\n[bold green]Processing {i}/{len(questions)}:[/bold green] {query[:80]}{'...' if len(query) > 80 else ''}")

        try:
            result = generator.generate(
                query=query,
                k=k,
                query_type=query_type,
                language=language,
                show_contexts=True,
                include_timing=include_timing,
                include_classification=include_classification,
            )

            contexts = result.get("contexts", [])
            structured_contexts = []

            for c in contexts:
                if not isinstance(c, dict):
                    try:
                        c = c.__dict__
                    except AttributeError:
                        try:
                            c = json.loads(json.dumps(c, default=lambda o: o.__dict__))
                        except Exception:
                            continue

                structured_contexts.append(
                    {
                        "text": c.get("text", ""),
                        "final_score": c.get("final_score", 0.0),
                        "source_file": c.get("source_file", ""),
                        "citation_text": c.get("citation_text", ""),
                    }
                )

            result_entry = {
                "question": query,
                "answer": result.get("answer", ""),
                "query_type": result.get("query_type", ""),
                "language": result.get("language", ""),
                "contexts": structured_contexts,
            }

            if include_classification and "classification" in result:
                result_entry["classification"] = result["classification"]

            if include_timing and "timing" in result:
                timing = result["timing"]
                result_entry["timing"] = {
                    "classification_time": getattr(timing, "classification_time", 0.0),
                    "embedding_time": timing.embedding_time,
                    "search_time": timing.search_time,
                    "reranking_time": timing.reranking_time,
                    "mmr_time": timing.mmr_time,
                    "context_preparation_time": timing.context_preparation_time,
                    "prompt_building_time": timing.prompt_building_time,
                    "llm_generation_time": timing.llm_generation_time,
                    "translation_time": timing.translation_time,
                    "language_detection_time": timing.language_detection_time,
                    "total_time": timing.total_time,
                }

            results.append(result_entry)
            console.print("[green]✓ Processed successfully[/green]")

        except Exception as e:
            console.print(f"[bold red]✗ Error processing question: {e}[/bold red]")
            results.append({"question": query, "answer": "", "query_type": "", "language": "", "contexts": [], "error": str(e)})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    successful = len([r for r in results if "error" not in r])
    failed = len(results) - successful

    summary_table = Table(title="Batch Processing Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total Questions", str(len(questions)))
    summary_table.add_row("Successful", str(successful))
    summary_table.add_row("Failed", str(failed))
    summary_table.add_row("Output File", output_file)

    console.print(summary_table)
    console.print(Panel("[bold green]Batch generation complete![/bold green]", border_style="green"))


# ========== SYSTEM MANAGEMENT COMMANDS ==========
@app.command()
def status(
    config_env: bool = typer.Option(False, "--env", help="Load configuration from environment variables"),
):
    """Show system status and health information."""
    print_header("WiQAS System Status")

    config = WiQASConfig.from_env() if config_env else WiQASConfig()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Checking system health...", total=None)
        health = check_system_health(config)
        progress.remove_task(task)

    table = Table(title="System Health", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Details", style="dim")

    status_icon = "✓" if health["ollama"] else "✗"
    status_color = "green" if health["ollama"] else "red"
    table.add_row("Ollama Service", f"[{status_color}]{status_icon}[/{status_color}]", "Connected" if health["ollama"] else "Not available")

    status_icon = "✓" if health["vector_store"] else "✗"
    status_color = "green" if health["vector_store"] else "red"
    doc_count = health.get("document_count", 0)
    table.add_row("Vector Store", f"[{status_color}]{status_icon}[/{status_color}]", f"{doc_count} documents" if health["vector_store"] else "Not available")

    status_icon = "✓" if health["embedding_model"] else "✗"
    status_color = "green" if health["embedding_model"] else "red"
    table.add_row("Embedding Model", f"[{status_color}]{status_icon}[/{status_color}]", config.rag.embedding.model if health["embedding_model"] else "Not available")

    status_icon = "✓" if health["knowledge_base"] else "✗"
    status_color = "green" if health["knowledge_base"] else "red"
    kb_files = health.get("knowledge_base_files", 0)
    table.add_row("Knowledge Base", f"[{status_color}]{status_icon}[/{status_color}]", f"{kb_files} files" if health["knowledge_base"] else "Directory not found")

    console.print(table)

    if health["errors"]:
        console.print("\n[bold red]Errors:[/bold red]")
        for error in health["errors"]:
            print_error(error)
    else:
        print_success("All systems operational")

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
        console.print(json.dumps(config.__dict__, indent=2, default=str, ensure_ascii=False))
    else:
        # System Configuration
        system_table = Table(title="System Configuration", show_header=True, header_style="bold cyan")
        system_table.add_column("Setting", style="yellow", width=30)
        system_table.add_column("Value", style="white")

        system_table.add_row("Data Directory", config.system.storage.data_directory)
        system_table.add_row("Knowledge Base Directory", config.system.storage.knowledge_base_directory)
        system_table.add_row("Log Level", config.system.logging.log_level)
        system_table.add_row("Log File", config.system.logging.log_file)

        console.print(system_table)
        console.print()

        # RAG Configuration
        rag_table = Table(title="RAG Configuration", show_header=True, header_style="bold cyan")
        rag_table.add_column("Setting", style="yellow", width=30)
        rag_table.add_column("Value", style="white")

        rag_table.add_row("LLM Model", config.rag.llm.model)
        rag_table.add_row("LLM Base URL", config.rag.llm.base_url)
        rag_table.add_row("LLM Temperature", str(config.rag.llm.temperature))
        rag_table.add_row("Embedding Model", config.rag.embedding.model)
        rag_table.add_row("Chunk Size", str(config.rag.chunking.chunk_size))
        rag_table.add_row("Chunk Overlap", str(config.rag.chunking.chunk_overlap))
        rag_table.add_row("Chunking Strategy", config.rag.chunking.strategy.value)
        rag_table.add_row("Reranking Enabled", str(config.rag.retrieval.enable_reranking))
        rag_table.add_row("Reranker Model", config.rag.reranker.model)
        rag_table.add_row("Cross-Lingual Enabled", str(config.rag.multilingual.enable_cross_lingual))

        console.print(rag_table)
        console.print()

        # Vector Store Configuration
        vector_table = Table(title="Vector Store Configuration", show_header=True, header_style="bold cyan")
        vector_table.add_column("Setting", style="yellow", width=30)
        vector_table.add_column("Value", style="white")

        vector_table.add_row("Persist Directory", config.rag.vectorstore.persist_directory)
        vector_table.add_row("Collection Name", config.rag.vectorstore.collection_name)
        vector_table.add_row("Distance Metric", config.rag.vectorstore.distance_metric)

        console.print(vector_table)

        console.print("\n[dim]Use --format json for complete configuration in JSON format[/dim]")


# ========== MAIN ENTRY POINT ==========
def main():
    """Main entry point for the WiQAS CLI application."""
    print_banner()
    app()


if __name__ == "__main__":
    main()
