"""
WiQAS Generation CLI Interface

Run end-to-end generation: retrieval + LLM answer generation.

Usage:
    python run_generator.py "<question>"
"""

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from src.generation.generator import WiQASGenerator
from src.utilities.config import WiQASConfig

app = typer.Typer(
    name="WiQAS Generator",
    help="WiQAS Answer Generation CLI (retrieval + LLM response)",
    add_completion=False,
)
console = Console()


@app.command()
def ask(
    query: str = typer.Argument(..., help="The question to ask"),
    k: int = typer.Option(5, "--results", "-k", help="Number of retrieval results"),
    query_type: str = typer.Option(None, "--type", "-t", help="Query type: Factual, Analytical, Procedural, Comparative, Exploratory (auto-detected if not specified)"),
    language: str = typer.Option(None, "--language", "-l", help="Response language: fil or en (auto-detected if not specified)"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON"),
    show_contexts: bool = typer.Option(True, "--show-contexts/--hide-contexts", help="Show retrieved contexts in console"),
    show_timing: bool = typer.Option(True, "--timing/--no-timing", help="Show timing breakdown"),
    show_classification: bool = typer.Option(True, "--classification/--no-classification", help="Show query classification info"),
):
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
                
                # Use citation if available, otherwise use source file
                source_display = citation_text if citation_text else source_file
                
                context_table.add_row(
                    str(idx),
                    text,
                    f"{final_score:.3f}",
                    source_display[:30] + "..." if len(source_display) > 30 else source_display
                )
        
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
            ("Query Classification", getattr(timing, 'classification_time', 0.0)),
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
                "classification_time": getattr(timing, 'classification_time', 0.0),
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

@app.command()
def batch_ask(
    input_file: str = typer.Argument(..., help="Path to input text file with questions (one per line or delimited by '?')"),
    output_file: str = typer.Option("batch_output.json", "--output", "-o", help="Path to output JSON file"),
    k: int = typer.Option(5, "--results", "-k", help="Number of retrieval results per question"),
    query_type: str = typer.Option(None, "--type", "-t", help="Query type for all questions (auto-detected if not specified)"),
    language: str = typer.Option(None, "--language", "-l", help="Response language for all questions (auto-detected if not specified)"),
    include_timing: bool = typer.Option(True, "--timing", help="Include timing information in output"),
    include_classification: bool = typer.Option(True, "--classification", help="Include classification info in output"),
    delimiter: str = typer.Option("?", "--delimiter", "-d", help="Question delimiter (default: '?')"),
):
    """
    Run batch question answering from a text file delimited by '?'.
    Saves results to a JSON file containing question, context, and answer.
    """
    console.print(Panel("[bold blue]WiQAS Batch Answer Generation[/bold blue]"))
    generator = WiQASGenerator(WiQASConfig.from_env())

    with open(input_file, encoding="utf-8") as f:
        content = f.read()

    # Split by '?', clean up, and re-append '?' for each question if needed
    if delimiter == "\\n":
        # Handle newline as delimiter
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
                    "classification_time": getattr(timing, 'classification_time', 0.0),
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
            
            console.print(f"[green]✓ Processed successfully[/green]")

        except Exception as e:
            console.print(f"[bold red]✗ Error processing question: {e}[/bold red]")
            results.append({
                "question": query,
                "answer": "",
                "query_type": "",
                "language": "",
                "contexts": [],
                "error": str(e)
            })

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
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
    console.print(Panel(f"[bold green]Batch generation complete![/bold green]", border_style="green"))


def main():
    app()


if __name__ == "__main__":
    main()
