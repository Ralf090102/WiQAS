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

from src.utilities.config import WiQASConfig
from src.generation.generator import WiQASGenerator

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
    query_type: str = typer.Option("Factual", "--type", "-t", help="Query type: Factual, Analytical, Procedural, Creative, Exploratory"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON"),
    show_contexts: bool = typer.Option(True, "--show-contexts/--hide-contexts", help="Show retrieved contexts in console"),
    show_timing: bool = typer.Option(True, "--timing/--no-timing", help="Show timing breakdown"),
):
    console.print(Panel("[bold blue]WiQAS Answer Generation[/bold blue]"))
    generator = WiQASGenerator(WiQASConfig.from_env())

    result = generator.generate(
        query=query,
        k=k,
        query_type=query_type,
        show_contexts=True,
        include_timing=show_timing,
    )

    answer = result["answer"]
    contexts = result.get("contexts", [])
    timing = result.get("timing")

    if show_contexts:
        context_strings = []
        for c in contexts:
            if isinstance(c, dict):
                text = c.get("text", "")
                final_score = c.get("final_score", 0.0)
                source_file = c.get("source_file", "")
                page = c.get("page", "")
                title = c.get("title", "")
                date = c.get("date", "")
                url = c.get("url", "")
                context_strings.append(
                    f"[bold]{source_file} - {title} | {page} | {url} | {date}\n{final_score}\n[/bold]\n{text}\n"
                )
            else:
                context_strings.append(str(c))

        console.print(Panel(
            "\n\n".join(context_strings),
            title=f"Retrieved Contexts (top {len(contexts)})",
            border_style="blue"
        ))

    if show_timing and timing:
        console.print(Panel(timing.format_timing_summary(), title="Performance Timing", border_style="blue"))

    if json_output:
        output = {
            "query": query,
            "answer": answer,
            "contexts": contexts,
        }
        if show_timing and timing:
            output["timing"] = {
                "embedding_time": timing.embedding_time,
                "search_time": timing.search_time,
                "reranking_time": timing.reranking_time,
                "mmr_time": timing.mmr_time,
                "context_preparation_time": timing.context_preparation_time,
                "prompt_building_time": timing.prompt_building_time,
                "llm_generation_time": timing.llm_generation_time,
                "total_time": timing.total_time
            }
        console.print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        console.print(Panel(answer, title="Answer", border_style="green"))

@app.command()
def batch_ask(
    input_file: str = typer.Argument(..., help="Path to input text file with questions delimited by '?'"),
    output_file: str = typer.Option("batch_output.json", "--output", "-o", help="Path to output JSON file"),
    k: int = typer.Option(5, "--results", "-k", help="Number of retrieval results per question"),
    query_type: str = typer.Option("Factual", "--type", "-t", help="Query type for all questions"),
    include_timing: bool = typer.Option(False, "--timing", help="Include timing information in output"),
):
    """
    Run batch question answering from a text file delimited by '?'.
    Saves results to a JSON file containing question, context, and answer.
    """
    console.print(Panel("[bold blue]WiQAS Batch Answer Generation[/bold blue]"))
    generator = WiQASGenerator(WiQASConfig.from_env())

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by '?', clean up, and re-append '?' for each question if needed
    raw_questions = [q.strip() for q in content.split("?") if q.strip()]
    questions = [q + "?" if not q.endswith("?") else q for q in raw_questions]

    results = []

    for i, query in enumerate(questions, 1):
        console.print(f"[bold green]Processing {i}/{len(questions)}:[/bold green] {query}")
        try:
            result = generator.generate(
                query=query,
                k=k,
                query_type=query_type,
                show_contexts=True,
                include_timing=include_timing,
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

                structured_contexts.append({
                    "text": c.get("text", ""),
                    "final_score": c.get("final_score", 0.0),
                    "source_file": c.get("source_file", "")
                })

            result_entry = {
                "question": query,
                "contexts": structured_contexts,
                "answer": result.get("answer", "")
            }
            
            if include_timing and "timing" in result:
                timing = result["timing"]
                result_entry["timing"] = {
                    "embedding_time": timing.embedding_time,
                    "search_time": timing.search_time,
                    "reranking_time": timing.reranking_time,
                    "mmr_time": timing.mmr_time,
                    "context_preparation_time": timing.context_preparation_time,
                    "prompt_building_time": timing.prompt_building_time,
                    "llm_generation_time": timing.llm_generation_time,
                    "total_time": timing.total_time
                }
            
            results.append(result_entry)

        except Exception as e:
            console.print(f"[bold red]Error processing question '{query}': {e}[/bold red]")
            results.append({
                "question": query,
                "contexts": [],
                "answer": "",
                "error": str(e)
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    console.print(Panel(f"[bold green]Batch generation complete![/bold green]\nSaved to: {output_file}", border_style="green"))

def main():
    app()

if __name__ == "__main__":
    main()
    
