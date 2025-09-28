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
):
    console.print(Panel("[bold blue]WiQAS Answer Generation[/bold blue]"))
    generator = WiQASGenerator(WiQASConfig.from_env())

    result = generator.generate(
        query=query,
        k=k,
        query_type=query_type,
        show_contexts=True,  
    )

    answer = result["answer"]
    contexts = result.get("contexts", [])

    if show_contexts:
        context_strings = []
        for c in contexts:
            if isinstance(c, dict):
                text = c.get("text", "")
                score = c.get("score", 0.0)
                meta = c.get("metadata", {})
                doc_id = c.get("document_id")
                context_strings.append(
                    f"[bold]{doc_id or ''} - {score}\n[/bold]\n{text}\n[dim]{meta}[/dim]"
                )
            else:
                context_strings.append(str(c))

        console.print(Panel(
            "\n\n".join(context_strings),
            title=f"Retrieved Contexts (top {len(contexts)})",
            border_style="blue"
        ))

    if json_output:
        output = {
            "query": query,
            "answer": answer,
            "contexts": contexts,
        }
        console.print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        console.print(Panel(answer, title="Answer", border_style="green"))

def main():
    app()

if __name__ == "__main__":
    main()
    
