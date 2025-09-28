import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

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
        console.print(Panel(
            "\n\n".join(contexts),
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
    
