
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
from src.server import start_server, stop_server
from src.ingestor import load_document, chunk_text
from src.embedder import store_chunks, clear_collection, clear_all, clear_document
from src.retriever import retrieve
from src.qa import answer_question

console = Console()


def ingest_docs():
    """Find all PDF and TXT files in the 'docs' folder and embed them"""
    docs_path = Path("./docs")
    files = list(docs_path.glob("*.pdf")) + list(docs_path.glob("*.txt"))

    if not files:
        console.print("[red]No documents found in the 'docs' folder! Please add some PDF or TXT files.[/red]")
        return
    
    for file in files:
        console.print(f"\n[blue]Processing document {file.name}...[/blue]")
        text = load_document(str(file))
        chunks = chunk_text(text)
        store_chunks(chunks, doc_name=file.name)

    console.print(f"[green]Finished all documents! Processed {len(files)} files.[/green]")


def main():
    print("Hello from Doc QA!")

    start_server()

    try:
        console.print("\n[bold cyan]Doc Q&A[/bold cyan] — drop PDFs or .txt files in /docs\n")

        ingest_docs()

        while True:
            question = Prompt.ask("[bold]Ask a question[/bold] or choose an option: \n"
            "[green]1.[/green] Clear all data\n"
            "[green]2.[/green] Clear collection\n"
            "[green]3.[/green] Clear document\n"
            "[green]4.[/green] Exit\n"
            "Your input: ")

            if question == "1":
                clear_all()

            elif question == "2":
                clear_collection()
            
            elif question == "3":
                doc_name = Prompt.ask("Enter the document name to clear (e.g. 'mydoc.txt'):")
                clear_document(doc_name)

            elif question.lower() == "exit" or question == "4":
                break

            chunks = retrieve(question)
            answer = answer_question(question, chunks)

            console.print(f"\n[bold green]Answer:[/bold green] {answer}\n")

    finally:

        stop_server()
    


if __name__ == "__main__":
    main()
