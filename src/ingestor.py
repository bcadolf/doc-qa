# loads files and splits into chunks

from pathlib import Path
from pypdf import PdfReader


def load_document(file_path: str) -> str:
    """Extract text from a document PDF or TXT file"""
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )
    
    elif path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    
    else: 
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split the test into chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

