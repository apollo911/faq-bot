"""Loading the company knowledge base from .txt and .pdf files."""

from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".txt", ".pdf"}


class KnowledgeBaseError(Exception):
    """The knowledge base is missing or contains no readable text."""


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def load_knowledge_base(directory: Path) -> str:
    """Return the text of every supported file in `directory`, each wrapped in a <document> tag.

    Files are read in name order so the result is identical between calls,
    which keeps the prompt cache valid. Files are only read, never modified.
    """
    if not directory.is_dir():
        raise KnowledgeBaseError(f"Knowledge base folder not found: {directory}")

    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    documents = []
    for path in files:
        text = _read_file(path).strip()
        if text:
            documents.append(f'<document name="{path.name}">\n{text}\n</document>')

    if not documents:
        raise KnowledgeBaseError(f"No .txt or .pdf files with text in {directory}")
    return "\n\n".join(documents)
