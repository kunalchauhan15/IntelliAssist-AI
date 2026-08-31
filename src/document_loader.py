from pathlib import Path

from pypdf import PdfReader
from docx import Document


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a text-based PDF.
    """

    reader = PdfReader(file_path)
    source = Path(file_path).name

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:
            text = page.extract_text()
        except Exception:
            text = ""

        if text and text.strip():

            pages.append({
                "text": text.strip(),
                "page": page_number,
                "source": source
            })

    if not pages:
        raise ValueError(
            "This PDF file is not supported because "
            "no readable text was found. "
            "Please upload another PDF, TXT, or DOCX file."
        )

    return pages


def load_txt(file_path: str) -> list[dict]:
    """
    Extract text from a TXT file.
    """

    path = Path(file_path)
    source = path.name

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    if not text.strip():
        raise ValueError(
            "This TXT file is empty. "
            "Please upload another PDF, TXT, or DOCX file."
        )

    return [{
        "text": text.strip(),
        "page": None,
        "source": source
    }]


def load_docx(file_path: str) -> list[dict]:
    """
    Extract text from a DOCX file.
    """

    document = Document(file_path)
    source = Path(file_path).name

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    if not paragraphs:
        raise ValueError(
            "This DOCX file does not contain readable text. "
            "Please upload another PDF, TXT, or DOCX file."
        )

    return [{
        "text": "\n\n".join(paragraphs),
        "page": None,
        "source": source
    }]


def load_document(file_path: str) -> list[dict]:
    """
    Automatically detect and load PDF, TXT, or DOCX.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            "Uploaded file could not be found."
        )

    extension = path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    elif extension == ".txt":
        return load_txt(file_path)

    elif extension == ".docx":
        return load_docx(file_path)

    else:
        raise ValueError(
            "Unsupported file format. "
            "Please upload a PDF, TXT, or DOCX file."
        )