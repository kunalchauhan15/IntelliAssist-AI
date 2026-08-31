import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """
    Clean extracted document text before further processing.

    Removes:
    - Extra spaces
    - Repeated newlines
    - Empty paragraphs
    - Unwanted control characters
    """

    if not text:
        return ""

    # Remove unwanted control characters
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    # Replace 3 or more consecutive newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove empty paragraphs
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    # Join cleaned lines
    text = "\n".join(lines)

    return text.strip()


def split_text(text: str) -> list[str]:
    """
    Clean the text and split it into smaller chunks.
    """

    text = clean_text(text)

    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)


def split_pages(pages: list[dict]) -> list[dict]:
    """
    Clean and split document pages into chunks
    while preserving source and page metadata.

    Each returned chunk contains:
    - text
    - page
    - source
    """

    if not pages:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for page in pages:
        text = clean_text(page.get("text", ""))
        page_number = page.get("page")
        source = page.get("source")

        if not text:
            continue

        page_chunks = splitter.split_text(text)

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": page_number,
                "source": source
            })

    return chunks