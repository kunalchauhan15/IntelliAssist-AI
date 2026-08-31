from sentence_transformers import SentenceTransformer


# Load the embedding model once.
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):
    """
    Generate embeddings for text chunks.

    Supports:
    1. List of strings
    2. List of dictionaries containing:
       {"text": "...", "page": 1}

    Only the text is converted into embeddings.
    Page information is preserved separately.
    """

    if not chunks:
        return []

    # If chunks contain dictionaries, extract only the text.
    if isinstance(chunks[0], dict):
        texts = [chunk["text"] for chunk in chunks]
    else:
        texts = chunks

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return embeddings


def generate_embedding(text: str):
    """
    Generate an embedding for a single text.
    """

    if not text:
        return None

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return embedding
