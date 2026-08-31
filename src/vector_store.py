import faiss
import numpy as np


def create_faiss_index(embeddings):
    """
    Create a FAISS index from embeddings.

    Supports embeddings as:
    - NumPy array
    - Python list
    """

    if embeddings is None:
        raise ValueError("Embeddings cannot be None.")

    # Convert embeddings to NumPy array
    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Validate embeddings
    if embeddings.size == 0:
        raise ValueError(
            "Cannot create FAISS index from empty embeddings."
        )

    # Make sure embeddings are 2-dimensional
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2-dimensional. "
            f"Received shape: {embeddings.shape}"
        )

    # Number of dimensions in each embedding
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings
    index.add(embeddings)

    return index


def save_faiss_index(
    index,
    path="vectorstore/faiss_index/index.faiss"
):
    """
    Save FAISS index to disk.
    """

    faiss.write_index(
        index,
        path
    )


def load_faiss_index(
    path="vectorstore/faiss_index/index.faiss"
):
    """
    Load FAISS index from disk.
    """

    return faiss.read_index(path)


def search_faiss(
    index,
    query_embedding,
    top_k=3
):
    """
    Search FAISS index and return
    relevant chunk indexes.
    """

    if index is None:
        raise ValueError(
            "FAISS index is not available."
        )

    if index.ntotal == 0:
        return [], []

    query_vector = np.asarray(
        [query_embedding],
        dtype="float32"
    )

    distances, indices = index.search(
        query_vector,
        min(top_k, index.ntotal)
    )

    return distances[0], indices[0]