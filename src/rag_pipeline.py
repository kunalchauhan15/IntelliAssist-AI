import os

from dotenv import load_dotenv

from src.embeddings import generate_embeddings
from src.vector_store import search_faiss


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()


# ============================================================
# GEMINI API KEY
# ============================================================

def get_gemini_api_key() -> str:
    """
    Get Gemini API key from environment variables.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError(
            "GEMINI_API_KEY is not configured. "
            "Please add your Gemini API key to the .env file."
        )

    return api_key


# ============================================================
# GEMINI ANSWER GENERATION
# ============================================================

def generate_answer(
    prompt: str,
    temperature: float = 0.0,
    model: str = "gemini-3.6-flash"
) -> str:
    """
    Generate an answer using Gemini.

    Phase 24:
    Standardized API error handling.
    """

    # --------------------------------------------------------
    # EMPTY PROMPT
    # --------------------------------------------------------

    if not prompt or not prompt.strip():

        return (
            "⚠ AI service temporarily unavailable.\n"
            "Please try again."
        )


    # --------------------------------------------------------
    # GET API KEY
    # --------------------------------------------------------

    try:

        api_key = get_gemini_api_key()

    except Exception:

        return (
            "⚠ AI service temporarily unavailable.\n"
            "Please try again."
        )


    # --------------------------------------------------------
    # CREATE GEMINI CLIENT
    # --------------------------------------------------------

    try:

        from google import genai

        client = genai.Client(
            api_key=api_key
        )

    except Exception:

        return (
            "⚠ AI service temporarily unavailable.\n"
            "Please try again."
        )


    # --------------------------------------------------------
    # GENERATE RESPONSE
    # --------------------------------------------------------

    try:

        try:

            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=float(temperature)
            )

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )

        except Exception:

            # SDK compatibility fallback

            response = client.models.generate_content(
                model=model,
                contents=prompt
            )


        # ----------------------------------------------------
        # EMPTY RESPONSE
        # ----------------------------------------------------

        if response is None:

            return (
                "⚠ AI service temporarily unavailable.\n"
                "Please try again."
            )


        if not getattr(response, "text", None):

            return (
                "⚠ AI service temporarily unavailable.\n"
                "Please try again."
            )


        return response.text.strip()


    # --------------------------------------------------------
    # API FAILURE
    # --------------------------------------------------------

    except Exception as e:

        error_text = str(e)

        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
            or "quota" in error_text.lower()
        ):

            return (
                "⚠️ **AI service quota exceeded.**\n\n"
                "Please try again later or check your "
                "Gemini API quota and billing settings."
            )

        return (
            "⚠ AI service temporarily unavailable.\n"
            "Please try again."
        )

# ============================================================
# SIMPLE TEXT SUMMARIZATION
# ============================================================

def summarize_text(text: str) -> str:
    """
    Generate a structured summary of the document text
    using Gemini.
    """

    if not text or not text.strip():

        return (
            "No document text is available for summarization."
        )


    prompt = f"""
You are IntelliAssist AI, a document summarization assistant.

Summarize the following document using ONLY the information
provided in the document text.

Follow this format:

### Document Summary

Write a concise overview of the document.

### Key Points

- Point 1
- Point 2
- Point 3
- Point 4
- Point 5

### Conclusion

Write a short conclusion based only on the document.

Rules:

1. Do not add information that is not present in the document.
2. Do not make up facts.
3. Keep the summary clear and easy to understand.
4. Focus on the main purpose, important topics, and conclusions.

Document:
-----------------
{text}
-----------------
"""

    return generate_answer(
        prompt
    )


# ============================================================
# DOCUMENT SUMMARIZATION
# ============================================================

def summarize_document(
    chunks: list[dict],
    batch_size: int = 5,
    temperature: float = 0.0,
    model: str = "gemini-3.6-flash"
) -> str:
    """
    Summarize a large document using batch-based
    summarization.
    """

    if not chunks:

        return (
            "No document content is available for summarization."
        )


    batch_summaries = []


    # ========================================================
    # PROCESS DOCUMENT CHUNKS IN BATCHES
    # ========================================================

    for i in range(
        0,
        len(chunks),
        batch_size
    ):

        batch = chunks[
            i:i + batch_size
        ]

        batch_text_parts = []


        for chunk in batch:

            source = chunk.get(
                "source",
                "Unknown"
            )

            page = chunk.get(
                "page",
                "Unknown"
            )

            text = chunk.get(
                "text",
                ""
            )

            if not text.strip():
                continue


            batch_text_parts.append(
                f"[Source: {source} | Page: {page}]\n{text}"
            )


        if not batch_text_parts:
            continue


        batch_text = "\n\n".join(
            batch_text_parts
        )


        prompt = f"""
You are IntelliAssist AI, a document summarization assistant.

Summarize ONLY the information provided below.

Focus on:

- Main ideas
- Important facts
- Important topics
- Findings or observations

Do not add information that is not present.

Document Content:
-----------------
{batch_text}
-----------------

Write a concise summary of this section.
"""


        summary = generate_answer(
            prompt,
            temperature=temperature,
            model=model
        )


        if summary and summary.strip():

            batch_summaries.append(
                summary.strip()
            )


    # ========================================================
    # CHECK BATCH SUMMARIES
    # ========================================================

    if not batch_summaries:

        return (
            "Unable to generate a document summary."
        )


    # ========================================================
    # COMBINE BATCH SUMMARIES
    # ========================================================

    combined_summary = "\n\n".join(
        batch_summaries
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    final_prompt = f"""
You are IntelliAssist AI.

Create a final summary of the document using ONLY
the section summaries provided below.

Use exactly this structure:

### Document Summary

Write a concise overview of the entire document.

### Key Points

- Important point 1
- Important point 2
- Important point 3
- Important point 4
- Important point 5

### Conclusion

Write a concise conclusion based only on the provided
section summaries.

Do not introduce information that is not present.

Section Summaries:
-----------------
{combined_summary}
-----------------
"""


    return generate_answer(
        final_prompt,
        temperature=temperature,
        model=model
    )


# ============================================================
# RETRIEVE RELEVANT DOCUMENT CONTEXT
# ============================================================

def retrieve_context(
    question: str,
    chunks: list[dict],
    index,
    top_k: int = 3,
    distance_threshold: float = 2.00
) -> list[dict]:
    """
    Retrieve the most relevant document chunks using FAISS.

    Handles both 1D and 2D NumPy arrays returned by FAISS.

    Lower FAISS L2 distance means better similarity.
    """

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if not question or not question.strip():
        return []

    if not chunks:
        return []

    if index is None:
        return []


    # --------------------------------------------------------
    # GENERATE QUESTION EMBEDDING
    # --------------------------------------------------------

    query_embeddings = generate_embeddings(
        [question.strip()]
    )


    if query_embeddings is None:
        return []


    if len(query_embeddings) == 0:
        return []


    query_embedding = query_embeddings[0]


    # --------------------------------------------------------
    # SEARCH FAISS
    # --------------------------------------------------------

    distances, indices = search_faiss(
        index,
        query_embedding,
        top_k=top_k
    )


    # --------------------------------------------------------
    # NORMALIZE FAISS OUTPUT
    #
    # FAISS commonly returns:
    #
    # distances -> [[0.12, 0.45, 0.91]]
    # indices   -> [[2,    5,    1   ]]
    #
    # But some implementations may return:
    #
    # distances -> [0.12, 0.45, 0.91]
    # indices   -> [2,    5,    1   ]
    #
    # Convert both into simple 1D arrays.
    # --------------------------------------------------------

    try:

        import numpy as np

        distances = np.asarray(
            distances
        ).reshape(-1)

        indices = np.asarray(
            indices
        ).reshape(-1)

    except Exception:

        return []


    # --------------------------------------------------------
    # PROCESS SEARCH RESULTS
    # --------------------------------------------------------

    context = []

    seen_sources_pages = set()


    for distance, index_position in zip(
        distances,
        indices
    ):

        # ----------------------------------------------------
        # CONVERT NUMPY VALUES TO PYTHON SCALARS
        # ----------------------------------------------------

        try:

            distance = float(
                distance
            )

            index_position = int(
                index_position
            )

        except (TypeError, ValueError):

            continue


        # ----------------------------------------------------
        # INVALID INDEX
        # ----------------------------------------------------

        if index_position < 0:
            continue


        if index_position >= len(chunks):
            continue


        # ----------------------------------------------------
        # SIMILARITY THRESHOLD
        # ----------------------------------------------------

        if distance > float(
            distance_threshold
        ):
            continue


        # ----------------------------------------------------
        # GET CHUNK
        # ----------------------------------------------------

        chunk = chunks[
            index_position
        ]


        if not isinstance(
            chunk,
            dict
        ):

            continue


        # ----------------------------------------------------
        # GET METADATA
        # ----------------------------------------------------

        page = chunk.get(
            "page"
        ) or 1


        source = chunk.get(
            "source",
            "Unknown"
        )


        text = chunk.get(
            "text",
            ""
        )


        # ----------------------------------------------------
        # VALIDATE TEXT
        # ----------------------------------------------------

        if not isinstance(
            text,
            str
        ):

            text = str(
                text
            )


        if not text.strip():
            continue


        # ----------------------------------------------------
        # AVOID DUPLICATE SOURCE + PAGE
        # ----------------------------------------------------

        source_page = (
            source,
            page
        )


        if source_page in seen_sources_pages:
            continue


        seen_sources_pages.add(
            source_page
        )


        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        context.append(
            {
                "text": text,
                "page": page,
                "source": source,
                "distance": distance
            }
        )


    # --------------------------------------------------------
    # RETURN CONTEXT
    # --------------------------------------------------------

    return context


# ============================================================
# BUILD RAG PROMPT
# ============================================================

def build_prompt(
    question: str,
    context: list[dict]
) -> str:
    """
    Build a strict document-grounded prompt.
    """

    context_parts = []


    for chunk in context:

        page = chunk.get(
            "page"
        )

        source = chunk.get(
            "source"
        )

        text = chunk.get(
            "text"
        )


        context_parts.append(
            f"[Source: {source} | Page: {page}]\n{text}"
        )


    context_text = "\n\n".join(
        context_parts
    )


    prompt = f"""
You are IntelliAssist AI, a document-based AI assistant.

Your task is to answer the user's question using ONLY
the information available in the provided document context.

Rules:

1. Do not use information from outside the provided context.

2. Do not make up or assume facts that are not present.

3. If the answer cannot be found in the context, say:
   "The information is not available in the uploaded document."

4. Give a clear and concise answer.

5. Explain the answer using the relevant information
   from the document when possible.

6. DO NOT create, modify, guess, or invent source names,
   page numbers, or citations.

7. Source citations will be added separately by the application.

Document Context:
-----------------
{context_text}
-----------------

User Question:
{question}

Answer:
"""


    return prompt.strip()


# ============================================================
# BUILD SOURCE CITATIONS
# ============================================================

def build_source_citations(
    context: list[dict]
) -> list[str]:
    """
    Build reliable source citations directly from
    retrieved chunk metadata.
    """

    if not context:
        return []


    citations = []

    seen = set()


    for chunk in context:

        source = chunk.get(
            "source",
            "Unknown"
        )

        page = chunk.get(
            "page"
        )

        section = chunk.get(
            "section"
        )


        # ====================================================
        # SECTION CITATION
        # ====================================================

        if section:

            citation = (
                f"{source} — Section: {section}"
            )

            citation_key = (
                source,
                "section",
                section
            )


        # ====================================================
        # PAGE CITATION
        # ====================================================

        elif page:

            citation = (
                f"{source} — Page {page}"
            )

            citation_key = (
                source,
                "page",
                page
            )


        # ====================================================
        # SOURCE ONLY
        # ====================================================

        else:

            citation = source

            citation_key = (
                source,
                "source"
            )


        if citation_key in seen:
            continue


        seen.add(
            citation_key
        )

        citations.append(
            citation
        )


    return citations
