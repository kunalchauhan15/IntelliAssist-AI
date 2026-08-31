"""
IntelliAssist AI
Phase 19 — Intent Analysis

This module detects what the user wants to do with
their query.

Supported intents:
- QUESTION
- SUMMARY
- SEARCH
- EXPLANATION
- COMPARISON
- UNKNOWN

This implementation works locally and does not require
a Gemini API request.
"""


# ============================================================
# INTENT KEYWORDS
# ============================================================

SUMMARY_KEYWORDS = {
    "summarize",
    "summary",
    "summarise",
    "overview",
    "brief",
    "short summary",
    "summarize this",
    "summarise this"
}


SEARCH_KEYWORDS = {
    "find",
    "search",
    "look for",
    "locate",
    "information about",
    "information on",
    "tell me about"
}


EXPLANATION_KEYWORDS = {
    "explain",
    "explanation",
    "why",
    "how does",
    "how do",
    "describe",
    "meaning",
    "what does",
    "what is meant by"
}


COMPARISON_KEYWORDS = {
    "compare",
    "comparison",
    "difference",
    "differences",
    "similarity",
    "similarities",
    "versus",
    "vs",
    "better than",
    "difference between"
}


QUESTION_WORDS = {
    "what",
    "who",
    "when",
    "where",
    "which",
    "whose",
    "whom",
    "can",
    "could",
    "would",
    "is",
    "are",
    "do",
    "does",
    "did"
}


# ============================================================
# INTENT ANALYSIS
# ============================================================

def analyze_intent(text: str) -> dict:
    """
    Detect the user's intent.

    Returns:

        {
            "intent": "QUESTION",
            "confidence": 95.0
        }
    """

    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not text or not text.strip():

        return {
            "intent": "UNKNOWN",
            "confidence": 100.0
        }


    # --------------------------------------------------------
    # NORMALIZE TEXT
    # --------------------------------------------------------

    query = text.lower().strip()


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    for keyword in SUMMARY_KEYWORDS:

        if keyword in query:

            return {
                "intent": "SUMMARY",
                "confidence": 95.0
            }


    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    for keyword in COMPARISON_KEYWORDS:

        if keyword in query:

            return {
                "intent": "COMPARISON",
                "confidence": 95.0
            }


    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    for keyword in EXPLANATION_KEYWORDS:

        if keyword in query:

            return {
                "intent": "EXPLANATION",
                "confidence": 90.0
            }


    # --------------------------------------------------------
    # SEARCH / INFORMATION RETRIEVAL
    # --------------------------------------------------------

    for keyword in SEARCH_KEYWORDS:

        if keyword in query:

            return {
                "intent": "SEARCH",
                "confidence": 90.0
            }


    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    words = query.replace("?", " ").split()

    if words:

        first_word = words[0]

        if first_word in QUESTION_WORDS:

            return {
                "intent": "QUESTION",
                "confidence": 95.0
            }


    # --------------------------------------------------------
    # QUESTION MARK FALLBACK
    # --------------------------------------------------------

    if "?" in query:

        return {
            "intent": "QUESTION",
            "confidence": 80.0
        }


    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return {
        "intent": "UNKNOWN",
        "confidence": 60.0
    }
