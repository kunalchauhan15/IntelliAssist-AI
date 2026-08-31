"""
IntelliAssist AI
Phase 18 — Sentiment Analysis

This module detects the sentiment of a user's query.

Supported sentiments:
- Positive
- Negative
- Neutral

The analysis is performed locally and does not require
a Gemini API request.
"""


# ============================================================
# SENTIMENT KEYWORDS
# ============================================================

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "happy",
    "helpful",
    "useful",
    "love",
    "like",
    "best",
    "perfect",
    "wonderful",
    "fantastic",
    "satisfied",
    "thank",
    "thanks",
    "easy",
    "success",
    "successful"
}


NEGATIVE_WORDS = {
    "bad",
    "poor",
    "worst",
    "terrible",
    "horrible",
    "frustrated",
    "frustrating",
    "angry",
    "hate",
    "problem",
    "problems",
    "error",
    "errors",
    "fail",
    "failed",
    "failure",
    "broken",
    "wrong",
    "difficult",
    "useless",
    "disappointed",
    "disappointing",
    "issue",
    "issues",
    "not",
    "doesn't",
    "doesnt",
    "cannot",
    "can't",
    "cant"
}


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of the provided text.

    Returns:
        {
            "sentiment": "Positive" | "Negative" | "Neutral",
            "confidence": float
        }
    """

    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not text or not text.strip():

        return {
            "sentiment": "Neutral",
            "confidence": 100.0
        }


    # --------------------------------------------------------
    # NORMALIZE TEXT
    # --------------------------------------------------------

    words = (
        text.lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace("!", " ")
        .replace("?", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace('"', " ")
        .replace("'", " ")
        .split()
    )


    # --------------------------------------------------------
    # COUNT SENTIMENT WORDS
    # --------------------------------------------------------

    positive_count = 0
    negative_count = 0


    for word in words:

        if word in POSITIVE_WORDS:
            positive_count += 1

        if word in NEGATIVE_WORDS:
            negative_count += 1


    # --------------------------------------------------------
    # CALCULATE TOTAL
    # --------------------------------------------------------

    total_sentiment_words = (
        positive_count +
        negative_count
    )


    # --------------------------------------------------------
    # NEUTRAL
    # --------------------------------------------------------

    if total_sentiment_words == 0:

        return {
            "sentiment": "Neutral",
            "confidence": 95.0
        }


    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    if positive_count > negative_count:

        confidence = (
            positive_count /
            total_sentiment_words
        ) * 100

        return {
            "sentiment": "Positive",
            "confidence": round(
                confidence,
                2
            )
        }


    # --------------------------------------------------------
    # NEGATIVE
    # --------------------------------------------------------

    if negative_count > positive_count:

        confidence = (
            negative_count /
            total_sentiment_words
        ) * 100

        return {
            "sentiment": "Negative",
            "confidence": round(
                confidence,
                2
            )
        }


    # --------------------------------------------------------
    # EQUAL POSITIVE AND NEGATIVE
    # --------------------------------------------------------

    return {
        "sentiment": "Neutral",
        "confidence": 50.0
    }
