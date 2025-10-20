from src.generation.context_preparer import prepare_contexts


def test_context_preparer_removes_exact_duplicates():
    """
    Test that exact duplicate contexts are removed.

    Input:
        - Two contexts with identical text but different scores.
    Expectation:
        - Only one context remains after processing.
        - The surviving context text is unchanged.
    """
    contexts = [
        {"content": "Fiesta is a celebration.", "final_score": 0.8},
        {"content": "Fiesta is a celebration.", "final_score": 0.8},
    ]
    result = prepare_contexts(contexts, include_citations=False)
    assert len(result) == 1
    assert result[0] == "Fiesta is a celebration."


def test_context_preparer_prefers_higher_score():
    """
    Test that when duplicates exist, the context with the higher score is kept.

    Input:
        - Two identical contexts with different scores.
    Expectation:
        - Only one context remains.
        - The higher-scored version is retained.
    """
    contexts = [
        {"content": "Ati-Atihan Festival is celebrated.", "final_score": 0.6},
        {"content": "Ati-Atihan Festival is celebrated.", "final_score": 0.9},
    ]
    result = prepare_contexts(contexts, return_scores=True, include_citations=False)

    assert len(result) == 1
    kept = result[0]
    assert kept["text"] == "Ati-Atihan Festival is celebrated."
    assert kept["final_score"] == 0.9


def test_context_preparer_prefers_longer_context():
    """
    Test that when scores are equal, the longer text is preferred.

    Input:
        - Two similar contexts with equal scores:
            1) Shorter sentence
            2) Longer sentence with more detail
    Expectation:
        - Only the longer sentence remains.
    """
    contexts = [
        {"content": "Karaoke is a popular pastime.", "final_score": 0.5},
        {"content": "Karaoke is a popular pastime in the Philippines.", "final_score": 0.5},
    ]
    result = prepare_contexts(contexts, include_citations=False)
    assert len(result) == 1
    assert result[0] == "Karaoke is a popular pastime in the Philippines."


def test_context_preparer_tie_breaker_longer_vs_higher_score():
    """
    Test tie-breaking when one context is longer but another has a higher score.

    Input:
        - Two similar contexts:
            1) Shorter sentence with higher score
            2) Longer sentence with lower score
    Expectation:
        - The shorter but higher-scored sentence is kept.
    """
    contexts = [
        {"content": "Karaoke is a popular pastime.", "final_score": 0.9},
        {"content": "Karaoke is a popular pastime in the Philippines.", "final_score": 0.5},
    ]
    result = prepare_contexts(contexts, include_citations=False)
    assert len(result) == 1
    assert result[0] == "Karaoke is a popular pastime."


def test_context_preparer_handles_non_similar_contexts():
    """
    Test that non-similar contexts are preserved independently.

    Input:
        - Two distinct sentences (different topics).
    Expectation:
        - Both contexts survive without merging.
    """
    contexts = [
        {"content": "Bayanihan is a Filipino tradition.", "final_score": 0.8},
        {"content": "Lechon is often served at fiestas.", "final_score": 0.7},
    ]
    result = prepare_contexts(contexts, include_citations=False)
    assert len(result) == 2
    assert "Bayanihan is a Filipino tradition." in result
    assert "Lechon is often served at fiestas." in result


def test_context_preparer_cleans_whitespace():
    """
    Test that unnecessary whitespace and line breaks are normalized.

    Input:
        - One context with irregular spacing and line breaks.
    Expectation:
        - Cleaned context contains single spaces only.
    """
    contexts = [
        {"text": "   Filipino    culture \n is diverse. ", "score": 0.8},
    ]
    result = prepare_contexts(contexts)
    assert len(result) == 1
    assert result[0] == "Filipino culture is diverse."


def test_context_preparer_removes_empty_after_cleaning():
    """
    Test that contexts with only whitespace are removed.

    Input:
        - One context containing only spaces and newlines.
    Expectation:
        - No contexts remain after cleaning.
    """
    contexts = [
        {"text": "    \n   ", "score": 0.5},
    ]
    result = prepare_contexts(contexts)
    assert len(result) == 0


def test_context_preparer_mixed_duplicates_and_uniques():
    """
    Test that duplicates are merged while unrelated contexts remain.

    Input:
        - Duplicate contexts about jeepneys (different scores).
        - One unique context about harana.
    Expectation:
        - Jeepney contexts are deduplicated into one.
        - Harana context remains intact.
        - Final result contains exactly two contexts.
    """
    contexts = [
        {"text": "Jeepneys are a mode of transport.", "score": 0.7},
        {"text": "Jeepneys are a mode of transport.", "score": 0.6},  # duplicate
        {"text": "Harana is a traditional serenade.", "score": 0.9},
    ]
    result = prepare_contexts(contexts)
    assert len(result) == 2
    assert "Jeepneys are a mode of transport." in result
    assert "Harana is a traditional serenade." in result


def test_context_preparer_cleans_repetitions():
    """
    Test that repeated phrases (unigrams to 4-grams) are collapsed.

    Input:
        - One context with repeated "Sinigang na" and "Sinigang sa".
    Expectation:
        - Repeated n-grams are collapsed to single occurrences.
        - Important sections like "Ang sinigang ay isang",
          "Mga uri ng sinigang", and "Mga sanggunian" remain intact.
        - No over-collapsing: at least one instance of each phrase is preserved.
    """
    contexts = [
        {
            "text": ("Ang sinigang ay isang. Mga uri ng sinigang " "Sinigang na Sinigang na Sinigang na " "Ayon sa pampaasim na sangkap " "Sinigang sa Sinigang sa Sinigang sa " "Mga sanggunian"),
            "score": 0.8,
        }
    ]

    result = prepare_contexts(contexts)
    assert len(result) == 1, f"Expected 1 context, got {len(result)}"
    cleaned = result[0]

    # unwanted repetitions must be gone
    assert "Sinigang na Sinigang na" not in cleaned, "bigram repetition still present"
    assert "Sinigang sa Sinigang sa" not in cleaned, "bigram repetition still present"

    # at least one copy of each phrase must remain
    assert "Sinigang na" in cleaned, "collapsed too aggressively, lost 'Sinigang na'"
    assert "Sinigang sa" in cleaned, "collapsed too aggressively, lost 'Sinigang sa'"

    # important beginnings and endings remain
    assert "Ang sinigang ay isang" in cleaned, "intro missing after cleaning"
    assert "Mga uri ng sinigang" in cleaned, "section header missing"
    assert "Mga sanggunian" in cleaned, "ending missing"
