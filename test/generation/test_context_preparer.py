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
        {"content": "   Filipino    culture \n is diverse. ", "final_score": 0.8},
    ]
    result = prepare_contexts(contexts, include_citations=False)
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
        {"content": "    \n   ", "final_score": 0.5},
    ]
    result = prepare_contexts(contexts, include_citations=False)
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
        {"content": "Jeepneys are a mode of transport.", "final_score": 0.7},
        {"content": "Jeepneys are a mode of transport.", "final_score": 0.6},  # duplicate
        {"content": "Harana is a traditional serenade.", "final_score": 0.9},
    ]
    result = prepare_contexts(contexts, include_citations=False)
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
            "content": (
                "Ang sinigang ay isang. Mga uri ng sinigang "
                "Sinigang na Sinigang na Sinigang na "
                "Ayon sa pampaasim na sangkap "
                "Sinigang sa Sinigang sa Sinigang sa "
                "Mga sanggunian"
            ),
            "final_score": 0.8,
        }
    ]

    result = prepare_contexts(contexts, include_citations=False)
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

def test_context_preparer_pdf_citation():
    """
    Test that PDF sources generate proper citations with title and page number.

    Input:
        - Context from a PDF file with page metadata.
    Expectation:
        - Citation includes formatted title and page number.
    """
    contexts = [
        {
            "content": "Machine learning is a subset of AI.",
            "final_score": 0.8,
            "source_file": "data/knowledge_base/machine-learning-intro.pdf",
            "page": 15,
        }
    ]
    result = prepare_contexts(contexts, include_citations=True)
    assert len(result) == 1
    assert "Machine learning is a subset of AI." in result[0]
    assert "[Source: Machine Learning Intro, p. 15]" in result[0]

def test_context_preparer_wikipedia_citation():
    """
    Test that Wikipedia sources generate proper citations with title and date.

    Input:
        - Context from Wikipedia with title and date metadata.
    Expectation:
        - Citation includes title, "Wikipedia", and formatted date.
    """
    contexts = [
        {
            "content": "Python is a programming language.",
            "final_score": 0.9,
            "source_file": "wikipedia/python_programming.html",
            "title": "Python (programming language)",
            "date": 1704067200,  # UNIX timestamp
        }
    ]
    result = prepare_contexts(contexts, include_citations=True)
    assert len(result) == 1
    assert "Python is a programming language." in result[0]
    assert "[Source: Python (programming language) (Wikipedia, accessed January 01, 2024)]" in result[0]

def test_context_preparer_news_site_citation():
    """
    Test that news site sources generate proper citations with title, date, and URL.

    Input:
        - Context from a news site with full metadata.
    Expectation:
        - Citation includes quoted title, date, and URL.
    """
    contexts = [
        {
            "content": "New AI regulations announced.",
            "final_score": 0.85,
            "source_file": "news_site/tech_news.html",
            "title": "AI Regulation Update",
            "date": "2024-06-15",
            "url": "https://example.com/ai-regulation",
        }
    ]
    result = prepare_contexts(contexts, include_citations=True)
    assert len(result) == 1
    assert "New AI regulations announced." in result[0]
    assert '"AI Regulation Update"' in result[0]
    assert "June 15, 2024" in result[0]
    assert "https://example.com/ai-regulation" in result[0]

def test_context_preparer_books_citation():
    """
    Test that book sources generate proper citations with title and page.

    Input:
        - Context from a book with page metadata.
    Expectation:
        - Citation includes title and page number.
    """
    contexts = [
        {
            "content": "Deep learning revolutionized computer vision.",
            "final_score": 0.9,
            "source_file": "books/deep_learning_textbook.pdf",
            "title": "Deep Learning",
            "page": 42,
        }
    ]
    result = prepare_contexts(contexts, include_citations=True)
    assert len(result) == 1
    assert "Deep learning revolutionized computer vision." in result[0]
    assert "[Source: Deep Learning, p. 42]" in result[0]

def test_context_preparer_no_citations():
    """
    Test that citations can be disabled via include_citations parameter.

    Input:
        - Context with source metadata, but include_citations=False.
    Expectation:
        - Result contains only text, no citation.
    """
    contexts = [
        {
            "content": "Context without citation.",
            "final_score": 0.7,
            "source_file": "data/knowledge_base/sample.pdf",
            "page": 10,
        }
    ]
    result = prepare_contexts(contexts, include_citations=False)
    assert len(result) == 1
    assert result[0] == "Context without citation."
    assert "[Source:" not in result[0]

def test_context_preparer_sorts_by_score():
    """
    Test that contexts are sorted by final_score in descending order.

    Input:
        - Three contexts with different scores.
    Expectation:
        - Results are ordered from highest to lowest score.
    """
    contexts = [
        {"content": "An example of a Low score context.", "final_score": 0.3},
        {"content": "This is a High score context.", "final_score": 0.9},
        {"content": "Medium score context that falls in between.", "final_score": 0.6},
    ]
    result = prepare_contexts(contexts, include_citations=False)
    assert len(result) == 3
    assert result[0] == "This is a High score context."
    assert result[1] == "Medium score context that falls in between."
    assert result[2] == "An example of a Low score context."

def test_context_preparer_return_scores():
    """
    Test that return_scores=True returns full metadata dictionaries.

    Input:
        - Contexts with various metadata.
    Expectation:
        - Result contains dicts with text, final_score, and metadata.
    """
    contexts = [
        {
            "content": "Scored context.",
            "final_score": 0.8,
            "source_file": "test.pdf",
            "page": 5,
        }
    ]
    result = prepare_contexts(contexts, return_scores=True, include_citations=False)
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["text"] == "Scored context."
    assert result[0]["final_score"] == 0.8
    assert result[0]["source_file"] == "test.pdf"
    assert result[0]["page"] == 5

def test_context_preparer_handles_string_contexts():
    """
    Test that plain string contexts (without metadata) are handled correctly.

    Input:
        - Simple string context.
    Expectation:
        - String is cleaned and returned with default score of 0.0.
    """
    contexts = ["Simple string context."]
    result = prepare_contexts(contexts, include_citations=False)
    assert len(result) == 1
    assert result[0] == "Simple string context."

def test_context_preparer_similarity_threshold():
    """
    Test that custom similarity threshold affects deduplication.

    Input:
        - Two somewhat similar contexts.
        - Low similarity threshold (0.5).
    Expectation:
        - Contexts are considered duplicates with low threshold.
    """
    contexts = [
        {"content": "The quick brown fox jumps.", "final_score": 0.7},
        {"content": "The quick brown fox leaps.", "final_score": 0.8},
    ]
    result = prepare_contexts(contexts, similarity_threshold=0.5, include_citations=False)
    # With a lower threshold, these should be considered similar
    assert len(result) == 1

def test_context_preparer_containment_check():
    """
    Test that shorter text contained in longer text is detected as duplicate.

    Input:
        - Short context fully contained in longer context.
    Expectation:
        - Only one context remains (the longer one with higher score).
    """
    contexts = [
        {"content": "Filipino cuisine is diverse and flavorful with influences from Spain.", "final_score": 0.9},
        {"content": "Filipino cuisine is diverse and flavorful with influences from Spain, China, and America.", "final_score": 0.7},
    ]
    result = prepare_contexts(contexts, include_citations=False)
    assert len(result) == 1
    # Higher score wins even though it's shorter
    assert result[0] == "Filipino cuisine is diverse and flavorful with influences from Spain."