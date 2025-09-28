from src.generation.context_preparer import prepare_contexts

def test_context_preparer_removes_exact_duplicates():
    contexts = [
        {"text": "Fiesta is a celebration.", "score": 0.8},
        {"text": "Fiesta is a celebration.", "score": 0.6},
    ]
    result = prepare_contexts(contexts)
    assert len(result) == 1
    assert result[0] == "Fiesta is a celebration."
    

def test_context_preparer_prefers_higher_score():
    contexts = [
        {"text": "Ati-Atihan Festival is celebrated.", "score": 0.6},
        {"text": "Ati-Atihan Festival is celebrated.", "score": 0.9},
    ]
    result = prepare_contexts(contexts, return_scores=True)

    assert len(result) == 1
    kept = result[0]
    assert kept["text"] == "Ati-Atihan Festival is celebrated."
    assert kept["score"] == 0.9

def test_context_preparer_prefers_longer_context():
    contexts = [
        {"text": "Karaoke is a popular pastime.", "score": 0.5},
        {"text": "Karaoke is a popular pastime in the Philippines.", "score": 0.5},
    ]
    result = prepare_contexts(contexts)
    assert len(result) == 1
    assert result[0] == "Karaoke is a popular pastime in the Philippines."

def test_context_preparer_tie_breaker_longer_vs_higher_score():
    contexts = [
        {"text": "Karaoke is a popular pastime.", "score": 0.9},
        {"text": "Karaoke is a popular pastime in the Philippines.", "score": 0.5},
    ]
    result = prepare_contexts(contexts)
    assert len(result) == 1
    assert result[0] == "Karaoke is a popular pastime."

def test_context_preparer_handles_non_similar_contexts():
    contexts = [
        {"text": "Bayanihan is a Filipino tradition.", "score": 0.8},
        {"text": "Lechon is often served at fiestas.", "score": 0.7},
    ]
    result = prepare_contexts(contexts)
    assert len(result) == 2
    assert "Bayanihan is a Filipino tradition." in result
    assert "Lechon is often served at fiestas." in result

def test_context_preparer_cleans_whitespace():
    contexts = [
        {"text": "   Filipino    culture \n is diverse. ", "score": 0.8},
    ]
    result = prepare_contexts(contexts)
    assert len(result) == 1
    assert result[0] == "Filipino culture is diverse."

def test_context_preparer_removes_empty_after_cleaning():
    contexts = [
        {"text": "    \n   ", "score": 0.5},
    ]
    result = prepare_contexts(contexts)
    assert len(result) == 0

def test_context_preparer_mixed_duplicates_and_uniques():
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
    contexts = [
        {
            "text": (
                "Ang sinigang ay isang. Mga uri ng sinigang "
                "Sinigang na Sinigang na Sinigang na "
                "Ayon sa pampaasim na sangkap "
                "Sinigang sa Sinigang sa Sinigang sa "
                "Mga sanggunian"
            ),
            "score": 0.8,
        }
    ]

    result = prepare_contexts(contexts)
    assert len(result) == 1, f"Expected 1 context, got {len(result)}"
    cleaned = result[0]

    print("\n--- CLEANED CONTEXT ---\n", cleaned, "\n-----------------------")

    assert "Sinigang na Sinigang na" not in cleaned, "bigram repetition still present"
    assert "Sinigang sa Sinigang sa" not in cleaned, "bigram repetition still present"

    assert "Sinigang na" in cleaned, "collapsed too aggressively, lost 'Sinigang na'"
    assert "Sinigang sa" in cleaned, "collapsed too aggressively, lost 'Sinigang sa'"

    assert "Ang sinigang ay isang" in cleaned, "intro missing after cleaning"
    assert "Mga uri ng sinigang" in cleaned, "section header missing"
    assert "Mga sanggunian" in cleaned, "ending missing"