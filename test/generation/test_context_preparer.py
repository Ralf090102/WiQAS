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