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
    result = prepare_contexts(contexts)
    assert len(result) == 1
    assert result[0] == "Ati-Atihan Festival is celebrated."