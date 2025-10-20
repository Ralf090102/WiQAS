import pytest
from src.generation.query_classifier import QueryClassifier, QueryClassification


class TestQueryClassifier:
    """Test suite for QueryClassifier."""

    @pytest.fixture
    def classifier(self):
        """Provide a QueryClassifier instance."""
        return QueryClassifier()

    # === Factual Query Tests ===
    
    def test_classify_factual_filipino(self, classifier):
        """Test factual query classification in Filipino."""
        query = "Ano ang Sinigang?"
        result = classifier.classify(query)
        
        assert result.query_type == "Factual"
        assert result.language == "fil"
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_factual_english(self, classifier):
        """Test factual query classification in English."""
        query = "What is Sinigang?"
        result = classifier.classify(query)
        
        assert result.query_type == "Factual"
        assert result.language == "en"

    def test_classify_factual_who_question(self, classifier):
        """Test factual query with 'who' question."""
        query = "Who is Jose Rizal?"
        result = classifier.classify(query)
        
        assert result.query_type == "Factual"
        assert result.language == "en"

    def test_classify_factual_when_question(self, classifier):
        """Test factual query with 'when' question."""
        query = "Kailan naganap ang EDSA Revolution?"
        result = classifier.classify(query)
        
        assert result.query_type == "Factual"
        assert result.language == "fil"

    def test_classify_factual_where_question(self, classifier):
        """Test factual query with 'where' question."""
        query = "Where is Intramuros located?"
        result = classifier.classify(query)
        
        assert result.query_type == "Factual"