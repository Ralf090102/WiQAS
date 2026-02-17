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

        # === Analytical Query Tests ===

    def test_classify_analytical_filipino(self, classifier):
        """Test analytical query classification in Filipino."""
        query = "Bakit mahalaga ang Ati-Atihan Festival sa kultura?"
        result = classifier.classify(query)

        assert result.query_type == "Analytical"
        assert result.language == "fil"

    def test_classify_analytical_english(self, classifier):
        """Test analytical query classification in English."""
        query = "Why are beliefs highly regarded in Philippine culture?"
        result = classifier.classify(query)

        assert result.query_type == "Analytical"
        assert result.language == "en"

    def test_classify_analytical_how_did(self, classifier):
        """Test analytical query with 'how did' pattern."""
        query = "How did Spanish colonization influence Filipino cuisine?"
        result = classifier.classify(query)

        assert result.query_type == "Analytical"

    def test_classify_analytical_importance(self, classifier):
        """Test analytical query asking about importance."""
        query = "What is the importance of family in Filipino culture?"
        result = classifier.classify(query)

        assert result.query_type == "Analytical"

    def test_classify_analytical_symbolism(self, classifier):
        """Test analytical query about symbolism."""
        query = "Ano ang sinisimbolo ng Philippine flag?"
        result = classifier.classify(query)

        assert result.query_type == "Analytical"

    # === Procedural Query Tests ===

    def test_classify_procedural_filipino(self, classifier):
        """Test procedural query classification in Filipino."""
        query = "Paano gumawa ng adobo?"
        result = classifier.classify(query)

        assert result.query_type == "Procedural"
        assert result.language == "fil"

    def test_classify_procedural_english(self, classifier):
        """Test procedural query classification in English."""
        query = "How to make lumpia?"
        result = classifier.classify(query)

        assert result.query_type == "Procedural"
        assert result.language == "en"

    def test_classify_procedural_steps(self, classifier):
        """Test procedural query asking for steps."""
        query = "What are the steps to prepare sinigang?"
        result = classifier.classify(query)

        assert result.query_type == "Procedural"

    def test_classify_procedural_process(self, classifier):
        """Test procedural query asking about process."""
        query = "Ano ang proseso ng paggawa ng bibingka?"
        result = classifier.classify(query)

        assert result.query_type == "Procedural"

    def test_classify_procedural_instructions(self, classifier):
        """Test procedural query asking for instructions."""
        query = "Give me instructions on how to cook lechon kawali."
        result = classifier.classify(query)

        assert result.query_type == "Procedural"

    # === Procedural Query Tests ===

    def test_classify_procedural_filipino(self, classifier):
        """Test procedural query classification in Filipino."""
        query = "Paano gumawa ng adobo?"
        result = classifier.classify(query)

        assert result.query_type == "Procedural"
        assert result.language == "fil"

    def test_classify_procedural_english(self, classifier):
        """Test procedural query classification in English."""
        query = "How to make lumpia?"
        result = classifier.classify(query)

        assert result.query_type == "Procedural"
        assert result.language == "en"

    def test_classify_procedural_steps(self, classifier):
        """Test procedural query asking for steps."""
        query = "What are the steps to prepare sinigang?"
        result = classifier.classify(query)

        assert result.query_type == "Procedural"

    def test_classify_procedural_process(self, classifier):
        """Test procedural query asking about process."""
        query = "Ano ang proseso ng paggawa ng bibingka?"
        result = classifier.classify(query)

        assert result.query_type == "Procedural"

    def test_classify_procedural_instructions(self, classifier):
        """Test procedural query asking for instructions."""
        query = "Give me instructions on how to cook lechon kawali."
        result = classifier.classify(query)

        assert result.query_type == "Procedural"

    # === Comparative Query Tests ===

    def test_classify_comparative_filipino(self, classifier):
        """Test comparative query classification in Filipino."""
        query = "Ano ang pagkakaiba ng Sinulog at Ati-Atihan?"
        result = classifier.classify(query)

        assert result.query_type == "Comparative"
        assert result.language == "fil"

    def test_classify_comparative_english(self, classifier):
        """Test comparative query classification in English."""
        query = "What is the difference between adobo and sinigang?"
        result = classifier.classify(query)

        assert result.query_type == "Comparative"
        assert result.language == "en"

    def test_classify_comparative_vs(self, classifier):
        """Test comparative query with 'vs' or 'versus'."""
        query = "Lumpia vs spring rolls: what's the difference?"
        result = classifier.classify(query)

        assert result.query_type == "Comparative"

    def test_classify_comparative_similar(self, classifier):
        """Test comparative query asking about similarity."""
        query = "How similar are Filipino and Spanish cultures?"
        result = classifier.classify(query)

        assert result.query_type == "Comparative"

    def test_classify_comparative_contrast(self, classifier):
        """Test comparative query with 'contrast' keyword."""
        query = "Compare and contrast Tagalog and Bisaya languages."
        result = classifier.classify(query)

        assert result.query_type == "Comparative"

    # === Exploratory Query Tests ===

    def test_classify_exploratory_filipino(self, classifier):
        """Test exploratory query classification in Filipino."""
        query = "Pakipaliwanag ang konteksto ng kasaysayan ng Pilipinas."
        result = classifier.classify(query)

        assert result.query_type == "Exploratory"
        assert result.language == "fil"

    def test_classify_exploratory_english(self, classifier):
        """Test exploratory query classification in English."""
        query = "Tell me about the background of Filipino festivals."
        result = classifier.classify(query)

        assert result.query_type == "Exploratory"
        assert result.language == "en"

    def test_classify_exploratory_overview(self, classifier):
        """Test exploratory query asking for overview."""
        query = "Give me an overview of Philippine history."
        result = classifier.classify(query)

        assert result.query_type == "Exploratory"

    def test_classify_exploratory_describe(self, classifier):
        """Test exploratory query asking to describe."""
        query = "Describe Filipino cuisine."
        result = classifier.classify(query)

        assert result.query_type == "Exploratory"

    def test_classify_exploratory_context(self, classifier):
        """Test exploratory query asking for context."""
        query = "What is the context of the Philippine Revolution?"
        result = classifier.classify(query)

        assert result.query_type == "Exploratory"
