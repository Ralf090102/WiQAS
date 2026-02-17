import re
from dataclasses import dataclass


@dataclass
class QueryClassification:
    query_type: str
    language: str
    confidence: float = 0.0


class QueryClassifier:
    QUERY_TYPE_KEYWORDS = {
        "Factual": [
            # Filipino
            r"\b(ano|sino|saan|kailan|ilan|alin)\b",
            r"\b(tawag|ibig\s+sabihin|kahulugan|depinisyon)\b",
            # English
            r"\b(what|who|where|when|which|define|meaning)\b",
            r"\b(is|are|was|were)\b.*\b(definition|called)\b",
        ],
        "Analytical": [
            # Filipino
            r"\b(bakit|paano|ano\s+ang\s+kahalagahan|ano\s+ang\s+papel)\b",
            r"\b(impluwensya|epekto|dahilan|resulta)\b",
            r"\b(simbolismo|sinisimbolo|kahulugan|representasyon)\b",
            # English
            r"\b(why|how\s+did|significance|importance|role|impact)\b",
            r"\b(analyze|explain|symbolism|represent|meaning|influence)\b",
            r"\b(cultural\s+significance|historical\s+context)\b",
        ],
        "Procedural": [
            # Filipino
            r"\b(paano\s+(gumawa|magluto|gawin|mag))\b",
            r"\b(hakbang|proseso|paraan|instruksyon)\b",
            r"\b(mga\s+hakbang|sundin|gawin)\b",
            # English
            r"\b(how\s+to|steps|process|procedure|instructions)\b",
            r"\b(make|cook|create|prepare|perform)\b",
            r"\b(guide|tutorial|method)\b",
        ],
        "Comparative": [
            # Filipino
            r"\b(pagkakaiba|pagkakatulad|ihambing)\b",
            r"\b(mas|kaysa|kumpara)\b",
            r"\b(katulad|kaiba)\b",
            # English
            r"\b(difference|similar|compare|contrast|versus|vs)\b",
            r"\b(alike|unlike|comparison|distinguish)\b",
            r"\b(better|worse|more|less)\s+than\b",
        ],
        "Exploratory": [
            # Filipino
            r"\b(paki(paliwanag|bigay\s+ng\s+overview))\b",
            r"\b(konteksto|background|kasaysayan)\b",
            r"\b(ano\s+ang\s+tungkol)\b",
            # English
            r"\b(overview|describe|tell\s+me\s+about|background)\b",
            r"\b(discuss|elaborate|explore|context)\b",
            r"\b(general|broad|comprehensive)\b",
        ],
    }

    FILIPINO_PATTERNS = [
        r"\b(ano|anong|sino|sinong|saan|saang|kailan|ilan|ilang|alin|aling)\b",
        r"\b(tawag|ibig\s+sabihin|kahulugan|depinisyon)\b",
        r"\b(bakit|paano|ano\s+ang\s+kahalagahan|ano\s+ang\s+papel)\b",
        r"\b(impluwensya|epekto|dahilan|resulta)\b",
        r"\b(simbolismo|kahulugan|representasyon)\b",
        r"\b(paano\s+(gumawa|magluto|gawin|mag))\b",
        r"\b(hakbang|proseso|paraan|instruksyon)\b",
        r"\b(mga\s+hakbang|sundin|gawin)\b",
        r"\b(pagkakaiba|pagkakatulad|ihambing)\b",
        r"\b(mas|kaysa|kumpara)\b",
        r"\b(katulad|kaiba)\b",
        r"\b(paki(paliwanag|bigay\s+ng\s+overview))\b",
        r"\b(konteksto|background|kasaysayan)\b",
        r"\b(ano\s+ang\s+tungkol)\b",
    ]

    ENGLISH_PATTERNS = [
        r"\b(what|who|where|when|which|define|meaning|is|the|they|this|these|those)\b",
        r"\b(is|are|was|were)\b.*\b(definition|called)\b",
        r"\b(why|how\s+did|significance|importance|role|impact)\b",
        r"\b(analyze|explain|symbolism|represent|meaning|influence)\b",
        r"\b(cultural\s+significance|historical\s+context)\b",
        r"\b(how\s+to|steps|process|procedure|instructions)\b",
        r"\b(make|cook|create|prepare|perform)\b",
        r"\b(guide|tutorial|method)\b",
        r"\b(difference|similar|compare|contrast|versus|vs)\b",
        r"\b(alike|unlike|comparison|distinguish)\b",
        r"\b(better|worse|more|less)\s+than\b",
        r"\b(overview|describe|tell\s+me\s+about|background)\b",
        r"\b(discuss|elaborate|explore|context)\b",
        r"\b(general|broad|comprehensive)\b",
    ]

    def __init__(self):
        self.type_patterns = {qtype: [re.compile(pattern, re.IGNORECASE) for pattern in patterns] for qtype, patterns in self.QUERY_TYPE_KEYWORDS.items()}
        self.fil_patterns = [re.compile(p, re.IGNORECASE) for p in self.FILIPINO_PATTERNS]
        self.en_patterns = [re.compile(p, re.IGNORECASE) for p in self.ENGLISH_PATTERNS]

    def classify_query_type(self, query: str) -> tuple[str, float]:
        scores = {qtype: 0 for qtype in self.type_patterns.keys()}

        for qtype, patterns in self.type_patterns.items():
            for pattern in patterns:
                matches = len(pattern.findall(query))
                scores[qtype] += matches

        query_lower = query.lower()

        analytical_triggers = ["why", "bakit", "how did", "significance", "importance", "role", "impact", "influence"]
        if any(t in query_lower for t in analytical_triggers):
            scores["Analytical"] += 3

        procedural_triggers = ["how to", "paano", "steps", "hakbang", "process", "procedure"]
        if any(t in query_lower for t in procedural_triggers):
            scores["Procedural"] += 3

        comparative_triggers = ["difference", "pagkakaiba", "compare", "contrast", "vs", "versus"]
        if any(t in query_lower for t in comparative_triggers):
            scores["Comparative"] += 3

        if max(scores.values()) == 0:
            return "Factual", 0.5

        priority_order = ["Analytical", "Procedural", "Comparative", "Exploratory", "Factual"]
        best_type = sorted(scores.items(), key=lambda kv: (-kv[1], priority_order.index(kv[0])))[0][0]

        total_matches = sum(scores.values())
        confidence = scores[best_type] / total_matches if total_matches > 0 else 0.5

        return best_type, min(confidence, 1.0)

    def detect_language(self, query: str) -> tuple[str, float]:
        fil_score = sum(1 for pattern in self.fil_patterns if pattern.search(query))
        en_score = sum(1 for pattern in self.en_patterns if pattern.search(query))

        total = fil_score + en_score

        if total == 0:
            return "fil", 0.5

        if fil_score > en_score:
            return "fil", fil_score / total
        else:
            return "en", en_score / total

    def classify(self, query: str) -> QueryClassification:
        query_type, type_confidence = self.classify_query_type(query)
        language, lang_confidence = self.detect_language(query)

        overall_confidence = (type_confidence + lang_confidence) / 2

        return QueryClassification(query_type=query_type, language=language, confidence=overall_confidence)

    # === Language Detection Tests ===

    def test_detect_language_mixed_code_switching(self, classifier):
        """Test language detection with code-switching."""
        query = "Ano ba ang meaning ng jeepney?"
        result = classifier.classify(query)

        # Should detect Filipino as primary language
        assert result.language == "fil"

    def test_detect_language_pure_filipino(self, classifier):
        """Test language detection with pure Filipino."""
        query = "Isang tradisyonal na pagkain ba ang Sinigang?"
        result = classifier.classify(query)

        assert result.language == "fil"

    def test_detect_language_pure_english(self, classifier):
        """Test language detection with pure English."""
        query = "Why is the jeepney an iconic vehicle in the Philippines?"
        result = classifier.classify(query)

        assert result.language == "en"

    def test_detect_language_no_markers(self, classifier):
        """Test language detection with minimal language markers."""
        query = "Festival celebration."
        language, confidence = classifier.detect_language(query)

        # Should default to Filipino with low confidence
        assert language in ["fil", "en"]
        assert 0.0 <= confidence <= 1.0

    # === Edge Cases and Special Scenarios ===

    def test_classify_ambiguous_defaults_to_factual(self, classifier):
        """Test that ambiguous queries default to Factual."""
        query = "Fiesta."
        result = classifier.classify(query)

        assert result.query_type == "Factual"
        assert result.confidence >= 0.0

    def test_classify_empty_query(self, classifier):
        """Test classification with empty query."""
        query = ""
        result = classifier.classify(query)

        assert result.query_type == "Factual"
        assert result.language in ["fil", "en"]

    def test_classify_very_short_query(self, classifier):
        """Test classification with very short query."""
        query = "Adobo?"
        result = classifier.classify(query)

        assert result.query_type == "Factual"

    def test_classify_query_type_confidence(self, classifier):
        """Test that confidence scores are calculated correctly."""
        query = "Bakit bakit bakit mahalaga ang kultura?"
        query_type, confidence = classifier.classify_query_type(query)

        assert query_type == "Analytical"
        assert 0.0 <= confidence <= 1.0

    def test_classify_multiple_type_indicators(self, classifier):
        """Test query with indicators for multiple types."""
        query = "Why and how to make adobo?"
        result = classifier.classify(query)

        assert result.query_type in ["Procedural", "Analytical"]
