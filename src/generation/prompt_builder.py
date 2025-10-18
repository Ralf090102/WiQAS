"""
Prompt Builder Module

This module defines classes and constants for constructing prompts used in WiQAS,
a RAG-driven Factoid Question Answering System specialized in Filipino culture.

Components:
    - FUNCTIONAL_GUIDELINES: Guidelines for shaping responses by type (Factual, Analytical, etc.).
    - EXEMPLARS: Few-shot examples demonstrating the expected QA style.
    - PromptTemplate: Encapsulates the logic for assembling system instructions,
      query, context, guidelines, and exemplars into a complete prompt.
    - PromptBuilder: Orchestrator that applies language detection (optional) and
      renders the final prompt via PromptTemplate.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

FUNCTIONAL_GUIDELINES = {
    "Factual": "Provide clear, concise, and accurate definitions, facts, or explanations. Focus on established knowledge and avoid unnecessary speculation. Use specific dates, names, and verifiable details when available.",
    "Analytical": "Offer thoughtful interpretation, highlight symbolism, make comparisons, and explain relationships or deeper meanings. Support reasoning with evidence or logical connections. Discuss cultural significance and historical context.",
    "Procedural": "Give structured, step-by-step instructions or processes. Ensure clarity, logical order, and completeness so the user can follow easily. Include materials needed and expected outcomes when relevant.",
    "Creative": "Generate original and engaging content such as stories, poems, dialogues, or imaginative scenarios. Emphasize creativity, style, and coherence while maintaining cultural authenticity.",
    "Exploratory": "Deliver broad, contextual, and descriptive overviews. Provide background, key themes, and relevant connections without going too narrow or rigid. Explore multiple perspectives when appropriate.",
    "Comparative": "Compare and contrast different concepts, practices, traditions, or historical elements. Highlight similarities, differences, and cultural significance. Provide balanced analysis of each element being compared.",
}

EXEMPLARS = [
    {
        "question": "Ano ang paboritong kulay ni Emilio Aguinaldo?",
        "context": "No relevant documents found.",
        "answer": "Walang sapat na impormasyon sa mga dokumentong ito tungkol sa paboritong kulay ni Emilio Aguinaldo. (Source: Biographical Note on Emilio Aguinaldo)",
    }
]


@dataclass
class QueryClassification:
    """
    Represents the classification result of a user query.

    Attributes:
        query_type (str): The functional type of the query (e.g., "Factual", "Analytical").
        language (str): Detected language code ("fil" for Filipino, "en" for English).
        confidence (float): Confidence score of the classification (0.0 to 1.0).
    """

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
            r"\b(simbolismo|kahulugan|representasyon)\b",
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
        r"\b(ang|ng|sa|mga|ay|na|at|para|kung|mga)\b",
        r"\b(ano|sino|saan|kailan|paano|bakit)\b",
        r"\b(ito|iyan|ako|ikaw|siya|kami|tayo|sila)\b",
    ]

    ENGLISH_PATTERNS = [
        r"\b(the|is|are|was|were|has|have|had)\b",
        r"\b(what|who|where|when|how|why)\b",
        r"\b(this|that|these|those|I|you|he|she|we|they)\b",
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


class PromptTemplate:
    """
    Defines the hierarchical structure for prompt construction.

    Template Sections:
        - System Instructions: Core principles guiding the model's behavior.
        - Context: Retrieved snippets formatted as bullet points.
        - Query: The user's question.
        - Guidelines: Response style instructions based on query type.
        - Exemplars: Few-shot examples illustrating the desired QA style.
    """

    def __init__(
        self,
        query: str,
        context: list[str] | list[dict[str, Any]],
        query_type: str = "Factual",
        language: str = "fil",
        include_exemplars: bool = True,
        use_detailed_context: bool = True,
    ):
        """
        Initialize a PromptTemplate instance.

        Args:
            query (str): User question.
            context (List[str]): Retrieved context snippets.
            query_type (str, optional): Response style guideline (default: "Factual").
            language (str, optional): Language code (default: "fil").
        """
        self.query = query
        self.context = context
        self.query_type = query_type
        self.language = language
        self.include_exemplars = include_exemplars
        self.use_detailed_context = use_detailed_context

    def _format_source_citation(self, ctx: dict[str, Any]) -> str:
        citation_text = ctx.get("citation_text")
        if citation_text:
            return f"[Source: {citation_text}]"

        source_file = ctx.get("normalized_source_file") or ctx.get("source_file", "")
        if source_file:
            return f"[Source: {source_file}]"

        return "[Source: Unknown]"

    def build_system_instructions(self) -> str:
        """
        Construct the system instructions section of the prompt.

        Returns:
            str: Instruction string containing principles of WiQAS such as
            factual accuracy, cultural faithfulness, and citation requirements.
        """

        citation_examples = (
            "Citation Format Examples:\n"
            "- For PDFs: (Source: Food Of The Philippines, p. 23)\n"
            "- For Wikipedia: (Source: Article Title (Wikipedia, accessed January 15, 2024))\n"
            '- For News: (Source: "Article Title", January 15, 2024. Retrieved from URL)\n'
            "- For Books: (Source: Book Title, p. 45)\n"
            "- Multiple sources: (Sources: Source1; Source2; Source3)\n\n"
        )

        return (
            "You are WiQAS, a RAG-driven Factoid Question Answering System specialized in Filipino culture. "
            "Your role is to generate answers grounded in the retrieved context from the knowledge base. "
            "Follow these principles:\n\n"
            "1. **Factual Accuracy**: Only use information found in the provided context. If the context "
            "does not contain enough information, clearly state: 'Walang sapat na impormasyon sa mga "
            "dokumentong ito tungkol sa [topic]' (Filipino) or 'There is insufficient information in "
            "these documents about [topic]' (English).\n\n"
            "2. **Cultural Faithfulness**: Ensure responses respect Filipino linguistic, historical, and "
            "cultural nuances. Preserve cultural authenticity when explaining concepts, practices, or traditions. "
            "Recognize the diversity within Filipino culture across regions and time periods.\n\n"
            "3. **Clarity & Precision**: Provide concise, factoid-style answers unless the question calls "
            "for elaboration. Avoid unnecessary speculation or overgeneralization. Use specific details like "
            "dates, names, and locations when available in the context.\n\n"
            "4. **Context-Aware Language**: Pay attention to Filipino semantic and linguistic nuances, including "
            "natural code-switching between Filipino and English as commonly practiced in Philippine discourse. "
            "When relevant, explain terms, transliterations, or culturally-specific phrases. Match the primary "
            "language of your response to the user's query language, but use the most appropriate language for "
            "specific terms and concepts. Provide translations or explanations when using terms that might not "
            "be familiar to the target audience.\n\n"
            f"5. **Detailed Citations**: Always reference sources at the end of your answer using the exact citation "
            "format provided in the context snippets. Each context snippet includes a [Source: ...] citation - "
            "use these citations directly in your answer. When multiple sources support your answer, list all of them. "
            "Citations must be accurate, detailed, and traceable. The generated answer should always end with the citation.\n\n"
            f"{citation_examples}"
            "6. **Handling Uncertainty**: If the context provides partial information, answer what you can and "
            "explicitly state what information is missing. Never fabricate details not present in the context.\n\n"
            "Remember: WiQAS is not a generic QA system—it is designed specifically to answer questions "
            "about Filipino culture accurately, faithfully, and in context. Your responses should demonstrate "
            "cultural competence and linguistic awareness appropriate for Filipino cultural topics."
        )

    def build_context_section(self) -> str:
        """
        Construct the context section of the prompt.

        Returns:
            str: Bullet-pointed list of context snippets or a fallback message.
        """
        if not self.context:
            return "No relevant documents found."

        formatted_contexts = []

        for i, ctx in enumerate(self.context, 1):
            if isinstance(ctx, dict):
                text = ctx.get("text", "")

                if self.use_detailed_context:
                    citation = self._format_source_citation(ctx)
                    formatted_contexts.append(f"[{i}] {text} {citation}")
                else:
                    formatted_contexts.append(f"[{i}] {text}")
            else:
                formatted_contexts.append(f"[{i}] {ctx}")

        return "\n\n".join(formatted_contexts)

    def build_query_section(self) -> str:
        """
        Construct the query section of the prompt.

        Returns:
            str: Formatted user question.
        """
        return f"User Question:\n{self.query}\n\n" f"[Detected Language: {self.language.upper()} | Query Type: {self.query_type}]"

    def build_guidelines(self) -> str:
        """
        Construct the response guidelines section of the prompt.

        Returns:
            str: Guidelines text derived from FUNCTIONAL_GUIDELINES and
            contextualized for the query type.
        """
        guideline = FUNCTIONAL_GUIDELINES.get(self.query_type)

        # Enhanced multilingual instructions
        if self.language == "fil":
            language_instruction = (
                "Respond primarily in Filipino (Tagalog), but use code-switching with English when:\n"
                "  • Technical terms are more commonly known in English\n"
                "  • Proper nouns or specific terminology from the source context is in English\n"
                "  • The context contains mixed language content that's better preserved as-is"
            )
        else:
            language_instruction = (
                "Respond in English, but include Filipino terms when:\n"
                "  • The Filipino term is culturally significant or has no direct English equivalent\n"
                "  • Proper nouns or cultural concepts are better understood in Filipino\n"
                "  • Direct quotes from Filipino sources should be preserved"
            )

        return (
            f"Response Guidelines ({self.query_type} Type):\n"
            f"{guideline}\n\n"
            f"Additional Requirements:\n"
            f"- {language_instruction}\n"
            f"- Use the EXACT citation format provided in each context snippet's [Source: ...] tag\n"
            f"- Maintain cultural sensitivity and authenticity\n"
            f"- If information is insufficient, state this clearly rather than speculating\n"
            f"- When relevant, provide cultural context and explanations for Filipino-specific terms\n"
        )

    def build_exemplars(self) -> str:
        """
        Construct the few-shot exemplar section of the prompt.

        Returns:
            str: Example question-context-answer triplets formatted for prompting.
        """
        if not self.include_exemplars:
            return ""

        relevant_exemplars = [ex for ex in EXEMPLARS if ex.get("language", "fil") == self.language or ex.get("query_type", "") == self.query_type]

        if not relevant_exemplars:
            relevant_exemplars = EXEMPLARS

        exemplars_text = []
        for ex in relevant_exemplars[:3]:
            exemplars_text.append(f"Example Question: {ex['question']}\n" f"Context Provided:\n{ex['context']}\n" f"Expected Answer: {ex['answer']}")
        return "\n\n".join(exemplars_text)

    def render(self) -> str:
        """
        Render the complete prompt by combining all sections.

        Returns:
            str: Fully constructed prompt string with system instructions,
            context, query, guidelines, and exemplars.
        """
        sections = [
            f"=== SYSTEM INSTRUCTIONS ===\n{self.build_system_instructions()}",
            f"=== RETRIEVED CONTEXT ===\n{self.build_context_section()}",
            f"=== USER QUERY ===\n{self.build_query_section()}",
            f"=== RESPONSE GUIDELINES ===\n{self.build_guidelines()}",
        ]

        if self.include_exemplars:
            sections.append(f"=== FEW-SHOT EXAMPLES ===\n{self.build_exemplars()}")

        sections.append("=== YOUR ANSWER ===")

        return "\n\n" + ("-" * 80 + "\n\n").join(sections)


class PromptBuilder:
    """
    Orchestrates construction of prompts with support for language detection.

    Attributes:
        detect_language_fn (Callable, optional): Function to infer language from query.
    """

    def __init__(self, detect_language_fn: Callable | None = None, use_classifier: bool = True, use_detailed_context: bool = True):
        """
        Initialize a PromptBuilder.

        Args:
            detect_language_fn (Callable, optional): Function that infers query language.
        """
        self.detect_language_fn = detect_language_fn
        self.use_classifier = use_classifier
        self.use_detailed_context = use_detailed_context
        self.classifier = QueryClassifier() if use_classifier else None

    def build_prompt(
        self,
        query: str,
        context: list[str] | list[dict[str, Any]],
        query_type: str | None = None,
        language: str | None = None,
        include_exemplars: bool = True,
    ) -> str:
        """
        Build the final prompt using PromptTemplate.

        Args:
            query (str): User question.
            context (List[str]): Retrieved context snippets.
            query_type (str, optional): Desired response style (default: "Factual").
            language (str, optional): Target response language (default: inferred or "fil").

        Returns:
            str: Fully constructed prompt string.
        """
        if self.use_classifier and self.classifier:
            classification = self.classifier.classify(query)

            if query_type is None:
                query_type = classification.query_type
            if language is None:
                language = classification.language

        if self.detect_language_fn and language is None:
            language = self.detect_language_fn(query)

        query_type = query_type or "Factual"
        language = language or "fil"

        template = PromptTemplate(
            query=query,
            context=context,
            query_type=query_type,
            language=language,
            include_exemplars=include_exemplars,
            use_detailed_context=self.use_detailed_context,
        )
        return template.render()
