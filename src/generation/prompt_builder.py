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

from typing import Callable, List, Optional

FUNCTIONAL_GUIDELINES = {
    "Factual": "Provide clear, concise, and accurate definitions, facts, or explanations. Focus on established knowledge and avoid unnecessary speculation.",
    "Analytical": "Offer thoughtful interpretation, highlight symbolism, make comparisons, and explain relationships or deeper meanings. Support reasoning with evidence or logical connections.",
    "Procedural": "Give structured, step-by-step instructions or processes. Ensure clarity, logical order, and completeness so the user can follow easily.",
    "Creative": "Generate original and engaging content such as stories, poems, dialogues, or imaginative scenarios. Emphasize creativity, style, and coherence.",
    "Exploratory": "Deliver broad, contextual, and descriptive overviews. Provide background, key themes, and relevant connections without going too narrow or rigid."
}

EXEMPLARS = [
    {
        "question": "Ano ang paboritong kulay ni Emilio Aguinaldo?",
        "context": "No relevant documents found.",
        "answer": "Walang sapat na impormasyon sa mga dokumentong ito tungkol sa paboritong kulay ni Emilio Aguinaldo. (Source: Biographical Note on Emilio Aguinaldo)"
    }
]

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

    def __init__(self, query: str, context: List[str], query_type: str = "Factual", language: str = "fil"):
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

    def build_system_instructions(self) -> str:
        """
        Construct the system instructions section of the prompt.

        Returns:
            str: Instruction string containing principles of WiQAS such as
            factual accuracy, cultural faithfulness, and citation requirements.
        """
        return (
            "You are WiQAS, a RAG-driven Factoid Question Answering System on Filipino culture. "
            "Your role is to generate answers grounded in the retrieved context from the knowledge base. "
            "Follow these principles:\n\n"
                "1. **Factual Accuracy**: Only use information found in the provided context. If the context "
                "does not contain enough information, clearly state that.\n"
                "2. **Cultural Faithfulness**: Ensure responses respect Filipino linguistic, historical, and "
                "cultural nuances. Preserve cultural authenticity when explaining concepts, practices, or traditions.\n"
                "3. **Clarity & Precision**: Provide concise, factoid-style answers unless the question calls "
                "for elaboration. Avoid unnecessary speculation or overgeneralization.\n"
                "4. **Context-Aware Language**: Pay attention to Filipino semantic and linguistic nuances. "
                "When relevant, explain terms, transliterations, or code-switched phrases.\n"
                "5. **Citations**: Reference sources when appropriate, using clear attribution.\n\n"
                "Remember: WiQAS is not a generic QA system—it is designed specifically to answer questions "
                "about Filipino culture accurately, faithfully, and in context."
        )
    
    def build_context_section(self) -> str:
        """
        Construct the context section of the prompt.

        Returns:
            str: Bullet-pointed list of context snippets or a fallback message.
        """
        if not self.context:
            return "No relevant documents found."
        return "\n\n".join([f"- {c}" for c in self.context])
    
    def build_query_section(self) -> str:
        """
        Construct the query section of the prompt.

        Returns:
            str: Formatted user question.
        """
        return f"User Question:\n{self.query}"
    
    def build_guidelines(self) -> str:
        """
        Construct the response guidelines section of the prompt.

        Returns:
            str: Guidelines text derived from FUNCTIONAL_GUIDELINES and
            contextualized for the query type.
        """
        guideline = FUNCTIONAL_GUIDELINES.get(self.query_type, FUNCTIONAL_GUIDELINES["Factual"])
        return f"Response Guidelines:\n{guideline}\nCite sources. Be culturally sensitive."

    def build_exemplars(self) -> str:
        """
        Construct the few-shot exemplar section of the prompt.

        Returns:
            str: Example question-context-answer triplets formatted for prompting.
        """
        exemplars_text = []
        for ex in EXEMPLARS:
            exemplars_text.append(
                f"Q: {ex['question']}\nContext: {ex['context']}\nA: {ex['answer']}"
            )
        return "\n\n".join(exemplars_text)
    
    def render(self) -> str:
        """
        Render the complete prompt by combining all sections.

        Returns:
            str: Fully constructed prompt string with system instructions,
            context, query, guidelines, and exemplars.
        """
        return (
            f"System Instructions:\n{self.build_system_instructions()}\n\n"
            f"Context:\n{self.build_context_section()}\n\n"
            f"{self.build_query_section()}\n\n"
            f"{self.build_guidelines()}\n\n"
            f"Few-shot Exemplars:\n{self.build_exemplars()}"
        )
    
class PromptBuilder:
    """
    Orchestrates construction of prompts with support for language detection.

    Attributes:
        detect_language_fn (Callable, optional): Function to infer language from query.
    """

    def __init__(self, detect_language_fn: Optional[Callable] = None):
        """
        Initialize a PromptBuilder.

        Args:
            detect_language_fn (Callable, optional): Function that infers query language.
        """
        self.detect_language_fn = detect_language_fn

    def build_prompt(
        self,
        query: str,
        context: List[str],
        query_type: str = "Factual",
        language: Optional[str] = None,
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
        if self.detect_language_fn and not language:
            language = self.detect_language_fn(query)
        language = language or "fil"

        template = PromptTemplate(query=query, context=context, query_type=query_type, language=language)
        return template.render()
