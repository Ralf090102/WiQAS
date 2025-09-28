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
    def __init__(self, query: str, context: List[str], query_type: str = "Factual", language: str = "fil"):
        self.query = query
        self.context = context
        self.query_type = query_type
        self.language = language

    def build_system_instructions(self) -> str:
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
        if not self.context:
            return "No relevant documents found."
        return "\n\n".join([f"- {c}" for c in self.context])
    
    def build_query_section(self) -> str:
        return f"User Question:\n{self.query}"
    
    def build_guidelines(self) -> str:
        guideline = FUNCTIONAL_GUIDELINES.get(self.query_type, FUNCTIONAL_GUIDELINES["Factual"])
        return f"Response Guidelines:\n{guideline}\nCite sources. Be culturally sensitive."

    def build_exemplars(self) -> str:
        exemplars_text = []
        for ex in EXEMPLARS:
            exemplars_text.append(
                f"Q: {ex['question']}\nContext: {ex['context']}\nA: {ex['answer']}"
            )
        return "\n\n".join(exemplars_text)
    
    def render(self) -> str:
        return (
            f"System Instructions:\n{self.build_system_instructions()}\n\n"
            f"Context:\n{self.build_context_section()}\n\n"
            f"{self.build_query_section()}\n\n"
            f"{self.build_guidelines()}\n\n"
            f"Few-shot Exemplars:\n{self.build_exemplars()}"
        )
    
class PromptBuilder:
    def __init__(self, detect_language_fn: Optional[Callable] = None):
        self.detect_language_fn = detect_language_fn

    def build_prompt(
        self,
        query: str,
        context: List[str],
        query_type: str = "Factual",
        language: Optional[str] = None,
    ) -> str:
        if self.detect_language_fn and not language:
            language = self.detect_language_fn(query)
        language = language or "fil"

        template = PromptTemplate(query=query, context=context, query_type=query_type, language=language)
        return template.render()
