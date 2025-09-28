from typing import List

FUNCTIONAL_GUIDELINES = {
    "Factual": "Provide clear, concise, and accurate definitions, facts, or explanations. Focus on established knowledge and avoid unnecessary speculation.",
    "Analytical": "Offer thoughtful interpretation, highlight symbolism, make comparisons, and explain relationships or deeper meanings. Support reasoning with evidence or logical connections.",
    "Procedural": "Give structured, step-by-step instructions or processes. Ensure clarity, logical order, and completeness so the user can follow easily.",
    "Creative": "Generate original and engaging content such as stories, poems, dialogues, or imaginative scenarios. Emphasize creativity, style, and coherence.",
    "Exploratory": "Deliver broad, contextual, and descriptive overviews. Provide background, key themes, and relevant connections without going too narrow or rigid."
}

class PromptTemplate:
    def __init__(self, query: str, context: List[str], query_type: str = "Factual", language: str = "fil"):
        self.query = query
        self.context = context
        self.query_type = query_type
        self.language = language