from typing import Any

from src.core.llm import generate_response
from src.retrieval.retriever import WiQASRetriever
from src.generation.context_preparer import ContextPreparer
from src.generation.prompt_builder import PromptBuilder
from src.utilities.config import AnswerGeneratorConfig, WiQASConfig

class WiQASGenerator:
    def __init__(self, config: WiQASConfig, answer_config: AnswerGeneratorConfig | None = None):
        self.config = config or WiQASConfig()
        self.answer_config = answer_config or self.config.rag.answer
        self.retriever = WiQASRetriever(self.config)

        self.context_preparer = ContextPreparer()
        self.prompt_builder = PromptBuilder()

    def _call_model(self, prompt: str) -> str:
        # print(prompt)  # debugging hook

        return generate_response(
            prompt=prompt,
            config=self.config,
            model=self.answer_config.model,
            temperature=self.answer_config.temperature,
            max_tokens=self.answer_config.max_tokens,
        )
    
    def generate(
        self,
        query: str,
        k: int = 5,
        query_type: str = "Factual",
        show_contexts: bool = False,
    ) -> dict[str, Any]:
        
        self.retriever._initialize_components()
        raw_results = self.retriever._perform_search(query, k=k, search_type="hybrid")
        contexts = [r.content for r in raw_results]
        prepared_contexts = self.context_preparer.prepare(contexts)
        prompt = self.prompt_builder.build_prompt(query, prepared_contexts, query_type=query_type)
        answer = self._call_model(prompt)

        return {
            "query": query,
            "answer": answer,
            "contexts": prepared_contexts if show_contexts else [],
        }
