from typing import Any

from src.core.llm import generate_response
from src.retrieval.retriever import WiQASRetriever
from src.generation.context_preparer import ContextPreparer
from src.generation.prompt_builder import PromptBuilder
from src.utilities.config import AnswerGeneratorConfig, WiQASConfig

class WiQASGenerator:
    """
    Main orchestrator for WiQAS retrieval-augmented answer generation.

    Attributes:
        config (WiQASConfig): Global configuration for WiQAS.
        answer_config (AnswerGeneratorConfig): Model-specific configuration.
        retriever (WiQASRetriever): Handles hybrid document retrieval.
        context_preparer (ContextPreparer): Cleans and deduplicates retrieved contexts.
        prompt_builder (PromptBuilder): Constructs the final structured prompt.
    """

    def __init__(self, config: WiQASConfig, answer_config: AnswerGeneratorConfig | None = None):
        """
        Initialize WiQASGenerator with system and answer configs.

        Args:
            config (WiQASConfig): Global WiQAS configuration.
            answer_config (AnswerGeneratorConfig | None, optional):
                Configuration for answer generation. If None, defaults
                to config.rag.answer.
        """
        self.config = config or WiQASConfig()
        self.answer_config = answer_config or self.config.rag.generator
        self.retriever = WiQASRetriever(self.config)

        self.context_preparer = ContextPreparer()
        self.prompt_builder = PromptBuilder()

    def _call_model(self, prompt: str) -> str:
        """
        Call the LLM with the constructed prompt.

        Args:
            prompt (str): Fully rendered LLM prompt.

        Returns:
            str: Model-generated answer text.
        """
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
        """
        Run the full WiQAS RAG pipeline and return a structured result.

        Steps:
            1. Retrieve contexts relevant to the query.
            2. Clean and deduplicate contexts.
            3. Build a structured prompt with instructions and context.
            4. Run the LLM to generate a grounded answer.

        Args:
            query (str): User question.
            k (int, optional): Number of retrieval results to fetch (default: 5).
            query_type (str, optional): Response style guideline (default: "Factual").
            show_contexts (bool, optional): Whether to return contexts in the output (default: False).

        Returns:
            dict[str, Any]: Structured output containing:
                - "query" (str): Original user query.
                - "answer" (str): Model-generated answer.
                - "contexts" (list[str]): Deduplicated contexts (only if show_contexts=True).
        """
        # retrieve
        self.retriever._initialize_components()
        raw_results = self.retriever.query(query, k=k, enable_mmr=False, formatted=False)
        contexts = [
            {
            "text": r.content,
            "score": getattr(r, "score", 0.0),
            "document_id": getattr(r, "document_id", None),
            "source": r.metadata.get("source") if hasattr(r, "metadata") and isinstance(r.metadata, dict) else None,
            }
            for r in raw_results
        ]

        # prepare contexts
        prepared_contexts = self.context_preparer.prepare(contexts, return_full=True)
      
        # build prompt
        prompt = self.prompt_builder.build_prompt(query, prepared_contexts, query_type=query_type)
        
        # generate answer
        answer = self._call_model(prompt)

        return {
            "query": query,
            "answer": answer,
            "contexts": prepared_contexts if show_contexts else [],
        }
