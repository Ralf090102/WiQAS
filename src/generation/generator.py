import time
from typing import Any

from src.generation.context_preparer import ContextPreparer
from src.generation.prompt_builder import PromptBuilder
from src.generation.query_classifier import QueryClassifier
from src.retrieval.retriever import WiQASRetriever
from src.utilities.config import AnswerGeneratorConfig, TimingBreakdown, WiQASConfig


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

    def __init__(
        self,
        config: WiQASConfig,
        answer_config: AnswerGeneratorConfig | None = None,
        use_query_classifier: bool = True,
    ):
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
        self.prompt_builder = PromptBuilder(use_classifier=use_query_classifier)
        self.query_classifier = QueryClassifier() if use_query_classifier else None

    def _call_model(self, prompt: str) -> str:
        if self.answer_config.backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = self.answer_config.model
            if model_id == "gemma2:9b":
                model_id = "aisingapore/Gemma-SEA-LION-v3-9B"

            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
            elif torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float16
            else:
                device = "cpu"
                dtype = torch.float32

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
            )
            hf_model.to(device)

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=self.answer_config.max_tokens,
                temperature=self.answer_config.temperature,
                do_sample=True,
            )

            print("Generated Response:", tokenizer.decode(outputs[0], skip_special_tokens=True).strip())

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if decoded.startswith(prompt):
                decoded = decoded[len(prompt) :].strip()
            return decoded

        else:
            from src.core.llm import generate_response

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
        query_type: str | None = None,
        language: str | None = None,
        show_contexts: bool = False,
        include_timing: bool = False,
        include_classification: bool = False,
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
            include_timing (bool, optional): Whether to include timing breakdown in results (default: False).

        Returns:
            dict[str, Any]: Structured output containing:
                - "query" (str): Original user query.
                - "answer" (str): Model-generated answer.
                - "contexts" (list[str]): Deduplicated contexts (only if show_contexts=True).
                - "timing" (TimingBreakdown): Component timing breakdown (only if include_timing=True).
        """
        # Initialize timing if requested
        timing = TimingBreakdown() if include_timing else None

        classification_info = None
        if self.query_classifier:
            if include_timing:
                classification_start = time.time()

            classification = self.query_classifier.classify(query)

            if include_timing:
                timing.classification_time = time.time() - classification_start

            # Use classification results if not explicitly provided
            if query_type is None:
                query_type = classification.query_type
            if language is None:
                language = classification.language

            # Store classification info if requested
            if include_classification:
                classification_info = {
                    "detected_type": classification.query_type,
                    "detected_language": classification.language,
                    "confidence": classification.confidence,
                    "used_type": query_type,
                    "used_language": language,
                }
        else:
            # Use defaults if classifier not available
            query_type = query_type or "Factual"
            language = language or "fil"

        # retrieve with timing
        self.retriever._initialize_components()
        if include_timing:
            # Get retrieval timing by calling with timing enabled
            retrieval_result = self.retriever.query(query, k=k, enable_mmr=True, formatted=False, include_timing=True)

            if isinstance(retrieval_result, dict) and "timing" in retrieval_result:
                # Extract retrieval timing
                retrieval_timing = retrieval_result["timing"]
                timing.embedding_time = retrieval_timing.embedding_time
                timing.search_time = retrieval_timing.search_time
                timing.reranking_time = retrieval_timing.reranking_time
                timing.mmr_time = retrieval_timing.mmr_time
                timing.translation_time = retrieval_timing.translation_time
                timing.language_detection_time = retrieval_timing.language_detection_time
                raw_results = retrieval_result["results"]
            else:
                raw_results = retrieval_result
        else:
            raw_results = self.retriever.query(query, k=k, enable_mmr=True, formatted=False)

        def get_meta(r, key):
            return r.metadata.get(key) if hasattr(r, "metadata") and isinstance(r.metadata, dict) else None

        contexts = [
            {
                "content": getattr(r, "content", None),
                "final_score": get_meta(r, "final_score"),
                "source_file": get_meta(r, "source_file"),
                "page": get_meta(r, "page"),
                "title": get_meta(r, "title"),
                "date": get_meta(r, "date"),
                "url": get_meta(r, "url"),
            }
            for r in raw_results
        ]

        # prepare contexts with timing
        if include_timing:
            context_start = time.time()
        prepared_contexts = self.context_preparer.prepare(contexts, include_citations=True, return_full=True)
        if include_timing:
            timing.context_preparation_time = time.time() - context_start

        # build prompt with timing and language awareness
        if include_timing:
            prompt_start = time.time()
        prompt = self.prompt_builder.build_prompt(query, prepared_contexts, query_type=query_type, language=language)
        if include_timing:
            timing.prompt_building_time = time.time() - prompt_start

        # generate answer with timing
        if include_timing:
            llm_start = time.time()
        raw_answer = self._call_model(prompt)

        # Remove prompt from answer if it was echoed by the model
        answer = raw_answer
        if raw_answer.startswith(prompt):
            answer = raw_answer[len(prompt) :].lstrip()

        if include_timing:
            timing.llm_generation_time = time.time() - llm_start
            # Calculate total time
            timing.total_time = (
                timing.embedding_time
                + timing.search_time
                + timing.reranking_time
                + timing.mmr_time
                + timing.context_preparation_time
                + timing.prompt_building_time
                + timing.llm_generation_time
                + timing.translation_time
                + timing.language_detection_time
                + getattr(timing, "classification_time", 0.0)
            )

        result = {
            "query": query,
            "answer": answer,
            "query_type": query_type,
            "language": language,
        }

        if show_contexts:
            result["contexts"] = prepared_contexts

        if include_classification and classification_info:
            result["classification"] = classification_info

        if include_timing:
            result["timing"] = timing

        return result
