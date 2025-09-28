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