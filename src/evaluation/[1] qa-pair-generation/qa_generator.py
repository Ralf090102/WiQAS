import json
import re
from pathlib import Path
from typing import Any

try:
    import requests
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install langchain langchain-community pypdf docx2txt requests tqdm")
    exit(1)


class OllamaQAGenerator:
    """Automatic pipeline for generating QA pairs from books using Ollama"""

    def __init__(
        self,
        model_name: str = "llama3.1",
        ollama_url: str = "http://127.0.0.1:11434",
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
        questions_per_chunk: int = 2,
        question_level: str = "high",
        bilingual: bool = True,
    ):
        """
        Initialize the QA generation pipeline with Ollama

        Args:
            model_name: Ollama model name (e.g., 'llama3.1', 'gemma2', 'mistral')
            ollama_url: URL for Ollama API
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks
            questions_per_chunk: Number of QA pairs to generate per chunk
            question_level: "high" for overview questions, "detailed" for specific questions
            bilingual: If True, alternate between Filipino and English questions
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.questions_per_chunk = questions_per_chunk
        self.question_level = question_level
        self.bilingual = bilingual

        # Test Ollama connection
        self._test_ollama_connection()

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""])

    def _test_ollama_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            print("Testing Ollama connection...")
            response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5, proxies={"http": None, "https": None})
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [m["name"] for m in models]

                if not any(self.model_name in m for m in available_models):
                    print(f"Warning: Model '{self.model_name}' not found in Ollama")
                    print(f"Available models: {', '.join(available_models)}")
                    print(f"\nTo pull the model, run: ollama pull {self.model_name}")
                else:
                    print(f"✓ Connected to Ollama - Using model: {self.model_name}")
            else:
                print("Warning: Could not connect to Ollama API")
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")

    def call_ollama(self, prompt: str, temperature: float = 0.7) -> str:
        """Call Ollama API with a prompt"""
        try:
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False, "temperature": temperature},
                timeout=120,
                proxies={"http": None, "https": None},
            )

            if response.status_code == 200:
                return response.json()["response"]
            else:
                print(f"Error: Ollama API returned status {response.status_code}")
                return ""
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""

    def load_document(self, file_path: str) -> str:
        """Load document from various formats"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"Loading document: {file_path}")

        # Select appropriate loader based on file extension
        if file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            text = "\n\n".join([doc.page_content for doc in documents])
        elif file_path.suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents = loader.load()
            text = documents[0].page_content
        elif file_path.suffix in [".docx", ".doc"]:
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()
            text = documents[0].page_content
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        print(f"Document loaded: {len(text)} characters")
        return text

    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks for processing"""
        print("Chunking text...")
        chunks = self.text_splitter.split_text(text)
        print(f"Created {len(chunks)} chunks")
        return chunks

    def generate_qa_from_chunk(self, chunk: str, chunk_index: int) -> list[dict[str, Any]]:
        """Generate QA pairs from a single chunk using Ollama"""

        # Adjust prompt based on question level
        if self.question_level == "high":
            question_guidance = """- Focus on HIGH-LEVEL, OVERVIEW questions (not overly specific details)
   - Ask about main concepts, key themes, important events, or significant figures
   - Avoid questions about minor details or specific numbers unless culturally significant
   - Questions should capture the essence of the passage, not trivia"""
        else:
            question_guidance = """- Questions can be detailed and specific
   - Focus on factoid questions (what, who, when, where)
   - Include both major and minor details"""

        # Add bilingual instruction if enabled
        if self.bilingual:
            # Alternate based on chunk index (even = Filipino, odd = English)
            if chunk_index % 2 == 0:
                language_instruction = """
LANGUAGE REQUIREMENT:
- Generate questions in FILIPINO (Tagalog)
- Answers should also be in Filipino
- Use natural, conversational Filipino"""
            else:
                language_instruction = """
LANGUAGE REQUIREMENT:
- Generate questions in ENGLISH
- Answers should also be in English
- Use clear, natural English"""
        else:
            language_instruction = ""

        prompt = f"""You are an expert in Filipino culture and language. Given the following text passage, generate {self.questions_per_chunk} high-quality question-answer pairs that focus on Filipino cultural context.

TEXT PASSAGE (for your reference only):
{chunk}
{language_instruction}

INSTRUCTIONS:
1. Generate exactly {self.questions_per_chunk} question-answer pairs this will be quiz questions that test knowledge about the text passage
2. Questions should be:
   - Factoid (what, who, when, where)
{question_guidance}
3. Do NOT use pronouns or vague references such as: "this", "it", "they", "the passage", "the text", "the list"
4. Must EXPLICITLY name: the subject (person, place, event, concept, dish, tradition, or object)
5. Every question and answer must be FULLY SELF-CONTAINED
6. A reader must understand BOTH the question and the answer WITHOUT seeing the text passage
7. Do NOT use pronouns without an explicit noun reference
8. Must NOT mention the existence of a text, recipe, page, passage, or list instead of a real-world entity
9. For each question, provide:
   - A factoid "answer" based on the text
   - A "cultural_golden_answer" that emphasizes Filipino cultural relevance
10. The cultural golden answer should:
   - Start with the same factoid answer if culturally relevant
   - Add specific Filipino cultural context, significance, or connections when applicable
   - If the fact has no special cultural significance, the cultural_golden_answer can be THE SAME as the answer
   - Examples:
     * Q: "Who was the first president?"
       Answer: "Emilio Aguinaldo"
       Cultural Golden: "Emilio Aguinaldo, who led the Philippine Revolution and established the First Philippine Republic in 1898"
     * Q: "What year did the event occur?"
       Answer: "1986"
       Cultural Golden: "1986" (same, unless the year has cultural significance like EDSA Revolution)
11. Focus on making the cultural connection explicit rather than treating facts as generic information
12. All answers must be directly supported by the text

OUTPUT FORMAT (strict JSON only, no other text):
{{
  "qa_pairs": [
    {{
      "question": "Question text here",
      "answer": "Factoid answer",
      "cultural_golden_answer": "Same factoid OR factoid with Filipino cultural context/significance",
      "language": "{'filipino' if self.bilingual and chunk_index % 2 == 0 else 'english'}",
      "cultural_relevance": 0.9
    }}
  ]
}}



Generate the QA pairs in JSON format now:"""

        try:
            response = self.call_ollama(prompt, temperature=0.7)

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                qa_data = json.loads(json_str)
                qa_pairs = qa_data.get("qa_pairs", [])

                # Add metadata
                for qa in qa_pairs:
                    qa["chunk_index"] = chunk_index
                    qa["context"] = chunk[:500]  # Store first 500 chars of context

                    if "cultural_golden_answer" not in qa:
                        qa["cultural_golden_answer"] = qa.get("answer", "")

                    if self.bilingual and "language" not in qa:
                        qa["language"] = "filipino" if chunk_index % 2 == 0 else "english"

                return qa_pairs
            else:
                print(f"Warning: Could not parse JSON from chunk {chunk_index}")
                return []

        except json.JSONDecodeError as e:
            print(f"JSON parsing error for chunk {chunk_index}: {e}")
            return []
        except Exception as e:
            print(f"Error generating QA for chunk {chunk_index}: {e}")
            return []

    def process_document(self, file_path: str, output_path: str = None) -> list[dict[str, Any]]:
        """
        Process a document and generate QA pairs

        Args:
            file_path: Path to the input document
            output_path: Path to save the output JSON (optional)

        Returns:
            List of QA pairs
        """

        text = self.load_document(file_path)

        # Chunk text
        chunks = self.chunk_text(text)

        # Generate QA pairs for each chunk
        all_qa_pairs = []
        print("\nGenerating QA pairs...")

        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            qa_pairs = self.generate_qa_from_chunk(chunk, i)

            for qa_pair in qa_pairs:
                print(qa_pair)

            all_qa_pairs.extend(qa_pairs)

        print(f"\nGenerated {len(all_qa_pairs)} valid QA pairs")

        # Prepare output
        output_data = {
            "metadata": {
                "source_file": str(file_path),
                "total_chunks": len(chunks),
                "total_qa_pairs": len(all_qa_pairs),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "question_level": self.question_level,
                "bilingual": self.bilingual,
                "model": self.model_name,
            },
            "qa_pairs": all_qa_pairs,
        }

        # Save to file if path provided
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Saved QA dataset to: {output_path}")

        return all_qa_pairs

    def batch_process(self, input_dir: str, output_dir: str):
        """Process multiple books in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Find all supported files
        supported_formats = [".pdf", ".txt", ".docx", ".doc"]
        files = [f for f in input_path.iterdir() if f.suffix in supported_formats]

        print(f"Found {len(files)} documents to process")

        all_datasets = []
        for file in files:
            print(f"\n{'='*60}")
            print(f"Processing: {file.name}")
            print("=" * 60)

            output_file = output_path / f"{file.stem}_qa_dataset.json"
            qa_pairs = self.process_document(str(file), str(output_file))
            all_datasets.append({"source": file.name, "qa_count": len(qa_pairs)})

        # Create summary
        summary = {
            "total_documents": len(files),
            "total_qa_pairs": sum(d["qa_count"] for d in all_datasets),
            "documents": all_datasets,
        }

        summary_file = output_path / "batch_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"Total documents processed: {summary['total_documents']}")
        print(f"Total QA pairs generated: {summary['total_qa_pairs']}")
        print(f"Summary saved to: {summary_file}")


def main():
    """Example usage with Ollama"""

    print("=" * 60)
    print("QA Generation Pipeline - Ollama Version")
    print("=" * 60)

    # high level for books
    # detailed for modules
    pipeline = OllamaQAGenerator(
        model_name="llama3.1",
        ollama_url="http://localhost:11434",
        chunk_size=4000,  # 1500, 2500, 4000
        chunk_overlap=400,  #  15-20%
        questions_per_chunk=1,
        question_level="high",  # "high" for overview, "detailed" for specific
        bilingual=True,  # Alternate between Filipino and English
    )

    # Example: Single document processing
    # pipeline.process_document(
    #     file_path="Filipino-Politics.pdf",
    #     output_path="qa_dataset.json"
    # )

    # Example: Batch processing
    pipeline.batch_process(input_dir="to_ingest/", output_dir="qa_datasets/")

    print("\n✓ Pipeline initialized successfully!")


if __name__ == "__main__":
    main()
