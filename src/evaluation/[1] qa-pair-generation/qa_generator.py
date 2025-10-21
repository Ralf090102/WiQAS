import json
import os
from typing import List, Dict, Any
from pathlib import Path
import re
import hashlib
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    from langchain.schema import Document
    import requests
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install langchain langchain-community pypdf docx2txt requests tqdm")
    exit(1)


# ==================== Configuration Classes ====================

@dataclass
class BaseConfig:
    """Base configuration class"""

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def validate(self) -> None:
        """Validate configuration values"""
        pass


@dataclass
class PreprocessingConfig(BaseConfig):
    """Text preprocessing configuration"""

    similarity_threshold: float = 0.85
    min_text_length: int = 50
    enable_deduplication: bool = True
    enable_normalization: bool = True

    def validate(self) -> None:
        """Validate preprocessing configuration"""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.min_text_length <= 0:
            raise ValueError("min_text_length must be positive")


# ==================== Text Preprocessor ====================

class TextPreprocessor:
    """Handles text normalization and deduplication"""

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()
        self.config.validate()
        self.similarity_threshold = self.config.similarity_threshold
        self.min_text_length = self.config.min_text_length

    def normalize_text(self, text: str) -> str:
        """
        Normalize text by cleaning and standardizing format.
        Includes intelligent word boundary detection for PDF text extraction issues.
        """
        if not text or not text.strip():
            return ""

        # Fix missing spaces
        text = self._fix_missing_spaces(text)

        # Remove excessive whitespace and normalize line endings
        text = re.sub(r"\s+", " ", text.strip())

        # Remove control characters but keep most printable characters
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # Remove leading numbers (verse/section numbers) from the beginning of text
        text = re.sub(r"^\d+\s+", "", text)

        # Normalize common text patterns
        text = re.sub(r"\.{2,}", "...", text)
        text = re.sub(r"\?{2,}", "?", text)
        text = re.sub(r"\!{2,}", "!", text)

        # Remove redundant spaces around punctuation
        text = re.sub(r"\s+([\.,:;!?])", r"\1", text)
        text = re.sub(r"([\.,:;!?])\s+", r"\1 ", text)

        # Normalize quotes
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r"[''']", "'", text)

        return text.strip()

    def _fix_missing_spaces(self, text: str) -> str:
        """Fix missing spaces in PDF-extracted text using pattern recognition."""
        # Basic camelCase detection (lowercase followed by uppercase)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

        # Numbers and letters
        text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
        text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)

        # Punctuation and letters
        text = re.sub(r"([a-zA-Z])([\.,:;!?])", r"\1 \2", text)
        text = re.sub(r"([\.,:;!?])([a-zA-Z])", r"\1 \2", text)

        # Prepositions
        text = re.sub(r"\bof([a-z]{3,})", r"of \1", text)
        text = re.sub(r"\bin([a-z]{3,})", r"in \1", text)
        text = re.sub(r"\bon([a-z]{3,})", r"on \1", text)
        text = re.sub(r"\bto([a-z]{3,})", r"to \1", text)
        text = re.sub(r"\bby([a-z]{3,})", r"by \1", text)
        text = re.sub(r"\bat([a-z]{3,})", r"at \1", text)
        text = re.sub(r"\bfor([a-z]{3,})", r"for \1", text)
        text = re.sub(r"\bwith([a-z]{3,})", r"with \1", text)
        text = re.sub(r"\bfrom([a-z]{3,})", r"from \1", text)
        text = re.sub(r"\binto([a-z]{3,})", r"into \1", text)

        # Articles/Conjunctions
        text = re.sub(r"\bthe([a-z]{3,})", r"the \1", text)
        text = re.sub(r"\band([a-z]{3,})", r"and \1", text)
        text = re.sub(r"\bbut([a-z]{3,})", r"but \1", text)
        text = re.sub(r"\bor([a-z]{3,})", r"or \1", text)

        # Word endings
        text = re.sub(r"\b(tion)([a-z]{2,})", r"\1 \2", text)
        text = re.sub(r"\b(ness)([a-z]{2,})", r"\1 \2", text)
        text = re.sub(r"\b(ment)([a-z]{2,})", r"\1 \2", text)
        text = re.sub(r"\b(able)([a-z]{2,})", r"\1 \2", text)
        text = re.sub(r"\b(ible)([a-z]{2,})", r"\1 \2", text)

        # Clean up any double spaces created
        text = re.sub(r"\s+", " ", text)

        return text

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using sequence matching."""
        if not text1 or not text2:
            return 0.0

        # Normalize texts for comparison
        norm_text1 = self.normalize_text(text1).lower()
        norm_text2 = self.normalize_text(text2).lower()

        if not norm_text1 or not norm_text2:
            return 0.0

        # Use sequence matcher for similarity
        matcher = SequenceMatcher(None, norm_text1, norm_text2)
        return matcher.ratio()

    def is_similar_content(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar enough to be considered duplicates."""
        similarity = self.calculate_similarity(text1, text2)
        return similarity >= self.similarity_threshold

    def is_valid_text(self, text: str) -> bool:
        """Check if text meets minimum quality requirements."""
        if not text or not text.strip():
            return False

        # Check minimum length
        if len(text.strip()) < self.min_text_length:
            return False

        meaningful_chars = re.sub(r"[^\w\s]", "", text)
        if len(meaningful_chars.strip()) < self.min_text_length * 0.7:
            return False

        return True

    def deduplicate_chunks(self, chunks: List[str]) -> List[str]:
        """
        Remove duplicate and similar chunks from a list.
        
        Args:
            chunks: List of text chunks to deduplicate
        
        Returns:
            List of deduplicated chunks
        """
        if not chunks:
            return chunks

        unique_chunks = []
        seen_hashes = set()
        duplicate_count = 0
        similar_count = 0

        print(f"Starting deduplication of {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            # Normalize the text content
            if self.config.enable_normalization:
                normalized_content = self.normalize_text(chunk)
            else:
                normalized_content = chunk

            if not self.is_valid_text(normalized_content):
                continue

            # Check for exact duplicates using hash
            content_hash = hashlib.md5(normalized_content.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                duplicate_count += 1
                continue

            is_similar = False
            if self.config.enable_deduplication:
                # Check against last 10 chunks for performance
                for existing_chunk in unique_chunks[-10:]:
                    if self.is_similar_content(normalized_content, existing_chunk):
                        is_similar = True
                        similar_count += 1
                        break

            if not is_similar:
                unique_chunks.append(normalized_content)
                seen_hashes.add(content_hash)

        removed_count = len(chunks) - len(unique_chunks)
        print(f"Deduplication complete: {len(unique_chunks)}/{len(chunks)} chunks kept")
        print(f"  - Exact duplicates removed: {duplicate_count}")
        print(f"  - Similar chunks removed: {similar_count}")
        print(f"  - Invalid chunks removed: {removed_count - duplicate_count - similar_count}")

        return unique_chunks


# ==================== QA Generator with Preprocessing ====================

class OllamaQAGenerator:
    """Automatic pipeline for generating QA pairs from books using Ollama with text preprocessing"""
    
    def __init__(self, 
                 model_name: str = "llama3.1:latest",
                 ollama_url: str = "http://127.0.0.1:11434",
                 chunk_size: int = 1500, 
                 chunk_overlap: int = 300, 
                 questions_per_chunk: int = 2,
                 question_level: str = "high",  
                 bilingual: bool = True,
                 preprocessing_config: PreprocessingConfig | None = None):
        """
        Initialize the QA generation pipeline with Ollama and text preprocessing
        
        Args:
            model_name: Ollama model name
            ollama_url: URL for Ollama API
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks
            questions_per_chunk: Number of QA pairs to generate per chunk
            question_level: "high" for overview questions, "detailed" for specific questions
            bilingual: If True, alternate between Filipino and English questions
            preprocessing_config: Configuration for text preprocessing
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.questions_per_chunk = questions_per_chunk
        self.question_level = question_level
        self.bilingual = bilingual
        
        # Initialize text preprocessor
        self.preprocessor = TextPreprocessor(preprocessing_config)
        print(f"✓ Text preprocessor initialized with:")
        print(f"  - Normalization: {self.preprocessor.config.enable_normalization}")
        print(f"  - Deduplication: {self.preprocessor.config.enable_deduplication}")
        print(f"  - Similarity threshold: {self.preprocessor.config.similarity_threshold}")
        print(f"  - Min text length: {self.preprocessor.config.min_text_length}")
        
        # Test Ollama connection
        self._test_ollama_connection()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def _test_ollama_connection(self):
        """Test if Ollama is running and model is available"""
        try:    
            print('Testing Ollama connection...')
            response = requests.get(
                "http://127.0.0.1:11434/api/tags",
                timeout=5,
                proxies={"http": None, "https": None}
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                
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
                f"http://127.0.0.1:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature
                },
                timeout=120,
                proxies={"http": None, "https": None}
            )
            
            if response.status_code == 200:
                return response.json()['response']
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
        if file_path.suffix == '.pdf':
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            text = "\n\n".join([doc.page_content for doc in documents])
        elif file_path.suffix == '.txt':
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            text = documents[0].page_content
        elif file_path.suffix in ['.docx', '.doc']:
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()
            text = documents[0].page_content
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        print(f"Document loaded: {len(text)} characters")
        
        # Apply text normalization if enabled
        if self.preprocessor.config.enable_normalization:
            print("Normalizing document text...")
            text = self.preprocessor.normalize_text(text)
            print(f"Normalized text: {len(text)} characters")
        
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks and apply preprocessing"""
        print("\nChunking text...")
        chunks = self.text_splitter.split_text(text)
        print(f"Created {len(chunks)} initial chunks")
        
        # Apply deduplication and validation
        if self.preprocessor.config.enable_deduplication or self.preprocessor.config.enable_normalization:
            chunks = self.preprocessor.deduplicate_chunks(chunks)
        
        return chunks
    
    def generate_qa_from_chunk(self, chunk: str, chunk_index: int) -> List[Dict[str, Any]]:
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

TEXT PASSAGE:
{chunk}
{language_instruction}

INSTRUCTIONS:
1. Generate exactly {self.questions_per_chunk} question-answer pairs
2. Questions should be:
   - Factoid (what, who, when, where)
{question_guidance}
3. For each question, provide:
   - A factoid "answer" based on the text
   - A "cultural_golden_answer" that emphasizes Filipino cultural relevance
4. The cultural golden answer should:
   - Start with the same factoid answer if culturally relevant
   - Add specific Filipino cultural context, significance, or connections when applicable
   - If the fact has no special cultural significance, the cultural_golden_answer can be THE SAME as the answer
5. Focus on making the cultural connection explicit rather than treating facts as generic information
6. ALL answers MUST be DIRECTLY SUPPORTED by the text passage - NO EXTERNAL KNOWLEDGE

OUTPUT FORMAT (strict JSON only, no other text):
{{
  "qa_pairs": [
    {{
      "question": "Question text here",
      "answer": "Factoid answer DIRECTLY from text",
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
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                qa_data = json.loads(json_str)
                qa_pairs = qa_data.get('qa_pairs', [])
                
                # Add metadata
                for qa in qa_pairs:
                    qa['chunk_index'] = chunk_index
                    qa['context'] = chunk

                    if 'cultural_golden_answer' not in qa:
                        qa['cultural_golden_answer'] = qa.get('answer', '')

                    if self.bilingual and 'language' not in qa:
                        qa['language'] = 'filipino' if chunk_index % 2 == 0 else 'english'
                
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
    
    def process_document(self, file_path: str, output_path: str = None) -> List[Dict[str, Any]]:
        """
        Process a document and generate QA pairs with preprocessing
        
        Args:
            file_path: Path to the input document
            output_path: Path to save the output JSON (optional)
        
        Returns:
            List of QA pairs
        """
        text = self.load_document(file_path)
        
        # Chunk text with preprocessing
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
                "preprocessing": {
                    "normalization_enabled": self.preprocessor.config.enable_normalization,
                    "deduplication_enabled": self.preprocessor.config.enable_deduplication,
                    "similarity_threshold": self.preprocessor.config.similarity_threshold,
                    "min_text_length": self.preprocessor.config.min_text_length
                }
            },
            "qa_pairs": all_qa_pairs
        }
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Saved QA dataset to: {output_path}")
        
        return all_qa_pairs
    
    def batch_process(self, input_dir: str, output_dir: str):
        """Process multiple books in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all supported files
        supported_formats = ['.pdf', '.txt', '.docx', '.doc']
        files = [f for f in input_path.iterdir() if f.suffix in supported_formats]
        
        print(f"Found {len(files)} documents to process")
        
        all_datasets = []
        for file in files:
            print(f"\n{'='*60}")
            print(f"Processing: {file.name}")
            print('='*60)
            
            output_file = output_path / f"{file.stem}_qa_dataset.json"
            qa_pairs = self.process_document(str(file), str(output_file))
            all_datasets.append({
                "source": file.name,
                "qa_count": len(qa_pairs)
            })
        
        # Create summary
        summary = {
            "total_documents": len(files),
            "total_qa_pairs": sum(d['qa_count'] for d in all_datasets),
            "documents": all_datasets,
            "preprocessing_config": self.preprocessor.config.model_dump()
        }
        
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"Total documents processed: {summary['total_documents']}")
        print(f"Total QA pairs generated: {summary['total_qa_pairs']}")
        print(f"Summary saved to: {summary_file}")


def main():
    """Example usage with Ollama and text preprocessing"""
    
    print("="*60)
    print("QA Generation Pipeline - Ollama Version with Text Preprocessing")
    print("="*60)
    
    # Configure preprocessing
    preprocessing_config = PreprocessingConfig(
        similarity_threshold=0.85,
        min_text_length=50,
        enable_deduplication=True,
        enable_normalization=True
    )
    
    # Initialize pipeline with preprocessing
    pipeline = OllamaQAGenerator(   
        model_name="llama3.1:latest",
        ollama_url="http://localhost:11434",
        chunk_size=4000,
        chunk_overlap=800,
        questions_per_chunk=2, 
        question_level="high",
        bilingual=True,
        preprocessing_config=preprocessing_config
    )
    
    # Example: Single document processing
    # pipeline.process_document(
    #     file_path="Filipino-Politics.pdf",
    #     output_path="qa_dataset.json"
    # )
    
    # Example: Batch processing
    pipeline.batch_process(
        input_dir="to_ingest/",
        output_dir="qa_datasets/"
    )
    
    print("\n✓ Pipeline initialized successfully!")

if __name__ == "__main__":
    main()