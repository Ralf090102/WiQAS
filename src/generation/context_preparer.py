from typing import Any, Dict, List, Union
from difflib import SequenceMatcher

class ContextPreparer:
    """
    Utility for preparing retrieved contexts for answer generation.
    
    Main Responsibilities:
        - Normalize whitespace.
        - Collapse repeated phrases (1–4 words repeated adjacently).
        - Deduplicate contexts by semantic similarity.
        - Resolve duplicates by preferring higher score or longer text.
        - Preserve Metadata 
    """

    def __init__(self, similarity_threshold: float = 0.7) -> None:
        """
        Initialize the context preparer.

        Args:
            similarity_threshold: 
                Minimum similarity ratio (0–1) for two texts to be
                considered duplicates. Defaults to 0.7.
        """
        self.similarity_threshold = similarity_threshold

    def _normalize_source_file(self, source_file: str) -> str:
        if not source_file:
            return ""
        
        patterns = [
            r'^[Dd]ata[/\\][Kk]nowledge_?[Bb]ase[/\\]',
            r'^[Dd]ata[/\\]',
            r'^knowledge_?base[/\\]',
        ]

        normalized = source_file
        for pattern in patterns:
            normalized = re.sub(pattern, '', normalized)
        
        return normalized

    def _extract_pdf_title(self, source_file: str) -> str:
        if not source_file or not source_file.endswith('.pdf'):
            return None
        
        normalized = self._normalize_source_file(source_file)
        
        filename = Path(normalized).stem
        
        title = filename.replace('-', ' ').replace('_', ' ')
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title

    def _categorize_source(self, source_file: str) -> str:
        if not source_file:
            return "unknown"
        
        source_lower = source_file.lower()
        
        if 'wikipedia' in source_lower:
            return "wikipedia"
        elif 'news_site' in source_lower or 'news-site' in source_lower:
            return "news_site"
        elif 'books' in source_lower or 'book' in source_lower:
            return "books"
        elif source_file.endswith('.pdf'):
            return "pdf"
        else:
            return "unknown"

    def _format_date(self, date_value: Any) -> Optional[str]:
        if not date_value:
            return None
        
        try:
            if isinstance(date_value, (int, float)):
                date_obj = datetime.fromtimestamp(int(date_value))
                return date_obj.strftime("%B %d, %Y")
            
            if isinstance(date_value, str) and date_value.isdigit():
                date_obj = datetime.fromtimestamp(int(date_value))
                return date_obj.strftime("%B %d, %Y")
            
            if isinstance(date_value, str):
                date_obj = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                return date_obj.strftime("%B %d, %Y")
                
        except (ValueError, TypeError, OSError) as e:
            logger.debug(f"Could not parse date {date_value}: {e}")
            
        return None

    def _format_citation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Format citation information based on source type."""
        source_file = context.get("source_file", "")
        normalized_source = self._normalize_source_file(source_file)
        source_type = self._categorize_source(normalized_source)
        
        citation = {
            "source_type": source_type,
            "citation_text": None,
            "normalized_source_file": normalized_source
        }
        
        if source_type == "pdf":
            title = self._extract_pdf_title(source_file)
            page = context.get("page")
            
            if title:
                citation["title"] = title
                if page is not None:
                    citation["citation_text"] = f"{title}, p. {page}"
                else:
                    citation["citation_text"] = title
                
        elif source_type == "wikipedia":
            title = context.get("title", "")
            date_value = context.get("date")
            
            if title:
                citation["title"] = title
                date_str = self._format_date(date_value)
                
                if date_str:
                    citation["date"] = date_str
                    citation["citation_text"] = f"{title} (Wikipedia, accessed {date_str})"
                else:
                    citation["citation_text"] = f"{title} (Wikipedia)"
                    
        elif source_type == "news_site":
            title = context.get("title", "")
            url = context.get("url", "")
            date_value = context.get("date")
            
            parts = []
            if title:
                citation["title"] = title
                parts.append(f'"{title}"')
            
            date_str = self._format_date(date_value)
            if date_str:
                citation["date"] = date_str
                parts.append(date_str)
            
            if url:
                citation["url"] = url
            
            if parts:
                citation_base = ", ".join(parts)
                if url:
                    citation["citation_text"] = f"{citation_base}. Retrieved from {url}"
                else:
                    citation["citation_text"] = citation_base
                
        elif source_type == "books":
            title = context.get("title", "")
            page = context.get("page")
            
            if title:
                citation["title"] = title
                if page is not None:
                    citation["citation_text"] = f"{title}, p. {page}"
                else:
                    citation["citation_text"] = title
        
        return citation

    def _remove_repetitive_phrases(self, text: str) -> str:
        tokens = text.split()
        out = []
        i = 0
        
        while i < len(tokens):
            collapsed = False
            for plen in range(4, 0, -1):
                if i + plen * 2 > len(tokens):
                    continue
                    
                phrase = tokens[i:i + plen]
                j = i + plen
                repeats = 1
                while j + plen <= len(tokens) and tokens[j:j + plen] == phrase:
                    repeats += 1
                    j += plen
                    
                if repeats > 1:
                    out.extend(phrase)  
                    i = j
                    collapsed = True
                    break
                    
            if not collapsed:
                out.append(tokens[i])
                i += 1

        return " ".join(out)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        return text.strip()

    def _clean_context(self, context: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(context, str):
            text = context
            final_score = 0.0
            metadata = {}
        else:
            text = context.get("content", context.get("text", ""))
            final_score = context.get("final_score", 0.0)
            
            metadata = {
                "source_file": context.get("source_file"),
                "page": context.get("page"),
                "title": context.get("title"),
                "date": context.get("date"),
                "url": context.get("url"),
            }

        text = self._normalize_whitespace(text)
        text = self._remove_repetitive_phrases(text)

        citation = self._format_citation({**metadata, "content": text})

        return {
            "text": text,
            "final_score": final_score,
            "length": len(text),
            **metadata,
            **citation
        }

    def _are_similar(self, a: str, b: str) -> bool:
        """
        Decide if two texts are similar enough to be considered duplicates.

        Uses two checks:
        - SequenceMatcher ratio above `similarity_threshold`.
        - Containment: shorter string fully contained in longer
          and at least 10 characters long.

        Args:
            a: First text.
            b: Second text.

        Returns:
            bool: True if texts are considered duplicates.
        """
        len_ratio = min(len(a), len(b)) / max(len(a), len(b))
        if len_ratio < 0.5:  
            return False
        
        ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
        if ratio >= self.similarity_threshold:
            return True
        
        shorter, longer = (a, b) if len(a) < len(b) else (b, a)
        if len(shorter) > 30 and shorter.lower() in longer.lower():
            return True
            
        return False
    
    def _deduplicate(self, contexts: list[dict]) -> list[dict]:
        """
        Deduplicate a list of cleaned contexts.

        Rules:
        - If two contexts are similar:
            * Keep the one with the higher score.
            * If scores are equal, keep the longer text.

        Args:
            contexts: List of dicts with "text" and "score".

        Returns:
            List of unique dicts.
        """
        unique = []
        for ctx in contexts:
            text, score = ctx["text"], ctx.get("score", 0.0)
            duplicate_found = False
            for kept in unique:
                if self._are_similar(text, kept["text"]):
                    duplicate_found = True
                    # keep the better one
                    if (score > kept.get("score", 0.0)) or (
                        score == kept.get("score", 0.0)
                        and len(text) > len(kept["text"])
                    ):
                        kept.update(ctx)
                    break
            if not duplicate_found:
                unique.append(ctx)
        return unique
    
    def prepare(self, contexts: List[Dict[str, Any]], return_full: bool = False) -> List[Union[str, Dict[str, Any]]]:
        """
        Clean and deduplicate contexts.

        Args:
            contexts: List of dicts with "text" and optional "score".

        Returns:
            List of cleaned, deduplicated context strings.
        """
        cleaned = [self._clean_context(c) for c in contexts]
        cleaned = [c for c in cleaned if c["text"]]  

        deduplicated = self._deduplicate(cleaned)
        return deduplicated if return_full else [c["text"] for c in deduplicated]
    
def prepare_contexts(contexts: List[Dict[str, Any]], return_scores: bool = False) -> List[Union[str, Dict[str, Any]]]:
    """
    Functional API: Clean and deduplicate contexts in one call.

    Args:
        contexts: List of dicts with "text" and optional "score".

    Returns:
        List of cleaned, deduplicated context strings.
    """
    preparer = ContextPreparer()
    cleaned = [preparer._clean_context(c) for c in contexts]
    cleaned = [c for c in cleaned if c["text"]]
    deduplicated = preparer._deduplicate(cleaned)

    if return_scores:
        return deduplicated
    return [c["text"] for c in deduplicated]