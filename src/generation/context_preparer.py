from typing import Any, Dict, List, Union, Optional
from difflib import SequenceMatcher
import re
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

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
            r'^[Dd]ata[\\/][Kk]nowledge_?[Bb]ase[\\/]',  
            r'^[Dd]ata[\\/]',                           
            r'^[Kk]nowledge_?[Bb]ase[\\/]',             
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
        """Convert various date formats to readable string."""
        if not date_value:
            return None
        
        try:
            # Try UNIX timestamp (handle both seconds and milliseconds)
            if isinstance(date_value, (int, float)):
                timestamp = int(date_value)
                # Check if timestamp is in milliseconds (13 digits)
                if timestamp > 10000000000:
                    timestamp = timestamp / 1000
                date_obj = datetime.fromtimestamp(timestamp)
                return date_obj.strftime("%B %d, %Y")
            
            # Try string timestamp
            if isinstance(date_value, str) and date_value.isdigit():
                timestamp = int(date_value)
                # Check if timestamp is in milliseconds (13 digits)
                if timestamp > 10000000000:
                    timestamp = timestamp / 1000
                date_obj = datetime.fromtimestamp(timestamp)
                return date_obj.strftime("%B %d, %Y")
            
            # Try ISO format string
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
    
    def _deduplicate(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            text = ctx["text"]
            score = ctx.get("final_score", 0.0)
            length = ctx.get("length", len(text))
            duplicate_found = False
            
            for kept in unique:
                if self._are_similar(text, kept["text"]):
                    duplicate_found = True
                    kept_score = kept.get("final_score", 0.0)
                    kept_length = kept.get("length", len(kept["text"]))
                    
                    if (score > kept_score) or (score == kept_score and length > kept_length):
                        kept.update(ctx)
                    break
                    
            if not duplicate_found:
                unique.append(ctx)
                
        return unique

    def _sort_contexts(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(contexts, key=lambda x: x.get("final_score", 0.0), reverse=True)
    
    def prepare(self, contexts: List[Dict[str, Any]], return_full: bool = False, include_citations: bool = True, sort_by_score: bool = True) -> List[Union[str, Dict[str, Any]]]:
        cleaned = []
        for c in contexts:
            cleaned_ctx = self._clean_context(c)
            if cleaned_ctx:  
                cleaned.append(cleaned_ctx)

        if not cleaned:
            logger.warning("No valid contexts after cleaning")
            return []

        deduplicated = self._deduplicate(cleaned)
        
        if sort_by_score:
            deduplicated = self._sort_contexts(deduplicated)
        
        if return_full:
            return deduplicated
        
        if include_citations:
            results = []
            for c in deduplicated:
                text = c["text"]
                citation_text = c.get("citation_text")
                if citation_text:
                    text = f"{text}\n[Source: {citation_text}]"
                results.append(text)
            return results
        
        return [c["text"] for c in deduplicated]
    
def prepare_contexts(contexts: List[Dict[str, Any]], return_scores: bool = False, include_citations: bool = True, similarity_threshold: float = 0.7) -> List[Union[str, Dict[str, Any]]]:
    """
    Functional API: Clean and deduplicate contexts in one call.

    Args:
        contexts: List of dicts with "text" and optional "score".

    Returns:
        List of cleaned, deduplicated context strings.
    """
    preparer = ContextPreparer(
        similarity_threshold=similarity_threshold
    )
    
    return preparer.prepare(
        contexts,
        return_full=return_scores,
        include_citations=include_citations
    )