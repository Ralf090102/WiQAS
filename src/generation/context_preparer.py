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

    def _clean_context(self, context: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Normalize a single context string and collapse repeated n-grams.

        - Collapses whitespace into single spaces.
        - Collapses repeated n-grams of length 1–4 (e.g.
          "Sinigang na Sinigang na Sinigang na" → "Sinigang na").
        - Preserves the score if provided.

        Args:
            context:
                Either a plain string or a dict containing keys:
                - "text" (str): The context text.
                - "score" (float, optional): Relevance score.

        Returns:
            dict: {"text": cleaned_text, "score": float}
        """
        if isinstance(context, str):
            text = " ".join(context.split())
            score = 0.0
        else:
            text = " ".join(context.get("text", "").split())
            score = context.get("score", 0.0)

        tokens = text.split()
        out = []
        i = 0
        while i < len(tokens):
            collapsed = False
            # Try longest possible phrase first (4 > 1 tokens)
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
                    out.extend(phrase) # keep a single instance
                    i = j
                    collapsed = True
                    break
            if not collapsed:
                out.append(tokens[i])
                i += 1

        cleaned_text = " ".join(out)

        return {
            "text": cleaned_text,
            "score": score,
            "metadata": context.get("source", {}),
            "document_id": context.get("document_id"),
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
        ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
        if ratio >= self.similarity_threshold:
            return True
        
        shorter, longer = (a, b) if len(a) < len(b) else (b, a)
        if len(shorter) > 10 and shorter.lower() in longer.lower():
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