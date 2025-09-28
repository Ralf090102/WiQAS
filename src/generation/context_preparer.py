from typing import Any, Dict, List, Union
from difflib import SequenceMatcher

import re

class ContextPreparer:

    def __init__(self, similarity_threshold: float = 0.7) -> None:
        self.similarity_threshold = similarity_threshold

    def _clean_context(self, context: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
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

        text = " ".join(out)
        return {"text": text, "score": score}

    def _are_similar(self, a: str, b: str) -> bool:
        ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
        if ratio >= self.similarity_threshold:
            return True
        
        shorter, longer = (a, b) if len(a) < len(b) else (b, a)
        if len(shorter) > 10 and shorter.lower() in longer.lower():
            return True
        return False
    
    def _deduplicate(self, contexts: list[dict]) -> list[dict]:
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
                        kept["text"] = text
                        kept["score"] = score
                    break
            if not duplicate_found:
                unique.append(ctx)
        return unique
    
    def prepare(self, contexts: List[Dict[str, Any]]) -> List[str]:
        cleaned = [self._clean_context(c) for c in contexts]
        cleaned = [c for c in cleaned if c["text"]]  

        deduplicated = self._deduplicate(cleaned)
        return [c["text"] for c in deduplicated]
    
    def prepare_contexts(contexts: List[Dict[str, Any]]) -> List[str]:
        return ContextPreparer().prepare(contexts)