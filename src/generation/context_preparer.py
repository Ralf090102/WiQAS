from typing import Any, Dict, Union
from difflib import SequenceMatcher

import re

class ContextPreparer:

    def __init__(self, similarity_threshold: float = 0.7) -> None:
        self.similarity_threshold = similarity_threshold

        def _clean_context(self, context: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
            if isinstance(context, str):
                text = " ".join(context.split())
            else:
                text = " ".join(context.get("text", "").split())

            text = re.sub(r'\b(\w+)( \1){2,}\b', r'\1', text)

    def _are_similar(self, a: str, b: str) -> bool:
        ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
        if ratio >= self.similarity_threshold:
            return True
        
        shorter, longer = (a, b) if len(a) < len(b) else (b, a)
        if len(shorter) > 10 and shorter.lower() in longer.lower():
            return True
        return False