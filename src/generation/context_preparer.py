from typing import Any, Dict, Union

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
