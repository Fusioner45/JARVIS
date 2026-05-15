import re
from typing import List, Tuple

class CommandParser:
    """Isole et nettoie les tags [CMD: ...] dans le flux du LLM."""

    @staticmethod
    def extract_all(text: str) -> List[str]:
        # Supporte les tags multi-lignes et les espaces variés
        pattern = r"\[\s*CMD\s*:\s*(.*?)\s*\]"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        return [f"[CMD: {m.strip()}]" for m in matches]

    @staticmethod
    def parse_call(cmd_tag: str) -> Tuple[str, List[str]]:
        """Extrait le nom et les arguments d'un tag avec support des quotes simples et doubles."""
        inner = cmd_tag.replace("[CMD:", "").replace("]", "").strip()
        if "(" not in inner:
            return inner, []

        name = inner.split("(")[0].strip()
        raw_args = inner[len(name):].strip("() ")

        # Regex robuste pour splitter par virgule SAUF si dans des quotes (simples ou doubles)
        # Gère : arg1, 'arg 2', "arg 3", 'l\'argument'
        pattern = r',(?=(?:[^\'"]*[\'"][^\'"]*[\'"])*[^\'"]*$)'
        args = [a.strip().strip("'\"") for a in re.split(pattern, raw_args)]

        return name, [a for a in args if a]
