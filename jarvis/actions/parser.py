import re
from typing import List, Tuple

class CommandParser:
    @staticmethod
    def extract_all(text: str) -> List[str]:
        pattern = r"\[\s*CMD\s* : \s*(.*?)\s*\]"
        # Regex compatible with spaces around colon
        matches = re.findall(r"\[\s*CMD\s*:\s*(.*?)\s*\]", text, re.DOTALL | re.IGNORECASE)
        return [f"[CMD: {m.strip()}]" for m in matches]

    @staticmethod
    def parse_call(cmd_tag: str) -> Tuple[str, List[str]]:
        inner = cmd_tag.replace("[CMD:", "").replace("]", "").strip()
        if "(" not in inner: return inner, []

        name = inner.split("(")[0].strip()
        raw_args = inner[len(name):].strip("() ")
        pattern = r',(?=(?:[^\'"]*[\'"][^\'"]*[\'"])*[^\'"]*$)'
        args = [a.strip().strip("'\"") for a in re.split(pattern, raw_args)]
        return name, [a for a in args if a]
