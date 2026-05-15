import re
from typing import List, Tuple
from jarvis.utils.logger import action_log as log

class CommandParser:
    """Robust regex-based command extractor (Phase 2)."""

    # Supporte [CMD: NAME('arg1', 'arg2')] et [CMD: NAME]
    TAG_PATTERN = re.compile(r'\[CMD:\s*([A-Z0-9_]+)(?:\((.*?)\))?\s*\]', re.IGNORECASE)

    @staticmethod
    def extract_all(text: str) -> List[str]:
        """Returns all [CMD: ...] blocks."""
        return [match.group(0) for match in CommandParser.TAG_PATTERN.finditer(text)]

    @staticmethod
    def parse_call(tag: str) -> Tuple[str, List[str]]:
        """Parses a single tag into (Name, [Args])."""
        match = CommandParser.TAG_PATTERN.match(tag)
        if not match:
            return "", []

        name = match.group(1).upper()
        args_str = match.group(2)

        if not args_str:
            return name, []

        # Extraction robuste des arguments gérant les guillemets et virgules
        # Pattern: '([^']*)' ou "([^"]*)"
        arg_pattern = re.compile(r"['\"](.*?)['\"]")
        args = arg_pattern.findall(args_str)

        # Fallback si pas de guillemets (mots simples séparés par virgules)
        if not args and args_str.strip():
            args = [a.strip() for a in args_str.split(',')]

        log.debug(f"Parsed command: {name} with args {args}")
        return name, args
