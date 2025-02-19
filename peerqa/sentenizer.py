import logging
import re
from typing import List, Union

logger = logging.getLogger(__name__)

SPECIAL_SPLIT_TOKEN = "%%%SPLIT%%%"


class Sentenizer:
    def __call__(self, text: Union[str, List[str]]) -> List[str]:
        raise NotImplementedError


class SentenizerPunctSplitter(Sentenizer):
    """Split sentences on '?' and '!'"""

    def __init__(self) -> None:
        self.pattern = re.compile(r"[!?]+(?=[\s\w\.\d-])", re.IGNORECASE)

    def __call__(self, text: Union[str, List[str]]) -> List[str]:
        if isinstance(text, str):
            text = [text]
        sentences = []
        for t in text:
            sentences.extend(self.pattern.split(t))

        return sentences


class SentenizerEnumerationCleaner(Sentenizer):
    """Removes Sentences that only contain enumerations."""

    def __init__(self) -> None:
        self.pattern = re.compile(r"^\d+(\s)*(\.)*$", re.IGNORECASE)

    def __call__(self, text: Union[str, List[str]]) -> List[str]:
        if isinstance(text, str):
            text = [text]
        sentences = [t for t in text if not self.pattern.search(t)]

        return sentences


class EmptySentenceCleaner(Sentenizer):
    """Removes sentences that do not contain any characters."""

    def __init__(self, min_chars: int = 3) -> None:
        self.min_chars = min_chars
        self.pattern = re.compile(r"[a-zA-Z]", re.IGNORECASE)

    def __call__(self, text: Union[str, List[str]]) -> List[str]:
        if isinstance(text, str):
            text = [text]

        sentences = [
            t for t in text if self.pattern.search(t) and len(t) >= self.min_chars
        ]

        return sentences


class SpecialTokenSentizer(Sentenizer):
    """Splits sentences on a special token."""

    def __call__(self, text: Union[str, List[str]]) -> List[str]:
        if isinstance(text, str):
            text = [text]
        sentences = []
        for t in text:
            sentences.extend([s for s in t.split(SPECIAL_SPLIT_TOKEN)])

        return sentences


class SentenizerPipeline(Sentenizer):
    def __init__(self, sentenizers: Union[List[str], List[Sentenizer]]) -> None:
        if isinstance(sentenizers, list) and isinstance(sentenizers[0], str):
            sentenizers = [sentenizer_cls[s]() for s in sentenizers]
        self.sentenizers = sentenizers

    def __call__(self, text: Union[str, List[str]]) -> List[str]:
        for sentenizer in self.sentenizers:
            text = sentenizer(text)
        return text


sentenizer_cls = {
    "empty": EmptySentenceCleaner,
    "enum": SentenizerEnumerationCleaner,
    "punct": SentenizerPunctSplitter,
    "special_token": SpecialTokenSentizer,
}
