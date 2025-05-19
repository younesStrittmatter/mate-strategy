from __future__ import annotations
import re, difflib
from typing import Type
from mate_strategy.rules import Rule


def excerptish_rule(source: str,
                    *,
                    window: int = 500,  # chars in comparison slice
                    stride: int = 250,  # slide overlap
                    threshold: float = 0.75  # difflib ratio
                    ) -> Type[Rule]:
    """
    Factory ⇒ subclass of :class:`Rule`.

    A candidate passes if, after collapsing whitespace/punctuation,
    **any** sliding window of *source* reaches `threshold` similarity
    (Jaro/Winkler-style ratio from difflib).
    """
    src_clean = re.sub(r"\W+", " ", source).lower()
    windows: list[str] = [
        src_clean[i:i + window]
        for i in range(0, len(src_clean), stride)
    ]

    class _ExcerptishRule(Rule):
        @classmethod
        def describe(cls) -> str:
            pct = int(threshold * 100)
            return f"closely match a passage in the source (≥ {pct}% similarity)"

        @classmethod
        def example(cls):
            # first ~20 words as a deterministic sample

            n = min(len(src_clean.split()) // 3, 30)
            return " ".join(src_clean.split()[:n]) + " …"

        @classmethod
        def validate(cls, v) -> bool:
            if not isinstance(v, str) or not v.strip():
                return False
            cand = re.sub(r"\W+", " ", v).lower()
            if len(cand) < 30:
                return False
            for chunk in windows:
                if difflib.SequenceMatcher(None, cand, chunk).ratio() >= threshold:
                    return True
            return False

    return _ExcerptishRule
