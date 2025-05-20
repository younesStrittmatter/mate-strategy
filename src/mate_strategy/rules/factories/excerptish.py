from __future__ import annotations
import re, difflib
from typing import Type
from mate_strategy.rules import Rule
from mate_strategy.rules.factories.utils.registry import register_rule


@register_rule
def excerptish_rule(source: str,
                    label: str = None,
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
    _label = label or "label"

    class _ExcerptishRule(Rule):
        @classmethod
        def describe(cls) -> str:
            return (f"list of matching passages.\n"
                    f"* ** Exhaustive **: include *everything* that contains information about *{_label}*.\n"
                    "  - It is better to be redundant than to lose information.\n"
                    "  - It is better to keep too much than too little.\n"
                    f"  - Keep passages that are necessary to understand information about the *{_label}*\n"
                    f"    even if they don't mention {_label}.\n"
                    f"    Make sure to provide the complete context for each passage.\n"
                    f"* Do *not* include unrelated passages or sentences.\n"
                    )

        @classmethod
        def example(cls):


            return ([f"... the grey fox jumped ...",
                     f"... ad minim veniam, quis nostrud  ..."])

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
