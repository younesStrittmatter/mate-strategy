from __future__ import annotations
from dataclasses import dataclass
from mate_strategy.schema import Schema
from mate_strategy.rules.factories import excerptish_rule


def excerptish_schema(source: str,
                      *,
                      similarity: float = 0.75) -> type[Schema]:
    ExRule = excerptish_rule(source, threshold=similarity)

    @dataclass
    class ExtractBlock(Schema):
        text: ExRule

    return ExtractBlock
