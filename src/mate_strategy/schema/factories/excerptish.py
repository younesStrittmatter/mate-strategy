from dataclasses import dataclass
from typing import List
from mate_strategy.schema import Schema
from mate_strategy.rules.factories import excerptish_rule


def excerptish_schema(source: str,
                      label: str = None,
                      similarity: float = 0.75) -> type[Schema]:
    """
    Builds a schema to extract text from a larger body of text.

    Examples:
        >>> txt = "This is a test. birds have wings and fish have fins."
        >>> schema = excerptish_schema(txt)
        >>> print(schema.prompt())

    """
    ex_rule = excerptish_rule(source, label, threshold=similarity)

    @dataclass
    class ExtractBlock(Schema):
        excerpts: ex_rule

    return ExtractBlock
