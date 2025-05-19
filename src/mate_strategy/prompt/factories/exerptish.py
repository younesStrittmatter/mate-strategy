from __future__ import annotations

from mate_strategy.prompt import Prompt
from mate_strategy.schema.factories import excerptish_schema


def excerptish_prompt(
        raw_text: str,
        template: str,
        *,
        similarity: float = 0.75,
) -> Prompt:
    """
    Builds a prompt to extract text from a larger body of text.

    Examples:

    """

    schema = excerptish_schema(
        raw_text,
        similarity=similarity,
    )

    tmpl = template

    return Prompt(tmpl, schema)
