from __future__ import annotations

from mate_strategy.prompt import Prompt
from mate_strategy.schema.factories import excerptish_schema


def excerptish_prompt(
        raw_text: str,
        template: str,
        label: str,
        *,
        similarity: float = 0.75,
) -> Prompt:
    """
    Builds a prompt to extract text from a larger body of text.

    Examples:
        >>> txt = "This is a test. birds have wings and fish have fins. insects have wings. Fins are sometimes called flippers. Wings are never called that."
        >>> tmplt = "Extract the things about {label} from the following source: {source}"
        >>> lbl = "wings"
        >>> prompt = excerptish_prompt(txt, tmplt, lbl)
        >>> print(prompt.render(label=lbl, source=txt))

    """

    schema = excerptish_schema(
        raw_text,
        label=label,
        similarity=similarity,
    )

    tmpl = template

    return Prompt(tmpl, schema)
