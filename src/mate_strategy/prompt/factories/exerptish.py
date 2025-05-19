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
    Generic helper → Prompt(schema).

    Parameters
    ----------
    raw_text        Article OCR dump
    labels          Full list of canonical labels
    target_label    The label to extract
    similarity      Fuzzy-matching threshold (0–1)
    template        Optional custom prompt string
                    (must include `{raw_text}` at minimum)
    fmt             Extra replacement dict for `template.format(**fmt)`
                    – lets callers inject task-specific instructions.
    """

    schema = excerptish_schema(
        raw_text,
        similarity=similarity,
    )

    tmpl = template

    return Prompt(tmpl, schema)
