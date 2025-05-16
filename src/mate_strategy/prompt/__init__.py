# prompt.py  ─────────────────────────────────────────────────────────
from dataclasses import dataclass
from string import Formatter
from typing import Type, Dict, Any
import warnings
from mate_strategy.parseable import Schema     # ← your new base

@dataclass
class Prompt:
    """
    A wrapper that combines
        • a *template*   (str.format-style)
        • a *Parseable*  answer schema

    Example:
        >>> from dataclasses import dataclass
        >>> from mate_strategy.parseable import Schema
        >>> from mate_strategy.rules.predefined import Interval

        We first define a schema for the answer. In this case we ask for a random number and a
        name.
        >>> @dataclass
        ... class Answer(Schema):
        ...     random_number: Interval[0, 10]
        ...     name: str
        ...     __example_overrides__ = {
        ...         "name": "Ben"
        ...     }

        We now create a prompt by providing a template and the answer schema.
        We can use {<...>} as placeholders in the template to insert values.
        >>> p = Prompt(
        ...     "We search for a random number and a name. The name shouldn't be {no_no_name}.",
        ...     Answer)

        We render the prompt with a specific value for z ...
        >>> print(p.render(no_no_name = "Norbert")) # doctest: +NORMALIZE_WHITESPACE
        We search for a random number and a name. The name shouldn't be Norbert.
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - random_number must be a number between 0 and 10
        - name must be a string
        <BLANKLINE>
        Example:
        {
          "random_number": 5,
          "name": "Ben"
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.


        ... we can validate a given answer against the schmema.
        >>> print(p.validate({"random_number": 5, "name": "Lisa"}))
        (True, None, None)

        ... and we can also check for invalid answers.
        >>> print(p.validate({"random_number": -1, "name": "Lisa"}))
        (False, '"random_number" is invalid.', '"random_number" must be a number between 0 and 10')
    """
    template: str
    schema  : Type[Schema]

    # ────────────────────────────────────────────────────────────────
    def _placeholders(self):
        """Return the set of field names in the template."""
        return {fname for _, fname, *_ in Formatter().parse(self.template) if fname}

    # ----------------------------------------------------------------
    def render(self, **values) -> str:
        """
        Render template and append the schema prompt.

        * Warns if user passes extra kwargs.
        * Warns if template placeholders remain unfilled.
        """
        ph   = self._placeholders()
        diff = set(values) - ph
        if diff:
            warnings.warn(f"Unused placeholders: {', '.join(diff)}", stacklevel=2)

        text = self.template.format(**values)

        if "{" in text or "}" in text:
            warnings.warn("Unresolved placeholders in prompt.", stacklevel=2)

        return text + "\n" + self.schema.prompt()

    def validate(self, data) -> tuple[bool, str | None, str | None]:
        """
        Run schema validation.

        Returns
        -------
        (True,  None,  None)          if the JSON passes
        (False, error_msg, expect_msg) otherwise
        """
        ok, *detail = self.schema.validate_with_error(data)
        if ok:
            return True, None, None

        # detail might be 1‑tuple (err) or 2‑tuple (err, expected)
        err = detail[0]
        exp = detail[1] if len(detail) > 1 else ""
        return False, err, exp

    # ----------------------------------------------------------------
    def validate_or_raise(self, data: Dict[str, Any]) -> None:
        """
        Run schema validation and raise ValueError on failure.
        """
        ok, *detail = self.schema.validate_with_error(data)
        if ok:
            return
        err, exp = (detail + ["<no details>"])[:2]
        raise ValueError(f"{err} :: {exp}")

    # convenience boolean wrapper
    def __call__(self, **kw) -> str:
        return self.render(**kw)
