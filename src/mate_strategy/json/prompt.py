from typing import Union

from src.mate_strategy.core.prompt import Prompt as _Prompt
from src.mate_strategy.json.response import Response


class Prompt:
    """
    A class to represent a prompt for a language model expecting a JSON response.

    Examples:
        >>> from mate_strategy.json.rules import Interval, OneOf
        >>> from mate_strategy.json.key  import Key
        >>> from mate_strategy.json.response import Response
        >>> response = Response([
        ...     Key("id",   int,  Interval(1, 100)),
        ...     Key("name", str,  OneOf(["Janet", "Alice", "Andrew", "Caroline"])),
        ...     Key("flag", bool, lambda x: isinstance(x, bool)),           # simple lambda
        ... ])

        # Can be used with a string prompt and a Response object
        >>> prompt = Prompt("Please provide the following information:", response)
        >>> print(prompt())         # doctest: +NORMALIZE_WHITESPACE
        Please provide the following information:
        Fill in **valid JSON** for the fields below.
        Rules
        • "id" must be an integer between 1 and 100.
        • "name" must be **one of** these examples: Alice, Andrew, Caroline, Janet.
        • "flag" must be a bool.
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.
        <BLANKLINE>
        Example of correct shape (values are placeholders):
        { "id": <int 1-100>, "name": <str Alice, Andrew, Caroline, Janet>, "flag": <bool …> }

        # We can also use placeholder in the prompt:
        >>> prompt = Prompt("Given that {name} is {age}, provide following information:", response)
        >>> print(prompt(name="John", age=32))        # doctest: +NORMALIZE_WHITESPACE
        Given that John is 32, provide following information:
        Fill in **valid JSON** for the fields below.
        Rules
        • "id" must be an integer between 1 and 100.
        • "name" must be **one of** these examples: Alice, Andrew, Caroline, Janet.
        • "flag" must be a bool.
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.
        <BLANKLINE>
        Example of correct shape (values are placeholders):
        { "id": <int 1-100>, "name": <str Alice, Andrew, Caroline, Janet>, "flag": <bool …> }

    """

    def __init__(self,
                 prompt: str,
                 response: Response):
        """
        Initializes the Prompt with a given string.

        Args:
            prompt (str): The prompt string.
        """
        self.prompt = _Prompt(prompt)
        self.response = response

    def __call__(self, **kwargs):
        """
        Returns the string representation of the Prompt.

        Returns:
            str: The prompt string.
        """
        _p = self.prompt
        if isinstance(self.prompt, _Prompt):
            _p = self.prompt(**kwargs)
        return _p + "\n" + self.response.prompt()

    def validate(self, data):
        """
        Validates the data against the prompt.

        Args:
            data (dict): The data to validate.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        return self.response(data)
