from dataclasses import dataclass


@dataclass
class Prompt:
    """
    AIPrompt is a class that represents a prompt for an AI model.

    It allows for the creation of a prompt with placeholders that can be
    Example:
        >>> prompt = Prompt("This is a {adjective} {noun}.")
        >>> prompt(adjective="beautiful", noun="day")
        'This is a beautiful day.'
    """

    def __init__(self, txt):
        self.txt = txt

    def __call__(self, **kwargs):
        """
                Fill the prompt with the given keyword arguments.
                """
        _prompt = self.txt
        for key, _ in kwargs.items():
            if '{' + key + '}' not in _prompt:
                print('[WARN] Key not found in prompt:', key)
        for key, value in kwargs.items():
            _prompt = _prompt.replace('{' + key + '}', str(value))
        return _prompt
