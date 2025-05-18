import re, random

from mate_strategy.rules import Rule

class NaturalNumber(Rule):
    @classmethod
    def describe(cls):
        return "integer (>= 1)"

    @classmethod
    def example(cls):
        return 3

    @classmethod
    def validate(cls, v):
        return isinstance(v, int) and v >= 1



class Interval(Rule):
    @classmethod
    def describe(cls):
        lo, hi = cls.__rule_params__
        return f"number between {lo} and {hi}"

    @classmethod
    def example(cls):
        lo, hi = cls.__rule_params__
        return (lo + hi) // 2

    @classmethod
    def validate(cls, v):
        lo, hi = cls.__rule_params__
        return lo <= v <= hi


class OneOf(Rule):
    @classmethod
    def describe(cls):
        return "one of " + ", ".join(map(repr, cls.__rule_params__))

    @classmethod
    def example(cls):
        return random.choice(cls.__rule_params__)

    @classmethod
    def validate(cls, v):
        return v in cls.__rule_params__


class Regex(Rule):
    @classmethod
    def describe(cls):
        pattern, = cls.__rule_params__
        return f'string matching regex "{pattern}"'

    @classmethod
    def example(cls):
        return "<match>"

    @classmethod
    def validate(cls, v):
        pattern, = cls.__rule_params__
        return isinstance(v, str) and re.fullmatch(pattern, v) is not None
