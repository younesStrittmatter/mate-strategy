# mate_strategy/rules/factories/registry.py
import sys
from functools import wraps
from types import ModuleType
from typing import Type
from mate_strategy.rules import Rule

def register_rule(fn):
    """
    Decorator for any *rule factory* that returns a subclass of Rule.
    It ensures the returned rule-class is visible in the module's globals
    so forward-reference strings in later Schemas resolve automatically.

    Usage
    -----
    @register_rule
    def excerptish_rule(...): ...
    """
    module: ModuleType = sys.modules[fn.__module__]

    @wraps(fn)
    def wrapper(*args, **kw) -> Type[Rule]:
        rule_cls = fn(*args, **kw)
        # put the class into the factory module's namespace
        module.__dict__[rule_cls.__name__] = rule_cls

        return rule_cls

    return wrapper