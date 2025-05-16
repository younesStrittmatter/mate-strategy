class RuleMeta(type):
    """
    Factory:  MyRule[params]  ➜  a *concrete* subclass that carries the params.
    A concrete rule must expose:
        - describe()  -> str
        - example()   -> Any
        - validate(v) -> bool
    """

    def __getitem__(cls, params):
        if not isinstance(params, tuple):
            params = (params,)

        # build a new subclass *on the fly*
        attrs = {"__rule_params__": params}
        name = f"{cls.__name__}_" + "_".join(map(str, params))
        return RuleMeta(name, (cls,), attrs)

    # allow direct instantiation of concrete subclasses
    def __call__(cls, *args, **kwargs):
        if cls is Rule:  # prevent instantiating the abstract base
            raise TypeError("Cannot instantiate Rule directly")
        return super().__call__(*args, **kwargs)


class Rule(metaclass=RuleMeta):
    """Abstract scalar rule – never annotate with plain Rule, only its subs."""

    # subclasses must implement the trio below
    @classmethod
    def describe(cls) -> str: ...  # human-readable constraint

    @classmethod
    def example(cls):  ...  # example value

    @classmethod
    def validate(cls, v): ...  # return bool
