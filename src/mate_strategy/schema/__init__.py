from dataclasses import dataclass, fields
from typing import (get_type_hints, get_origin, get_args,
                    Any, Callable, Tuple, Union, Optional,
                    Annotated)
import json, inspect
from mate_strategy.rules import Rule

NoneType = type(None)


def constraint(path: str, desc: str, *, fix: Callable[[dict], None] | None = None):
    def wrap(fn):
        fn.__constraint_info__ = (path, desc, fix or (lambda d: None))
        return fn

    return wrap


# helper for bullet-style indent
def _indent(level): return "  " * level


def _list_of_schema(t):
    return Schema._is_list(t) and Schema._is_schema(Schema._origin(get_args(t)[0]))


def _tuple_of_schema(t):
    return Schema._is_tuple(t) and any(
        Schema._is_schema(Schema._origin(a)) for a in get_args(t)
    )


class Schema:
    """
    A class that can emit a prompt and validate a JSON object.

    Examples:
        >>> import random
        >>> random.seed(42)  # for reproducible tests

        We create a schmas class by subclassing :class:`Schema` ...
        >>> @dataclass
        ... class MySchema(Schema):
        ...     x: int
        ...     y: str

        ... we can use this to create prompts
        >>> print(MySchema.prompt()) # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - x
            • integer      (ex: 42)
        - y
            • string      (ex: "example")
        <BLANKLINE>
        Example:
        {
          "x": 42,
          "y": "example"
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.

        ... or to validate a JSON object
        >>> MySchema.validate_with_error({"x": 42, "y": "example"})
        (True,)

        ... the *lazy* validation will fail at the *first* error
        >>> MySchema.validate_with_error({"x": "example", "y": 42})
        (False, '"x" must be integer', '"x" must be integer')

        >>> MySchema.validate_with_error({"x": 42, "y": 42})
        (False, '"y" must be string', '"y" must be string')

        ... and we can get a repair prompt:
        >>> print(MySchema.repair_prompt({"x": 42, "y": 42}))  # doctest: +NORMALIZE_WHITESPACE
        The JSON below is invalid.
        <BLANKLINE>
        Problem:
        - "y" must be string
          Expected:
         "y" must be string
        <BLANKLINE>
        Rules:
        - x
            • integer      (ex: 42)
        - y
            • string      (ex: "example")
        <BLANKLINE>
        Current value (invalid):
        ```json
        {
          "x": 42,
          "y": 42
        }
        ```
        <BLANKLINE>
        Reply with **only** the corrected JSON object.

        *****
        Rules
        *****

        We can also add more complicated rules.
        Either predefined
        >>> from mate_strategy.rules.predefined import Interval, OneOf
        >>> @dataclass
        ... class MySchema(Schema):
        ...    x: Interval[20, 100]  # bounded integer
        ...    y: OneOf['green', 'red', 'blue']

        ... again, we can use this to create prompts
        >>> print(MySchema.prompt()) # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - x
            • must be a number between 20 and 100      (ex: 60)
        - y
            • must be one of 'green', 'red', 'blue'      (ex: "blue")
        <BLANKLINE>
        Example:
        {
          "x": 60,
          "y": "green"
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.

        ... and validate with errors
        >>> MySchema.validate_with_error({"x": 42, "y": "red"})
        (True,)

        ... again the validation is *lazy*
        >>> MySchema.validate_with_error({"x": 10, "y": "yellow"})
        (False, '"x" is invalid.', '"x" must be a number between 20 and 100')

        ... and we can get a repair prompt:
        >>> print(MySchema.repair_prompt({"x": 42, "y": "yellow"}))  # doctest: +NORMALIZE_WHITESPACE
        The JSON below is invalid.
        <BLANKLINE>
        Problem:
        - "y" is invalid.
          Expected:
         "y" must be one of 'green', 'red', 'blue'
        <BLANKLINE>
        Rules:
        - x
            • must be a number between 20 and 100      (ex: 60)
        - y
            • must be one of 'green', 'red', 'blue'      (ex: "green")
        <BLANKLINE>
        Current value (invalid):
        ```json
        {
          "x": 42,
          "y": "yellow"
        }
        ```
        <BLANKLINE>
        Reply with **only** the corrected JSON object.

        ************
        Custom Rules
        ************
        We can also create custom rules by providing
        a class that inherits from :class:`Rule` and implements the
        :meth:`describe`, :meth: `example`, and :meth:`validate` methods:
        >>> from mate_strategy.rules import Rule
        >>> class MyRule(Rule):
        ...     @classmethod
        ...     def describe(cls):
        ...         return "must be between 0 and 1 or between 2 and 3"
        ...     @classmethod
        ...     def example(cls):
        ...         return random.choice([random.random(), random.random() + 2])
        ...     @classmethod
        ...     def validate(cls, v):
        ...         return 0 <= v <= 1 or 2 <= v <= 3

        we can use this in the schema just as any other rule:
        >>> @dataclass
        ... class MySchema(Schema):
        ...     x: MyRule

        >>> print(MySchema.prompt()) # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - x
            • must be between 0 and 1 or between 2 and 3
              (ex: 0.74155049…)
        <BLANKLINE>
        Example:
        {
          "x": 0.7364712141640124
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.

        >>> MySchema.validate_with_error({"x": 0.5})
        (True,)

        >>> MySchema.validate_with_error({"x": 2.5})
        (True,)

        >>> MySchema.validate_with_error({"x": 1.5})
        (False, '"x" is invalid.', '"x" must be between 0 and 1 or between 2 and 3')

        ... for all, we can use List, Tuples, Unions or Optional
        >>> from typing import List, Tuple, Union, Optional
        >>> @dataclass
        ... class ListSchema(Schema):
        ...     a: List[int]
        >>> @dataclass
        ... class TupleSchema(Schema):
        ...     a: Tuple[int, int]
        >>> @dataclass
        ... class UnionSchema(Schema):
        ...     a: Union[int, str]
        >>> @dataclass
        ... class OptionalSchema(Schema):
        ...     a: Optional[int]
        ...     b: int

        ... we can get prompts:
        >>> print(ListSchema.prompt()) # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - a
            • list of integers
              (ex: [42])
        <BLANKLINE>
        Example:
        {
          "a": [
            42
          ]
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.

        >>> print(TupleSchema.prompt()) # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - a
            • tuple ⟨el[0] integer, el[1] integer⟩
        <BLANKLINE>
        Example:
        {
          "a": [
            42,
            42
          ]
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.

        >>> print(UnionSchema.prompt()) # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - a
            • choose **one** of:
                        1. integer
                        2. string
        <BLANKLINE>
        Example:
        {
          "a": 42
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.

        >>> print(OptionalSchema.prompt()) # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - a
          • (Optional) Key can be *missing* or:
            - integer
            - None
        - b
          • integer
            (ex: 42)
        <BLANKLINE>
        Example:
        {
          "a": 42,
          "b": 42
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.

        ... and we can validate these:
        >>> ListSchema.validate_with_error({"a": [1, 2, 3]})
        (True,)

        >>> ListSchema.validate_with_error({"a": 1})
        (False, '"a" must be a list, got int', '"a" must be list')

        >>> TupleSchema.validate_with_error({"a": (1, 2)})
        (True,)

        >>> TupleSchema.validate_with_error({"a": (1, 2, 3)})
        (False, '"a" must have length 2, got 3', '"a" must be length-2 tuple')

        >>> UnionSchema.validate_with_error({"a": 1})
        (True,)

        >>> UnionSchema.validate_with_error({"a": "example"})
        (True,)

        >>> UnionSchema.validate_with_error({"a": [1]})
        (False, '"a" matches none of the allowed alternatives.', '"a" must be integer or "a" must be string')

        >>> OptionalSchema.validate_with_error({"a": 1, "b": 2})
        (True,)

        >>> OptionalSchema.validate_with_error({"a": None, "b": 2})
        (True,)

        >>> OptionalSchema.validate_with_error({"b": 2})
        (True,)

        >>> OptionalSchema.validate_with_error({"a": 1})
        (False, '"b" is missing.', '"b" must be present.')




        >>> from typing import List
        >>> @dataclass
        ... class MySchema(Schema):
        ...     x: List[MyRule]
        ...     y: Tuple[int, str]
        ...     z: Union[int, str]

        >>> print(MySchema.prompt()) # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - x
            • list of must be between 0 and 1 or between 2 and 3s
              (ex: [0.5904925…)
        - y
            • tuple ⟨el[0] integer, el[1] string⟩
        - z
            • choose **one** of:
                        1. integer
                        2. string
        <BLANKLINE>
        Example:
        {
          "x": [
            0.21863797480360336
          ],
          "y": [
            42,
            "example"
          ],
          "z": 42
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.

        ... schemas can also be nested:
        >>> @dataclass
        ... class OuterSchema(Schema):
        ...     a: MySchema
        ...     b: str

        >>> print(OuterSchema.prompt()) # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - a
            • MySchema
          - a.x
              • list of must be between 0 and 1 or between 2 and 3s
                (ex: [2.7160196…)
          - a.y
              • tuple ⟨el[0] integer, el[1] string⟩
          - a.z
              • choose **one** of:
                        1. integer
                        2. string
        - b
            • string
              (ex: "example")
        <BLANKLINE>
        Example:
        {
          "a": {
            "x": [
              0.2204406220406967
            ],
            "y": [
              42,
              "example"
            ],
            "z": 42
          },
          "b": "example"
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.

        ... and we can validate this:
        >>> OuterSchema.validate_with_error({"a": {"x": [0.5], "y": [42, "example"], "z": "hi"}, "b": "example"})
        (True,)

        >>> OuterSchema.validate_with_error({"a": {"x": [0.5], "y": [42, 42]}, "b": "example"})
        (False, '"a.y[1]" must be string', '"a.y[1]" must be string')

        ... and get a repair prompt:
        >>> print(OuterSchema.repair_prompt({"a": {"z": [0.5], "y": [42, 42]}, "b": "example"}))  # doctest: +NORMALIZE_WHITESPACE
        The JSON below is invalid.
        <BLANKLINE>
        Problem:
        - "a.x" is missing.
          Expected:
         "a.x" must be present.
        <BLANKLINE>
        Rules:
        - a
            • MySchema
          - a.x
              • list of must be between 0 and 1 or between 2 and 3s
                (ex: [2.1596593…)
          - a.y
              • tuple ⟨el[0] integer, el[1] string⟩
          - a.z
              • choose **one** of:
                        1. integer
                        2. string
        - b
            • string
              (ex: "example")
        <BLANKLINE>
        Current value (invalid):
        ```json
        {
          "a": {
            "z": [
              0.5
            ],
            "y": [
              42,
              42
            ]
          },
          "b": "example"
        }
        ```
        <BLANKLINE>
        Reply with **only** the corrected JSON object.

        ... this  also works with Unions (Lists, ...):
        >>> @dataclass
        ... class OuterSchema(Schema):
        ...     x: Union[MyRule, List[MySchema]]
        >>> print(OuterSchema.prompt()) # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - x
          • choose **one** of:
            1. must be between 0 and 1 or between 2 and 3
            2. list of MySchema
            - x[].x
              • list of must be between 0 and 1 or between 2 and 3s
                (ex: [2.1554794…)
            - x[].y
              • tuple ⟨el[0] integer, el[1] string⟩
            - x[].z
              • choose **one** of:
                1. integer
                2. string
        <BLANKLINE>
        Example:
        {
          "x": 2.3799273006373376
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.

        ****************
        Cross Validation
        ****************
        We can also add cross validation rules using the @constraint decorator.
        >>> @dataclass
        ... class MySchema(Schema):
        ...     x: int
        ...     y: int
        ...
        ...     @constraint("y", "must be odd if x is even and vice versa")
        ...     def _(data):
        ...         x = data["x"]
        ...         y = data["y"]
        ...         if not x % 2:
        ...             return y % 2
        ...         if x % 2:
        ...             return not y % 2

        >>> print(MySchema.prompt()) # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - x
          • integer
            (ex: 42)
        - y
          • integer
            (ex: 42)
          - y must be odd if x is even and vice versa
        <BLANKLINE>
        Example:
        {
          "x": 42,
          "y": 42
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.

        ... if necessary, we can also fix the example:
        >>> def make_even_odd(d):
        ...     d["x"] = d["y"] + 1

        >>> @dataclass
        ... class MySchema(Schema):
        ...     x: int
        ...     y: int
        ...
        ...     @constraint("y", "must be odd if x is even and vice versa", fix=make_even_odd)
        ...     def _(data):
        ...         x = data["x"]
        ...         y = data["y"]
        ...         if not x % 2:
        ...             return y % 2
        ...         if x % 2:
        ...             return not y % 2

        >>> print(MySchema.prompt()) # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - x
          • integer
            (ex: 42)
        - y
          • integer
            (ex: 42)
          - y must be odd if x is even and vice versa
        <BLANKLINE>
        Example:
        {
          "x": 43,
          "y": 42
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.


        >>> print(MySchema.validate_with_error({"y": 1, "x": 42}))
        (True,)

        >>> print(MySchema.validate_with_error({"y": 2, "x": 42}))
        (False, '"y" violates constraint.', '"y" must be odd if x is even and vice versa')

        ... instead of fixing the example with a function, we can also override the defaults:
        >>> @dataclass
        ... class MySchema(Schema):
        ...     x: int
        ...     y: int
        ...     __example_overrides__ = {
        ...         'x': 43,
        ...         'y': 44
        ...     }

        >>> print(MySchema.prompt()) # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - x
          • integer
            (ex: 42)
        - y
          • integer
            (ex: 42)
        <BLANKLINE>
        Example:
        {
          "x": 43,
          "y": 44
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.

    """
    _declared_constraints: list[tuple[str, str, staticmethod]] = []
    __additional_examples__: list[dict] = []

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls._declared_constraints = []

        for v in cls.__dict__.values():
            if hasattr(v, "__constraint_info__"):
                data = v.__constraint_info__  # tuple of 2 or 3 items
                if len(data) == 2:
                    path, desc = data
                    fix = lambda d: None  # no-op fixer
                else:
                    path, desc, fix = data  # new (path, desc, fix)

                cls._declared_constraints.append((path, desc, staticmethod(v), fix))

    @staticmethod
    def _origin(t):
        return get_origin(t) or t

    @staticmethod
    def _is_rule(t):
        try:
            return issubclass(t, Rule)
        except TypeError:
            return False

    @staticmethod
    def _is_schema(t):
        try:
            return issubclass(t, Schema)
        except TypeError:
            return False

    @staticmethod
    def _is_list(t):
        return Schema._origin(t) is list

    @staticmethod
    def _is_tuple(t):
        return Schema._origin(t) is tuple

    @staticmethod
    def _list_of_schema(t):
        return Schema._is_list(t) and Schema._is_schema(
            Schema._origin(get_args(t)[0])
        )

    @staticmethod
    def _tuple_of_schema(t):
        return Schema._is_tuple(t) and any(
            Schema._is_schema(Schema._origin(a)) for a in get_args(t)
        )

    @staticmethod
    def _is_union(t):
        return get_origin(t) is Union

    @staticmethod
    def _is_optional(t):
        return Schema._is_union(t) and NoneType in get_args(t) and len(get_args(t)) == 2

    # ──────────────────────────────────────────────────────────────────────z

    @classmethod
    def _describe_type(cls, typ: Any) -> str:
        if cls._is_rule(typ):
            return cls._origin(typ).describe()

        # primitives
        if typ in (str, int, float, bool):
            return {str: "string", int: "integer",
                    float: "float", bool: "boolean"}[typ]

        # list
        if cls._is_list(typ):
            elem = get_args(typ)[0]
            if cls._is_schema(cls._origin(elem)):
                inner = cls._origin(elem)
                hdr = getattr(inner, "_doc_header", "")
                return f"list of {inner.__name__}" + (f" – {hdr}" if hdr else "")
            return f"list of {cls._describe_type(elem)}s"

        # tuple
        if cls._is_tuple(typ):
            parts = []
            for i, t in enumerate(get_args(typ)):
                desc = cls._describe_type(t)
                parts.append(f"el[{i}] {desc}")
            return f"tuple ⟨{', '.join(parts)}⟩"

        # optional
        if cls._is_optional(typ):
            inner = next(t for t in get_args(typ) if t is not NoneType)
            return f"(optional) {cls._describe_type(inner)}"

        # union  → numbered list, each line already indented five levels
        if cls._is_union(typ):
            alts = [cls._describe_type(t) for t in get_args(typ)]
            return ("choose **one** of:\n" +
                    "\n".join(f"{_indent(5)}{i + 1}. {d}"
                              for i, d in enumerate(alts)))

        # nested Schema
        if cls._is_schema(cls._origin(typ)):
            target = cls._origin(typ)
            head = getattr(target, "_doc_header", "")
            return f"{target.__name__}" + (f"  – {head}" if head else "")

        return str(typ)

    @classmethod
    def _example_for_type(cls, typ: Any):
        if cls._is_rule(typ):
            return typ.example()

        if cls._is_list(typ):
            elem = get_args(typ)[0] if get_args(typ) else Any
            return [cls._example_for_type(elem)]

        if cls._is_tuple(typ):
            return [cls._example_for_type(t) for t in get_args(typ)]  # lists → valid JSON

        if cls._is_optional(typ):
            non_none = next(t for t in get_args(typ) if t is not NoneType)
            return cls._example_for_type(non_none)

        if cls._is_union(typ):
            # pick first alternative for deterministic example
            first = get_args(typ)[0]
            return cls._example_for_type(first)

        if cls._is_schema(cls._origin(typ)):
            return cls._origin(typ).example()

        if typ is str:
            return "example"
        if typ is int:
            return 42
        if typ is float:
            return 3.14
        if typ is bool:
            return True

        return f"<{typ}>"

    # ───────────────────── field-introspection ──────────────────────
    @classmethod
    def _field_types(cls) -> dict[str, Any]:
        return {f.name: get_type_hints(cls)[f.name] for f in fields(cls)}

    @classmethod
    def _infer_rule(cls, name: str, typ: Any) -> Any:
        """Override in subclasses to inject pre-configured rule objects."""
        return None

    # ────────────────────────── rendering ───────────────────────────

    # ────────────────────────── rendering ───────────────────────────
    # ────────────────────────── rendering ───────────────────────────
    @classmethod
    def rules(cls, prefix: str = "", _lvl: int = 0) -> list[str]:
        """Produce a pretty, fully-nested rule list.

        `_lvl` is the current indentation level (0 for top-level).  Each level
        indents two spaces; nested items simply call `rules(..., _lvl+1)`."""
        IND = "  " * _lvl
        lines: list[str] = []

        # -------- collect @constraint rules grouped by their first key -------
        grouped: dict[str, list[tuple[str, str]]] = {}
        for path, desc, *_ in cls._declared_constraints:
            head, *rest = path.split(".", 1)
            grouped.setdefault(head, []).append((".".join(rest), desc))

        # -------- helper to append child-schema rules -------------------------
        def _emit_child(inner_cls, child_prefix: str):
            for l in inner_cls.rules(child_prefix, _lvl + 1):
                lines.append(l)

        # ---------------------------------------------------------------------
        for name, typ in cls._field_types().items():
            full = f"{prefix}{name}"
            IND2 = IND + "  "  # one level deeper
            IND3 = IND + "    "  # two levels deeper

            # label
            lines.append(f"{IND}- {full}")

            if cls._is_optional(typ):
                # inner type ≠ NoneType
                inner = next(t for t in get_args(typ) if t is not NoneType)
                desc = cls._describe_type(inner).replace("\n", "\n" + IND3)

                # headline for an optional field
                lines.append(f"{IND2}• (Optional) Key can be *missing* or:")
                lines.append(f"{IND3}- {desc}")  # the real value
                lines.append(f"{IND3}- None")  # explicit null

                # ――― NEW: append the note (if any) ―――
                note = ""
                if get_origin(inner) is Annotated:  # e.g. Annotated[int, "note"]
                    _, note = get_args(inner)
                elif issubclass(cls, AnnotatedSchema):  # doc-string note
                    note = cls._field_notes.get(name, "")



                    # ---- recurse if the *inner* itself is / contains Schemas ----
                child_lvl = _lvl + 2
                if cls._is_schema(cls._origin(inner)):
                    lines.extend(cls._origin(inner).rules(full + ".", child_lvl))
                elif cls._list_of_schema(inner):
                    inner_schema = cls._origin(get_args(inner)[0])
                    lines.extend(inner_schema.rules(full + "[].", child_lvl))
                elif cls._tuple_of_schema(inner):
                    for j, part in enumerate(get_args(inner)):
                        if cls._is_schema(cls._origin(part)):
                            lines.extend(
                                cls._origin(part).rules(f"{full}[{j}].", child_lvl)
                            )


            # ────────────────────────────── Union[T, …] ────────────────────────────
            elif cls._is_union(typ):
                lines.append(f"{IND2}• choose **one** of:")
                for idx, alt in enumerate(get_args(typ), 1):
                    desc = cls._describe_type(alt).replace("\n", "\n" + IND3)
                    lines.append(f"{IND3}{idx}. {desc}")
                    child_lvl = _lvl + 3  # one indent deeper
                    if cls._is_schema(cls._origin(alt)):
                        lines.extend(cls._origin(alt).rules(full + ".", child_lvl))
                    elif cls._list_of_schema(alt):
                        inner = cls._origin(get_args(alt)[0])
                        lines.extend(inner.rules(full + "[].", child_lvl))
                    elif cls._tuple_of_schema(alt):
                        for j, part in enumerate(get_args(alt)):
                            if cls._is_schema(cls._origin(part)):
                                lines.extend(
                                    cls._origin(part).rules(f"{full}[{j}].",
                                                            child_lvl))
            # --- NON-UNION branch --------------------------------------------
            else:
                desc = cls._describe_type(typ).replace("\n", "\n" + IND3)
                lines.append(f"{IND2}• {desc}")

                # micro-example for primitive / simple list only
                if (not cls._is_schema(cls._origin(typ))
                        and not cls._is_tuple(typ)
                        and not (cls._is_list(typ)
                                 and cls._is_schema(cls._origin(get_args(typ)[0])))):
                    ex = json.dumps(cls._example_for_type(typ))[:10]
                    lines.append(f"{IND3}(ex: {ex}{'…' if len(ex) == 10 else ''})")

                # recurse into nested Schemas
                if cls._is_schema(cls._origin(typ)):
                    _emit_child(cls._origin(typ), full + ".")
                elif cls._is_list(typ) and cls._is_schema(cls._origin(get_args(typ)[0])):
                    _emit_child(cls._origin(get_args(typ)[0]), full + "[].")
                elif cls._is_tuple(typ):
                    for i, part in enumerate(get_args(typ)):
                        if cls._is_schema(cls._origin(part)):
                            _emit_child(cls._origin(part), f"{full}[{i}].")

            # inline constraints anchored at this head
            for tail, desc in grouped.get(name, []):
                p = f"{full}.{tail}" if tail else full
                lines.append(f"{IND2}- {p} {desc}")

        # object-level constraints (no head key)
        for tail, desc in grouped.get("", []):
            lines.append(f"{IND}- {desc}")

        return lines

    @classmethod
    def _apply_path(cls, obj: dict, path: str, value):
        """Assign `value` into a nested dict via dot‐separated path."""
        parts = path.split(".")
        head, *tail = parts
        if not tail:
            obj[head] = value
        else:
            cls._apply_path(obj.setdefault(head, {}), ".".join(tail), value)

    @classmethod
    def example(cls):

        ex = {n: cls._example_for_type(t) for n, t in cls._field_types().items()}

        for _, _, _, fix in getattr(cls, "_declared_constraints", []):
            fix(ex)
        for path, value in getattr(cls, "__example_overrides__", {}).items():
            cls._apply_path(ex, path, value)

        return ex

    @classmethod
    def _all_examples(cls) -> list[dict]:
        """Base example + any hard-coded extras."""
        base = cls.example()  # current single example
        extras = getattr(cls, "__additional_examples__", [])
        # normalise: allow user to supply one dict or a list
        if extras and isinstance(extras, dict):
            extras = [extras]
        return [base, *extras]

    @classmethod
    def prompt(cls) -> str:
        lead = "Fill in **valid JSON** for the fields below."
        if getattr(cls, "_doc_header", ""):
            lead += f" – **{cls._doc_header}**"

        rule_block = "\n".join(cls.rules())

        # render every example in a fenced block for readability
        ex_blocks = "\n\n".join(
            "Example {}:\n```json\n{}\n```".format(i + 1,
                                                   json.dumps(ex, indent=2))
            for i, ex in enumerate(cls._all_examples())
        )

        return (
            f"{lead}\n\nRules\n{rule_block}\n\n{ex_blocks}"
            "\n\nReturn **only** the JSON object — no code-fences, no comments."
        )

    # ────────────────────────── validation ─────────────────────────
    # ────────────────────────── validation helper ─────────────────────────
    @classmethod
    def _validate_value(cls, key: str, val, typ: Any, prefix: str = ""):
        """
        Validate a single value against its declared type/rule.

        Returns
        -------
        None                      – if the value is valid
        (err, expected) : tuple   – if the value is invalid
        """
        full = f"{prefix}{key}"  # fully-qualified path at this level

        # 1️⃣  scalar Rule --------------------------------------------------
        if cls._is_rule(typ):
            if not typ.validate(val):
                return (f'"{full}" is invalid.',
                        f'"{full}" {typ.describe()}')
            return None

        # 2️⃣  list ---------------------------------------------------------
        if cls._is_list(typ):
            if not isinstance(val, list):
                return (f'"{full}" must be a list, got {type(val).__name__}',
                        f'"{full}" must be list')
            elem_typ = get_args(typ)[0] if get_args(typ) else Any
            for i, v in enumerate(val):
                err = cls._validate_value(f"{key}[{i}]", v, elem_typ, prefix)
                if err:
                    return err
            return None

        # 3️⃣  tuple --------------------------------------------------------
        if cls._is_tuple(typ):
            if not isinstance(val, (list, tuple)):
                return (f'"{full}" must be a tuple, got {type(val).__name__}',
                        f'"{full}" must be tuple')
            parts = get_args(typ)
            if len(val) != len(parts):
                return (f'"{full}" must have length {len(parts)}, got {len(val)}',
                        f'"{full}" must be length-{len(parts)} tuple')
            for i, (v, sub_t) in enumerate(zip(val, parts)):
                err = cls._validate_value(f"{key}[{i}]", v, sub_t, prefix)
                if err:
                    return err
            return None

        # 4️⃣  Optional[T] --------------------------------------------------
        if cls._is_optional(typ):
            inner = next(t for t in get_args(typ) if t is not NoneType)
            if val is None:
                return None
            return cls._validate_value(key, val, inner, prefix)

        # 5️⃣  Union[...] ---------------------------------------------------
        if cls._is_union(typ):
            expected_parts = []
            for alt in get_args(typ):
                err = cls._validate_value(key, val, alt, prefix)
                if err is None:
                    return None  # at least one branch OK
                expected_parts.append(err[1])
            return (f'"{full}" matches none of the allowed alternatives.',
                    " or ".join(expected_parts))

        # 6️⃣  nested Schema -----------------------------------------------
        base = cls._origin(typ)
        if cls._is_schema(base):
            if not isinstance(val, dict):
                return (f'"{full}" must be an object',
                        f'"{full}" must be an object')

            ok, *reason = base.validate_with_error(val)
            if ok:
                return None

            # ── patch the inner error messages with outer path ────────────
            if len(reason) == 2:
                err_msg, exp_msg = reason

                def _prepend_outer_path(msg: str) -> str:
                    # only change messages that start with "quoted path"
                    if msg.startswith('"'):
                        end = msg.find('"', 1)
                        if end != -1:
                            inner_path = msg[1:end]
                            if not inner_path.startswith(full + "."):
                                return f'"{full}.{inner_path}"' + msg[end + 1:]
                    return msg

                return (f'"{full}.{err_msg[1:]}'
                        if err_msg.startswith('"') else err_msg,
                        _prepend_outer_path(exp_msg))

            # fallback – propagate unchanged
            return False, *reason

        # 7️⃣  primitives ---------------------------------------------------
        if typ is str and not isinstance(val, str):
            return (f'"{full}" must be string', f'"{full}" must be string')
        if typ is int and not isinstance(val, int):
            return (f'"{full}" must be integer', f'"{full}" must be integer')
        if typ is float and not isinstance(val, (int, float)):
            return (f'"{full}" must be float', f'"{full}" must be float')
        if typ is bool and not isinstance(val, bool):
            return (f'"{full}" must be boolean', f'"{full}" must be boolean')

        # ------------------------------------------------------------------
        return None  # nothing complained ⇒ valid

    @classmethod
    def validate_with_error(cls, data: dict):
        """Return  (True,)  if OK,
            else (False, short_msg, long_msg)."""

        # unexpected keys ------------------------------------------------
        for k in data:
            if k not in cls._field_types():
                return False, f'"{k}" is not a valid field.', f'"{k}" is not expected here.'

        # per-field validation ------------------------------------------
        for name, typ in cls._field_types().items():
            if name not in data:
                if cls._is_optional(typ):
                    continue  # missing optional is fine
                return False, f'"{name}" is missing.', f'"{name}" must be present.'

            err = cls._validate_value(name, data[name], typ)
            if err:  # _validate_value returns None or (err, exp)
                return False, err[0], err[1]

        ok, *reason = cls.validate_cross(data)  # <── call the hook
        if not ok:
            return False, *reason

        return True,

    # ────────────────────────── repair-prompt ───────────────────────
    @classmethod
    def repair_prompt(cls, bad_data: dict) -> str:
        ok, *reason = cls.validate_with_error(bad_data)
        reason_msg = ""
        if not ok and len(reason) == 2:
            err, exp = reason
            reason_msg = f"Problem:\n- {err}\n  Expected:\n {exp}\n\n"
        return (
                "The JSON below is invalid.\n\n"
                f"{reason_msg}"
                "Rules:\n" + "\n".join(cls.rules()) + "\n\n"
                                                      "Current value (invalid):\n"
                                                      f"```json\n{json.dumps(bad_data, indent=2)}\n```\n\n"
                                                      "Reply with **only** the corrected JSON object."
        )

    @classmethod
    def validate_cross(cls, data: dict):
        for path, desc, fn, _ in cls._declared_constraints:
            if not fn(data):
                return (
                    False,
                    f'"{path}" violates constraint.',
                    f'"{path}" {desc}',
                )
        return True,

import re, inspect
from typing import get_origin, get_args

_BRACKETS_RE = re.compile(r"\[[^\]]*\]$")       # matches [], [0], [17] …

def _base_name(part: str) -> str:
    """'inner[]' -> 'inner',  'coords[0]' -> 'coords'."""
    return _BRACKETS_RE.sub("", part)


def _unwrap_container(typ):
    """List[T] ➜ T   |   Tuple[A,B] ➜ (A or B, depending on index)."""
    origin = get_origin(typ) or typ

    # Optional[T]  →  T
    if origin is Union and type(None) in get_args(typ) and len(get_args(typ)) == 2:
        inner = next(t for t in get_args(typ) if t is not type(None))
        return _unwrap_container(inner)

    # List[T]  →  T
    if origin is list:
        return _unwrap_container(get_args(typ)[0])

    # plain schema, rule, primitive …
    return origin

def _note_for_path(root_cls, parts):
    """
    Follow a ‘inner[].name’-style path through nested AnnotatedSchema
    definitions and return the first matching field-note.
    """
    if not parts or not (inspect.isclass(root_cls)
                         and issubclass(root_cls, AnnotatedSchema)):
        return None

    raw_head, *rest = parts
    head = _base_name(raw_head)                        # <- strip “[]” / “[n]”

    # note on the current class (if any)
    note = getattr(root_cls, "_field_notes", {}).get(head)

    # done?
    if not rest:
        return note

    # descend into the type of this field ────────────────────────────
    typ = root_cls.__annotations__.get(head)
    if typ is None:
        return note                                      # nothing to follow

    target = _unwrap_container(typ)

    # Tuple[...] is a special case – choose element by index if we have one
    origin = get_origin(typ) or typ
    if origin is tuple and "[" in raw_head:
        idx = int(raw_head[raw_head.find('[')+1:raw_head.find(']')] or 0)
        tup_args = get_args(typ)
        if idx < len(tup_args):
            target = _unwrap_container(tup_args[idx])

    # Union[..., …] – pick the first schema alternative
    if get_origin(target) is Union:
        for alt in get_args(target):
            cand = _unwrap_container(alt)  # <── unwrap first
            if inspect.isclass(cand) and issubclass(cand, AnnotatedSchema):
                target = cand  # descend into this one
                break

    return _note_for_path(target, rest) or note

# ---------------------------------------------------------------------
#  AnnotatedSchema  –  opt-in “Schema with doc-string annotations”
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
def _field_note(owner_cls, field: str) -> str | None:
    if not inspect.isclass(owner_cls):
        return None
    if not issubclass(owner_cls, AnnotatedSchema):
        return None
    return getattr(owner_cls, "_field_notes", {}).get(field)


import inspect
from typing import Annotated   # stdlib, already available

class AnnotatedSchema(Schema):
    """
    Extend :class:`Schema` with human-readable annotations.

    Put them directly in the subclass doc-string:

      • *first* non-blank line ...... → short header
      • lines like “field: note …” .. → per-field note

    Example
    -------
    >>> @dataclass
    ... class LevelSchema(AnnotatedSchema):
    ...     '''
    ...     Single level in a factorial design
    ...
    ...     name  : The concrete value (e.g. "red")
    ...     weight: Optional repetition weight
    ...     '''
    ...     name  : str
    ...     weight: Optional[int]
    ...
    >>> print(LevelSchema.prompt())       # doctest:+NORMALIZE_WHITESPACE
    Fill in **valid JSON** for the fields below. – **Single level in a factorial design**
    <BLANKLINE>
    Rules
    - name  – The concrete value (e.g. "red")
      • string
        (ex: "example")
    - weight  – Optional repetition weight
      • (Optional) Key can be *missing* or:
        - integer
        - None
    <BLANKLINE>
    Example:
    {
      "name": "example",
      "weight": 42
    }
    <BLANKLINE>
    Return **only** the JSON object — no code-fences, no comments.

    >>> @dataclass
    ... class OuterSchema(AnnotatedSchema):
    ...     '''
    ...     Outer level in a factorial design
    ...
    ...     inner  : Single level in a factorial design
    ...     weight: Optional repetition weight
    ...     '''
    ...     inner  : LevelSchema
    ...     weight: Optional[int]
    ...
    >>> print(OuterSchema.prompt()) # doctest:+NORMALIZE_WHITESPACE
    Fill in **valid JSON** for the fields below. – **Outer level in a factorial design**
    <BLANKLINE>
    Rules
    - inner  – Single level in a factorial design
      • LevelSchema – Single level in a factorial design
      - inner.name  – The concrete value (e.g. "red")
        • string
          (ex: "example")
      - inner.weight  – Optional repetition weight
        • (Optional) Key can be *missing* or:
          - integer
          - None
    - weight  – Optional repetition weight
      • (Optional) Key can be *missing* or:
        - integer
        - None
    <BLANKLINE>
    Example:
    {
      "inner": {
        "name": "example",
        "weight": 42
      },
      "weight": 42
    }
    <BLANKLINE>
    Return **only** the JSON object — no code-fences, no comments.

    >>> from typing import List
    >>> @dataclass
    ... class OuterSchema(AnnotatedSchema):
    ...     '''
    ...     Outer level in a factorial design
    ...
    ...     inner: List of single level in a factorial design
    ...     weight: Optional repetition weight
    ...     '''
    ...     inner: List[LevelSchema]
    ...     weight: Optional[int]
    ...
    >>> print(OuterSchema.prompt())
    Fill in **valid JSON** for the fields below. – **Outer level in a factorial design**
    <BLANKLINE>
    Rules
    - inner  – List of single level in a factorial design
      • list of LevelSchema – Single level in a factorial design
      - inner[].name  – The concrete value (e.g. "red")
        • string
          (ex: "example")
      - inner[].weight  – Optional repetition weight
        • (Optional) Key can be *missing* or:
          - integer
          - None
    - weight  – Optional repetition weight
      • (Optional) Key can be *missing* or:
        - integer
        - None
    <BLANKLINE>
    Example:
    {
      "inner": [
        {
          "name": "example",
          "weight": 42
        }
      ],
      "weight": 42
    }
    <BLANKLINE>
    Return **only** the JSON object — no code-fences, no comments.


    >>> @dataclass
    ... class Deep(AnnotatedSchema):
    ...     '''
    ...     Deep schema
    ...     id: deepest
    ...     '''
    ...     id: int

    >>> @dataclass
    ... class Outer(AnnotatedSchema):
    ...     '''
    ...     Outer schema
    ...     nest : crazy types
    ...     '''
    ...     nest: Union[
    ...         List[Deep],
    ...         Optional[Tuple[Deep, int]],
    ...         Union[Tuple[Deep, List[Deep]], List[Deep]]
    ...     ]

    >>> print(Outer.prompt()) # doctest: +NORMALIZE_WHITESPACE
    Fill in **valid JSON** for the fields below. – **Outer schema**
    <BLANKLINE>
    Rules
    - nest  – crazy types
      • choose **one** of:
        1. list of Deep – Deep schema
          - nest[].id  – deepest
            • integer
              (ex: 42)
        2. tuple ⟨el[0] Deep – Deep schema, el[1] integer⟩
          - nest[0].id  – deepest
            • integer
              (ex: 42)
        3. <class 'NoneType'>
        4. tuple ⟨el[0] Deep – Deep schema, el[1] list of Deep – Deep schema⟩
          - nest[0].id  – deepest
            • integer
              (ex: 42)
    <BLANKLINE>
    Example:
    {
      "nest": [
        {
          "id": 42
        }
      ]
    }
    <BLANKLINE>
    Return **only** the JSON object — no code-fences, no comments.
    """

    # parsed once per subclass -----------------------------------------
    _doc_header: str          = ""
    _field_notes: dict[str, str] = {}

    # -----------------------  meta-processing  ------------------------
    def __init_subclass__(cls):
        super().__init_subclass__()          # keep Schema setup first

        doc = inspect.getdoc(cls) or ""
        if not doc:
            return                           # nothing to parse

        lines = [l.rstrip() for l in doc.splitlines()]

        # header  = first non-blank line
        for ln in lines:
            if ln.strip():
                cls._doc_header = ln.strip()
                break

        # “field: note”  or  “field – note”
        cls._field_notes = {}
        for ln in lines:
            if ":" in ln:
                field, note = ln.split(":", 1)
                cls._field_notes[field.strip()] = note.strip()
            elif "–" in ln:
                field, note = ln.split("–", 1)
                cls._field_notes[field.strip()] = note.strip()

    # ------------------------  cosmetic layer  ------------------------
    @classmethod
    def _describe_type(cls, typ: Any) -> str:
        """
        Just like Schema._describe_type, but if *typ* itself is a Schema
        that has a doc_header we append “ – header”.
        """
        if cls._is_schema(cls._origin(typ)):
            base = cls._origin(typ)
            hdr  = getattr(base, "_doc_header", "")
            return f"{base.__name__} – {hdr}" if hdr else base.__name__
        # otherwise defer to the parent implementation
        return super()._describe_type(typ)

    @classmethod
    def _describe_optional(cls, typ):
        """
        Pretty-print Optional[T] without appending the field-note again.
        """
        inner = cls._unwrap_optional(typ)

        # get the *type* portion before any " – note"
        inner_desc = cls._describe_type(inner).split(" – ")[0]

        return [
            "• (Optional) Key can be *missing* or:",
            f"  - {inner_desc}",
            "  - None",
        ]

    # ------------------------------------------------------------------
    @classmethod
    def rules(cls, prefix: str = "", _lvl: int = 0) -> list[str]:
        parent_lines = super().rules(prefix, _lvl)

        patched: list[str] = []
        for ln in parent_lines:
            stripped = ln.lstrip()
            if stripped.startswith("- "):

                full_path = stripped[2:].split()[0]  # e.g.  'nest[].id'
                parts = full_path.split(".")

                # ── NEW ── drop leading components that belong to outer schemas
                while parts and parts[0].split("[", 1)[0] not in cls.__annotations__:
                    parts = parts[1:]  # discard 'nest[]'

                note = _note_for_path(cls, parts)
                if note and " – " not in ln:  # no double patch
                    ln = f"{ln}  – {note}"

            patched.append(ln)
        return patched

    # ------------------------------------------------------------------
    @classmethod
    def prompt(cls) -> str:
        lead = "Fill in **valid JSON** for the fields below."
        if cls._doc_header:
            lead += f" – **{cls._doc_header}**"

        rule_block = "\n".join(cls.rules())

        # NEW – render every example (base + extras)
        ex_blocks = "\n\n".join(
            f"Example {i + 1}:\n```json\n{json.dumps(ex, indent=2)}\n```"
            for i, ex in enumerate(cls._all_examples())
        )

        return (
            f"{lead}\n\nRules\n{rule_block}"
            f"\n\n{ex_blocks}"
            "\n\nReturn **only** the JSON object — no code-fences, no comments."
        )
