from dataclasses import dataclass, fields
from typing import get_type_hints, get_origin, get_args, Any, Callable, Tuple, Union
import json
from mate_strategy.rules import Rule


def constraint(path: str, desc: str, *, fix: Callable[[dict], None] | None = None):
    def wrap(fn):
        fn.__constraint_info__ = (path, desc, fix or (lambda d: None))
        return fn

    return wrap


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
        - x must be a integer
        - y must be a string
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
          Expected: "y" must be string
        <BLANKLINE>
        Rules:
        - x must be a integer
        - y must be a string
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
        - x must be a number between 20 and 100
        - y must be one of 'green', 'red', 'blue'
        <BLANKLINE>
        Example:
        {
          "x": 60,
          "y": "blue"
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
        - x must be a number between 20 and 100
        - y must be one of 'green', 'red', 'blue'
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
        ...         return "must be between 0 and 1 or be a string"
        ...     @classmethod
        ...     def example(cls):
        ...         return random.choice([random.random(), "example"])
        ...     @classmethod
        ...     def validate(cls, v):
        ...         return isinstance(v, str) or 0 <= v <= 1

        we can use this in the schema just as any other rule:
        >>> @dataclass
        ... class MySchema(Schema):
        ...     x: MyRule

        >>> print(MySchema.prompt()) # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - x must be between 0 and 1 or be a string
        <BLANKLINE>
        Example:
        {
          "x": "example"
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.

        >>> MySchema.validate_with_error({"x": 0.5})
        (True,)

        >>> MySchema.validate_with_error({"x": "example"})
        (True,)

        >>> MySchema.validate_with_error({"x": 2})
        (False, '"x" is invalid.', '"x" must be between 0 and 1 or be a string')

        ... for all, we can use List, Tuples and Unions
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
        - x must be a list. An element of the list must be between 0 and 1 or be a string
        - y must be a tuple (el[0] must be a integer, el[1] must be a string)
        - z must satisfy ONE of: must be a integer or must be a string
        <BLANKLINE>
        Example:
        {
          "x": [
            0.24489185380347622
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
        - a MySchema
          - a.x must be a list. An element of the list must be between 0 and 1 or be a string
          - a.y must be a tuple (el[0] must be a integer, el[1] must be a string)
          - a.z must satisfy ONE of: must be a integer or must be a string
        - b must be a string
        <BLANKLINE>
        Example:
        {
          "a": {
            "x": [
              0.7364712141640124
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
        (False, '"y[1]" must be string', '"y[1]" must be string')

        ... and get a repair prompt:
        >>> print(OuterSchema.repair_prompt({"a": {"z": [0.5], "y": [42, 42]}, "b": "example"}))  # doctest: +NORMALIZE_WHITESPACE
        The JSON below is invalid.
        <BLANKLINE>
        Problem:
        - "x" is missing.
          Expected:
         "x" must be present.
        <BLANKLINE>
        Rules:
        - a MySchema
          - a.x must be a list. An element of the list must be between 0 and 1 or be a string
          - a.y must be a tuple (el[0] must be a integer, el[1] must be a string)
          - a.z must satisfy ONE of: must be a integer or must be a string
        - b must be a string
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
        - x must be a integer
        - y must be a integer
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
        - x must be a integer
        - y must be a integer
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
        - x must be a integer
        - y must be a integer
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
    def _is_union(t):
        return get_origin(t) is Union

    # ──────────────────────────────────────────────────────────────────────z

    @classmethod
    def _describe_type(cls, typ: Any) -> str:
        # scalar Rule ------------------------------------------------------
        if cls._is_rule(typ):
            return typ.describe()

        # primitives
        if typ in (str, int, float, bool):
            return \
                {str: "must be a string", int: "must be a integer", float: "must be a float", bool: "must be boolean"}[
                    typ]

        # list
        if cls._is_list(typ):
            elem = get_args(typ)[0] if get_args(typ) else Any
            return f"must be a list. An element of the list {cls._describe_type(elem)}"

        # tuple
        if cls._is_tuple(typ):
            _descriptions = [cls._describe_type(t) for t in get_args(typ)]
            _p = [f'el[{i}] {_d}' for i, _d in enumerate(_descriptions)]
            parts = ", ".join(_p)
            return f"must be a tuple ({parts})"

        # union
        if cls._is_union(typ):
            return "must satisfy ONE of: " + " or ".join(
                cls._describe_type(t) for t in get_args(typ)
            )

        if cls._is_schema(cls._origin(typ)):
            return cls._origin(typ).__name__

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

    @classmethod
    def rules(cls, prefix: str = "") -> list[str]:
        """
        Build a fully nested rule list, including:
            • per-field rules (scalar, list, tuple, nested Schema, custom rule)
            • @constraint-decorated cross-field rules, shown right after the
              field block they reference.
        """
        lines: list[str] = []

        # ── 1. bucket constraints by their top-level key ──────────────────
        constraints_by_head: dict[str, list[tuple[str, str]]] = {}
        for path, desc, _, _ in cls._declared_constraints:
            head, *rest = path.split(".", 1)
            tail = rest[0] if rest else ""  # "" means constraint on the head itself
            constraints_by_head.setdefault(head, []).append((tail, desc))

        # ── 2. walk dataclass fields in declared order ────────────────────
        for name, typ in cls._field_types().items():
            full = f"{prefix}{name}"

            def _recurse(inner, path):
                lines.extend(f"  {l}" for l in inner.rules(path))

            # 2-a  custom _infer_rule() override?
            custom = cls._infer_rule(name, typ)
            if isinstance(custom, Schema):
                lines.append(f"- {full} is a {custom.__class__.__name__}")
                lines.extend(f"  {l}" for l in custom.rules(full + "."))
            else:
                # 2-b  normal per-field rule
                lines.append(f"- {full} {cls._describe_type(typ)}")

                # recurse into collections that contain Schemas
                if cls._is_list(typ):
                    elem = get_args(typ)[0]
                    if cls._is_schema(cls._origin(elem)):
                        lines.extend(
                            f"  {l}" for l in cls._origin(elem).rules(full + "[].")
                        )
                elif cls._is_tuple(typ):
                    for i, part in enumerate(get_args(typ)):
                        if cls._is_schema(cls._origin(part)):
                            lines.extend(
                                f"  {l}" for l in cls._origin(part).rules(f"{full}[{i}].")
                            )
                elif cls._is_union(typ):
                    for part in get_args(typ):
                        if cls._is_schema(cls._origin(part)):
                            _recurse(cls._origin(part), f"{full}[].")
                elif cls._is_schema(cls._origin(typ)):
                    # field *is* a Schema → recurse
                    lines.extend(f"  {l}" for l in cls._origin(typ).rules(full + "."))

            # 2-c  add any constraints that start with this field
            for tail, desc in constraints_by_head.get(name, []):
                indent = "  " if tail else ""
                path = f"{full}.{tail}" if tail else full
                lines.append(f"{indent}- {path} {desc}")

        # ── 3. constraints on the *whole object* (no head) ────────────────
        for tail, desc in constraints_by_head.get("", []):
            lines.append(f"- {desc}")  # path == whole object; indent 0

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
    def prompt(cls) -> str:
        return (
                "Fill in **valid JSON** for the fields below.\n\nRules\n"
                + "\n".join(cls.rules())
                + "\n\nExample:\n"
                + json.dumps(cls.example(), indent=2)
                + "\n\nReturn **only** the JSON object — no code-fences, no comments."
        )

    # ────────────────────────── validation ─────────────────────────
    @classmethod
    def _validate_value(cls, key: str, val, typ: Any, prefix=""):
        """Return (err, expected) if invalid, else None."""
        full = f"{prefix}{key}"

        # 1 scalar rules
        if cls._is_rule(typ):
            if not typ.validate(val):
                return (
                    f'"{full}" is invalid.',
                    f'"{full}" {typ.describe()}',
                )
            return None

        # 2 lists
        if cls._is_list(typ):
            if not isinstance(val, list):
                return f'"{full}" must be a list, got {type(val).__name__}', f'"{full}" must be list'
            elem_typ = get_args(typ)[0] if get_args(typ) else Any
            for i, v in enumerate(val):
                err = cls._validate_value(f"{key}[{i}]", v, elem_typ, prefix)
                if err:
                    return err
            return None

        # 3 tuples
        if cls._is_tuple(typ):
            if not isinstance(val, (list, tuple)):
                return f'"{full}" must be a tuple, got {type(val).__name__}', f'"{full}" must be tuple'
            parts = get_args(typ)
            if len(val) != len(parts):
                return (
                    f'"{full}" must have length {len(parts)}, got {len(val)}',
                    f'"{full}" must be length-{len(parts)} tuple',
                )
            for i, (v, sub_t) in enumerate(zip(val, parts)):
                err = cls._validate_value(f"{key}[{i}]", v, sub_t, prefix)
                if err:
                    return err
            return None

        # 4 union
        if cls._is_union(typ):
            expected_msgs = []
            for alt in get_args(typ):
                err = cls._validate_value(key, val, alt, prefix)
                if err is None:
                    return None
                expected_msgs.append(err[1])
            return (
                f'"{full}" matches none of the allowed alternatives.',
                " or ".join(expected_msgs),
            )

        # 5 nested Schemas
        base = cls._origin(typ)
        if cls._is_schema(base):
            if not isinstance(val, dict):
                return f'"{full}" must be an object', f'"{full}" must be an object'
            ok, *reason = base.validate_with_error(val)
            return None if ok else reason  # (err, expected)

        # 6 primitives
        if typ is str and not isinstance(val, str):
            return f'"{full}" must be string', f'"{full}" must be string'
        if typ is int and not isinstance(val, int):
            return f'"{full}" must be integer', f'"{full}" must be integer'
        if typ is float and not isinstance(val, (int, float)):
            return f'"{full}" must be float', f'"{full}" must be float'

        return None

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
