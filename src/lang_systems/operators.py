"""Generic operator registry and ``@operator`` decorator.

Completely independent of quantum code -- it can be used for anything.

Operator kinds are named by **how the operator is written**, which is usually
what you care about:

    kind          operands   example
    ----------    --------   -------------------------
    nullary       0          ``pi``
    unary_pre     1          ``-x``, ``~x``
    unary_post    1          ``x!``, ``x†``
    binary        2          ``a <-> b``
    ternary       3          ``a ? b : c``
    callable      any        ``f(a, b, c)``
    matrix        0          a gate factory returning an ``np.ndarray``

``kind`` is optional and defaults to ``"callable"``. Common synonyms are
accepted (``"unary"``/``"prefix"`` -> ``unary_pre``, ``"infix"`` -> ``binary``,
...), so existing code keeps working. As a shortcut you can write
``@operator.binary("CNOT")`` instead of ``@operator(kind="binary", name="CNOT")``.

:class:`OperatorRegistry` provides the same decorators in an isolated namespace,
which is useful in tests or when importing only a few named operators.
"""

from contextlib import contextmanager
from functools import wraps
from importlib import import_module
from types import ModuleType
from typing import Any, Callable, Iterator, Literal
import ast
import inspect
import keyword
import re


OperatorKind = Literal[
    "nullary",
    "unary_pre",
    "unary_post",
    "binary",
    "ternary",
    "callable",
    "matrix",
    "bracket",
]

#: Every accepted kind, in a convenient order for docs/validation.
OPERATOR_KINDS: tuple[str, ...] = (
    "nullary",
    "unary_pre",
    "unary_post",
    "binary",
    "ternary",
    "callable",
    "matrix",
    "bracket",
)

#: Friendly synonyms, so the old names and obvious variations still work.
_KIND_ALIASES: dict[str, str] = {
    "constant": "nullary",
    "value": "nullary",
    "zero": "nullary",
    "unary": "unary_pre",
    "prefix": "unary_pre",
    "pre": "unary_pre",
    "postfix": "unary_post",
    "post": "unary_post",
    "infix": "binary",
    "nary": "callable",
    "any": "callable",
    "func": "callable",
    "gate": "matrix",
    "gate_matrix": "matrix",
    "braket": "bracket",
    "delimiter": "bracket",
}

#: A ``{}`` in a bracket pattern marks where one operand appears in source.
BRACKET_PLACEHOLDER = "{}"


#: Operators that the language knows syntactically even before their runtime
#: implementation is imported.  Keeping these names here means ``parse()`` does
#: not have to import the quantum simulator just to recognise ``a <-> b``.
BUILTIN_OPERATOR_KINDS: dict[str, str] = {
    "<->": "binary",
    "<-i->": "binary",
    "<-s->": "binary",
    "h": "unary_pre",
    "~[x]": "unary_pre",
    "~[y]": "unary_pre",
    "~[z]": "unary_pre",
    "idm": "unary_pre",
    "s": "unary_pre",
    "sdg": "unary_pre",
    "t": "unary_pre",
    "tdg": "unary_pre",
}


def normalize_kind(kind: str) -> str:
    """Return the canonical kind name for `kind` (or raise ``ValueError``)."""
    if not isinstance(kind, str):
        raise ValueError(f"Operator kind must be a string, got {type(kind).__name__}.")
    key = _KIND_ALIASES.get(kind.strip().lower(), kind.strip().lower())
    if key not in OPERATOR_KINDS:
        raise ValueError(
            f"Unknown operator kind {kind!r}. Choose one of: {', '.join(OPERATOR_KINDS)}."
        )
    return key


def bracket_parts(pattern: str) -> tuple[str, ...] | None:
    """Split a bracket pattern on ``{}`` placeholders.

    ``"|{}>"`` becomes ``("|", ">")``, ``"<{}|{}>"`` becomes
    ``("<", "|", ">")``. ``None`` means `pattern` is not a bracket pattern.
    """
    if not isinstance(pattern, str) or BRACKET_PLACEHOLDER not in pattern:
        return None
    parts = tuple(pattern.split(BRACKET_PLACEHOLDER))
    if len(parts) < 2 or not any(parts):
        return None
    return parts


# Global operator registry
_global_operators: dict[str, Callable] = {}
_operator_metadata: dict[str, dict[str, Any]] = {}

# ``OperatorRegistry.capture()`` temporarily points decorator registrations at
# a private registry, so importing a module can be isolated from global state.
_registration_stack: list[tuple[dict[str, Callable], dict[str, dict[str, Any]]]] = []


def _registration_target() -> tuple[dict[str, Callable], dict[str, dict[str, Any]]]:
    if _registration_stack:
        return _registration_stack[-1]
    return _global_operators, _operator_metadata


def _mark(obj, name: str, kind: str, requires_instance: bool) -> None:
    obj._operator_name = name
    obj._is_operator = True
    obj._requires_instance = requires_instance
    obj._kind = kind


_PYTHON_OPERATORS = frozenset({
    "+", "-", "*", "**", "/", "//", "%", "@", "<<", ">>", "&", "|", "^", "~",
    "<", ">", "<=", ">=", "==", "!=", "<>", ":=", "=",
    "+=", "-=", "*=", "/=", "%=", "@=", "&=", "|=", "^=", "<<=", ">>=", "**=", "//=",
    "(", ")", "[", "]", "{", "}", ",", ":", ".", ";", "...", "->",
})
_PYTHON_SYNTAX_NAMES = _PYTHON_OPERATORS | frozenset(keyword.kwlist)


def _register(name: str, kind: str, func: Callable) -> Callable:
    target, metadata = _registration_target()
    return _register_into(target, metadata, name, kind, func)


def _register_into(
    target: dict[str, Callable],
    metadata: dict[str, dict[str, Any]],
    name: str,
    kind: str,
    func: Callable,
) -> Callable:
    if not isinstance(name, str) or not name:
        raise ValueError("Operator name must be a non-empty string")
    if name in _PYTHON_SYNTAX_NAMES:
        raise ValueError(f"Operator name {name!r} is reserved Python syntax")
    kind = normalize_kind(kind)

    # Handle staticmethod/classmethod
    if isinstance(func, (staticmethod, classmethod)):
        func = func.__func__

    # For matrix kind, call once now to validate and cache the matrix.
    if kind == "matrix":
        import numpy as np

        try:
            inspect.signature(func).bind()
        except TypeError as error:
            parameters = list(inspect.signature(func).parameters.values())
            if not parameters or parameters[0].name != "self":
                raise ValueError(
                    f"Matrix operator '{name}' must be callable without arguments"
                ) from error
            # Instance method: defer the call until an instance is supplied.
            def instance_wrapper(instance):
                return func(instance)

            _mark(instance_wrapper, name, kind, True)
            target[name] = instance_wrapper
            metadata[name] = {"kind": kind, "requires_instance": True}
            _mark(func, name, kind, True)
            return func

        matrix = func()
        if not isinstance(matrix, np.ndarray):
            raise ValueError(f"Operator function '{name}' must return a numpy array")

        _mark(func, name, kind, False)
        func._operator_matrix = matrix  # type: ignore[attr-defined]
        target[name] = func
        metadata[name] = {"kind": kind, "requires_instance": False}
        return func

    # unary_pre / unary_post / binary / ternary / nullary / callable
    code = getattr(func, "__code__", None)
    requires_instance = bool(code and "self" in code.co_varnames)
    _mark(func, name, kind, requires_instance)
    target[name] = func
    metadata[name] = {"kind": kind, "requires_instance": requires_instance}

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _mark(wrapper, name, kind, requires_instance)
    return wrapper


def _register_bracket_into(
    target: dict[str, Callable],
    metadata: dict[str, dict[str, Any]],
    pattern: str,
    func: Callable,
    kind: OperatorKind = "bracket",
) -> Callable:
    parts = bracket_parts(pattern)
    if parts is None:
        raise ValueError(
            "Bracket operator patterns need at least one '{}' placeholder and "
            "some literal bracket text"
        )
    result = _register_into(target, metadata, pattern, normalize_kind(kind), func)
    entry = metadata.setdefault(pattern, {})
    entry["is_bracket"] = True
    entry["bracket_parts"] = parts
    result._is_bracket_operator = True  # type: ignore[attr-defined]
    result._bracket_parts = parts  # type: ignore[attr-defined]
    return result


def register_bracket(
    pattern: str,
    func: Callable,
    *,
    kind: OperatorKind = "bracket",
) -> Callable:
    """Register `func` as a delimiter-style operator pattern.

    Use ``{}`` for each operand, e.g. ``"|{}>"`` or ``"<{}|{}>"``.
    """
    target, metadata = _registration_target()
    return _register_bracket_into(target, metadata, pattern, func, kind)


class operator:
    """Register a callable under an arbitrary name.

    The name can be any non-reserved string: capital letters, symbols, unicode,
    etc. Python keywords and operator tokens cannot be registered because they
    would hijack ordinary Python syntax. Works as a function, method, or class
    decorator (or with any Python callable).

    Examples::

        @operator(kind="matrix", name="h")
        def h_MAT(): ...

        @operator.binary("<->")
        def swap(control, target): ...

        @operator.unary_post("†")
        def adjoint(qubit): ...

        @operator.bracket("<{}|{}>")
        def inner(bra, ket): ...

    ``kind`` may be omitted (default ``"callable"``).
    """

    def __init__(
        self,
        name: str,
        kind: OperatorKind = "callable",
        *,
        bracket: bool = False,
    ):
        self.name = name
        self.kind = kind
        self.is_bracket = bracket

    def __call__(self, func: Callable) -> Callable:
        if self.is_bracket:
            return register_bracket(self.name, func, kind=self.kind) #type: ignore
        return _register(self.name, self.kind, func) #type: ignore

    # ─── Convenience constructors ───

    @classmethod
    def nullary(cls, name: str) -> "operator":
        return cls(name, "nullary")

    @classmethod
    def unary_pre(cls, name: str) -> "operator":
        return cls(name, "unary_pre")

    @classmethod
    def unary_post(cls, name: str) -> "operator":
        return cls(name, "unary_post")

    @classmethod
    def binary(cls, name: str) -> "operator":
        return cls(name, "binary")

    @classmethod
    def ternary(cls, name: str) -> "operator":
        return cls(name, "ternary")

    @classmethod
    def callable(cls, name: str) -> "operator": 
        return cls(name, "callable")

    @classmethod
    def matrix(cls, name: str) -> "operator":
        return cls(name, "matrix")

    @classmethod
    def bracket(cls, pattern: str, *, kind: OperatorKind = "bracket") -> "operator":
        """Register a delimiter pattern where each ``{}`` holds one operand."""
        return cls(pattern, kind, bracket=True)


# ─── Static operator declarations ───


_DECLARATION_RE = re.compile(r"@(?P<decorator>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*){0,2})\s*\((?P<args>[^)]*)\)")
_IMPORT_OPERATOR_RE = re.compile(r"from\s+[\w.]+\s+import\s+operator(?:\s+as\s+(?P<alias>\w+))?")
_IMPORT_MODULE_RE = re.compile(r"import\s+lang_systems\.operators(?:\s+as\s+(?P<alias>\w+))?")


def _declaration_arguments(args: str) -> tuple[str | None, str]:
    """Read ``name`` and ``kind`` from an ``@operator(...)`` argument list."""
    try:
        call = ast.parse(f"operator({args})", mode="eval").body
    except SyntaxError:
        return None, "callable"
    name: str | None = None
    kind = "callable"
    if isinstance(call, ast.Call):
        if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
            name = call.args[0].value
        for keyword in call.keywords:
            if not isinstance(keyword.value, ast.Constant):
                continue
            if keyword.arg == "name" and isinstance(keyword.value.value, str):
                name = keyword.value.value
            elif keyword.arg == "kind" and isinstance(keyword.value.value, str):
                kind = keyword.value.value
    return name, kind


def _register_placeholder_operator(name: str, kind: str) -> None:
    """Register a parse-only stand-in for an operator declared in source."""
    canonical = normalize_kind(kind)
    if canonical == "matrix":
        def placeholder(): #type: ignore
            import numpy as np
            return np.zeros((2, 2), dtype=complex)
    else:
        def placeholder(*args, **kwargs):
            return None
    if canonical == "bracket":
        register_bracket(name, placeholder)
        return
    operator(name, kind)(placeholder) #type: ignore


def register_declared_operators(source: str) -> None:
    """Register operators declared in ``source`` without executing the module.

    A text scan is used instead of :func:`ast.parse` because the file declaring
    an operator may itself contain Quantar syntax that Python cannot parse.
    Only decorator arguments need to be valid Python literals.
    """
    aliases = {"operator"}
    modules = {"operators"}
    for match in _IMPORT_OPERATOR_RE.finditer(source):
        aliases.add(match.group("alias") or "operator")
    for match in _IMPORT_MODULE_RE.finditer(source):
        modules.add(match.group("alias") or "operators")

    for match in _DECLARATION_RE.finditer(source):
        parts = match.group("decorator").split(".")
        kind: str | None = None
        if len(parts) == 1 and parts[0] in aliases:
            pass
        elif len(parts) == 2 and parts[0] in aliases:
            kind = parts[1]
        elif len(parts) == 2 and parts[0] in modules and parts[1] == "operator":
            pass
        elif len(parts) == 3 and parts[0] in modules and parts[1] == "operator":
            kind = parts[2]
        else:
            continue

        name, default_kind = _declaration_arguments(match.group("args"))
        if not name:
            continue
        _register_placeholder_operator(name, kind or default_kind)


# ─── Public API ───

def get_global_operator(name: str):
    """Retrieve a registered operator by name.

    Returns ``(operator, kind)``; ``kind`` is one of :data:`OPERATOR_KINDS`, or
    ``None`` when the name is not registered.
    """
    op = _global_operators.get(name)
    if op is None:
        return None, None
    return op, getattr(op, "_kind", "callable")


def get_operator(name: str):
    """Alias for get_global_operator (backwards compatibility)."""
    return get_global_operator(name)


def get_operator_kind(name: str) -> str | None:
    """Get the canonical kind of a registered operator."""
    meta = _operator_metadata.get(name)
    return meta.get("kind") if meta else None


def get_known_operator_kind(name: str) -> str | None:
    """Get the kind of a registered or built-in operator."""
    kind = get_operator_kind(name)
    if kind is not None:
        return kind
    return BUILTIN_OPERATOR_KINDS.get(name)


def is_known_operator(name: str) -> bool:
    """Return whether `name` is a registered or built-in operator."""
    return name in _global_operators or name in BUILTIN_OPERATOR_KINDS


def is_bracket_operator(name: str) -> bool:
    """Return whether `name` is a registered delimiter-style pattern."""
    return bool(_operator_metadata.get(name, {}).get("is_bracket"))


def get_bracket_parts(name: str) -> tuple[str, ...] | None:
    """Return the literal pieces around a bracket pattern's ``{}`` slots."""
    meta = _operator_metadata.get(name)
    if not meta or not meta.get("is_bracket"):
        return None
    parts = meta.get("bracket_parts")
    return tuple(parts) if parts else bracket_parts(name)


def list_bracket_operators() -> list[str]:
    """List registered delimiter-style operator patterns."""
    return [name for name in _global_operators if is_bracket_operator(name)]


def get_operator_requires_instance(name: str) -> bool:
    """Check if an operator requires an instance (is a method)."""
    meta = _operator_metadata.get(name)
    return meta.get("requires_instance", False) if meta else False


def get_global_operator_matrix(name: str):
    """Retrieve a registered matrix operator by name."""
    op, kind = get_global_operator(name)
    if kind == "matrix" and callable(op):
        if _operator_metadata.get(name, {}).get("requires_instance"):
            return op
        return op()
    return None


def get_operator_matrix(name: str):
    """Alias for get_global_operator_matrix (backwards compatibility)."""
    return get_global_operator_matrix(name)


def list_operators(kind: str | None = None) -> list[str]:
    """List registered operator names, optionally filtered to one `kind`."""
    if kind is None:
        return list(_global_operators.keys())
    wanted = normalize_kind(kind)
    return [name for name in _global_operators if _operator_metadata.get(name, {}).get("kind") == wanted]

def list_known_operators() -> dict[str, str]:
    """Return every registered and built-in operator mapped to its kind."""
    names = set(list_operators()) | set(BUILTIN_OPERATOR_KINDS)
    return {
        name: get_known_operator_kind(name) or "callable"
        for name in names
    }


class OperatorRegistry:
    """An isolated collection of operators.

    Instances are handy in tests, or when you want to work with a few named
    operators without touching the process-wide registry used by ``qtr``.

    Example::

        ops = OperatorRegistry()

        @ops.binary("<~>")
        def reverse(left, right):
            return right, left

        reverse = ops["<~>"]        # import one operator
        ops.apply("<~>", 1, 2)      # (2, 1)
        ops.import_global("h")      # copy specific global operators

    Decorators run inside :meth:`capture` register here instead of globally::

        with ops.capture():
            import my_quantar_operators
    """

    def __init__(self) -> None:
        self._operators: dict[str, Callable] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    # ─── Registration ───

    def register(
        self,
        name: str,
        func: Callable,
        *,
        kind: OperatorKind = "callable",
    ) -> Callable:
        """Register `func` under `name` in this registry."""
        return _register_into(self._operators, self._metadata, name, kind, func)

    def operator(self, name: str, kind: OperatorKind = "callable") -> Callable[[Callable], Callable]:
        """Return a decorator that registers a callable under `name`."""
        def decorate(func: Callable) -> Callable:
            return self.register(name, func, kind=kind)
        return decorate

    def nullary(self, name: str) -> Callable[[Callable], Callable]:
        return self.operator(name, "nullary")

    def unary_pre(self, name: str) -> Callable[[Callable], Callable]:
        return self.operator(name, "unary_pre")

    def unary_post(self, name: str) -> Callable[[Callable], Callable]:
        return self.operator(name, "unary_post")

    def binary(self, name: str) -> Callable[[Callable], Callable]:
        return self.operator(name, "binary")

    def ternary(self, name: str) -> Callable[[Callable], Callable]:
        return self.operator(name, "ternary")

    def callable(self, name: str) -> Callable[[Callable], Callable]:
        return self.operator(name, "callable")

    def matrix(self, name: str) -> Callable[[Callable], Callable]:
        return self.operator(name, "matrix")

    def bracket(
        self,
        pattern: str,
        *,
        kind: OperatorKind = "bracket",
    ) -> Callable[[Callable], Callable]:
        """Return a decorator for a delimiter pattern with ``{}`` operand slots."""
        def decorate(func: Callable) -> Callable:
            return _register_bracket_into(self._operators, self._metadata, pattern, func, kind)
        return decorate

    # ─── Lookup ───

    def get(self, name: str) -> Callable | None:
        """Return the callable registered under `name`, or ``None``."""
        return self._operators.get(name)

    def get_operator(self, name: str) -> tuple[Callable | None, str | None]:
        """Return ``(operator, kind)`` like :func:`get_global_operator`."""
        op = self.get(name)
        if op is None:
            return None, None
        return op, self.kind(name)

    def __getitem__(self, name: str) -> Callable:
        op = self.get(name)
        if op is None:
            raise KeyError(f"Unknown operator: {name!r}")
        return op

    def __contains__(self, name: object) -> bool:
        return name in self._operators

    def __iter__(self) -> Iterator[str]:
        return iter(self._operators)

    def __len__(self) -> int:
        return len(self._operators)

    def __repr__(self) -> str:
        return f"OperatorRegistry({len(self._operators)} operator(s))"

    def kind(self, name: str) -> str | None:
        """Return the canonical kind of a registered operator."""
        meta = self._metadata.get(name)
        return meta.get("kind") if meta else None

    def requires_instance(self, name: str) -> bool:
        """Return whether an operator is an instance method."""
        meta = self._metadata.get(name)
        return bool(meta and meta.get("requires_instance"))

    def is_bracket(self, name: str) -> bool:
        """Return whether `name` is a delimiter-style pattern."""
        return bool(self._metadata.get(name, {}).get("is_bracket"))

    def bracket_parts(self, name: str) -> tuple[str, ...] | None:
        """Return the literal pieces around a bracket pattern's slots."""
        meta = self._metadata.get(name)
        if not meta or not meta.get("is_bracket"):
            return None
        parts = meta.get("bracket_parts")
        return tuple(parts) if parts else bracket_parts(name)

    def names(self, kind: str | None = None) -> list[str]:
        """List names in this registry, optionally filtered to one `kind`."""
        if kind is None:
            return list(self._operators)
        wanted = normalize_kind(kind)
        return [name for name in self._operators if self.kind(name) == wanted]

    def get_matrix(self, name: str):
        """Return the matrix for a registered matrix operator, if available."""
        op = self.get(name)
        if self.kind(name) != "matrix" or not callable(op):
            return None
        if self.requires_instance(name):
            return op
        return op()

    def apply(self, name: str, *args, **kwargs) -> Any:
        """Execute a registered operator with the given arguments."""
        op = self.get(name)
        if op is None:
            raise ValueError(f"Unknown operator: '{name}'")
        if self.kind(name) == "matrix" and callable(op):
            try:
                return op()
            except TypeError as error:
                if "missing 1 required positional argument: 'self'" in str(error):
                    raise ValueError(f"Operator '{name}' is a method and requires an instance") from error
                raise
        return op(*args, **kwargs)

    # ─── Importing specific operators ───

    def _copy_registered(
        self,
        target: dict[str, Callable],
        metadata: dict[str, dict[str, Any]],
        name: str,
    ) -> Callable:
        op = target.get(name)
        if op is None:
            raise KeyError(f"Unknown operator: {name!r}")
        meta = metadata.get(name, {})
        self._operators[name] = op
        entry = dict(meta)
        entry.setdefault("kind", getattr(op, "_kind", "callable"))
        entry.setdefault("requires_instance", False)
        self._metadata[name] = entry
        return op

    def _copy_marked(self, name: str, value: Callable) -> Callable:
        declared_kind = getattr(value, "_kind", None)
        self._operators[name] = value
        entry: dict[str, Any] = {
            "kind": normalize_kind(declared_kind) if declared_kind else "callable",
            "requires_instance": bool(getattr(value, "_requires_instance", False)),
        }
        if getattr(value, "_is_bracket_operator", False):
            entry["is_bracket"] = True
            entry["bracket_parts"] = tuple(getattr(value, "_bracket_parts", ()))
        self._metadata[name] = entry
        return value

    def import_global(self, *names: str) -> tuple[Callable, ...]:
        """Copy named operators from the process-wide registry into this one."""
        for name in names:
            if name not in _global_operators:
                raise KeyError(f"Unknown global operator: {name!r}")
            self._copy_registered(_global_operators, _operator_metadata, name)
        return tuple(self._operators[name] for name in names)

    def import_module(self, module: str | ModuleType, *names: str) -> tuple[Callable, ...]:
        """Copy operators registered by `module` into this registry.

        ``names`` are registration names (for example ``"h"``), not necessarily
        Python attribute names. Decorators executed during the import are
        captured, so importing does not mutate the process-wide registry. With
        no ``names``, all operators registered by the module are copied.
        """
        staged = OperatorRegistry()
        staged._operators.update(_global_operators)
        staged._metadata.update({key: dict(value) for key, value in _operator_metadata.items()})
        baseline = {name: id(op) for name, op in staged._operators.items()}

        with staged.capture():
            loaded = import_module(module) if isinstance(module, str) else module
        if not isinstance(loaded, ModuleType):
            raise TypeError(
                "import_module() expects a module name or module object, "
                f"got {type(loaded).__name__}"
            )

        marked = {
            getattr(value, "_operator_name", attribute): value
            for attribute, value in vars(loaded).items()
            if getattr(value, "_is_operator", False)
        }
        if names:
            selected = tuple(names)
        else:
            selected = tuple(
                name for name, op in staged._operators.items()
                if baseline.get(name) != id(op)
            )
            selected += tuple(name for name in marked if name not in selected)

        for name in selected:
            if name in staged._operators:
                self._copy_registered(staged._operators, staged._metadata, name)
            elif name in marked:
                self._copy_marked(name, marked[name])
            else:
                raise AttributeError(f"Module {loaded.__name__!r} has no operator {name!r}")
        return tuple(self._operators[name] for name in selected)

    @contextmanager
    def capture(self) -> Iterator["OperatorRegistry"]:
        """Route ``@operator`` registrations into this registry temporarily."""
        _registration_stack.append((self._operators, self._metadata))
        try:
            yield self
        finally:
            _registration_stack.pop()

    def clear(self) -> None:
        """Remove all operators from this registry."""
        self._operators.clear()
        self._metadata.clear()


def load_operators(*modules: str | ModuleType) -> tuple[ModuleType, ...]:
    """Import operator modules so their registrations take effect.

    Strings are imported with :func:`importlib.import_module`; module objects
    are accepted as-is. Call this before :func:`lang_systems.parse` when the
    source uses operators defined by another package::

        from lang_systems import parse
        from lang_systems.operators import load_operators

        load_operators("my_package.quantar_ops")
        tree = parse("a >< b\\n")

    Returns the loaded modules in the order given.
    """
    loaded = []
    for module in modules:
        if isinstance(module, str):
            loaded.append(import_module(module))
        elif isinstance(module, ModuleType):
            loaded.append(module)
        else:
            raise TypeError(
                "load_operators() expects module names or module objects, "
                f"got {type(module).__name__}"
            )
    return tuple(loaded)


def clear_operators():
    """Clear all registered operators (useful for testing)."""
    _global_operators.clear()
    _operator_metadata.clear()


def apply_operator(name: str, *args, **kwargs) -> Any:
    """Execute a registered operator by name with the given arguments.

    Raises:
        ValueError: If the operator is unknown or needs an instance.
    """
    op, kind = get_operator(name)
    if op is None:
        raise ValueError(f"Unknown operator: '{name}'")

    meta = _operator_metadata.get(name, {})
    if meta.get("kind") == "matrix" and callable(op):
        try:
            return op()
        except TypeError as error:
            if "missing 1 required positional argument: 'self'" in str(error):
                raise ValueError(f"Operator '{name}' is a method and requires an instance") from error
            raise

    return op(*args, **kwargs)


__all__ = [
    "operator",
    "register_bracket",
    "OperatorRegistry",
    "OperatorKind",
    "OPERATOR_KINDS",
    "normalize_kind",
    "BUILTIN_OPERATOR_KINDS",
    "BRACKET_PLACEHOLDER",
    "bracket_parts",
    "is_bracket_operator",
    "get_bracket_parts",
    "list_bracket_operators",
    "get_operator",
    "get_global_operator",
    "get_global_operator_matrix",
    "get_operator_kind",
    "get_known_operator_kind",
    "is_known_operator",
    "get_operator_requires_instance",
    "get_operator_matrix",
    "list_operators",
    "load_operators",
    "clear_operators",
    "register_declared_operators",
    "apply_operator",
    "list_known_operators",
]
