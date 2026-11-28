"""Build a Python AST for Quantar source.

Quantar adds a handful of infix operators to Python's syntax.  Reimplementing
CPython's grammar would be wasteful, so this module takes the opposite route:

1. recursively find custom infix expressions in the pegen parse tree;
2. replace each one with a unique sentinel call;
3. hand the resulting text to :func:`ast.parse`;
4. replace the sentinel calls with :class:`QuantumOp` nodes.

The result is ordinary Python AST plus custom :class:`QuantumOp` nodes, which
can then be inspected with :mod:`lang_systems.visitors` or lowered back to calls.
"""

from __future__ import annotations

import ast
import bisect
import token
import tokenize
from dataclasses import dataclass
from types import CodeType
from typing import Sequence
from uuid import uuid4

from lang_systems.operators import BUILTIN_OPERATOR_KINDS, get_known_operator_kind
from lang_systems.quantar import parse
from lang_systems.quantar_tokens import tokenize_source

_SENTINEL_PREFIX = "__quantar_operator_"
_SENTINEL_SUFFIX = "__"


class QuantumOp(ast.expr):
    """A Quantar infix operation, stored as a first-class AST node."""

    _fields = ("operator", "operands", "kind")

    def __init__(
        self,
        operator: str,
        operands: list[ast.expr],
        kind: str = "binary",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.operator = operator
        self.operands = operands
        self.kind = kind


@dataclass
class _Rewrite:
    name: str
    operator: str
    kind: str
    start: tuple[int, int]
    end: tuple[int, int]


@dataclass
class _Rendered:
    """Rewritten text plus the original character offset for every character."""

    text: str
    offsets: list[int]


class _Source:
    """Map token positions (line, column) to source offsets and slices."""

    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.starts: list[int] = []
        offset = 0
        for line in self.lines:
            self.starts.append(offset)
            offset += len(line)

    def offset(self, position: tuple[int, int]) -> int:
        row, column = position
        if row < 1 or row > len(self.starts):
            return len(self.text)
        return self.starts[row - 1] + column

    def slice(self, start: tuple[int, int], end: tuple[int, int]) -> str:
        return self.text[self.offset(start) : self.offset(end)]

    def line(self, row: int) -> str:
        if 1 <= row <= len(self.lines):
            return self.lines[row - 1]
        return ""

    def byte_column(self, position: tuple[int, int]) -> int:
        _, column = position
        return len(self.line(position[0])[:column].encode("utf-8"))

    def position_for_offset(self, offset: int) -> tuple[int, int]:
        """Return a ``(line, UTF-8 byte column)`` pair for a character offset."""
        if not self.starts:
            return 1, 0
        offset = max(0, min(offset, len(self.text)))
        row = max(1, bisect.bisect_right(self.starts, offset))
        start = self.starts[row - 1]
        return row, len(self.text[start:offset].encode("utf-8"))


def _first_token(node) -> tokenize.TokenInfo | None:
    if isinstance(node, tokenize.TokenInfo):
        return node
    if isinstance(node, list):
        for item in node:
            found = _first_token(item)
            if found is not None:
                return found
    return None


def _last_token(node) -> tokenize.TokenInfo | None:
    if isinstance(node, tokenize.TokenInfo):
        return node
    if isinstance(node, list):
        for item in reversed(node):
            found = _last_token(item)
            if found is not None:
                return found
    return None


def _bracket_operator(node) -> tokenize.TokenInfo | None:
    """Return the token if `node` is a bracket-delimited operator expression."""
    if isinstance(node, tokenize.TokenInfo):
        if getattr(node, "_quantar_bracket", False):
            return node
    return None


def _custom_operator(node) -> tokenize.TokenInfo | None:
    """Return the operator token if `node` is a custom binary expression."""
    if not isinstance(node, list) or len(node) != 3:
        return None
    left, middle, right = node
    if not isinstance(middle, tokenize.TokenInfo):
        return None
    if middle.type not in (token.OP, token.NAME):
        return None
    if get_known_operator_kind(middle.string) is None:
        return None
    if left is None or right is None:
        return None
    return middle


def _direct_custom_nodes(node) -> list:
    """Return the outermost custom expressions inside a standard node."""
    found = []

    def visit(value):
        if _custom_operator(value) is not None or _bracket_operator(value) is not None:
            found.append(value)
            return
        if isinstance(value, list):
            for item in value:
                visit(item)

    visit(node)
    return found


def _node_text(node, source: _Source) -> str:
    if isinstance(node, tokenize.TokenInfo):
        return node.string
    if not isinstance(node, list):
        return ""
    first = _first_token(node)
    last = _last_token(node)
    if first is None or last is None:
        return ""
    return source.slice(first.start, last.end)


def _render_piece(text: str, offset: int) -> _Rendered:
    """Build a rendered fragment whose characters map to one source offset."""
    return _Rendered(text, [offset] * len(text))


def _join_rendered(pieces: Sequence[_Rendered]) -> _Rendered:
    return _Rendered(
        "".join(piece.text for piece in pieces),
        [offset for piece in pieces for offset in piece.offsets],
    )


class _SubSource(_Source):
    """A source view for an operand slice, mapped back to original offsets."""

    def __init__(self, text: str, base_offset: int):
        super().__init__(text)
        self.base_offset = base_offset

    def _local_offset(self, position: tuple[int, int]) -> int:
        return super().offset(position)

    def offset(self, position: tuple[int, int]) -> int:
        return self.base_offset + self._local_offset(position)

    def slice(self, start: tuple[int, int], end: tuple[int, int]) -> str:
        return self.text[self._local_offset(start):self._local_offset(end)]


def _render_operand(
    span: tuple[tuple[int, int], tuple[int, int]],
    source: _Source,
    rewrites: list[_Rewrite],
) -> _Rendered:
    text = source.slice(*span)
    if not text.strip():
        return _render_piece(text, source.offset(span[0]))
    view = _SubSource(text, source.offset(span[0]))
    try:
        operand_tree = parse(text)
    except (SyntaxError, tokenize.TokenError):
        return _render_piece(text, source.offset(span[0]))
    return _render(operand_tree, view, rewrites)


def _render_bracket(node: tokenize.TokenInfo, source: _Source, rewrites: list[_Rewrite]) -> _Rendered:
    pattern = getattr(node, "_quantar_pattern", node.string)
    slots = getattr(node, "_quantar_slots", ())
    kind = get_known_operator_kind(pattern) or "bracket"
    first, last = node.start, node.end
    name = f"{_SENTINEL_PREFIX}{uuid4().hex}{_SENTINEL_SUFFIX}"
    rewrites.append(_Rewrite(name, pattern, kind, first, last))

    pieces = [
        _render_piece(f"{name}(", source.offset(first)),
        _render_piece(f"{pattern!r}, ", source.offset(first)),
    ]
    for index, slot in enumerate(slots):
        if index:
            pieces.append(_render_piece(", ", source.offset(slot[0])))
        pieces.append(_render_piece("(", source.offset(slot[0])))
        pieces.append(_render_operand(slot, source, rewrites))
        pieces.append(_render_piece(")", source.offset(slot[1])))
    pieces.append(_render_piece(")", source.offset(last)))
    return _join_rendered(pieces)


def _render(node, source: _Source, rewrites: list[_Rewrite]) -> _Rendered:
    bracket = _bracket_operator(node)
    if bracket is not None:
        return _render_bracket(bracket, source, rewrites)
    operator = _custom_operator(node)
    if operator is not None:
        left, _, right = node
        first = _first_token(node)
        last = _last_token(node)
        if first is None or last is None:
            return _render_piece(_node_text(node, source), source.offset(first.start if first else (1, 0)))

        name = f"{_SENTINEL_PREFIX}{uuid4().hex}{_SENTINEL_SUFFIX}"
        # The kind describes the syntax (``a <-> b`` is binary), not the
        # runtime implementation, which may be a matrix or a callable.
        kind = BUILTIN_OPERATOR_KINDS.get(operator.string, "binary")
        rewrites.append(_Rewrite(name, operator.string, kind, first.start, last.end))

        left_part = _render(left, source, rewrites)
        right_part = _render(right, source, rewrites)
        return _join_rendered([
            _render_piece(f"{name}(", source.offset(first.start)),
            _render_piece(f"{operator.string!r}, (", source.offset(operator.start)),
            left_part,
            _render_piece("), (", source.offset(operator.end)),
            right_part,
            _render_piece("))", source.offset(last.end)),
        ])

    if isinstance(node, tokenize.TokenInfo):
        return _render_piece(node.string, source.offset(node.start))

    text = _node_text(node, source)
    if not text:
        return _Rendered("", [])

    first = _first_token(node)
    if first is None:
        return _render_piece(text, 0)
    base = source.offset(first.start)
    offsets = list(range(base, base + len(text)))
    replacements: list[tuple[int, int, _Rendered]] = []
    for child in _direct_custom_nodes(node):
        child_first = _first_token(child)
        child_last = _last_token(child)
        if child_first is None or child_last is None:
            continue
        start = source.offset(child_first.start) - base
        end = source.offset(child_last.end) - base
        replacements.append((start, end, _render(child, source, rewrites)))

    for start, end, child in sorted(replacements, reverse=True):
        text = text[:start] + child.text + text[end:]
        offsets = offsets[:start] + child.offsets + offsets[end:]
    return _Rendered(text, offsets)


def python_compatible_source(source: str) -> str:
    """Replace Quantar infix operators with Python comparisons, preserving positions.

    This is a lightweight mirror for external Python language servers. Each
    Quantar operator is replaced with ``==`` (or ``<`` for single-character
    operators) and padding spaces of equal UTF-16 width, so lines and editor
    columns remain aligned. A comparison binds at the same level as Quantar's
    custom operators, so ``a <-> b | c and d`` keeps its original grouping.
    """
    try:
        parsed = parse(source)
    except (SyntaxError, tokenize.TokenError):
        try:
            ast.parse(source)
        except SyntaxError:
            return source
        return source

    operators: list[tuple[tokenize.TokenInfo, bool]] = []

    def collect(node) -> None:
        bracket = _bracket_operator(node)
        if bracket is not None:
            operators.append((bracket, True))
            return
        operator = _custom_operator(node)
        if operator is not None:
            operators.append((operator, False))
            collect(node[0])
            collect(node[2])
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(parsed)
    if not operators:
        return source

    source_map = _Source(source)
    rewritten = source
    replacements = []
    for operator, whole_expression in operators:
        start = source_map.offset(operator.start)
        end = source_map.offset(operator.end)
        if whole_expression:
            original = source_map.slice(operator.start, operator.end)
            filler = list(original)
            width = sum(character not in "\r\n" for character in original)
            if width >= 4:
                replacement = "None"
            elif width == 3:
                replacement = "..."
            elif width == 2:
                replacement = "[]"
            else:
                replacement = "0"
            for index, character in enumerate(filler):
                if character in "\r\n":
                    continue
                filler[index] = replacement[index] if index < len(replacement) else " "
            replacements.append((start, end, "".join(filler)))
            continue
        utf16_width = len(operator.string.encode("utf-16-le")) // 2
        if utf16_width >= 2:
            replacement = "==" + (" " * (utf16_width - 2))
        else:
            replacement = "<"
        replacements.append((start, end, replacement))

    for start, end, replacement in sorted(replacements, reverse=True):
        rewritten = rewritten[:start] + replacement + rewritten[end:]

    try:
        ast.parse(rewritten)
    except SyntaxError:
        # Keep broken/incomplete user input intact so the Python server can
        # still provide whatever recovery it supports.
        return source
    return rewritten


def _line_starts(text: str) -> list[int]:
    starts = []
    offset = 0
    for line in text.splitlines(keepends=True):
        starts.append(offset)
        offset += len(line)
    return starts


class _RestorePositions(ast.NodeTransformer):
    """Map rewritten AST positions back onto the original Quantar source."""

    def __init__(self, rendered: _Rendered, source: _Source):
        self.rendered = rendered
        self.source = source
        self.lines = rendered.text.splitlines()
        self.starts = _line_starts(rendered.text)

    def _rewritten_offset(self, line: int, byte_column: int) -> int:
        if not self.lines:
            return 0
        index = max(0, min(line - 1, len(self.lines) - 1))
        text = self.lines[index]
        prefix = text.encode("utf-8")[: max(0, byte_column)]
        try:
            character_column = len(prefix.decode("utf-8"))
        except UnicodeDecodeError:
            character_column = len(text)
        return self.starts[index] + min(character_column, len(text))

    def _original_position(self, line: int, byte_column: int, *, is_end: bool = False) -> tuple[int, int]:
        if not self.rendered.offsets:
            return 1, 0
        rewritten = self._rewritten_offset(line, byte_column)
        # An end position is the boundary *after* the last character. Mapping
        # that boundary directly would pick up the following rendered fragment,
        # which may be an inserted parenthesis mapped to the operator token.
        index = max(0, rewritten - (1 if is_end else 0))
        original = self.rendered.offsets[min(index, len(self.rendered.offsets) - 1)]
        if is_end:
            original = min(original + 1, len(self.source.text))
        return self.source.position_for_offset(original)

    def visit(self, node: ast.AST):
        if hasattr(node, "lineno") and hasattr(node, "col_offset"):
            line, column = self._original_position(node.lineno, node.col_offset) #type: ignore[attr-defined]
            node.lineno = line #type: ignore[attr-defined]
            node.col_offset = column #type: ignore[attr-defined]
        if hasattr(node, "end_lineno") and hasattr(node, "end_col_offset"):
            line, column = self._original_position(node.end_lineno, node.end_col_offset, is_end=True) #type: ignore[attr-defined]
            node.end_lineno = line #type: ignore[attr-defined]
            node.end_col_offset = column #type: ignore[attr-defined]
        return self.generic_visit(node)


class _ReplaceSentinels(ast.NodeTransformer):
    def __init__(self, rewrites: list[_Rewrite], source: _Source):
        self.rewrites = {rewrite.name: rewrite for rewrite in rewrites}
        self.source = source

    def visit_Call(self, node: ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id in self.rewrites):
            return self.generic_visit(node)

        node = self.generic_visit(node) #type: ignore
        rewrite = self.rewrites[node.func.id] #type: ignore
        if len(node.args) < 2:
            return node

        result = QuantumOp(rewrite.operator, list(node.args[1:]), rewrite.kind)
        result.lineno = rewrite.start[0]
        result.col_offset = self.source.byte_column(rewrite.start)
        result.end_lineno = rewrite.end[0]
        result.end_col_offset = self.source.byte_column(rewrite.end)
        return result


class LowerQuantumOps(ast.NodeTransformer):
    """Lower :class:`QuantumOp` nodes to ``__quantar_apply__(name, *operands)``."""

    def visit_QuantumOp(self, node: QuantumOp):
        node = self.generic_visit(node) #type: ignore
        call = ast.Call(
            func=ast.Name(id="__quantar_apply__", ctx=ast.Load()),
            args=[ast.Constant(node.operator), *node.operands],
            keywords=[],
        )
        return ast.copy_location(call, node)


def parse_ast(source: str, filename: str = "<quantar>") -> ast.Module:
    """Parse Quantar source into a Python AST containing :class:`QuantumOp` nodes."""
    source_map = _Source(source)
    rewrites: list[_Rewrite] = []
    try:
        parsed = parse(source)
        rendered = _render(parsed, source_map, rewrites)
        tree = ast.parse(rendered.text, filename=filename)
        tree = _RestorePositions(rendered, source_map).visit(tree) #type: ignore[assignment]
    except SyntaxError as quantar_error:
        # Keep Quantar's parser for custom operators, but let CPython accept
        # ordinary Python syntax supported by the running Python version.
        try:
            return ast.parse(source, filename=filename)
        except SyntaxError:
            raise quantar_error
    tree = _ReplaceSentinels(rewrites, source_map).visit(tree)
    tree._quantar_rewrites = rewrites
    ast.fix_missing_locations(tree)
    return tree


def lower_ast(tree: ast.AST) -> ast.AST:
    """Return `tree` with all :class:`QuantumOp` nodes lowered to calls."""
    lowered = LowerQuantumOps().visit(tree)
    ast.fix_missing_locations(lowered)
    return lowered


def compile_ast(tree: ast.AST, filename: str = "<quantar>", mode: str = "exec") -> CodeType:
    """Compile an AST produced by :func:`parse_ast`."""
    return compile(lower_ast(tree), filename, mode) #type: ignore


def tokenize_ast(source: str):
    """Return the token stream used while building an AST (useful for tooling)."""
    return tokenize_source(source)


build_ast = parse_ast


__all__ = [
    "QuantumOp",
    "LowerQuantumOps",
    "parse_ast",
    "python_compatible_source",
    "build_ast",
    "lower_ast",
    "compile_ast",
    "tokenize_ast",
]
