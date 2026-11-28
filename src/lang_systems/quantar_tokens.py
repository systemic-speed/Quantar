"""Tokenizer glue for Quantar's symbolic and bracket operators.

Python's tokenizer splits a symbol such as ``<->`` into ``<``, ``-``, ``>``, so
the generated parser could never match it. This module post-processes the token
stream and merges maximal runs of adjacent ``OP``/``NAME`` tokens that together
spell an operator known to the language into a single ``OP`` token.

Bracket patterns use ``{}`` for operand slots (``"|{}>"``, ``"<{}|{}>"``).
Those are merged across the operand tokens into one ``QUANTUM_BRACKET`` token
that carries the slot positions for the AST builder. Identifier operators
(``CNOT``) need no help -- they are ordinary ``NAME`` tokens.
"""

import io
import token
import tokenize
from collections.abc import Iterator

from lang_systems.operators import (
    BUILTIN_OPERATOR_KINDS,
    get_bracket_parts,
    is_bracket_operator,
    list_operators,
)

_MERGEABLE = (token.OP, token.NAME)
_BRACKET_MERGEABLE = frozenset({token.OP, token.NAME})
_BRACKET_BREAKS = frozenset({
    token.NEWLINE, token.NL, token.INDENT, token.DEDENT,
    token.ENDMARKER, token.COMMENT,
})
_CLOSING_OPS = frozenset({")", "]", "}"})
_PREFIX_KEYWORDS = frozenset({
    "and", "or", "not", "in", "is", "if", "else", "return", "yield",
    "assert", "await", "lambda", "for", "while", "with", "as",
})


def symbolic_operators() -> list[str]:
    """Known operator names that are not plain identifiers, longest first."""
    names = set(list_operators()) | set(BUILTIN_OPERATOR_KINDS)
    symbols = [
        name for name in names
        if not name.isidentifier() and not is_bracket_operator(name)
    ]
    return sorted(symbols, key=len, reverse=True)


def bracket_patterns() -> list[tuple[str, tuple[str, ...]]]:
    """Registered bracket patterns with their literal pieces, longest first."""
    patterns = []
    for name in list_operators():
        parts = get_bracket_parts(name)
        if parts is not None:
            patterns.append((name, parts))
    return sorted(
        patterns,
        key=lambda item: (sum(len(part) for part in item[1]), len(item[0])),
        reverse=True,
    )


def _match(tokens: list[tokenize.TokenInfo], start: int, symbol: str):
    """Try to match `symbol` against adjacent tokens from `start`.

    Returns (merged_token, index_after) or None.
    """
    position = 0
    index = start
    while position < len(symbol):
        if index >= len(tokens):
            return None
        current = tokens[index]
        if current.type not in _MERGEABLE:
            return None
        piece = current.string
        if not symbol.startswith(piece, position):
            return None
        if index > start and tokens[index - 1].end != current.start:
            return None
        position += len(piece)
        index += 1

    if position != len(symbol):
        return None

    first, last = tokens[start], tokens[index - 1]
    merged = tokenize.TokenInfo(
        type=token.OP,
        string=symbol,
        start=first.start,
        end=last.end,
        line=first.line,
    )
    return merged, index


class _BracketToken(tokenize.TokenInfo):
    """A merged bracket expression carrying its operator pattern and slots."""


def _can_start_bracket(tokens: list[tokenize.TokenInfo], index: int) -> bool:
    """Avoid stealing Python's ``a | b > c`` / ``a < b | c`` expressions."""
    if index <= 0:
        return True
    previous = tokens[index - 1]
    if previous.type in _BRACKET_BREAKS:
        return True
    if previous.type == token.NAME:
        return previous.string in _PREFIX_KEYWORDS
    if previous.type == token.OP:
        return previous.string not in _CLOSING_OPS
    return False


def _match_literal(tokens: list[tokenize.TokenInfo], index: int, literal: str) -> int | None:
    """Match literal bracket text, allowing spaces between tokens."""
    position = 0
    while position < len(literal):
        if index >= len(tokens):
            return None
        current = tokens[index]
        if current.type not in _BRACKET_MERGEABLE:
            return None
        if not literal.startswith(current.string, position):
            return None
        position += len(current.string)
        index += 1
    return index


def _slot_breaks(tokens: list[tokenize.TokenInfo], start: int, end: int) -> bool:
    return any(item.type in _BRACKET_BREAKS for item in tokens[start:end])


def _first_bracket_match(
    tokens: list[tokenize.TokenInfo],
    index: int,
    patterns: list[tuple[str, tuple[str, ...]]],
):
    for pattern, parts in patterns:
        matched = _match_bracket(tokens, index, pattern, parts, patterns)
        if matched is not None:
            return matched
    return None


def _find_slot_end(
    tokens: list[tokenize.TokenInfo],
    index: int,
    literal: str,
    patterns: list[tuple[str, tuple[str, ...]]],
) -> tuple[int, int] | None:
    candidate = index
    while candidate <= len(tokens):
        if _slot_breaks(tokens, index, candidate):
            return None
        after = _match_literal(tokens, candidate, literal)
        if after is not None and candidate > index:
            return candidate, after
        nested = _first_bracket_match(tokens, candidate, patterns)
        if nested is not None:
            candidate = nested[1]
            continue
        candidate += 1
    return None


def _match_bracket(
    tokens: list[tokenize.TokenInfo],
    start: int,
    pattern: str,
    parts: tuple[str, ...],
    patterns: list[tuple[str, tuple[str, ...]]],
):
    if not _can_start_bracket(tokens, start):
        return None

    index = start
    if parts[0]:
        matched = _match_literal(tokens, index, parts[0])
        if matched is None:
            return None
        index = matched

    slots: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for literal in parts[1:]:
        if index >= len(tokens):
            return None
        slot_start = tokens[index].start
        found = _find_slot_end(tokens, index, literal, patterns)
        if found is None:
            return None
        end, index = found
        slots.append((slot_start, tokens[end - 1].end))

    if index <= start:
        return None
    first, last = tokens[start], tokens[index - 1]
    merged = _BracketToken(token.OP, pattern, first.start, last.end, first.line)
    merged._quantar_bracket = True  # type: ignore[attr-defined]
    merged._quantar_pattern = pattern  # type: ignore[attr-defined]
    merged._quantar_parts = parts  # type: ignore[attr-defined]
    merged._quantar_slots = tuple(slots)  # type: ignore[attr-defined]
    return merged, index


def merge_bracket_operators(
    tokens: list[tokenize.TokenInfo],
    patterns: list[tuple[str, tuple[str, ...]]],
):
    """Replace bracket expressions with one token each, preserving positions."""
    index = 0
    while index < len(tokens):
        for pattern, parts in patterns:
            matched = _match_bracket(tokens, index, pattern, parts, patterns)
            if matched is not None:
                merged, index = matched
                yield merged
                break
        else:
            yield tokens[index]
            index += 1


def merge_operators(tokens: list[tokenize.TokenInfo], symbols: list[str]):
    """Yield `tokens`, replacing adjacent runs that spell a symbol with one OP token."""
    index = 0
    total = len(tokens)
    while index < total:
        current = tokens[index]
        if current.type in _MERGEABLE:
            for symbol in symbols:
                matched = _match(tokens, index, symbol)
                if matched is not None:
                    merged, index = matched
                    yield merged
                    break
            else:
                yield current
                index += 1
        else:
            yield current
            index += 1


def tokenize_source(source: str) -> Iterator[tokenize.TokenInfo]:
    """Tokenize source, merging symbolic and bracket operators."""
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    tokens = list(merge_operators(tokens, symbolic_operators()))
    return merge_bracket_operators(tokens, bracket_patterns())
