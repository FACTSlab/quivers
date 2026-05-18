"""Typed accessors for the 0.11.0 unified option block.

Every declaration in the 0.11.0 surface carries options as a
``tuple[OptionEntry, ...]``. Each entry's value lives in the
:class:`OptionValue` tagged union (flag / name / number / string /
list / call). The compiler frequently asks ``what is the value of
this option, decoded to T?'' for concrete T in {str, int, float,
bool, tuple[str, ...], OptionCall}; this module is the single
typed seam where that decoding happens.

The rules are deliberate:

* Missing key returns the supplied default (or None).
* Wrong shape (e.g. asking for an int when the entry is a string)
  is a :class:`CompileError` at the declaration's source location.
* Identifier-valued options (``role=latent``) decode through
  ``get_option_name``; string-valued options (``path="lex.tsv"``)
  decode through ``get_option_string``. The surface distinguishes
  the two, and the compiler should not silently coerce.
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    OptionCall,
    OptionEntry,
    OptionFlag,
    OptionList,
    OptionName,
    OptionNumber,
    OptionString,
    OptionValue,
)
from quivers.dsl.compiler._prelude import CompileError


def find_option(
    options: tuple[OptionEntry, ...], key: str
) -> OptionEntry | None:
    """Linear scan for the first entry whose key matches.

    The unified option block is small and bounded; a list scan is
    both clearer and faster than dict construction at the call site.
    """
    for entry in options:
        if entry.key == key:
            return entry
    return None


def has_option(options: tuple[OptionEntry, ...], key: str) -> bool:
    """Whether ``key`` appears in ``options`` (any value, including flag)."""
    return find_option(options, key) is not None


def _at(line: int, col: int, entry: OptionEntry | None) -> tuple[int, int]:
    """Pick the most precise (line, col) we have for an option error."""
    if entry is not None:
        return entry.line, entry.col
    return line, col


def get_option_flag(options: tuple[OptionEntry, ...], key: str) -> bool:
    """Whether ``[key]`` appears as a bare flag (OptionFlag value).

    Distinct from :func:`has_option`: ``key=true`` would pass
    ``has_option`` but not ``get_option_flag``.
    """
    entry = find_option(options, key)
    return entry is not None and isinstance(entry.value, OptionFlag)


def get_option_name(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = 0,
    col: int = 0,
    default: str | None = None,
) -> str | None:
    """Decode an identifier-valued option to its name.

    Matches ``key=identifier`` (and only that). ``role=latent``,
    ``init=auto``, ``reduction=logsumexp`` all hit this path.
    """
    entry = find_option(options, key)
    if entry is None:
        return default
    if not isinstance(entry.value, OptionName):
        ln, cl = _at(line, col, entry)
        raise CompileError(
            f"option {key!r}: expected identifier value, got "
            f"{type(entry.value).__name__}",
            ln,
            cl,
        )
    return entry.value.value


def get_option_string(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = 0,
    col: int = 0,
    default: str | None = None,
) -> str | None:
    """Decode a string-literal-valued option."""
    entry = find_option(options, key)
    if entry is None:
        return default
    if not isinstance(entry.value, OptionString):
        ln, cl = _at(line, col, entry)
        raise CompileError(
            f"option {key!r}: expected string literal, got "
            f"{type(entry.value).__name__}",
            ln,
            cl,
        )
    return entry.value.value


def get_option_int(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = 0,
    col: int = 0,
    default: int | None = None,
) -> int | None:
    """Decode a numeric option as a non-negative integer.

    The grammar's ``OptionNumber`` carries a Python float; this
    accessor enforces integrality. ``depth=8``, ``hidden_dim=64``,
    ``replicate=3`` all hit this path.
    """
    entry = find_option(options, key)
    if entry is None:
        return default
    if not isinstance(entry.value, OptionNumber):
        ln, cl = _at(line, col, entry)
        raise CompileError(
            f"option {key!r}: expected numeric value, got "
            f"{type(entry.value).__name__}",
            ln,
            cl,
        )
    value = entry.value.value
    if not value.is_integer():
        ln, cl = _at(line, col, entry)
        raise CompileError(
            f"option {key!r}: expected integer, got {value!r}", ln, cl,
        )
    return int(value)


def get_option_float(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = 0,
    col: int = 0,
    default: float | None = None,
) -> float | None:
    """Decode a numeric option as a Python float."""
    entry = find_option(options, key)
    if entry is None:
        return default
    if not isinstance(entry.value, OptionNumber):
        ln, cl = _at(line, col, entry)
        raise CompileError(
            f"option {key!r}: expected numeric value, got "
            f"{type(entry.value).__name__}",
            ln,
            cl,
        )
    return entry.value.value


def get_option_name_list(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = 0,
    col: int = 0,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Decode a list-of-identifiers option (``over=[a, b]``).

    Accepts two surface shapes for unary convenience:

    * ``[over=[a, b, c]]`` -> OptionList of OptionNames.
    * ``[over=a]`` -> single OptionName, lifted to ``(a,)``.
    """
    entry = find_option(options, key)
    if entry is None:
        return default
    v = entry.value
    if isinstance(v, OptionName):
        return (v.value,)
    if isinstance(v, OptionList):
        names: list[str] = []
        for item in v.items:
            if not isinstance(item, OptionName):
                ln, cl = _at(line, col, entry)
                raise CompileError(
                    f"option {key!r}: list items must be identifiers, "
                    f"got {type(item).__name__}",
                    ln,
                    cl,
                )
            names.append(item.value)
        return tuple(names)
    ln, cl = _at(line, col, entry)
    raise CompileError(
        f"option {key!r}: expected identifier or list-of-identifiers, "
        f"got {type(v).__name__}",
        ln,
        cl,
    )


def get_option_call(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = 0,
    col: int = 0,
) -> OptionCall | None:
    """Decode a callable-shaped option (``on=program(NAME)``)."""
    entry = find_option(options, key)
    if entry is None:
        return None
    if not isinstance(entry.value, OptionCall):
        ln, cl = _at(line, col, entry)
        raise CompileError(
            f"option {key!r}: expected call-shaped value (e.g. "
            f"``program(NAME)``), got {type(entry.value).__name__}",
            ln,
            cl,
        )
    return entry.value


def get_option_value(
    options: tuple[OptionEntry, ...], key: str
) -> OptionValue | None:
    """Untyped option lookup; returns the raw :class:`OptionValue`.

    For paths where the compiler dispatches on the value shape itself
    (e.g. ``loss.on`` accepts both ``[global]`` flag and ``on=...``
    call).
    """
    entry = find_option(options, key)
    return entry.value if entry is not None else None


__all__ = [
    "find_option",
    "get_option_call",
    "get_option_flag",
    "get_option_float",
    "get_option_int",
    "get_option_name",
    "get_option_name_list",
    "get_option_string",
    "get_option_value",
    "has_option",
]
