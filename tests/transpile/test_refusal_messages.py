"""Every refusal the transpiler can raise must explain itself.

A refusal reaches the user as
[`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct], whose
message is rendered from a structured `kinds` list by
[`user_facing_message`][quivers.transpile._diagnostics.user_facing_message].
That renderer ends in a fallback that names the tag and asks for a bug
report, which is the right behaviour for a tag nobody has seen and the
wrong thing for a user to hit: it tells them the transpile failed
without telling them what to do about it.

This module holds the line that no kind the transpiler actually emits
reaches that fallback. The kinds are read off `docs/transpile-support.md`,
which `test_support_matrix_docs` pins to a fresh measurement over every
gallery program and construct fixture against every backend, so the set
checked here is the set the pipeline currently produces rather than a
list that drifts as backends change.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import quivers.transpile

from quivers.transpile._diagnostics import user_facing_message
from tests.transpile import support_matrix

#: Text the terminal fallback in `_render_kind` emits. A rendered
#: message containing this is a kind with no explanation registered.
UNEXPLAINED = "no explanation registered"

#: Backends the support page reports on, used to check that a kind
#: explains itself whichever target refused it.
TARGETS: tuple[str, ...] = (
    "qvr-stan",
    "qvr-numpyro",
    "qvr-pyro",
    "qvr-pymc",
    "qvr-edward2",
    "qvr-gen",
    "qvr-turing",
    "qvr-webppl",
    "qvr-jags",
    "qvr-bugs",
    "qvr-church",
)


def _emitted_kinds() -> tuple[str, ...]:
    """Every kind quoted in the published support matrix.

    The page records each refusal's reported kinds in fenced `text`
    blocks, one kind per line.
    """
    page = support_matrix.DOC_PATH.read_text(encoding="utf-8")
    kinds: set[str] = set()
    for block in re.findall(r"```text\n(.*?)```", page, re.DOTALL):
        for line in block.splitlines():
            candidate = line.strip().strip("`")
            if candidate and " " not in candidate:
                kinds.add(candidate)
    if not kinds:
        raise AssertionError(
            f"{support_matrix.DOC_PATH.name} quoted no refusal kinds, so "
            f"this test would vacuously pass. The page's format changed; "
            f"update the reader."
        )
    return tuple(sorted(kinds))


#: Root of the package whose raise sites are scanned statically.
TRANSPILE_ROOT = Path(quivers.transpile.__file__).parent


def _kind_literal(node: ast.expr) -> str | None:
    """The concrete kind a literal or f-string argument spells.

    An f-string's interpolations are stamped with a placeholder: the
    renderer dispatches on the literal prefixes, so a placeholder in
    a value slot exercises the same branch the real value would.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            str(part.value) if isinstance(part, ast.Constant) else "X"
            for part in node.values
        )
    return None


def _static_kinds() -> tuple[str, ...]:
    """Every kind spelled at a literal `UnsupportedConstruct` raise.

    The measured page only records kinds the gallery and fixture
    programs actually provoke, so a raise site no example reaches
    would go unchecked. Reading the raise sites off the source covers
    those too, which is how the mis-spelled family kinds were found.
    """
    kinds: set[str] = set()
    for path in sorted(TRANSPILE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name != "UnsupportedConstruct":
                continue
            for argument in node.args[1:]:
                if not isinstance(argument, ast.List | ast.Tuple):
                    continue
                for element in argument.elts:
                    kind = _kind_literal(element)
                    if kind:
                        kinds.add(kind)
    if not kinds:
        raise AssertionError(
            "no `UnsupportedConstruct` raise sites were found under "
            f"{TRANSPILE_ROOT}, so this test would vacuously pass."
        )
    return tuple(sorted(kinds))


EMITTED_KINDS: tuple[str, ...] = tuple(
    sorted(set(_emitted_kinds()) | set(_static_kinds()))
)


@pytest.mark.parametrize("kind", EMITTED_KINDS)
@pytest.mark.parametrize("target", TARGETS)
def test_every_emitted_kind_renders_an_explanation(
    target: str, kind: str
) -> None:
    """No emitted kind falls through to the report-a-bug fallback."""
    message = user_facing_message(target, (kind,))
    assert UNEXPLAINED not in message, (
        f"transpiling to {target} can refuse with kind {kind!r}, and the "
        f"user is told only that the tag has no explanation registered. "
        f"Register one in `_diagnostics` so the refusal says what the "
        f"construct means, why this target has no form for it, and what "
        f"to write instead."
    )


@pytest.mark.parametrize("kind", EMITTED_KINDS)
@pytest.mark.parametrize("target", TARGETS)
def test_every_explanation_frames_the_tag_as_prose(
    target: str, kind: str
) -> None:
    """A message is prose about the tag, never the tag echoed back.

    Some kinds carry their own detail after a structured head, so a
    message may legitimately quote most of its kind; what it may not
    do is hand the raw tag back, which would leave the user exactly
    where the structured token did.
    """
    message = user_facing_message(target, (kind,))
    assert message.strip() != kind.strip(), (
        f"{target} renders kind {kind!r} as the tag itself, which "
        f"explains nothing the tag did not."
    )
    assert len(message.split()) >= 8, (
        f"{target} renders kind {kind!r} as {message!r}, too terse to "
        f"say what the construct is and what to write instead."
    )
