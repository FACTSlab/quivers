"""Register the authoritative QVR lexer during MkDocs builds.

The project is installed before both local and CI documentation builds, so the
site can use the same tree-sitter-backed lexer as the REPL and Python tooling.
Keeping a second regex approximation here allowed the documentation vocabulary
to drift from the grammar; in particular, ``sample`` and ``define`` were
rendered as ordinary identifiers.
"""

import pygments.lexers

from quivers.dsl.pygments_lexer import QvrLexer


# Intercept get_lexer_by_name so codehilite / pymdownx-highlight pick up the
# in-tree lexer for the ``qvr`` alias regardless of entry-point cache state.
_original_get_lexer_by_name = pygments.lexers.get_lexer_by_name


def _patched_get_lexer_by_name(_alias, **options):
    if _alias == "qvr":
        return QvrLexer(**options)
    return _original_get_lexer_by_name(_alias, **options)


pygments.lexers.get_lexer_by_name = _patched_get_lexer_by_name

# Patch references captured by codehilite / pymdownx at import time. Each is
# best-effort because MkDocs installations may omit one of the extensions.
try:
    import markdown.extensions.codehilite as _ch

    _ch.get_lexer_by_name = _patched_get_lexer_by_name
except ImportError:
    pass

try:
    import pymdownx.highlight as _hl

    if hasattr(_hl, "get_lexer_by_name"):
        _hl.get_lexer_by_name = _patched_get_lexer_by_name
except ImportError:
    pass


def on_startup(**kwargs):
    """MkDocs hook entry point; module import performs registration."""
    pass
