"""MkDocs hook that registers the QVR Pygments lexer at build time.

Self-contained — does not import from the quivers package so it
works without ``pip install`` and without torch.

Monkey-patches ``pygments.lexers.get_lexer_by_name`` to intercept
the ``qvr`` alias and return our lexer directly.
"""

import pygments.lexers
from pygments.lexer import RegexLexer, words
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
)


class QvrLexer(RegexLexer):
    """Pygments lexer for .qvr (quivers DSL) files."""

    name = "QVR"
    aliases = ["qvr"]
    filenames = ["*.qvr"]
    mimetypes = ["text/x-qvr"]

    tokens = {
        "root": [
            # comments
            (r"#.*$", Comment.Single),

            # declaration keywords
            (
                words(
                    (
                        "quantale", "category", "object", "rule",
                        "latent", "observed",
                        "space", "continuous", "stochastic",
                        "discretize", "embed", "program", "output",
                    ),
                    suffix=r"\b",
                ),
                Keyword.Declaration,
            ),

            # program body keywords
            (
                words(
                    ("draw", "observe", "let", "return", "where", "type"),
                    suffix=r"\b",
                ),
                Keyword.Reserved,
            ),

            # quantale names
            (
                words(
                    (
                        "product_fuzzy", "boolean", "lukasiewicz",
                        "godel", "tropical",
                    ),
                    suffix=r"\b",
                ),
                String.Symbol,
            ),

            # space constructors
            (
                words(
                    (
                        "Euclidean", "UnitInterval", "Simplex",
                        "PositiveReals", "FreeMonoid",
                    ),
                    suffix=r"\b",
                ),
                Name.Class,
            ),

            # distribution families
            (
                words(
                    (
                        "Normal", "LogitNormal", "Beta", "Bernoulli",
                        "Uniform", "TruncatedNormal", "Dirichlet",
                        "Exponential", "HalfCauchy", "HalfNormal",
                        "LogNormal", "Gamma", "Categorical",
                        "MultivariateNormal", "LowRankMVN",
                        "RelaxedBernoulli", "RelaxedOneHotCategorical",
                        "Wishart", "Flow",
                    ),
                    suffix=r"\b",
                ),
                Name.Decorator,
            ),

            # built-in rule schemas
            (
                words(
                    (
                        "evaluation", "harmonic_composition",
                        "crossed_composition", "adjunction_units",
                        "tensor_introduction", "tensor_projection",
                        "modal_introduction", "modal_elimination",
                        "commutative_evaluation",
                    ),
                    suffix=r"\b",
                ),
                Name.Constant,
            ),

            # built-in type constructors
            (
                words(
                    ("slash", "diamond", "box", "unit"),
                    suffix=r"\b",
                ),
                Name.Constant,
            ),

            # built-in functions and combinators
            (
                words(
                    (
                        "sigmoid", "exp", "log", "abs", "softplus",
                        "identity", "parser", "repeat", "scan",
                        "stack", "fan", "marginalize",
                    ),
                    suffix=r"\b",
                ),
                Name.Builtin,
            ),

            # arrows — all structural arrows share Keyword.Type
            (r"->", Keyword.Type),
            (r"<-", Keyword.Type),
            (r">=>", Keyword.Type),
            (r"=>", Keyword.Type),
            # tilde (distribution binding)
            (r"~", Keyword.Type),
            # backslash (category constructor)
            (r"\\", Keyword.Type),
            # composition operators
            (r">>", Operator),
            (r"<<", Operator),
            # tensor product
            (r"@", Operator),
            # arithmetic
            (r"[+\-*/]", Operator),
            # assignment
            (r"=", Operator),

            # numbers
            (r"-?\d+\.\d+", Number.Float),
            (r"-?\d+", Number.Integer),

            # option keys inside brackets: scale=, hidden_dim=, etc.
            (r"[a-z_]+(?==)", Name.Attribute),

            # punctuation
            (r"[(),:.\[\]]", Punctuation),

            # capitalized identifiers (type names, user objects)
            (r"[A-Z]\w*", Name.Class),

            # regular identifiers (variables, morphism names)
            (r"[a-z_]\w*", Name.Variable),

            # whitespace
            (r"\s+", Text),
        ],
    }


# monkey-patch get_lexer_by_name to intercept "qvr"
_original_get_lexer_by_name = pygments.lexers.get_lexer_by_name


def _patched_get_lexer_by_name(_alias, **options):
    if _alias == "qvr":
        return QvrLexer(**options)

    return _original_get_lexer_by_name(_alias, **options)


pygments.lexers.get_lexer_by_name = _patched_get_lexer_by_name

# also patch the reference that codehilite/pymdownx may have already imported
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
    """MkDocs hook entry point."""
    pass
