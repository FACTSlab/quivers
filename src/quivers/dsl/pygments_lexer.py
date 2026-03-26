"""Pygments lexer for the QVR domain-specific language.

Registers the ``qvr`` language alias so that code blocks tagged
``qvr`` in MkDocs, Sphinx, or any Pygments-based renderer get
proper syntax highlighting.

Registration happens via the ``[project.entry-points]`` table
in ``pyproject.toml``::

    [project.entry-points."pygments.lexers"]
    qvr = "quivers.dsl.pygments_lexer:QvrLexer"
"""

from __future__ import annotations

from pygments.lexer import RegexLexer, words
from pygments.token import (
    Comment,
    Generic,
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

            # declaration keywords — blue/purple tones via Keyword
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

            # program body keywords — bold keyword color
            (
                words(
                    ("draw", "observe", "let", "return", "where", "type"),
                    suffix=r"\b",
                ),
                Keyword.Reserved,
            ),

            # quantale names — string-like (green tones)
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

            # space constructors — type/class (teal/cyan tones)
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

            # distribution families — decorator color (distinct from types)
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

            # built-in rule schemas — distinct from functions
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

            # built-in type constructors (used in constructors=[...])
            (
                words(
                    ("slash", "diamond", "box", "unit"),
                    suffix=r"\b",
                ),
                Name.Constant,
            ),

            # built-in functions and combinators — builtin color
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

            # numbers — number color
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
