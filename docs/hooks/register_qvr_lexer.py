"""MkDocs hook that registers the QVR Pygments lexer at build time.

Self-contained: does not import from the quivers package so it
works without ``pip install`` and without torch.

Monkey-patches ``pygments.lexers.get_lexer_by_name`` to intercept
the ``qvr`` alias and return our lexer directly. Also patches the
copies of ``get_lexer_by_name`` that ``markdown.extensions.codehilite``
and ``pymdownx.highlight`` capture at import time.

The lexer is a regex-based approximation. The authoritative
lexer (``quivers.dsl.pygments_lexer.QvrLexer``) drives on the
in-tree tree-sitter grammar but requires building a shared
library, which is too heavy for a docs build pipeline.
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

            # composition-rule declaration keywords
            (
                words(
                    (
                        "algebra",
                        "semigroupoid",
                        "bilinear_form",
                        "composition_rule",
                    ),
                    suffix=r"\b",
                ),
                Keyword.Declaration,
            ),

            # operadic-contraction declaration keywords
            (
                words(
                    ("contraction", "rule", "wiring"),
                    suffix=r"\b",
                ),
                Keyword.Declaration,
            ),

            # module-level declaration keywords
            (
                words(
                    (
                        "category", "object", "rule",
                        "latent", "observed",
                        "space", "kernel",
                        "discretize", "embed", "program", "output",
                        "alias", "bundle", "atoms", "schema",
                        "signature", "compressor",
                        "encoder", "decoder", "loss",
                        "deduction", "semiring", "lexicon",
                    ),
                    suffix=r"\b",
                ),
                Keyword.Declaration,
            ),

            # program body keywords
            (
                words(
                    (
                        "observe", "let", "return", "where",
                        "type", "marginalize", "in", "for", "over",
                        "via", "export",
                    ),
                    suffix=r"\b",
                ),
                Keyword.Reserved,
            ),

            # effect-signature tags
            (
                words(
                    ("Pure", "Sample", "Score", "Marginal"),
                    suffix=r"\b",
                ),
                Keyword.Type,
            ),

            # algebra names (built-in catalogue)
            (
                words(
                    (
                        "product_fuzzy", "boolean", "lukasiewicz",
                        "godel", "tropical", "max_plus", "log_prob",
                        "markov", "real", "probability", "counting",
                        "material_impl", "reichenbach",
                        "boolean_dual", "dual_lukasiewicz",
                        "dual_godel",
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
                        "FreeResiduated", "FinSet", "Mor", "Space",
                        "Object", "Real",
                    ),
                    suffix=r"\b",
                ),
                Name.Class,
            ),

            # distribution families
            (
                words(
                    (
                        "Normal", "Bernoulli", "Beta", "Uniform",
                        "Poisson", "Binomial", "Dirichlet",
                        "Exponential", "HalfCauchy", "HalfNormal",
                        "LogNormal", "Gamma", "Categorical",
                        "MultivariateNormal", "LowRankMVN",
                        "RelaxedBernoulli", "RelaxedOneHotCategorical",
                        "Wishart", "Flow", "LogitNormal",
                        "TruncatedNormal",
                    ),
                    suffix=r"\b",
                ),
                Name.Decorator,
            ),

            # change-of-base singletons + constructors
            (
                words(
                    (
                        "expectation", "log_prob", "max_plus",
                        "material_implication", "threshold",
                        "boolean_embedding", "probability_clamp",
                        "probability_to_real",
                        "counting_from_real", "counting_to_real",
                        "softmax", "l1_normalize", "l2_normalize",
                        "bayes_invert",
                    ),
                    suffix=r"\b",
                ),
                Name.Builtin,
            ),

            # built-in rule schemas (deduction)
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

            # built-in type constructors / modalities
            (
                words(
                    ("slash", "diamond", "box", "unit"),
                    suffix=r"\b",
                ),
                Name.Constant,
            ),

            # let-arithmetic + program builtins
            (
                words(
                    (
                        "sigmoid", "exp", "log", "abs", "softplus",
                        "tanh", "relu", "log1p", "sqrt",
                        "softmax", "log_softmax", "softmax_rows",
                        "identity", "parser", "repeat", "scan",
                        "stack", "fan", "marginalize",
                        "sum", "prod", "cumsum", "mean",
                        "logsumexp", "logsumexp_over",
                        "cholesky_quad_form",
                        "from_data", "freeze",
                        "zeros", "ones",
                    ),
                    suffix=r"\b",
                ),
                Name.Builtin,
            ),

            # arrows; all structural arrows share Keyword.Type
            (r"->", Keyword.Type),
            (r"<-", Keyword.Type),
            (r">=>", Keyword.Type),
            (r"=>", Keyword.Type),
            (r"\|->", Keyword.Type),
            (r"\|-", Keyword.Type),
            # tilde (distribution binding)
            (r"~>", Operator),
            (r"~", Keyword.Type),
            # backslash (category constructor)
            (r"\\", Keyword.Type),
            # composition operators
            (r">>>", Operator),
            (r">>", Operator),
            (r"<<", Operator),
            (r"\*>", Operator),
            (r"\|\|>", Operator),
            (r"\?>", Operator),
            (r"&&>", Operator),
            (r"\+>", Operator),
            (r"\$>", Operator),
            (r"%>", Operator),
            # tensor product
            (r"@", Operator),
            # effect-signature marker
            (r"!", Operator),
            # arithmetic + assignment
            (r"[+\-*/]", Operator),
            (r"=", Operator),

            # string literals (wiring spec, from_data key, etc.)
            (r'"[^"]*"', String.Double),

            # numbers
            (r"-?\d+\.\d+", Number.Float),
            (r"-?\d+", Number.Integer),

            # option keys inside brackets: scale=, hidden_dim=, etc.
            (r"[a-z_]+(?==)", Name.Attribute),

            # punctuation
            (r"[(),:.\[\]{}]", Punctuation),

            # capitalised identifiers (type names, user objects)
            (r"[A-Z]\w*", Name.Class),

            # regular identifiers (variables, morphism names)
            (r"[a-z_]\w*", Name.Variable),

            # whitespace
            (r"\s+", Text),
        ],
    }


# Intercept get_lexer_by_name so codehilite / pymdownx-highlight
# pick up our lexer for the ``qvr`` alias regardless of whether
# the entry-point registration is active.
_original_get_lexer_by_name = pygments.lexers.get_lexer_by_name


def _patched_get_lexer_by_name(_alias, **options):
    if _alias == "qvr":
        return QvrLexer(**options)
    return _original_get_lexer_by_name(_alias, **options)


pygments.lexers.get_lexer_by_name = _patched_get_lexer_by_name

# Patch the references already captured by codehilite / pymdownx
# at import time. Each is best-effort: missing modules are
# silently skipped.
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
    """MkDocs hook entry point. Module-level patching has already
    fired by the time this is called; the function exists so
    MkDocs accepts the file as a hook."""
    pass
