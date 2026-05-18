"""Top-level statement AST nodes for the Every Statement variant corresponds 1:1 to a top-level tree-sitter
production in ``grammars/qvr/grammar.js``. The :class:`Statement`
tagged union discriminates via the ``kind`` field.

Sixteen Statement subclasses:

* :class:`CompositionDecl` 
* :class:`CategoryDecl`
* :class:`RuleDecl`
* :class:`SchemaDecl`
* :class:`TypeDecl` 
* :class:`MorphismDecl` 
* :class:`BundleDecl`
* :class:`ContractionDecl`
* :class:`LetDecl`
* :class:`ExportDecl`
* :class:`DeductionDecl`
* :class:`SignatureDecl`
* :class:`EncoderDecl`
* :class:`DecoderDecl`
* :class:`LossDecl`
* :class:`ProgramDecl`

Every declaration carries a ``docs: tuple[str, ...]`` field.
The ``option_block`` surface is modelled as
``options: dict[str, OptionValue]``; structured values use the
:class:`OptionValue` tagged union.
"""

from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes._shared import (
    CompositionLevel,
    OptionCall,
    OptionEntry,
    OptionFlag,
    OptionList,
    OptionName,
    OptionNumber,
    OptionString,
    OptionValue,
)
from quivers.dsl.ast_nodes.expressions import Expr
from quivers.dsl.ast_nodes.let_expressions import LetExprNode
from quivers.dsl.ast_nodes.program_steps import ProgramStep
from quivers.dsl.ast_nodes.structural import (
    BinderDecl,
    ConstructorDecl,
    EdgeKindDecl,
    EncoderInitRule,
    EncoderMessageRule,
    EncoderRule,
    EncoderUpdateRule,
    EncoderVarInit,
    SortDecl,
    SortDim,
    VertexKindDecl,
)
from quivers.dsl.ast_nodes.types import TypeExpr

# ---------------------------------------------------------------------------
# Statement root
# ---------------------------------------------------------------------------

class Statement(dx.TaggedUnion, discriminator="kind"):
    """Sum of top-level statement kinds."""

# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

class CompositionRuleEntry(dx.Model):
    """One entry of a ``composition`` body block.

    Function-valued entries (``tensor_op``, ``join``, ``negation``,
    ``meet``) declare a lambda over named parameters; value-valued
    entries (``unit``, ``zero``) declare a numeric literal. The
    ``params`` tuple is empty for value-valued entries.
    """

    key: str
    params: tuple[str, ...] = ()
    body: LetExprNode
    line: int = 0
    col: int = 0

class CompositionDecl(Statement):
    """``composition NAME [at LEVEL] [: body]`` declaration.

    ``level`` records which algebraic level the
    declaration advertises; the optional body defines the rule's
    operations inline.
    """

    name: str
    level: CompositionLevel | None = None
    body: tuple[CompositionRuleEntry, ...] = ()
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["composition_decl"] = "composition_decl"

# ---------------------------------------------------------------------------
# category
# ---------------------------------------------------------------------------

class CategoryDecl(Statement):
    """``category NAME, NAME, ...`` declaration."""

    names: tuple[str, ...]
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["category_decl"] = "category_decl"

# ---------------------------------------------------------------------------
# rule (top-level CCG/Lambek)
# ---------------------------------------------------------------------------

class RuleDecl(Statement):
    """``rule NAME(variables) : premises => conclusion`` declaration.

    Premises and conclusion are :class:`TypeExpr` patterns drawn from
    the unified type-expression family.
    """

    name: str
    variables: tuple[str, ...]
    premises: tuple[TypeExpr, ...]
    conclusion: TypeExpr
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["rule_decl"] = "rule_decl"

# ---------------------------------------------------------------------------
# schema (pattern-polymorphic morphism schema)
# ---------------------------------------------------------------------------

class SchemaParameter(dx.Model):
    """One ``names : type`` group inside a schema's parameter list."""

    names: tuple[str, ...]
    type_expr: TypeExpr
    line: int = 0
    col: int = 0

class SchemaDecl(Statement):
    """``schema NAME(parameters) : DOM -> COD`` declaration."""

    name: str
    parameters: tuple[SchemaParameter, ...]
    domain: TypeExpr
    codomain: TypeExpr
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["schema_decl"] = "schema_decl"

# ---------------------------------------------------------------------------
# type: collapses object / space / alias / type-alias
# ---------------------------------------------------------------------------

class TypeInitializer(dx.TaggedUnion, discriminator="kind"):
    """Sum of ``type NAME : VALUE`` value shapes.

    The grammar's ``_type_value`` choice corresponds 1:1 to this
    union.
    """

class TypeEnumSet(TypeInitializer):
    """``type Atoms : {NP, S, VP}`` enum-set initializer."""

    elements: tuple[str, ...]
    line: int = 0
    col: int = 0
    kind: Literal["type_enum_set"] = "type_enum_set"

class TypeFreeResiduated(TypeInitializer):
    """``type Cat : FreeResiduated(Atoms, depth=4, ops=[/, \\])`` initializer."""

    generators: str
    depth: int = 1
    ops: tuple[str, ...] = ("slash",)
    line: int = 0
    col: int = 0
    kind: Literal["type_free_residuated"] = "type_free_residuated"

class TypeFreeMonoid(TypeInitializer):
    """``type Words : FreeMonoid(Atoms, max_length=N)`` initializer."""

    generators: str
    max_length: int
    line: int = 0
    col: int = 0
    kind: Literal["type_free_monoid"] = "type_free_monoid"

class TypeFromExpr(TypeInitializer):
    """``type X : <type-expression>`` for every non-special-form value.

    Covers integer cardinalities, identifiers, products, coproducts,
    slashes, effect-applies, and constructor calls (FinSet, Real,
    Simplex, ...).
    """

    expr: TypeExpr
    kind: Literal["type_from_expr"] = "type_from_expr"

class TypeDecl(Statement):
    """``type NAME : VALUE`` declaration.

    The VALUE's shape picks the
    discrete-vs-continuous and reference-vs-initializer distinction;
    the compiler reads :attr:`init` to decide what kind of object
    (FinSet, EnumSet, FreeResiduated, FreeMonoid, ContinuousSpace) to
    register in the env.
    """

    name: str
    init: TypeInitializer
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["type_decl"] = "type_decl"

# ---------------------------------------------------------------------------
# morphism: single keyword + ~ initializer
# ---------------------------------------------------------------------------

type MorphismRole = Literal[
    "latent", "observed", "kernel", "embed", "discretize", "let"
]
"""Role attribute on a morphism declaration.

``latent`` (default learnable point estimate), ``observed`` (fixed
structural input), ``kernel`` (parametric Markov kernel with a family
prior), ``embed`` (FinSet -> ContinuousSpace boundary),
``discretize`` (ContinuousSpace -> FinSet boundary), ``let``
(deterministic let-bound morphism). Required; the compiler rejects
declarations without ``role`` in their option block.
"""

class MorphismInitFamily(dx.Model):
    """``~ Family(args)`` family-call initializer (e.g. ``~ Normal(0, 1)``)."""

    family: str
    args: tuple[str | float, ...] = ()
    line: int = 0
    col: int = 0

class MorphismDecl(Statement):
    """``morphism NAME : DOM -> COD [options] [~ init]``.

    The morphism's role
    travels in the option block; the compiler reads ``options["role"]``
    to pick the runtime construction.

    Initializer (``~`` clause, optional):

    * ``init_family``: ``~ Family(args)`` family call (used for
      kernel priors, latent priors, distribution-driven init).
    * ``init_expr``: ``~ <expression>`` arbitrary morphism-expression
      (used for explicit values, composition pipelines, ``~ auto``
      named recipes).

    The two init slots are mutually exclusive; exactly zero or one
    is populated.
    """

    name: str
    domain: TypeExpr
    codomain: TypeExpr
    options: tuple[OptionEntry, ...] = ()
    init_family: MorphismInitFamily | None = None
    init_expr: Expr | None = None
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["morphism_decl"] = "morphism_decl"

# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------

class BundleDecl(Statement):
    """``bundle NAME = [rule1, rule2, ...]`` first-class schema bundle.

    Binds NAME to a tuple of schema references; ``parser(rules=NAME)``
    and ``chart_fold(binary=NAME, ...)`` resolve the bundle by name
    and splice its members into the rule list.
    """

    name: str
    rules: tuple[str, ...]
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["bundle_decl"] = "bundle_decl"

# ---------------------------------------------------------------------------
# contraction
# ---------------------------------------------------------------------------

class ContractionInput(dx.Model):
    """One input wire of a :class:`ContractionDecl` declaration."""

    name: str
    input_domain: TypeExpr
    input_codomain: TypeExpr
    line: int = 0
    col: int = 0

class ContractionDecl(Statement):
    """``contraction NAME(inputs) : DOM -> COD [options]``.

    The option block carries:

    * ``rule=<NAME>`` (required): names the composition rule whose
      ``join`` is the fold operation for contracted axes.
    * ``share=[ax1, ax2, ...]``: keep listed axes element-wise even
      when they appear in multiple inputs.
    * ``wiring="<einsum>"``: explicit escape hatch; verbatim
      einsum string.
    """

    name: str
    inputs: tuple[ContractionInput, ...]
    domain: TypeExpr
    codomain: TypeExpr
    options: tuple[OptionEntry, ...] = ()
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["contraction_decl"] = "contraction_decl"

# ---------------------------------------------------------------------------
# let / export
# ---------------------------------------------------------------------------

class LetDecl(Statement):
    """``let NAME = EXPR [where: nested-lets]`` value binding.

    Unlike a :class:`MorphismDecl` with ``role=let``, this is a
    value-level let: its RHS is an arbitrary Expr, not a morphism
    signature with an init.

    The ``where`` field is typed as ``tuple[Statement, ...]`` (the
    union root) because didactic does not yet accept self-referential
    forward refs in field annotations. The parser only ever writes
    :class:`LetDecl` instances into the tuple.
    """

    name: str
    expr: Expr
    where: tuple[Statement, ...] = ()
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["let_decl"] = "let_decl"

class ExportDecl(Statement):
    """``export EXPR`` module-level export."""

    expr: Expr
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["export_decl"] = "export_decl"

# ---------------------------------------------------------------------------
# deduction
# ---------------------------------------------------------------------------

class SequentRule(dx.Model):
    """A named sequent inside a deduction block.

    ``rule NAME : premises |- conclusion``. Premise and conclusion
    type-expressions may contain single-uppercase wildcards
    (``X``, ``Y``) that bind to actual category subexpressions when
    the rule fires.
    """

    name: str
    premises: tuple[TypeExpr, ...]
    conclusion: TypeExpr
    line: int = 0
    col: int = 0

class LexiconEntry(dx.Model):
    """A single entry in a deduction's lexicon block."""

    word: str
    category: TypeExpr
    lf: LetExprNode
    options: tuple[OptionEntry, ...] = ()
    line: int = 0
    col: int = 0

class DeductionDecl(Statement):
    """``deduction NAME : DOM -> COD [options] : body``.

    Body is an indented block of ``atoms``, ``rule``, ``lexicon``
    entries. The option block carries ``semiring=...``, ``start=...``,
    ``depth=...``, ``axioms=...``, ``signature=...``, ``encoder=...``.
    """

    name: str
    domain: TypeExpr
    codomain: TypeExpr
    options: tuple[OptionEntry, ...] = ()
    atoms: tuple[str, ...] = ()
    rules: tuple[SequentRule, ...] = ()
    lexicon: tuple[LexiconEntry, ...] = ()
    lexicon_from_file: str | None = None
    lexicon_from_file_options: tuple[OptionEntry, ...] = ()
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["deduction_decl"] = "deduction_decl"

# ---------------------------------------------------------------------------
# signature
# ---------------------------------------------------------------------------

class SignatureDecl(Statement):
    """``signature NAME[(params)] : body``."""

    name: str
    params: tuple[str, ...] = ()
    sorts: tuple[SortDecl, ...] = ()
    constructors: tuple[ConstructorDecl, ...] = ()
    binders: tuple[BinderDecl, ...] = ()
    vertex_kinds: tuple[VertexKindDecl, ...] = ()
    edge_kinds: tuple[EdgeKindDecl, ...] = ()
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["signature_decl"] = "signature_decl"

# ---------------------------------------------------------------------------
# encoder / decoder / loss
# ---------------------------------------------------------------------------

class EncoderDecl(Statement):
    """``encoder NAME : SIG[(sig_args)] [options] [: body]``.

    The option block carries ``factory=...`` (factory-backed form)
    plus any factory keyword arguments. Body entries are
    ``dim``, ``iterations``, ``readout``, per-op rules,
    init/message/update rules, var_init rules.
    """

    name: str
    signature: str
    sig_args: tuple[str, ...] = ()
    options: tuple[OptionEntry, ...] = ()
    dims: tuple[SortDim, ...] = ()
    iterations: int | None = None
    readout: LetExprNode | None = None
    op_rules: tuple[EncoderRule, ...] = ()
    init_rules: tuple[EncoderInitRule, ...] = ()
    message_rules: tuple[EncoderMessageRule, ...] = ()
    update_rules: tuple[EncoderUpdateRule, ...] = ()
    var_inits: tuple[EncoderVarInit, ...] = ()
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["encoder_decl"] = "encoder_decl"

class DecoderDecl(Statement):
    """``decoder NAME : SIG[(sig_args)] [options] : body``."""

    name: str
    signature: str
    sig_args: tuple[str, ...] = ()
    options: tuple[OptionEntry, ...] = ()
    dims: tuple[SortDim, ...] = ()
    structure: LetExprNode | None = None
    structure_arg: str | None = None
    primitive: LetExprNode | None = None
    primitive_arg: str | None = None
    factor: LetExprNode | None = None
    factor_arg: str | None = None
    binder_select: LetExprNode | None = None
    binder_select_arg: str | None = None
    recursive_default: bool = True
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["decoder_decl"] = "decoder_decl"

class LossDecl(Statement):
    """``loss NAME [options] : body``.

    Option keys: ``weight=<expr>``, ``on=<attachment>``. The
    ``on`` value is an :class:`OptionCall` of one of ``program(NAME)``,
    ``deduction(NAME)``, ``encoder(NAME)``, ``decoder(NAME)``,
    ``rule(NAME, in=DEDUCTION)``, ``chart(of=DEDUCTION)``, or an
    :class:`OptionFlag` ``global``.
    """

    name: str
    options: tuple[OptionEntry, ...] = ()
    body: LetExprNode
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["loss_decl"] = "loss_decl"

# ---------------------------------------------------------------------------
# program
# ---------------------------------------------------------------------------

class ProgramParam(dx.TaggedUnion, discriminator="kind"):
    """Sum of typed-program-parameter variants (parametric programs)."""

class ObjectParam(ProgramParam):
    """Object-typed program parameter: ``G : FinSet`` / ``Space`` / ``Object``."""

    name: str
    universe: Literal["FinSet", "Space", "Object"]
    line: int = 0
    col: int = 0
    kind: Literal["object_param"] = "object_param"

class ScalarParam(ProgramParam):
    """Scalar-typed program parameter: ``s : Real`` / ``Nat``."""

    name: str
    scalar_kind: Literal["Real", "Nat"]
    line: int = 0
    col: int = 0
    kind: Literal["scalar_param"] = "scalar_param"

class MorphismParam(ProgramParam):
    """Morphism-typed program parameter: ``f : Mor[A, B]``."""

    name: str
    domain: TypeExpr
    codomain: TypeExpr
    line: int = 0
    col: int = 0
    kind: Literal["morphism_param"] = "morphism_param"

class ProgramDecl(Statement):
    """``program NAME[(params)] : DOM -> COD [options] : body``.

    Body is a sequence of program steps (``sample``, ``observe``,
    ``marginalize``, ``let``) terminated by a ``return`` step.
    Effects, posterior-modifier (``over=<model>``), and
    parametric/concrete type-params live in the option block.
    """

    name: str
    params: tuple[str, ...] | None = None
    type_params: tuple[ProgramParam, ...] | None = None
    domain: TypeExpr
    codomain: TypeExpr
    options: tuple[OptionEntry, ...] = ()
    draws: tuple[ProgramStep, ...] = ()
    return_vars: tuple[str, ...] = ()
    return_labels: tuple[str, ...] | None = None
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["program_decl"] = "program_decl"

__all__ = [
    "BundleDecl",
    "CategoryDecl",
    "CompositionDecl",
    "CompositionRuleEntry",
    "ContractionDecl",
    "ContractionInput",
    "DecoderDecl",
    "DeductionDecl",
    "EncoderDecl",
    "ExportDecl",
    "LetDecl",
    "LexiconEntry",
    "LossDecl",
    "MorphismDecl",
    "MorphismInitFamily",
    "MorphismParam",
    "MorphismRole",
    "ObjectParam",
    "OptionCall",
    "OptionEntry",
    "OptionFlag",
    "OptionList",
    "OptionName",
    "OptionNumber",
    "OptionString",
    "OptionValue",
    "ProgramDecl",
    "ProgramParam",
    "RuleDecl",
    "ScalarParam",
    "SchemaDecl",
    "SchemaParameter",
    "SequentRule",
    "SignatureDecl",
    "Statement",
    "TypeDecl",
    "TypeEnumSet",
    "TypeFreeMonoid",
    "TypeFreeResiduated",
    "TypeFromExpr",
    "TypeInitializer",
]
