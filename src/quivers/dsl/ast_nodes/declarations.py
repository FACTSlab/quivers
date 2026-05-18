"""Top-level statement AST nodes (everything that can appear at file scope)."""

from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes._shared import (
    AxisSpec,
    CompositionLevel,
    MorphismPrior,
)
from quivers.dsl.ast_nodes.expressions import Expr
from quivers.dsl.ast_nodes.let_expressions import LetExprNode
from quivers.dsl.ast_nodes.program_steps import ProgramStep
from quivers.dsl.ast_nodes.spaces import SpaceExpr
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
# Statement root + composition-rule entry helper
# ---------------------------------------------------------------------------


class Statement(dx.TaggedUnion, discriminator="kind"):
    """Sum of top-level statement kinds."""


class CompositionRuleEntry(dx.Model):
    """One entry of a composition-rule body block.

    Function-valued entries (``tensor_op``, ``join``, ``negation``,
    ``meet``) declare a lambda; value-valued entries (``unit``,
    ``zero``) declare a numeric literal. The ``params`` tuple is
    empty for value-valued entries.
    """

    key: str
    params: tuple[str, ...] = ()
    body: LetExprNode
    line: int = 0
    col: int = 0


class AlgebraDecl(Statement):
    """Composition-rule selection: ``algebra <name>``,
    ``semigroupoid <name>``, ``bilinear_form <name>``, or
    ``composition_rule <name>``, with an optional inline body.

    Without a body the declaration looks up ``name`` in the
    compiler's :data:`_ALGEBRA_REGISTRY` and verifies the
    registered rule matches the keyword's algebraic level. With a
    body, the declaration *defines* a fresh composition rule
    named ``name`` whose operations come from the supplied
    expressions; the keyword fixes the rule's level.
    """

    name: str
    declared_level: CompositionLevel = "algebra"
    body: tuple[CompositionRuleEntry, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["algebra_decl"] = "algebra_decl"


class CategoryDecl(Statement):
    """Category atom declaration: ``category <name>``."""

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["category_decl"] = "category_decl"


class RuleDecl(Statement):
    """Rule-of-inference declaration.

    Premises and conclusion are :class:`TypeExpr` patterns drawn from
    the unified type-expression family: ``TypeName``, ``TypeProduct``,
    ``TypeSlash`` (residuated), and ``TypeEffectApply`` (effect-typed).
    """

    name: str
    variables: tuple[str, ...]
    premises: tuple[TypeExpr, ...]
    conclusion: TypeExpr
    line: int = 0
    col: int = 0
    kind: Literal["rule_decl"] = "rule_decl"


class SchemaDecl(Statement):
    """Pattern-polymorphic morphism schema declaration.

    Surface form: ``schema r[X, Y : Cat] : (X/Y) * Y -> X``.

    Parameters are encoded as two parallel tuples; :attr:`parameter_names`
    holds, for each parameter group, the tuple of variable names (e.g.
    ``("X", "Y")`` for ``X, Y : Cat``); :attr:`parameter_types` holds the
    corresponding type expressions. The arity invariant
    ``len(parameter_names) == len(parameter_types)`` is enforced via a
    dx.axiom.

    Arity (binary vs. unary) is derived from the domain shape: a
    top-level :class:`TypeProduct` with two components produces a
    binary schema; any other domain shape produces a unary schema.
    """

    name: str
    parameter_names: tuple[tuple[str, ...], ...]
    parameter_types: tuple[TypeExpr, ...]
    domain: TypeExpr
    codomain: TypeExpr
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["schema_decl"] = "schema_decl"

    __axioms__ = (
        dx.axiom(
            "length parameter_names == length parameter_types",
            message="schema parameter_names and parameter_types must align",
        ),
    )


# ---------------------------------------------------------------------------
# object initializer (the `=` RHS of ObjectDecl) and ObjectDecl itself
# ---------------------------------------------------------------------------


class ObjectInitializer(dx.TaggedUnion, discriminator="kind"):
    """Sum of object-initializer kinds for the ``=`` form of ObjectDecl."""


class EnumSetLiteral(ObjectInitializer):
    """A ``{NP, S, VP}``-shaped enum-set initializer."""

    elements: tuple[str, ...]
    line: int = 0
    col: int = 0
    kind: Literal["enum_set_literal"] = "enum_set_literal"


class FreeResiduatedExpr(ObjectInitializer):
    """A ``FreeResiduated(generators, depth=, ops=[...])`` initializer."""

    generators: str
    depth: int = 1
    ops: tuple[str, ...] = ("slash",)
    line: int = 0
    col: int = 0
    kind: Literal["free_residuated_expr"] = "free_residuated_expr"


class FreeMonoidExpr(ObjectInitializer):
    """A ``FreeMonoid(generators, max_length=)`` initializer."""

    generators: str
    max_length: int
    line: int = 0
    col: int = 0
    kind: Literal["free_monoid_expr"] = "free_monoid_expr"


class ObjectDecl(Statement):
    """Object declaration.

    Three surface forms:

    - ``object X : 3`` for anonymous-element FinSet of cardinality 3.
      ``type_expr`` carries the TypeExpr; ``init`` is None.
    - ``object Atoms = {NP, S, VP}`` for EnumSet of named atoms.
      ``init`` carries an :class:`EnumSetLiteral`; ``type_expr`` is None.
    - ``object Cat = FreeResiduated(Atoms, depth=4)`` for residuated
      category universe. ``init`` carries a :class:`FreeResiduatedExpr`.

    Doc comments (``##``-prefixed lines) immediately preceding the
    declaration are accumulated into :attr:`docs`.
    """

    name: str
    type_expr: TypeExpr | None = None
    init: ObjectInitializer | None = None
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["object_decl"] = "object_decl"


class MorphismDecl(Statement):
    """Morphism declaration: ``latent|observed <name> : <dom> -> <cod>``.

    ``morphism_kind`` distinguishes ``"latent"`` from ``"observed"``;
    ``kind`` is the dx.TaggedUnion discriminator and is fixed for this
    variant.
    """

    morphism_kind: Literal["latent", "observed"]
    name: str
    domain: TypeExpr
    codomain: TypeExpr
    init_expr: Expr | None = None
    options: dict[str, str] = dx.field(default_factory=dict)
    # Parameter prior on the morphism's representing tensor.  Legal
    # only on ``latent`` declarations; promotes the morphism from
    # a free-parameter point estimate to a random morphism whose
    # tensor is drawn from the named family.
    prior: MorphismPrior | None = None
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["morphism_decl"] = "morphism_decl"


class SpaceDecl(Statement):
    """Space declaration: ``space <name> : <space_expr>``."""

    name: str
    space_expr: SpaceExpr
    line: int = 0
    col: int = 0
    kind: Literal["space_decl"] = "space_decl"


class TypeAliasDecl(Statement):
    """Space-level alias declaration: ``type Latent = Euclidean(16)``."""

    name: str
    space_expr: SpaceExpr
    line: int = 0
    col: int = 0
    kind: Literal["type_alias_decl"] = "type_alias_decl"


class AliasDecl(Statement):
    """Object-level type alias: ``alias Sentence = Cat / NP``.

    Binds ``name`` to the resolved :class:`SetObject` of ``type_expr``
    in the compiler's object environment. The alias is transparent:
    every later occurrence of ``name`` resolves to the underlying
    object, with no reference-counting or recursion bound.
    """

    name: str
    type_expr: TypeExpr
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["alias_decl"] = "alias_decl"


class BundleDecl(Statement):
    """First-class schema-bundle binding.

    Surface form: ``bundle CCG = [forward_app, backward_app,
    harmonic_composition]``. Binds ``name`` to a tuple of schema
    references; ``parser(rules=CCG, ...)`` and
    ``chart_fold(binary=CCG, ...)`` resolve the bundle by name and
    splice its members into the rule list.
    """

    name: str
    rules: tuple[str, ...]
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["bundle_decl"] = "bundle_decl"


class KernelDecl(Statement):
    """Markov-kernel declaration: ``kernel f : A -> B [~ Family ...]``.

    Without a ``~`` clause, declares a lookup-table kernel on finite
    sets: a categorical kernel :math:`A \\to D(B)` realised as a
    learnable matrix of conditional probabilities.

    With a ``~ Family [options] [axes]`` clause, declares a parametric
    kernel :math:`A \\to G(B)` whose family's parameters are produced
    from the input by a parameter network at sample time.  The
    optional ``axes`` clause configures the family's event/batch
    decomposition over codomain factors.
    """

    name: str
    domain: TypeExpr
    codomain: TypeExpr
    family: str | None = None
    options: dict[str, str] = dx.field(default_factory=dict)
    axes: AxisSpec | None = None
    replicate: int | None = None
    line: int = 0
    col: int = 0
    kind: Literal["kernel_decl"] = "kernel_decl"


class DiscretizeDecl(Statement):
    """Discretize boundary: ``discretize <name> : <space> -> <n_bins>``."""

    name: str
    space_name: str
    n_bins: int
    options: dict[str, str] = dx.field(default_factory=dict)
    line: int = 0
    col: int = 0
    kind: Literal["discretize_decl"] = "discretize_decl"


class EmbedDecl(Statement):
    """Embed boundary: ``embed <name> : <finset> -> <space>``."""

    name: str
    domain_name: str
    codomain_name: str
    replicate: int | None = None
    line: int = 0
    col: int = 0
    kind: Literal["embed_decl"] = "embed_decl"


# ---------------------------------------------------------------------------
# program parameters (used by ProgramDecl)
# ---------------------------------------------------------------------------


class ProgramParam(dx.TaggedUnion, discriminator="kind"):
    """Sum of typed-program-parameter variants.

    A *parametric* program declaration takes a list of typed
    parameters (objects, scalars, or morphisms) and denotes a
    dependent kernel

    .. math::

        \\Pi (p_1 : P_1) \\ldots \\Pi (p_n : P_n).\\ \\mathbf{Kern}(\\mathrm{dom}(p),\\, \\mathrm{cod}(p))

    in the indexed family of Kleisli arrows over the parameter
    category. Each call site substitutes specific arguments, yielding
    a concrete Kern-morphism with fresh latent factors inlined into
    the caller's trace; the freshness corresponds to the fact that
    distinct call sites contribute distinct factors to the parent's
    joint kernel.
    """


class ObjectParam(ProgramParam):
    """Object-typed program parameter: ``G : FinSet`` / ``Space`` / ``Object``.

    Denotes a dependent quantification over an object of the
    relevant subcategory: ``FinSet`` ranges over finite-set objects,
    ``Space`` over continuous spaces, ``Object`` over either.
    """

    name: str
    universe: Literal["FinSet", "Space", "Object"]
    line: int = 0
    col: int = 0
    kind: Literal["object_param"] = "object_param"


class ScalarParam(ProgramParam):
    """Scalar-valued program parameter: ``s : Real`` / ``Nat``.

    Denotes a dependent quantification over a hom-object of scalar
    type (real or nonnegative-integer values, used as
    hyperparameters and cardinalities respectively).
    """

    name: str
    scalar_kind: Literal["Real", "Nat"]
    line: int = 0
    col: int = 0
    kind: Literal["scalar_param"] = "scalar_param"


class MorphismParam(ProgramParam):
    """Morphism-typed program parameter: ``f : Mor[A, B]``.

    Denotes a dependent quantification over the hom-set
    :math:`\\mathbf{Kern}(A, B)`; the body may reference ``f`` as a
    family in any plate-draw or draw step whose codomain matches
    ``B``.
    """

    name: str
    domain: TypeExpr
    codomain: TypeExpr
    line: int = 0
    col: int = 0
    kind: Literal["morphism_param"] = "morphism_param"


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
    """Operadic n-ary contraction declaration.

    Surface form (type-driven, inferred wiring)::

        contraction op_apply (
            arg1 : A -> B,
            arg2 : A -> C,
            kernel : (B * C) -> D
        ) : A -> D
            rule product_fuzzy

    The typed signature determines the einsum implicitly: each axis
    in the output (here ``A`` and ``D``) propagates; each axis that
    appears in >= 2 inputs but not in the output (``B``, ``C``) is
    contracted via the rule's join.

    Two opt-in disambiguators handle cases the inference cannot
    derive from the signature alone:

    * ``share T1, T2, ...`` keeps the listed axes element-wise
      (broadcast / propagated) even when they appear in multiple
      inputs.  Stored on :attr:`shared_axes`.
    * ``wiring "<einsum>"`` is the explicit escape hatch.  Stored
      verbatim on :attr:`wiring_spec`.

    Exactly zero or one of those clauses appears in source.  The
    compiler dispatches: ``wiring_spec`` non-empty -> use it
    verbatim; otherwise build the einsum from the typed signature
    plus ``shared_axes``.

    Compiles to a callable that wraps
    :class:`~quivers.core.wiring.EinsumWiring`.
    """

    name: str
    inputs: tuple[ContractionInput, ...]
    domain: TypeExpr
    codomain: TypeExpr
    rule_name: str
    wiring_spec: str = ""
    shared_axes: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["contraction_decl"] = "contraction_decl"


# ---------------------------------------------------------------------------
# programs, let bindings, exports
# ---------------------------------------------------------------------------


class ProgramDecl(Statement):
    """Monadic program block, the unique program-form in QVR.

    A program is either *concrete* (no ``type_params``), denoting a
    single Kern-morphism ``dom -> cod``, or *parametric* (with
    ``type_params``), denoting a dependent family of Kern-morphisms
    indexed by the parameters. Parametric programs are not compiled
    into a runtime ``MonadicProgram`` directly; the compiler stores
    them as templates and inlines a freshly-renamed copy of the body
    at each call site.

    Effects and posterior modifier:

    * ``effects`` carries the capability set declared after ``!``:
      ``frozenset({"Sample", "Score"})``, ``frozenset({"Marginal"})``,
      ``frozenset({"Pure"})``, etc. ``None`` means unannotated (the
      compiler infers the set from the body but does not enforce a
      restriction).
    * ``over_model`` declares the program is a *posterior block*
      over another program's latents, replacing the standalone
      ``posterior`` keyword. A program with ``over_model`` set is
      routed by the compiler to the posterior registry; its body
      must be deterministic (``Pure`` or at most ``Pure | Marginal``).
    """

    name: str
    params: tuple[str, ...] | None
    domain: TypeExpr
    codomain: TypeExpr
    draws: tuple[ProgramStep, ...]
    return_vars: tuple[str, ...]
    return_labels: tuple[str, ...] | None = None
    effects: frozenset[str] | None = None
    over_model: str | None = None
    type_params: tuple[ProgramParam, ...] | None = None
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["program_decl"] = "program_decl"


class LetDecl(Statement):
    """Let binding: ``let <name> = <expr> [where let_decl+]``.

    The optional ``where`` clause carries nested :class:`LetDecl`
    statements; the field is typed as ``tuple[Statement, ...]`` (the
    union root) because didactic does not yet accept self-referential
    forward refs in field annotations. The parser only ever writes
    :class:`LetDecl` instances into the tuple.
    """

    name: str
    expr: Expr
    where: tuple[Statement, ...] | None = None
    line: int = 0
    col: int = 0
    kind: Literal["let_decl"] = "let_decl"


class ExportDecl(Statement):
    """Module-level export: ``export <expr>``.

    Any number per module; each selects a top-level morphism /
    posterior / deduction for the compiled output.  Semantically a
    public binding in the module namespace.
    """

    expr: Expr
    line: int = 0
    col: int = 0
    kind: Literal["export_decl"] = "export_decl"


# ---------------------------------------------------------------------------
# deductions
# ---------------------------------------------------------------------------


class SequentRule(dx.Model):
    """A named sequent-style inference rule inside a deduction block.

    ``rule name : premises |- conclusion``. Patterns may contain
    single-uppercase-identifier wildcards (e.g. ``X``, ``Y``) that
    bind to actual category subexpressions when the rule fires.
    """

    name: str
    premises: tuple[TypeExpr, ...]
    conclusion: TypeExpr
    line: int = 0
    col: int = 0


class LexiconEntry(dx.Model):
    """A single entry in a deduction's lexicon block.

    Maps a literal word string to a (category, logical-form)
    pair, with an optional learnable log-weight. Multiple
    entries with the same ``word`` give the model a latent
    disjunction over (cat, lf) options for that word: at
    chart-construction time, each becomes an independent span
    axiom and the chart's semiring aggregates them (the
    standard semiring-parsing realisation of latent lexical
    categories / logical forms).

    Either or both of ``category`` and ``lf`` may carry
    structural metavariables; under Curry-Howard these are
    types and proof witnesses, and the lexicon is a
    *family of typed terms* in the deduction's term algebra.
    Both slots are syntactically open enough to express
    type-logical grammars, dependent-type systems,
    Datalog facts, edit-distance alignment moves, or any other
    item algebra the deduction operates over.

    Attributes
    ----------
    word : str
        The literal lexical token (typically a surface word in
        an NLP setting, but may be any string-keyed item label).
    category : TypeExpr
        The entry's syntactic / type-theoretic category. Used
        as the leftmost field of the span axiom emitted at chart
        time.
    lf : LetExprNode
        The entry's logical form / proof witness, as a
        let-sublanguage expression evaluated at axiom-injection
        time.
    learnable : bool
        Whether to allocate a per-entry :class:`nn.Parameter`
        log-weight (default scale ``0.0``) that the optimizer
        can adjust during training.
    """

    word: str
    category: TypeExpr
    lf: LetExprNode
    learnable: bool = False
    line: int = 0
    col: int = 0


class DeductionDecl(Statement):
    """A weighted-deduction-system declaration.

    Surface form::

        deduction NAME : Domain -> Codomain {
            atoms { A, B, ... }
            rule r1 : premises |- conclusion
            rule r2 : ...
            semiring  SemiringName
            start     StartSymbol
            depth     N

            # One of three optional axiom-source forms:
            lexicon {                          # inline lexicon block
                "every" : Cat = lf @ learnable
                ...
            }
            lexicon from "path/to/lexicon.tsv" with learnable
                                               # file-loaded lexicon
            axioms = some_kernel_morphism      # general axiom source
        }

    Categorical denotation: the system denotes a
    :math:`\\mathcal{V}`-presheaf-valued morphism
    :math:`\\mathrm{Domain} \\to \\mathbf{Set}^{I^{\\mathrm{op}}}_{K}`,
    computed as the least pre-fixed point of the rule-system
    functor in the :math:`K`-enriched lattice of charts. The
    axiom-source field declares the kernel
    :math:`\\mathrm{Input} \\to \\mathrm{List}(I \\times K)` that
    produces the chart's initial items from an input value;
    ``lexicon`` is a sugar specialization for the
    label-indexed-lookup case.
    """

    name: str
    domain: TypeExpr
    codomain: TypeExpr
    atoms: tuple[str, ...]
    rules: tuple[SequentRule, ...]
    semiring: str | None = None
    start: str | None = None
    depth: int | None = None
    lexicon: tuple[LexiconEntry, ...] = ()
    lexicon_from_file: str | None = None
    lexicon_from_file_learnable: bool = False
    axioms_source: str | None = None
    item_signature: str | None = None
    item_encoder: str | None = None
    line: int = 0
    col: int = 0
    kind: Literal["deduction_decl"] = "deduction_decl"


# ---------------------------------------------------------------------------
# structural compression: signatures, encoders, decoders, losses
# ---------------------------------------------------------------------------


class SignatureDecl(Statement):
    """A signature block declaring an algebra over which encoders,
    decoders, and rules are defined.

    A signature may be **inductive** (sorts + constructors + binders)
    or **graph-shaped** (vertex_kinds + edge_kinds), or both.
    """

    name: str
    params: tuple[str, ...] = ()
    sorts: tuple[SortDecl, ...] = ()
    constructors: tuple[ConstructorDecl, ...] = ()
    binders: tuple[BinderDecl, ...] = ()
    vertex_kinds: tuple[VertexKindDecl, ...] = ()
    edge_kinds: tuple[EdgeKindDecl, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["signature_decl"] = "signature_decl"


class EncoderDecl(Statement):
    """An algebra homomorphism from an inductive or graph signature
    to a fixed-dimension vector carrier.

    Two surface forms.

    *Explicit*: list per-constructor rules, dims, iterations, and
    readout in a ``{ ... }`` body::

        encoder C over Sig {
            dim Term = 64
            App(f, x)        |-> mlp_app([f, x])
            Lam(ty, body)    |-> mlp_lam([ty, body])
            var_init(ty)     |-> mlp_var_init(ty)
            iterations 4                       # graph-only
            init Atom(a)     |-> atom_embed[a] # graph-only
            message[e](s, t) |-> mlp_msg([s, t])
            update[V](s, m)  |-> gru_update(s, m)
            readout          |-> mean_pool
        }

    *Factory-backed*: invoke a builder from the
    :mod:`quivers.structural.shapes` registry (``rnn_encoder``,
    ``transformer_encoder``, ``bow_encoder``, ``tree_lstm_encoder``,
    ``gnn_encoder``, ...) with optional ``[k=v]`` overrides::

        encoder C over Sig using rnn_encoder
        encoder C over Sig using transformer_encoder [dim=128]
        encoder C over Sig using gnn_encoder [iterations=4, dim=64]

    The two forms are mutually exclusive: a declaration that sets
    :attr:`factory` to a non-empty string leaves every per-rule
    field at its default; the explicit form leaves :attr:`factory`
    empty and populates the per-rule tuples.
    """

    name: str
    signature: str
    sig_args: tuple[str, ...] = ()
    dims: tuple[SortDim, ...] = ()
    op_rules: tuple[EncoderRule, ...] = ()
    init_rules: tuple[EncoderInitRule, ...] = ()
    message_rules: tuple[EncoderMessageRule, ...] = ()
    update_rules: tuple[EncoderUpdateRule, ...] = ()
    iterations: int | None = None
    readout: LetExprNode | None = None
    var_inits: tuple[EncoderVarInit, ...] = ()
    factory: str = ""
    factory_options: dict[str, str] = dx.field(default_factory=dict)
    line: int = 0
    col: int = 0
    kind: Literal["encoder_decl"] = "encoder_decl"


class DecoderDecl(Statement):
    """A Kleisli coalgebraic decoder over an inductive or graph
    signature.

    Surface form::

        decoder D over Sig depth 8 {
            dim Term = 64
            structure(v)      |-> structure_logits(v)
            primitive(v)      |-> primitive_logits(v)
            factor(v)         |-> factor_split(v)
            binder_select(v)  |-> binder_logits(v)
            body              |-> recursive
        }
    """

    name: str
    signature: str
    sig_args: tuple[str, ...] = ()
    depth: int = 8
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
    line: int = 0
    col: int = 0
    kind: Literal["decoder_decl"] = "decoder_decl"


class LossAttachment(dx.Model):
    """Where a loss fires.

    ``attachment_kind`` is one of:

    * ``"global"`` for fires once per training step.
    * ``"program"`` for fires after a named program invocation.
    * ``"deduction"`` for fires after a named deduction's chart build.
    * ``"encoder"`` / ``"decoder"`` for fires after the named call.
    * ``"rule"`` for fires on each application of a named rule.
    * ``"chart"`` for fires once on the chart of a named deduction.
    """

    attachment_kind: Literal[
        "global", "program", "deduction", "encoder", "decoder", "rule", "chart"
    ] = "global"
    target: str | None = None
    rule_deduction: str | None = None


class LossDecl(Statement):
    """A weighted scalar loss, attachable to any training site."""

    name: str
    weight: LetExprNode | None = None
    attachment: LossAttachment = dx.field(default_factory=LossAttachment)
    body: LetExprNode
    line: int = 0
    col: int = 0
    kind: Literal["loss_decl"] = "loss_decl"


__all__ = [
    "Statement",
    "CompositionRuleEntry",
    "AlgebraDecl",
    "CategoryDecl",
    "RuleDecl",
    "SchemaDecl",
    "ObjectInitializer",
    "EnumSetLiteral",
    "FreeResiduatedExpr",
    "FreeMonoidExpr",
    "ObjectDecl",
    "MorphismDecl",
    "SpaceDecl",
    "TypeAliasDecl",
    "AliasDecl",
    "BundleDecl",
    "KernelDecl",
    "DiscretizeDecl",
    "EmbedDecl",
    "ProgramParam",
    "ObjectParam",
    "ScalarParam",
    "MorphismParam",
    "ContractionInput",
    "ContractionDecl",
    "ProgramDecl",
    "LetDecl",
    "ExportDecl",
    "SequentRule",
    "LexiconEntry",
    "DeductionDecl",
    "SignatureDecl",
    "EncoderDecl",
    "DecoderDecl",
    "LossAttachment",
    "LossDecl",
]
