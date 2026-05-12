"""Program Theory: a panproto protocol for the resolved (post-compilation) DSL.

The :data:`QVR_PROGRAM_PROTOCOL` protocol describes the static structure of
a compiled `.qvr` program — every object, space, morphism, and output the
program declares — as a panproto :class:`~panproto.Schema`. This sits one
layer above the syntactic ``qvr`` protocol from
``panproto-grammars-all``: the syntactic protocol carries the AST as
parsed, this protocol carries the AST after the resolution layer
(``_resolve_type``, ``_resolve_space``) has run.

Why have it
-----------
Once two programs share this protocol, panproto's structural diff,
auto-lens-generation, and breaking-change detection apply to compiled
programs as a whole. Two `.qvr` files that compile to structurally
equivalent programs produce equal Schemas; two that diverge surface
their divergence through :func:`panproto.diff_schemas`.

Vertex kinds
------------

Top-level container:
    ``program``
        the root vertex; one per compiled module.

Discrete objects (mirrors :mod:`quivers.core.objects`):
    ``finset`` ``product_set`` ``coproduct_set`` ``free_monoid`` ``empty_set``

Continuous spaces (mirrors :mod:`quivers.continuous.spaces`):
    ``euclidean`` ``simplex`` ``positive_reals`` ``product_space``

Top-level declarations:
    ``object_decl`` ``space_decl`` ``morphism_decl``
    ``continuous_morphism_decl`` ``stochastic_morphism_decl``
    ``discretize_decl`` ``embed_decl`` ``output_decl``

Edge kinds
----------

``decl``
    ``program -> *_decl``: each declaration is a child of the root program.
``binds_to``
    ``object_decl -> set_object`` / ``space_decl -> space``.
``component``
    ``product_set | coproduct_set -> set_object``;
    ``product_space -> space``: structural recursion into composite types.
``domain`` / ``codomain``
    ``morphism_decl | continuous_morphism_decl | stochastic_morphism_decl ->
    set_object | space``.
``output``
    ``program -> output_decl``.

Constraint sorts carry the per-vertex scalar metadata: ``name``,
``cardinality``, ``dim``, ``low``, ``high``, ``family``, ``modality``,
``morphism_kind``, ``replicate``, ``n_bins``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import panproto

from quivers.continuous.spaces import (
    ContinuousSpace,
    Euclidean,
    PositiveReals,
    ProductSpace,
    Simplex,
)
from quivers.categorical.monoidal import EmptySet
from quivers.core.objects import (
    CoproductSet,
    FinSet,
    EnumSet,
    FreeMonoid,
    FreeResiduated,
    ProductSet,
    SetObject,
)
from quivers.dsl.ast_nodes import ExprIdent, ExprIdentity

if TYPE_CHECKING:
    from quivers.dsl.compiler import Compiler


# ---------------------------------------------------------------------------
# protocol definition
# ---------------------------------------------------------------------------

_OBJECT_KINDS = [
    "program",
    # discrete
    "finset",
    "product_set",
    "coproduct_set",
    "free_monoid",
    "empty_set",
    "enum_set",
    "free_residuated",
    # continuous
    "euclidean",
    "simplex",
    "positive_reals",
    "product_space",
    # declarations
    "object_decl",
    "space_decl",
    "morphism_decl",
    "continuous_morphism_decl",
    "stochastic_morphism_decl",
    "discretize_decl",
    "embed_decl",
    "output_decl",
    "schema_decl",
]

_SET_OBJECT_KINDS = [
    "finset",
    "product_set",
    "coproduct_set",
    "free_monoid",
    "empty_set",
    "enum_set",
    "free_residuated",
]
_SPACE_KINDS = ["euclidean", "simplex", "positive_reals", "product_space"]
_DOMAIN_KINDS = _SET_OBJECT_KINDS + _SPACE_KINDS
_DECL_KINDS = [
    "object_decl",
    "space_decl",
    "morphism_decl",
    "continuous_morphism_decl",
    "stochastic_morphism_decl",
    "discretize_decl",
    "embed_decl",
    "output_decl",
]

_MORPHISM_DECL_KINDS = [
    "morphism_decl",
    "continuous_morphism_decl",
    "stochastic_morphism_decl",
    "discretize_decl",
    "embed_decl",
]

_EDGE_RULES = [
    {"edge_kind": "decl", "src_kinds": ["program"], "tgt_kinds": _DECL_KINDS},
    # ``binds_to`` covers both object_decl→set_object and space_decl→space:
    # panproto requires a unique edge_kind label, so the two cases share one
    # rule whose src/tgt kinds are unioned. The Schema-level validator then
    # accepts any valid pairing across the union.
    {
        "edge_kind": "binds_to",
        "src_kinds": ["object_decl", "space_decl"],
        "tgt_kinds": _SET_OBJECT_KINDS + _SPACE_KINDS,
    },
    # ``component`` covers ProductSet/CoproductSet → set-object children
    # and ProductSpace → space|set-object children (mixed-domain products
    # land here too).
    {
        "edge_kind": "component",
        "src_kinds": ["product_set", "coproduct_set", "product_space"],
        "tgt_kinds": _DOMAIN_KINDS,
    },
    {
        "edge_kind": "generators",
        "src_kinds": ["free_monoid", "free_residuated"],
        "tgt_kinds": ["finset", "enum_set"],
    },
    {
        "edge_kind": "domain",
        "src_kinds": _MORPHISM_DECL_KINDS,
        "tgt_kinds": _DOMAIN_KINDS,
    },
    {
        "edge_kind": "codomain",
        "src_kinds": _MORPHISM_DECL_KINDS,
        "tgt_kinds": _DOMAIN_KINDS,
    },
    {"edge_kind": "output", "src_kinds": ["program"], "tgt_kinds": ["output_decl"]},
]

_CONSTRAINT_SORTS = [
    "name",
    "cardinality",
    "max_length",
    "dim",
    "low",
    "high",
    "family",
    "modality",
    "morphism_kind",
    "replicate",
    "n_bins",
    "quantale",
]

QVR_PROGRAM_PROTOCOL: panproto.Protocol = panproto.define_protocol(
    {
        "name": "qvr_program",
        # Reuse the brat schema/instance theories — both shape graphs with
        # vertices, edges, and constraint metadata; the kinds and rules above
        # are what specialise them to the compiled-quivers shape.
        "schema_theory": "ThBratSchema",
        "instance_theory": "ThBratInstance",
        "edge_rules": _EDGE_RULES,
        "obj_kinds": _OBJECT_KINDS,
        "constraint_sorts": _CONSTRAINT_SORTS,
    }
)


# ---------------------------------------------------------------------------
# extraction: compiled program -> panproto.Schema
# ---------------------------------------------------------------------------


class _SchemaWriter:
    """Helper that emits set-object / space subgraphs into a SchemaBuilder.

    Each emitted SetObject / ContinuousSpace instance gets its own vertex,
    keyed in the cache by Python object identity. Structural deduplication
    (one vertex per ``__eq__``-equivalent value) would collapse the
    components of e.g. ``ProductSet(components=(N, N))`` into a single
    target — and ``component`` edges between the same source and target
    can't repeat under panproto's edge-set semantics — so we keep distinct
    occurrences distinct.
    """

    def __init__(self, builder: panproto.SchemaBuilder) -> None:
        self._builder = builder
        self._object_ids: dict[int, str] = {}
        self._counter = 0

    def _fresh(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    # -- discrete -------------------------------------------------------

    def write_set_object(self, obj: SetObject) -> str:
        """Emit vertices/constraints for a SetObject; return the root vid."""
        cached = self._object_ids.get(id(obj))
        if cached is not None:
            return cached
        if isinstance(obj, FinSet):
            vid = self._fresh("finset")
            self._builder.vertex(vid, "finset")
            self._builder.constraint(vid, "name", obj.name)
            self._builder.constraint(vid, "cardinality", str(obj.cardinality))
        elif isinstance(obj, ProductSet):
            vid = self._fresh("product_set")
            self._builder.vertex(vid, "product_set")
            for child in obj.components:
                cvid = self.write_set_object(child)
                self._builder.edge(vid, cvid, "component")
        elif isinstance(obj, CoproductSet):
            vid = self._fresh("coproduct_set")
            self._builder.vertex(vid, "coproduct_set")
            for child in obj.components:
                cvid = self.write_set_object(child)
                self._builder.edge(vid, cvid, "component")
        elif isinstance(obj, FreeMonoid):
            vid = self._fresh("free_monoid")
            self._builder.vertex(vid, "free_monoid")
            self._builder.constraint(vid, "max_length", str(obj.max_length))
            gvid = self.write_set_object(obj.generators)
            self._builder.edge(vid, gvid, "generators")
        elif isinstance(obj, EmptySet):
            vid = self._fresh("empty_set")
            self._builder.vertex(vid, "empty_set")
        elif isinstance(obj, EnumSet):
            vid = self._fresh("enum_set")
            self._builder.vertex(vid, "enum_set")
            self._builder.constraint(vid, "name", obj.name)
            for elem in obj.elements:
                self._builder.constraint(vid, "element", elem)
        elif isinstance(obj, FreeResiduated):
            vid = self._fresh("free_residuated")
            self._builder.vertex(vid, "free_residuated")
            self._builder.constraint(vid, "depth", str(obj.depth))
            for op in obj.ops:
                self._builder.constraint(vid, "op", op)
            gvid = self.write_set_object(obj.generators)
            self._builder.edge(vid, gvid, "generators")
        else:
            raise TypeError(f"unsupported SetObject variant: {type(obj).__name__}")
        self._object_ids[id(obj)] = vid
        return vid

    # -- continuous -----------------------------------------------------

    def write_space(self, space: ContinuousSpace) -> str:
        cached = self._object_ids.get(id(space))
        if cached is not None:
            return cached
        if isinstance(space, Euclidean):
            vid = self._fresh("euclidean")
            self._builder.vertex(vid, "euclidean")
            self._builder.constraint(vid, "name", space.name)
            self._builder.constraint(vid, "dim", str(space.dim))
            if space.low is not None:
                self._builder.constraint(vid, "low", str(space.low))
            if space.high is not None:
                self._builder.constraint(vid, "high", str(space.high))
        elif isinstance(space, Simplex):
            vid = self._fresh("simplex")
            self._builder.vertex(vid, "simplex")
            self._builder.constraint(vid, "name", space.name)
            self._builder.constraint(vid, "dim", str(space.dim))
        elif isinstance(space, PositiveReals):
            vid = self._fresh("positive_reals")
            self._builder.vertex(vid, "positive_reals")
            self._builder.constraint(vid, "name", space.name)
            self._builder.constraint(vid, "dim", str(space.dim))
        elif isinstance(space, ProductSpace):
            vid = self._fresh("product_space")
            self._builder.vertex(vid, "product_space")
            for child in space.components:
                if isinstance(child, ContinuousSpace):
                    cvid = self.write_space(child)
                else:
                    cvid = self.write_set_object(child)
                self._builder.edge(vid, cvid, "component")
        else:
            raise TypeError(
                f"unsupported ContinuousSpace variant: {type(space).__name__}"
            )
        self._object_ids[id(space)] = vid
        return vid

    def write_any(self, target: object) -> str:
        if isinstance(target, ContinuousSpace):
            return self.write_space(target)
        if isinstance(target, SetObject):
            return self.write_set_object(target)
        raise TypeError(f"unsupported domain/codomain: {type(target).__name__}")


# ---------------------------------------------------------------------------
# the public extractor
# ---------------------------------------------------------------------------


def extract_program_schema(compiler: "Compiler") -> panproto.Schema:
    """Produce a :class:`panproto.Schema` for a compiled program.

    Walks the compiler's resolved environment (objects, spaces, morphisms)
    and emits a graph of vertices and edges in the
    :data:`QVR_PROGRAM_PROTOCOL` protocol. The returned schema validates
    against that protocol and is suitable for :func:`panproto.diff_schemas`,
    :func:`panproto.auto_generate_lens`, and the rest of panproto's
    schema-level operations.

    Parameters
    ----------
    compiler
        A :class:`~quivers.dsl.compiler.Compiler` after :meth:`compile_env`
        (or :meth:`compile`) has populated the resolved environments.

    Returns
    -------
    panproto.Schema
        A program-level Schema in the ``qvr_program`` protocol.
    """
    builder = QVR_PROGRAM_PROTOCOL.schema()
    writer = _SchemaWriter(builder)

    builder.vertex("program", "program")
    if compiler._quantale is not None:
        builder.constraint("program", "quantale", type(compiler._quantale).__name__)

    # object decls
    for name, obj in compiler._objects.items():
        decl_vid = f"object_decl::{name}"
        builder.vertex(decl_vid, "object_decl")
        builder.constraint(decl_vid, "name", name)
        builder.edge("program", decl_vid, "decl")
        target_vid = writer.write_set_object(obj)
        builder.edge(decl_vid, target_vid, "binds_to")

    # space decls
    for name, space in compiler._spaces.items():
        decl_vid = f"space_decl::{name}"
        builder.vertex(decl_vid, "space_decl")
        builder.constraint(decl_vid, "name", name)
        builder.edge("program", decl_vid, "decl")
        target_vid = writer.write_space(space)
        builder.edge(decl_vid, target_vid, "binds_to")

    # morphism decls — the compiler's _morphisms env holds named primitive
    # morphisms; we record them as morphism_decl vertices with domain/codomain.
    # Composite morphisms (let-bindings, output) are derived rather than
    # recorded directly; the structural Diff-able layer is the named decls.
    for name, morphism in compiler._morphisms.items():
        kind = _classify_morphism_kind(morphism)

        decl_vid = f"{kind}::{name}"
        builder.vertex(decl_vid, kind)
        builder.constraint(decl_vid, "name", name)
        builder.edge("program", decl_vid, "decl")

        dom = getattr(morphism, "domain", None)
        cod = getattr(morphism, "codomain", None)
        if dom is not None:
            dom_vid = writer.write_any(dom)
            builder.edge(decl_vid, dom_vid, "domain")
        if cod is not None:
            cod_vid = writer.write_any(cod)
            builder.edge(decl_vid, cod_vid, "codomain")

    # output decl — the compiler's `_output_expr` holds the AST expression
    # whose compilation produces the program's root morphism. We record an
    # output_decl vertex carrying the expression's source-text form (when
    # available) as a name constraint; the structural diff cares about
    # presence/absence of an output, not its detailed shape.
    if compiler._output_expr is not None:
        out_vid = "output_decl"
        builder.vertex(out_vid, "output_decl")
        # ExprIdent / ExprIdentity carry a single name; composite expressions
        # don't have a single canonical name, so mark them as "<composite>".
        expr = compiler._output_expr
        if isinstance(expr, ExprIdent):
            label = expr.name
        elif isinstance(expr, ExprIdentity):
            label = f"identity({expr.object_name})"
        else:
            label = "<composite>"
        builder.constraint(out_vid, "name", label)
        builder.edge("program", out_vid, "output")

    return builder.build()


def _classify_morphism_kind(morphism: object) -> str:
    """Classify a runtime morphism into the program-theory vertex kind.

    The compiler's ``_morphisms`` env holds primitive morphisms produced
    by every kind of morphism declaration (``MorphismDecl``,
    ``ContinuousMorphismDecl``, ``StochasticMorphismDecl``,
    ``DiscretizeDecl``, ``EmbedDecl``). Classification routes through
    ``isinstance`` rather than module/class name string-matching so the
    boundaries are explicit.
    """
    # Imports are local because these modules form a long dependency
    # chain; importing at module top level would deepen the import graph
    # for callers that only need QVR_PROGRAM_PROTOCOL itself.
    from quivers.continuous.boundaries import Discretize, Embed
    from quivers.continuous.morphisms import ContinuousMorphism
    from quivers.stochastic.morphisms import StochasticMorphism

    if isinstance(morphism, Discretize):
        return "discretize_decl"
    if isinstance(morphism, Embed):
        return "embed_decl"
    if isinstance(morphism, StochasticMorphism):
        return "stochastic_morphism_decl"
    if isinstance(morphism, ContinuousMorphism):
        return "continuous_morphism_decl"
    return "morphism_decl"


# ---------------------------------------------------------------------------
# Deduction-system protocol
# ---------------------------------------------------------------------------


_DEDUCTION_OBJECT_KINDS = [
    "deduction_system",
    "deduction_rule",
    "deduction_atom",
    "deduction_premise",
    "deduction_conclusion",
]

_DEDUCTION_EDGE_RULES = [
    {
        "edge_kind": "decl",
        "src_kinds": ["deduction_system"],
        "tgt_kinds": ["deduction_rule"],
    },
    {
        "edge_kind": "atom",
        "src_kinds": ["deduction_system"],
        "tgt_kinds": ["deduction_atom"],
    },
    {
        "edge_kind": "premise",
        "src_kinds": ["deduction_rule"],
        "tgt_kinds": ["deduction_premise"],
    },
    {
        "edge_kind": "conclusion",
        "src_kinds": ["deduction_rule"],
        "tgt_kinds": ["deduction_conclusion"],
    },
]

_DEDUCTION_CONSTRAINT_SORTS = [
    "name",
    "semiring",
    "start",
    "depth",
    "pattern",
]


QVR_DEDUCTION_PROTOCOL: panproto.Protocol = panproto.define_protocol(
    {
        "name": "qvr_deduction",
        "schema_theory": "ThBratSchema",
        "instance_theory": "ThBratInstance",
        "edge_rules": _DEDUCTION_EDGE_RULES,
        "obj_kinds": _DEDUCTION_OBJECT_KINDS,
        "constraint_sorts": _DEDUCTION_CONSTRAINT_SORTS,
    }
)
"""Panproto protocol for a weighted deductive system.

Vertex kinds:
    * ``deduction_system`` — the top-level declaration.
    * ``deduction_atom`` — an atom of the item algebra.
    * ``deduction_rule`` — a named sequent-style inference rule.
    * ``deduction_premise`` — one premise pattern of a rule.
    * ``deduction_conclusion`` — a rule's conclusion pattern.

Edges:
    * ``decl`` — system :math:`\\to` rule.
    * ``atom`` — system :math:`\\to` atom.
    * ``premise`` — rule :math:`\\to` premise.
    * ``conclusion`` — rule :math:`\\to` conclusion.

Constraint sorts carry the system's semiring, start, depth, and
each pattern's textual form. Schema morphisms over this protocol
correspond to specialisations of deduction systems
(e.g., :math:`\\mathsf{CCG} \\subset \\mathsf{Lambek} \\subset \\mathsf{MultimodalLambek}`).
"""


def extract_deduction_schema(compiler: "Compiler") -> panproto.Schema:
    """Produce a :class:`panproto.Schema` for the compiler's
    deduction-system environment.

    Walks the compiler's ``_deductions`` registry and emits one
    panproto vertex per deduction system, one per rule, one per
    atom, and per-premise / per-conclusion pattern vertices. The
    returned schema validates against
    :data:`QVR_DEDUCTION_PROTOCOL` and is suitable for
    :func:`panproto.diff_schemas` and
    :func:`panproto.auto_generate_lens` operations over deduction
    systems.
    """
    builder = QVR_DEDUCTION_PROTOCOL.schema()
    deductions = getattr(compiler, "_deductions", {})
    for name, system in deductions.items():
        sys_vid = f"deduction:{name}"
        builder.vertex(sys_vid, "deduction_system")
        builder.constraint(sys_vid, "name", name)
        builder.constraint(sys_vid, "semiring", system.semiring.__class__.__name__)
        for rule_idx, rule in enumerate(system.rules):
            rule_vid = f"{sys_vid}/rule:{rule.name}"
            builder.vertex(rule_vid, "deduction_rule")
            builder.constraint(rule_vid, "name", rule.name)
            builder.edge(sys_vid, rule_vid, "decl")
            for prem_idx, premise in enumerate(rule.premises):
                p_vid = f"{rule_vid}/premise:{prem_idx}"
                builder.vertex(p_vid, "deduction_premise")
                builder.constraint(p_vid, "pattern", repr(premise))
                builder.edge(rule_vid, p_vid, "premise")
            conc_vid = f"{rule_vid}/conclusion"
            builder.vertex(conc_vid, "deduction_conclusion")
            builder.constraint(conc_vid, "pattern", repr(rule.conclusion))
            builder.edge(rule_vid, conc_vid, "conclusion")
    return builder.build()
