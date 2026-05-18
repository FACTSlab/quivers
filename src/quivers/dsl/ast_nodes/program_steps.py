"""Program-block step AST nodes.

The parser emits exclusively :class:`BindStep` and :class:`LetStep` for
program bodies. The compiler expands a BindStep into one of the four
specialized forms below at the entry to ``_compile_program``, based on
the bind's ``mode`` and ``index`` fields:

* ``sample, no index`` -> DrawStep
* ``sample, with idx`` -> PlateDrawStep
* ``score, no index``  -> DrawStep with is_observed=True
* ``score, with idx``  -> VectorisedObserveStep
* ``marginal``         -> MarginalizeStep (the scope steps are expanded
                          inline; the variable is registered for that scope)

The compiler-only forms are not part of the public surface; they are an
internal IR consumed by the rest of the compiler / template-expansion /
runtime step-builder machinery.
"""

from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes._shared import AxisSpec
from quivers.dsl.ast_nodes.let_expressions import LetExprNode
from quivers.dsl.ast_nodes.types import TypeExpr


class ProgramStep(dx.TaggedUnion, discriminator="kind"):
    """Sum of program-block step node kinds."""


class BindStep(ProgramStep):
    """A Kleisli bind inside a program block: the unified step shape.

    Surface forms:

    .. code-block:: qvr

        v        <- F(args)                              # mode=sample, scalar
        v : A    <- F(args)                              # mode=sample, A-indexed plate
        (a, b)   <- F(args)                              # destructuring tuple bind
        observe v        <- F(args)                      # mode=score, scalar
        observe r : N    <- F(theta[N])                  # mode=score, N-indexed
        marginalize c    <- F(args) in { steps }         # mode=marginal, scoped
        marginalize c : A <- F(args) in { steps }        # mode=marginal, A-indexed

    Categorical denotation:

    * ``mode="sample"`` extends the trace by a fresh Kleisli arrow
      :math:`\\Phi \\to \\mathcal{G}(\\Phi \\times K)`. When ``index``
      is non-``None`` the iso
      :math:`\\mathbf{Kern}(\\mathbf{1}, K^A) \\cong \\mathbf{Kern}(A, K)`
      lifts the per-fiber family to an indexed family.
    * ``mode="score"`` is a sub-probabilistic Kleisli arrow
      :math:`\\Phi \\to \\mathcal{G}_{\\le 1}(\\Phi)` clamping the
      bound coordinate to a runtime-supplied observation; the
      indexed form denotes the batched-likelihood kernel
      :math:`\\prod_{n} p_F(r_{\\mathrm{obs}}(n); \\theta(n, \\phi))`.
    * ``mode="marginal"`` introduces a coordinate, executes the
      scope's steps with that coordinate in trace context, and at
      the end of the scope pushes forward through the projection
      :math:`\\pi_{\\Phi} : \\Phi \\times C \\to \\Phi` (logsumexp for
      discrete, fibrewise integration for continuous). The
      coordinate is local to ``scope``.

    Attributes
    ----------
    vars : tuple[str, ...]
        Bound names. For sample mode, may be a tuple for
        destructuring; score and marginal modes always carry a
        single name.
    index : TypeExpr | None
        Optional index-set annotation; non-``None`` for plate /
        vectorized / indexed-marginalize forms.
    morphism : str
        Family / morphism name on the kernel-expression RHS.
    args : tuple
        Family arguments. Strings of the form ``"name[Index]"`` are
        bracket-indexed family sections, categorically sections of
        an ``Index``-indexed family.
    mode : Literal["sample", "score", "marginal"]
        Kleisli-bind mode.
    scope : tuple[ProgramStep, ...] | None
        Integration scope; non-``None`` iff ``mode == "marginal"``.
    """

    vars: tuple[str, ...]
    morphism: str
    args: tuple[str | float, ...] | None = None
    index: TypeExpr | None = None
    mode: Literal["sample", "score", "marginal"] = "sample"
    scope: tuple[ProgramStep, ...] | None = None
    # Axis-role clause: ``over <axes> [iid over <axes>]`` on the
    # family invocation.  Configures the event/batch decomposition
    # of the family over the named axes of the step's type
    # annotation (``: T``).  Required when ``family.event_rank > 0``
    # and ambiguous from the annotation alone; rejected at compile
    # time on mismatch with the family's event rank.
    axes: AxisSpec | None = None
    # ``over G`` on the marginalize-mode bind declares the grouping
    # plate.  ``over_obj`` is the single plate name; ``over_objs``
    # is the tuple of plate names when the user wrote a type
    # product (e.g. ``over G * H``).  Unused on score / sample
    # binds.
    over: str | None = None
    over_objs: tuple[str, ...] | None = None
    # ``via idx`` (single fibration) or ``via product(idx_a, idx_b)``
    # (product fibration) on a score-mode bind inside a grouped
    # marginalize body.  Every observe inside the body carries its
    # own ``via`` clause naming the fibration into the shared
    # grouping plate.  Unused on sample / marginal binds.
    via: str | None = None
    via_axes: tuple[str, ...] | None = None
    # `reduction = logsumexp | sum | mean`.
    reduction: str | None = None
    line: int = 0
    col: int = 0
    kind: Literal["bind_step"] = "bind_step"


class LetStep(ProgramStep):
    """A deterministic ``let`` binding inside a program block.

    The ``value`` field always holds a :class:`LetExprNode`; bare floats
    and bare identifier aliases are wrapped in :class:`LetExprLiteral` and
    :class:`LetExprVar` respectively at parse time.
    """

    name: str
    value: LetExprNode
    line: int = 0
    col: int = 0
    kind: Literal["let_step"] = "let_step"


class DrawStep(ProgramStep):
    """Internal compiler IR: a scalar sample or score step.

    Synthesised from a :class:`BindStep` with no index annotation;
    ``is_observed`` distinguishes sample (``False``) from score
    (``True``).
    """

    vars: tuple[str, ...]
    morphism: str
    args: tuple[str | float, ...] | None = None
    is_observed: bool = False
    line: int = 0
    col: int = 0
    kind: Literal["draw_step"] = "draw_step"


class PlateDrawStep(ProgramStep):
    """Internal compiler IR: an A-indexed sample step.

    Synthesised from a :class:`BindStep` with ``mode='sample'`` and
    an index annotation. Categorically a Kern-morphism
    :math:`A \\to \\mathcal{G}(K)` realised as a single tensor of
    shape ``(|A|, *K.shape)``.
    """

    name: str
    index: TypeExpr
    codomain: TypeExpr
    morphism: str
    args: tuple[str | float, ...] | None = None
    line: int = 0
    col: int = 0
    kind: Literal["plate_draw_step"] = "plate_draw_step"


class VectorisedObserveStep(ProgramStep):
    """Internal compiler IR: an A-indexed score step.

    Synthesised from a :class:`BindStep` with ``mode='score'`` and
    an index annotation. Denotes the sub-probabilistic kernel
    :math:`\\Phi \\to \\mathcal{G}_{\\le 1}(\\Phi)` with score
    :math:`\\prod_{n} p_F(r_{\\mathrm{obs}}(n); \\theta(n, \\phi))`.
    """

    index_var: str
    index_set: TypeExpr
    morphism: str
    args: tuple[str | float, ...] | None = None
    response_var: str = ""
    # ``via <idx>`` clause on the originating observe surface step.
    # Inside a grouped marginalize body this names the per-observe
    # fibration into the shared grouping plate; the product form
    # ``via product(...)`` populates ``fibration_axes`` instead.
    # Outside a grouped body both fields are unused.
    fibration_var: str | None = None
    fibration_axes: tuple[str, ...] | None = None
    line: int = 0
    col: int = 0
    kind: Literal["vectorized_observe_step"] = "vectorized_observe_step"


class GroupedLatentInitStep(ProgramStep):
    """Internal compiler IR: initialize the latent's environment
    slot to ``torch.arange(class_size)`` at the start of a grouped
    marginalize block's body.

    The body's downstream ``let`` and ``observe`` steps then see the
    latent as a length-``K`` index tensor; any arithmetic involving
    the latent broadcasts across the class axis. The terminal
    captured observe (see :class:`GroupedBodyObserveStep`) overwrites
    this slot with the per-(N, K) log-likelihood tensor the
    marginalize step consumes.
    """

    latent_name: str
    class_size: int
    line: int = 0
    col: int = 0
    kind: Literal["grouped_latent_init_step"] = "grouped_latent_init_step"


class GroupedBodyObserveStep(ProgramStep):
    """Internal compiler IR: a captured observe inside a grouped
    marginalize block.

    The body of a grouped marginalize block ends with an observe
    step whose per-row log-likelihood depends on the latent. Rather
    than accumulating the scalar log-density into the program-level
    joint (the normal observe path), this captured form:

    1. Computes ``family.log_prob(theta, response)`` per row,
       broadcasting ``theta`` across the class axis if it carries
       one (because upstream ``let`` steps referenced the latent).
    2. Stores the resulting ``(N, K)`` tensor at the marginalize
       block's latent slot, where the
       :class:`MarginalizeStep`'s runtime callable picks it up,
       applies the prior, and reduces.

    Categorically: the captured observe is the body's
    contribution to the right Kan extension along the fibration in
    :math:`\\mathbf{Kern}`: the per-(row, class) log-likelihood
    tensor that the per-group accumulator scatter-adds.
    """

    response_var: str
    morphism: str
    args: tuple[str | float, ...] | None = None
    index_set: TypeExpr | None = None
    index_var: str = ""
    latent_name: str = ""
    # Per-observe fibration into the shared grouping plate.
    # ``fibration_var`` carries a single-axis fibration's name;
    # ``fibration_axes`` carries a product-fibration's tuple of
    # axis names.  Exactly one is set inside a grouped body; the
    # other is ``None``.
    fibration_var: str | None = None
    fibration_axes: tuple[str, ...] | None = None
    # The env slot the captured-observe's (N_m, K) per-row
    # per-class log-likelihood is written to.  Unique per observe
    # inside a single grouped body so the surrounding
    # MarginalizeStep can collect each axis's contribution
    # separately and pair it with the right fibration.
    ll_slot: str = ""
    line: int = 0
    col: int = 0
    kind: Literal["grouped_body_observe_step"] = "grouped_body_observe_step"


class GroupedObserveEntry(dx.Model):
    """One entry in a grouped :class:`MarginalizeStep`'s
    ``body_observes`` list: the pairing of an env slot (where
    the captured observe writes its ``(N_m, K)`` per-row
    per-class log-likelihood) with the fibration that carries
    those rows into the shared grouping plate.

    Exactly one of ``fibration_var`` and ``fibration_axes`` is
    non-``None``: a single-axis fibration uses
    ``fibration_var``; a product fibration uses
    ``fibration_axes`` (whose arity must match the marginalize
    header's product-plate arity).

    Both ``None`` flags a nested-marginalize entry: the inner
    block has already performed its own scatter, so the outer
    block consumes the ``(|G|, K)`` tensor at ``ll_slot``
    directly with no further fibration.
    """

    ll_slot: str
    fibration_var: str | None = None
    fibration_axes: tuple[str, ...] | None = None


class MarginalizeStep(ProgramStep):
    """Internal compiler IR: a marginalisation reduction.

    The :class:`BindStep` for marginalize is expanded by the
    compiler into: (1) a sample step that introduces the
    coordinate, (2) the scope's steps, (3) this MarginalizeStep
    that pushes forward through the projection
    :math:`\\pi_{\\Phi} : \\Phi \\times C \\to \\Phi`.

    When the surface block carries ``over G via idx``, the
    reduction is fibred: the body's per-row log-density tensor
    of shape ``(N, K)`` is scatter-added along ``via_var`` to
    shape ``(|G|, K)``, the categorical prior ``probs_var``
    contributes ``log probs[g, k]`` per (group, class), and the
    final log-sum-exp over the class axis is summed over groups.
    This denotes the right Kan extension along the fibration
    :math:`r : \\text{Resp} \\to G` in :math:`\\mathbf{Kern}`,
    followed by integration of the class axis under the
    categorical prior.
    """

    var_name: str
    class_size: int = 0
    probs_var: str | None = None
    over_obj: str | None = None
    # Product grouping plate: a tuple of plate names whose
    # cardinalities multiply to give the flat group cardinality.
    # ``None`` for a single grouping plate; in that case
    # ``over_obj`` carries the singleton name.
    over_objs: tuple[str, ...] | None = None
    body_ll_var: str | None = None
    # Grouped form: ordered tuple of per-observe entries.  Each
    # entry pairs an env slot (where the observe writes its
    # ``(N_m, K)`` log-likelihood) with the fibration into the
    # shared grouping plate.  See :class:`GroupedObserveEntry`
    # for the field semantics.  ``None`` outside a grouped body.
    body_observes: tuple[GroupedObserveEntry, ...] | None = None
    # Per-group reduction over the class axis. ``None`` defaults
    # to ``"logsumexp"`` at the runtime call site.
    reduction: str | None = None
    line: int = 0
    col: int = 0
    kind: Literal["marginalize_step"] = "marginalize_step"


__all__ = [
    "ProgramStep",
    "BindStep",
    "LetStep",
    "DrawStep",
    "PlateDrawStep",
    "VectorisedObserveStep",
    "GroupedLatentInitStep",
    "GroupedBodyObserveStep",
    "GroupedObserveEntry",
    "MarginalizeStep",
]
