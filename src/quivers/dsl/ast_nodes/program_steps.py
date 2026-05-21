"""Program-block step AST nodes (). The
parser emits one AST class per surface step keyword. The internal
IR forms (``DrawStep``, ``PlateDrawStep``, ``VectorisedObserveStep``,
``GroupedLatentInitStep``, ``GroupedBodyObserveStep``,
``MarginalizeStep`` (compiler-only),
``GroupedObserveEntry``) remain unchanged: the compiler expands each
surface step into one of these specialized IR shapes at the entry to
``_compile_program``.

Surface AST (what the parser emits):

* `SampleStep` - ``sample v[: A] <- F(args) [options]``
* `ObserveStep` - ``observe v[: A] <- F(args) [options]``
* `MarginalizeStep` - ``marginalize c[: K] <- F(args) [options] : scope``
* `LetStep` - ``let x = expr``
* `ReturnStep` - ``return v`` or ``return (a, b, ...)``

Compiler-only IR (synthesized at compile time):

* `BindStep` - the unified Kleisli-bind shape used internally
  by the compiler when expanding surface steps.
* `DrawStep`, `PlateDrawStep`,
  `VectorisedObserveStep` - specializations.
* `GroupedLatentInitStep`, `GroupedBodyObserveStep`,
  `GroupedObserveEntry` - grouped-marginalize machinery.
"""

from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes._shared import AxisSpec, OptionEntry
from quivers.dsl.ast_nodes.let_expressions import LetExprNode
from quivers.dsl.ast_nodes.objects import ObjectExpr

class ProgramStep(dx.TaggedUnion, discriminator="kind"):
    """Sum of program-block step node kinds."""

# ---------------------------------------------------------------------------
# Surface program steps
# ---------------------------------------------------------------------------

class SampleStep(ProgramStep):
    """``sample vars[: index] <- morphism(args) [options]``.

    Surface forms::

        sample v <- F(args)                         # scalar draw
        sample v : A <- F(args)                     # A-indexed plate
        sample [a, b] <- F(args)                    # destructuring tuple

    Optional ``[options]`` block carries axis-role and other
    family-level config (move #9 ``[over=[...], iid=[...]]``).
    """

    vars: tuple[str, ...]
    morphism: str
    args: tuple[str | float, ...] | None = None
    index: ObjectExpr | None = None
    axes: AxisSpec | None = None
    options: tuple[OptionEntry, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["sample_step"] = "sample_step"

class ObserveStep(ProgramStep):
    """``observe var[: index] <- morphism(args) [options]``.

    Scored Kleisli bind; the bound coordinate is clamped at runtime
    by the ``observations`` dict, making the resulting kernel
    sub-probabilistic.
    """

    var: str
    morphism: str
    args: tuple[str | float, ...] | None = None
    index: ObjectExpr | None = None
    axes: AxisSpec | None = None
    via: str | None = None
    via_axes: tuple[str, ...] | None = None
    options: tuple[OptionEntry, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["observe_step"] = "observe_step"

class MarginalizeStep(ProgramStep):
    """``marginalize var[: index] <- morphism(args) [options] : scope``.

    Introduces a coordinate, executes the scope's steps with that
    coordinate in trace context, and at the end of the scope pushes
    forward through the projection (logsumexp for discrete, fibrewise
    integration for continuous). When the option block carries
    ``over=...``, the reduction is fibred (grouped marginalize); the
    ``via`` clause on each inner observe names the per-observe
    fibration.
    """

    var: str
    morphism: str
    args: tuple[str | float, ...] | None = None
    index: ObjectExpr | None = None
    over: str | None = None
    over_objs: tuple[str, ...] | None = None
    reduction: str | None = None
    options: tuple[OptionEntry, ...] = ()
    scope: tuple[ProgramStep, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["marginalize_step"] = "marginalize_step"

class LetStep(ProgramStep):
    """``let name = value`` deterministic step.

    The value is a let-arithmetic expression evaluated against the
    surrounding program env.
    """

    name: str
    value: LetExprNode
    line: int = 0
    col: int = 0
    kind: Literal["let_step"] = "let_step"

class ScoreStep(ProgramStep):
    """``score name = value`` log-density factor step.

    Like `LetStep` in that ``name`` is bound to the evaluated
    expression's value for downstream reference, but distinct in
    that the value is *additionally* added to the program's
    ``log_joint``. This is the deduction-style analog of
    `ObserveStep`: the chart's `goal_weight` is a
    log-density factor on the corpus, and ``score log_Z = chart.goal_weight()``
    expresses that contribution as a first-class step.
    """

    name: str
    value: LetExprNode
    line: int = 0
    col: int = 0
    kind: Literal["score_step"] = "score_step"

class ReturnStep(ProgramStep):
    """``return v`` or ``return (a, b)`` or ``return (a: x, b: y)`` step.

    Always the terminal step of a program body. ``labels`` is
    non-``None`` for the labelled-tuple form ``return (a: x, b: y)``.
    """

    vars: tuple[str, ...]
    labels: tuple[str, ...] | None = None
    line: int = 0
    col: int = 0
    kind: Literal["return_step"] = "return_step"

# ---------------------------------------------------------------------------
# Compiler-only IR (synthesized from surface steps at compile time)
# ---------------------------------------------------------------------------


class GroupedObserveEntry(dx.Model):
    """One entry in a `GroupedMarginalizeStep`'s
    ``body_observes`` list.

    Pairs an env slot (where the captured observe writes its
    ``(N_m, K)`` log-likelihood) with the fibration that carries
    those rows into the shared grouping plate.
    """

    ll_slot: str
    fibration_var: str | None = None
    fibration_axes: tuple[str, ...] | None = None
    line: int = 0
    col: int = 0


class GroupedMarginalizeStep(ProgramStep):
    """Internal compiler IR: a marginalisation pushforward.

    The compiler lowers a surface `MarginalizeStep` into this
    shape after expanding the scope. ``class_size`` is the resolved
    cardinality of the latent index; ``probs_var`` names the env
    slot holding the family's probability tensor; ``over_obj`` /
    ``over_objs`` carry the grouping object (single or product);
    ``body_ll_var`` names the env slot that the grouped observe
    pushed its (N_m, K) log-likelihood into; ``body_observes`` lists
    the (ll_slot, fibration) entries that the runtime callable
    consumes for grouped pushforward.
    """

    var_name: str
    class_size: int
    probs_var: str | None = None
    over_obj: str | None = None
    over_objs: tuple[str, ...] | None = None
    body_ll_var: str = ""
    body_observes: tuple[GroupedObserveEntry, ...] | None = None
    reduction: str | None = None
    line: int = 0
    col: int = 0
    kind: Literal["grouped_marginalize_step"] = "grouped_marginalize_step"


class BindStep(ProgramStep):
    """Internal compiler IR: a unified Kleisli bind.

    Synthesized from surface `SampleStep` / `ObserveStep`
    / `MarginalizeStep` during template instantiation. The
    surface AST never carries this shape directly.
    """

    vars: tuple[str, ...]
    morphism: str
    args: tuple[str | float, ...] | None = None
    index: ObjectExpr | None = None
    mode: Literal["sample", "score", "marginal"] = "sample"
    scope: tuple[ProgramStep, ...] | None = None
    axes: AxisSpec | None = None
    over: str | None = None
    over_objs: tuple[str, ...] | None = None
    via: str | None = None
    via_axes: tuple[str, ...] | None = None
    reduction: str | None = None
    line: int = 0
    col: int = 0
    kind: Literal["bind_step"] = "bind_step"

class DrawStep(ProgramStep):
    """Internal compiler IR: a scalar sample or score step.

    Synthesised from a `SampleStep`/`ObserveStep` with
    no index annotation; ``is_observed`` distinguishes sample
    (``False``) from score (``True``).
    """

    vars: tuple[str, ...]
    morphism: str
    args: tuple[str | float, ...] | None = None
    is_observed: bool = False
    axes: AxisSpec | None = None
    line: int = 0
    col: int = 0
    kind: Literal["draw_step"] = "draw_step"

class PlateDrawStep(ProgramStep):
    """Internal compiler IR: an A-indexed sample step.

    Synthesised from a `SampleStep` with an index annotation;
    realises a Kern-morphism ``A -> Kern(K)`` as a tensor of shape
    ``(|A|, *K.shape)``.
    """

    name: str
    index: ObjectExpr
    codomain: ObjectExpr
    morphism: str
    args: tuple[str | float, ...] | None = None
    axes: AxisSpec | None = None
    line: int = 0
    col: int = 0
    kind: Literal["plate_draw_step"] = "plate_draw_step"

class VectorisedObserveStep(ProgramStep):
    """Internal compiler IR: an A-indexed score step.

    Synthesised from an `ObserveStep` with an index
    annotation; denotes the sub-probabilistic kernel
    ``Phi -> Kern_{<=1}(Phi)`` with score
    ``prod_n p_F(r_obs(n); theta(n, phi))``.
    """

    index_var: str
    index_set: ObjectExpr
    morphism: str
    args: tuple[str | float, ...] | None = None
    response_var: str = ""
    fibration_var: str | None = None
    fibration_axes: tuple[str, ...] | None = None
    line: int = 0
    col: int = 0
    kind: Literal["vectorized_observe_step"] = "vectorized_observe_step"

class GroupedLatentInitStep(ProgramStep):
    """Internal compiler IR: initialise the latent's env slot to
    ``torch.arange(class_size)`` at the start of a grouped
    marginalize block's body.
    """

    latent_name: str
    class_size: int
    line: int = 0
    col: int = 0
    kind: Literal["grouped_latent_init_step"] = "grouped_latent_init_step"

class GroupedBodyObserveStep(ProgramStep):
    """Internal compiler IR: a captured observe inside a grouped
    marginalize block.
    """

    response_var: str
    morphism: str
    args: tuple[str | float, ...] | None = None
    index_set: ObjectExpr | None = None
    index_var: str = ""
    latent_name: str = ""
    class_size: int = 0
    fibration_var: str | None = None
    fibration_axes: tuple[str, ...] | None = None
    ll_slot: str = ""
    line: int = 0
    col: int = 0
    kind: Literal["grouped_body_observe_step"] = "grouped_body_observe_step"


__all__ = [
    "BindStep",
    "DrawStep",
    "GroupedBodyObserveStep",
    "GroupedLatentInitStep",
    "GroupedMarginalizeStep",
    "GroupedObserveEntry",
    "LetStep",
    "MarginalizeStep",
    "ObserveStep",
    "PlateDrawStep",
    "ProgramStep",
    "ReturnStep",
    "SampleStep",
    "VectorisedObserveStep",
]
