"""Monadic programs: sequenced probabilistic programs as ContinuousMorphisms.

A MonadicProgram defines a ContinuousMorphism via monadic sequencing
of draw steps. Each step samples from a named morphism, optionally
conditioned on previously drawn variables, and binds the result.
The program returns one or more of the bound variables as its output.

This corresponds to the Kleisli composition pattern used in
probabilistic programming languages like PDS (Grove & White),
where a sequence of ``let' x ~ D in ...`` bindings threads
probabilistic state through a generative model.

Features
--------
- Single and tuple returns: ``return x`` or ``return (x, y, z)``
- Named input parameters for product-domain sub-programs
- Multi-argument draw steps: ``draw z ~ f(x, y)``
- Destructuring draws from tuple-returning sub-programs:
  ``draw (a, b) ~ sub_prog(x)``

Example
-------
Given morphisms f : A -> B and g : B -> C, the monadic program::

    program p : A -> C
        draw x ~ f
        draw y ~ g(x)
        return y

is equivalent to the composition f >> g, but the program form
allows fan-out (using the input in multiple draws) and
non-linear variable dependency graphs.

PDS-style nested programs::

    program cg_update(y, z) : Belief * Belief -> Truth * Truth
        draw c ~ bern_c(y)
        draw d ~ bern_d(z)
        return (c, d)

    program factivityPrior : Entity -> Truth * Truth * Truth
        draw x ~ prior_x
        draw y ~ prior_y
        draw z ~ prior_z
        draw b ~ bern_b(x)
        draw (c, d) ~ cg_update(y, z)
        return (b, c, d)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast

import torch

from quivers.continuous.morphisms import AnySpace, ContinuousMorphism
from quivers.continuous.program_steps import _LetSpec, _ScoreSpec, _StepSpec


def _lookup_arg(
    env: "dict[str, torch.Tensor]",
    arg,
) -> "torch.Tensor":
    """Resolve a draw / observe argument against the environment.

    Two admissible arg shapes:

    * A bare identifier string (``"mu"``) returns ``env["mu"]``
      directly. Legacy compiler-synthesized step specs use this
      form.
    * A [`DrawArgIndex`][quivers.dsl.ast_nodes.DrawArgIndex] tagged
      variant (identified structurally via `arg.kind == "index"`)
      gathers ``env[arg.name]`` at every index tensor listed in
      ``arg.indices``. For a single-index reference this is the
      standard plate-gather (``mu[cls]`` with ``mu`` of shape
      ``(K, ...)`` and ``cls`` an integer tensor of shape ``(N,)``
      returns a tensor of shape ``(N, ...)``); multi-index refs
      apply the gathers left to right.

    The structured `DrawArgIndex` is the AST form the parser
    produces for the surface ``name[idx]`` notation. Central
    resolution here keeps every step-spec walker on the same code
    path. Structural identification (via `kind`) avoids importing
    the AST-node class from this leaf module and the
    resulting import cycle through the DSL compiler.
    """
    if getattr(arg, "kind", None) == "index":
        if arg.name not in env:
            raise KeyError(arg.name)
        tensor = env[arg.name]
        for ix in arg.indices:
            if ix not in env:
                raise KeyError(ix)
            tensor = tensor[env[ix]]
        return tensor
    if isinstance(arg, str) and arg in env:
        return env[arg]
    raise KeyError(arg)


class ProgramEvaluator(Protocol):
    """What runs a program: the reference machine under the handler stack.

    The evaluator lives in `quivers.effects`, which builds on programs;
    it installs itself here when imported, so a program's own methods
    reach it without the program package depending on the effects
    package at import.
    """

    def sample_program(
        self,
        program: MonadicProgram,
        x: torch.Tensor,
        sample_shape: torch.Size,
        observations: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run the program forward.

        Parameters
        ----------
        program : MonadicProgram
            The program.
        x : torch.Tensor
            Program input.
        sample_shape : torch.Size
            Leading sample dimensions.
        observations : Mapping[str, torch.Tensor] or None
            Values to clamp observed variables to, and host data.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            The program's return value.
        """
        ...

    def log_joint(
        self,
        program: MonadicProgram,
        x: torch.Tensor,
        intermediates: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Score the program at values for all of its draws.

        Parameters
        ----------
        program : MonadicProgram
            The program.
        x : torch.Tensor
            Program input.
        intermediates : Mapping[str, torch.Tensor]
            A value for every draw, and host data.

        Returns
        -------
        torch.Tensor
            The joint log density.
        """
        ...


_EVALUATOR: ProgramEvaluator | None = None


def install_evaluator(evaluator: ProgramEvaluator) -> None:
    """Install the evaluator every program's methods run through.

    Parameters
    ----------
    evaluator : ProgramEvaluator
        The evaluator.
    """
    global _EVALUATOR
    _EVALUATOR = evaluator


def _evaluator() -> ProgramEvaluator:
    """The installed evaluator.

    Returns
    -------
    ProgramEvaluator
        The evaluator `quivers.effects` installed.

    Raises
    ------
    RuntimeError
        If `quivers.effects` has not been imported.
    """
    if _EVALUATOR is None:
        raise RuntimeError(
            "no program evaluator is installed; import `quivers.effects`, "
            "whose interpreter runs programs on the reference machine"
        )
    return _EVALUATOR


class MonadicProgram(ContinuousMorphism):
    """A probabilistic program defined by monadic sequencing of draw steps.

    Each draw step samples from a ContinuousMorphism and binds the
    result to one or more named variables. Later steps can reference
    earlier bindings as their input. The program's output is the
    value(s) of the designated return variable(s).

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        The program's input space.
    codomain : SetObject or ContinuousSpace
        The program's output space.
    steps : list[tuple]
        Each entry is either (var_names, morphism, arg_names) for draw
        steps, or (var_names, None, value) for let bindings where
        value is a float constant or str variable reference.
    return_vars : tuple[str, ...]
        Name(s) of the bound variable(s) whose value(s) are
        the program output.
    params : tuple[str, ...] or None
        Named input parameters for product-domain programs.
        When set, the program input is split along the feature
        dimension and each component is pre-bound in the env.
    return_labels : tuple[str, ...] or None
        Optional labels for tuple return fields. When set, the
        output dict uses these labels as keys instead of the
        variable names. Length must match return_vars.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        steps: list[tuple],
        return_vars: tuple[str, ...],
        params: tuple[str, ...] | None = None,
        return_labels: tuple[str, ...] | None = None,
        effect_set: frozenset[str] | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        self._return_vars = return_vars
        self._return_is_single = len(return_vars) == 1
        self._params = params
        self._return_labels = return_labels
        # The effect-row annotation. None when unannotated;
        # otherwise carries the declared capability set
        # (Sample / Score / Marginal / Pure) for introspection by
        # downstream inference / dispatch code.
        self.effect_set: frozenset[str] | None = effect_set
        self._step_specs: list[_StepSpec | _LetSpec | _ScoreSpec] = []

        # compute input component dimensions for param splitting
        if params is not None and len(params) > 1:
            self._param_dims = self._compute_component_dims(domain)
            self._param_is_continuous = self._compute_component_continuous(domain)

        else:
            self._param_dims = None
            self._param_is_continuous = None

        # register each morphism as a named submodule so parameters
        # are visible to optimizers; let bindings become _LetSpec
        for step in steps:
            # The fourth tuple element is `is_observed` for draw steps
            # (when morph is not None) and `is_score` for let steps
            # (when morph is None). The two flags are mutually
            # exclusive by the morph-None partition, so the slot is
            # reused without ambiguity.
            if len(step) == 4:
                var_names, morph, arg_or_value, fourth_flag = step

            else:
                var_names, morph, arg_or_value = step
                fourth_flag = False

            if morph is None:
                # let-or-score binding: arg_or_value is float | str | callable.
                # A True `is_score` flag (only meaningful when the value is a
                # callable) routes the result through `total += score(env)`
                # in `log_joint` instead of contributing 0.
                if fourth_flag is True and callable(arg_or_value):
                    self._step_specs.append(_ScoreSpec(var_names[0], arg_or_value))

                else:
                    self._step_specs.append(_LetSpec(var_names[0], arg_or_value))

            else:
                is_observed = fourth_flag
                key = f"_step_{var_names[0]}"
                from quivers.core.morphisms import as_torch_module

                wrapped = as_torch_module(morph)
                self.add_module(key, wrapped)
                # The runtime accesses ``morph`` via
                # ``self._modules[key]`` and expects an object whose
                # ``rsample`` / ``log_prob`` methods are callable.
                # When ``morph`` was already an `nn.Module`
                # (a ContinuousMorphism), the wrapped value is the
                # morphism itself and those methods are available.
                # When ``morph`` is a backend-agnostic
                # `Morphism` (e.g. a ComposedMorphism from
                # the V-Cat hierarchy), the wrapper exposes the
                # morphism's parameters; the categorical object is
                # attached as ``wrapped._morphism`` so a runtime
                # path that needs it can recover via
                # `extract_morphism`.
                self._step_specs.append(
                    _StepSpec(var_names, key, arg_or_value, is_observed)
                )

    @staticmethod
    def _compute_component_dims(space: AnySpace) -> list[int]:
        """Compute per-component feature dimensions for a product space.

        Parameters
        ----------
        space : AnySpace
            A product space (ProductSet or similar).

        Returns
        -------
        list[int]
            Feature dimensions for each component.
        """
        from quivers.core.objects import ProductSet
        from quivers.continuous.spaces import ContinuousSpace, ProductSpace

        if isinstance(space, (ProductSet, ProductSpace)):
            dims = []

            for c in space.components:
                if isinstance(c, ContinuousSpace):
                    dims.append(c.dim)

                else:
                    # discrete FinSet component: 1 dimension (index)
                    dims.append(1)

            return dims

        # non-product: single component
        if isinstance(space, ContinuousSpace):
            return [space.dim]

        return [1]

    @staticmethod
    def _compute_component_continuous(space: AnySpace) -> list[bool]:
        """Determine which components of a product space are continuous.

        Parameters
        ----------
        space : AnySpace
            A product space.

        Returns
        -------
        list[bool]
            True for continuous components, False for discrete.
        """
        from quivers.core.objects import ProductSet
        from quivers.continuous.spaces import ContinuousSpace, ProductSpace

        if isinstance(space, (ProductSet, ProductSpace)):
            return [isinstance(c, ContinuousSpace) for c in space.components]

        return [isinstance(space, ContinuousSpace)]

    def _resolve_input(
        self,
        spec: _StepSpec,
        x: torch.Tensor,
        env: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Resolve the input tensor for a draw step.

        Parameters
        ----------
        spec : _StepSpec
            The step specification.
        x : torch.Tensor
            The raw program input.
        env : dict[str, torch.Tensor]
            Current variable environment.

        Returns
        -------
        torch.Tensor
            Input tensor for the morphism.
        """
        if spec.args is None:
            return x

        if len(spec.args) == 1:
            return self._promote_rank(_lookup_arg(env, spec.args[0]))

        # multiple args: stack along the feature dimension. When the
        # morphism declares per-parameter event ranks and dims (an
        # inline distribution), the stacking interprets each argument by
        # its declared role rather than guessing from tensor rank: a
        # rank-0 position is a per-row scalar; a rank->=1 position is a
        # vector feature of its declared dim, arriving either as a
        # shared ``(D,)`` vector or a per-row ``(N, D)`` block.
        parts = [_lookup_arg(env, a) for a in spec.args]
        morph = self._modules.get(spec.morphism_name)
        event_ranks = getattr(morph, "_param_event_ranks", None)
        param_spec = getattr(morph, "_param_spec", None)
        if event_ranks is not None and param_spec is not None:
            var_ranks = tuple(
                event_ranks[i] for i, (k, _) in enumerate(param_spec) if k == "var"
            )
            var_dims = tuple(int(v) for k, v in param_spec if k == "var")
            if len(var_ranks) == len(parts):
                return self._stack_params(parts, var_ranks, var_dims)
        return self._stack_tensors(parts)

    @staticmethod
    def _stack_params(
        parts: list[torch.Tensor],
        event_ranks: tuple[int, ...],
        dims: tuple[int, ...],
    ) -> torch.Tensor:
        """Stack inline-distribution parameters into a single input.

        Each parameter is reshaped to a per-row ``(batch, dim)`` block
        using its declared event rank and dim, and the blocks are
        concatenated along the feature axis. A shared parameter (no
        leading batch: a scalar ``()``/``(1,)`` or a vector ``(dim,)``)
        reshapes to a batch-1 block and broadcasts against any per-row
        parameters. When every parameter is shared, the result is a
        single batch-1 row; the response batch is supplied downstream by
        the observed value.
        """
        shaped: list[torch.Tensor] = []
        for t, rank, dim in zip(parts, event_ranks, dims, strict=True):
            tf = t.float()
            if rank >= 1:
                # Vector feature of size ``dim``.
                if tf.dim() <= 1:
                    # Shared vector ``(dim,)`` (or a scalar broadcast to
                    # the vector width): one row.
                    shaped.append(tf.reshape(1, -1))
                else:
                    shaped.append(tf.reshape(tf.shape[0], -1))
            else:
                # Per-row scalar.
                if tf.dim() == 0:
                    shaped.append(tf.reshape(1, 1))
                elif tf.dim() == 1:
                    shaped.append(tf.unsqueeze(-1))
                else:
                    shaped.append(tf.reshape(tf.shape[0], -1))
        batch = max((t.shape[0] for t in shaped), default=1)
        broadcast: list[torch.Tensor] = []
        for t in shaped:
            if t.shape[0] == batch:
                broadcast.append(t)
            elif t.shape[0] == 1:
                broadcast.append(t.expand(batch, *t.shape[1:]))
            else:
                raise RuntimeError(
                    "_stack_params: incompatible batch sizes "
                    f"{[t.shape[0] for t in shaped]}; expected each to be "
                    f"1 (shared) or {batch} (per-row)"
                )
        return torch.cat(broadcast, dim=-1)

    @staticmethod
    def _promote_rank(t: torch.Tensor) -> torch.Tensor:
        """Promote a rank-1 continuous tensor to ``(batch, 1)``.

        Integer tensors (which feed ``nn.Embedding`` lookups in
        ``_LookupSource``) are returned unchanged so discrete program
        inputs continue to index correctly.

        Parameters
        ----------
        t : torch.Tensor
            Tensor to promote.

        Returns
        -------
        torch.Tensor
            ``t.unsqueeze(-1)`` if ``t`` is a 1-D floating-point tensor,
            otherwise ``t`` unchanged.
        """
        if t.dtype in (
            torch.long,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.bool,
        ):
            return t
        if t.dim() == 1:
            return t.unsqueeze(-1)
        return t

    @staticmethod
    def _stack_tensors(parts: list[torch.Tensor]) -> torch.Tensor:
        """Stack tensors along feature dimension.

        Handles 1D (discrete, shape ``(batch,)``) and 2D (continuous,
        shape ``(batch, d)``) tensors by unsqueezing 1D tensors before
        concatenation, and broadcasts scalar / size-1 leading dims
        against the maximum batch size present in the input list.
        The size-1 broadcast covers the hierarchical-model case where
        a scalar latent (e.g. ``sigma``) is concatenated with a
        per-row tensor (e.g. ``mu``) ahead of an indexed observe.

        Parameters
        ----------
        parts : list[torch.Tensor]
            Tensors to stack.

        Returns
        -------
        torch.Tensor
            Concatenated tensor along dim=-1.
        """
        expanded = []
        for p in parts:
            if p.dim() == 1:
                expanded.append(p.unsqueeze(-1).float())
            else:
                expanded.append(p.float())

        batch = max(t.shape[0] for t in expanded)
        broadcast = []
        for t in expanded:
            if t.shape[0] == batch:
                broadcast.append(t)
            elif t.shape[0] == 1:
                broadcast.append(t.expand(batch, *t.shape[1:]))
            else:
                raise RuntimeError(
                    f"_stack_tensors: incompatible batch sizes "
                    f"{[t.shape[0] for t in expanded]}; expected each to be "
                    f"1 (scalar) or {batch} (per-row)"
                )

        return torch.cat(broadcast, dim=-1)

    def _bind_result(
        self,
        spec: _StepSpec,
        result: torch.Tensor | dict[str, torch.Tensor],
        env: dict[str, torch.Tensor],
    ) -> None:
        """Bind the morphism result to variable(s) in the env.

        Parameters
        ----------
        spec : _StepSpec
            The step specification.
        result : torch.Tensor or dict[str, torch.Tensor]
            The morphism output. A dict for tuple-returning
            sub-programs.
        env : dict[str, torch.Tensor]
            Variable environment (mutated in place).
        """
        if len(spec.vars) == 1:
            # simple binding
            if isinstance(result, dict):
                # sub-program returned dict but we're binding to single var
                # — shouldn't happen if types are correct
                env[spec.vars[0]] = result  # type: ignore[assignment]

            else:
                env[spec.vars[0]] = result

        else:
            # destructuring: unpack dict from sub-program
            if isinstance(result, dict):
                for var_name in spec.vars:
                    env[var_name] = result[var_name]

            else:
                # tensor result from product-codomain morphism: split
                # along feature dim
                morph = self._modules[spec.morphism_name]
                assert morph is not None
                morph_cm = cast(ContinuousMorphism, morph)
                dims = self._compute_component_dims(morph_cm.codomain)
                splits = torch.split(result, dims, dim=-1)

                for var_name, chunk in zip(spec.vars, splits):
                    env[var_name] = chunk.squeeze(-1) if chunk.shape[-1] == 1 else chunk

    @property
    def observed_names(self) -> set[str]:
        """Return the set of variable names marked as observed in the DSL."""
        names = set()

        for spec in self._step_specs:
            if isinstance(spec, _StepSpec) and spec.is_observed:
                for v in spec.vars:
                    names.add(v)

        return names

    def rsample(  # type: ignore[override]
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
        observations: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run the program forward, returning the designated output(s).

        The program runs on the reference machine under the active
        handler stack, so a `clamp`, `do`, or `trace` in scope applies
        to the draws the run makes.

        Parameters
        ----------
        x : torch.Tensor
            Program input.
        sample_shape : torch.Size
            Additional leading sample dimensions; each element is one
            independent run of the program.
        observations : dict[str, torch.Tensor] or None
            Values to clamp observed variables to, keyed by variable
            name, together with any host data the steps read by name.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            The value of the return variable(s). Returns a tensor
            for single-variable returns, or a dict keyed by variable
            name for tuple returns.
        """
        return _evaluator().sample_program(self, x, sample_shape, observations)

    def _apply_categorical_morphism(
        self,
        morph,
        inp: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        """Apply a V-enriched `Morphism` to a batched input
        tensor as a deterministic step.

        The morphism's tensor has shape ``(*dom.shape, *cod.shape)``
        with values in the algebra's lattice. For a batched input
        ``inp`` of shape ``(batch, *dom.shape)``, the V-enriched
        action is the algebra tensor product followed by join over
        the domain axes — equivalent to a matrix-vector contraction
        when the algebra is product-fuzzy / Markov / boolean. We
        delegate that contraction to the active algebra's
        ``composition_kernel`` so a single deterministic-bind path
        handles every supported algebra uniformly.
        """
        m_tensor = morph.tensor
        # Broadcast inp to (batch, *dom.shape). For a one-hot input
        # of shape (batch,) into a finite domain, treat the values
        # as integer indices and gather the corresponding rows.
        dom_shape = tuple(morph.domain.shape)
        cod_shape = tuple(morph.codomain.shape)
        if inp.dim() == 1 or (inp.dim() == 2 and inp.shape[-1] == 1):
            idx = inp.reshape(-1).to(torch.long)
            if idx.numel() != batch:
                idx = idx[:batch] if idx.numel() > batch else idx.expand(batch)
            # m_tensor has shape (*dom_shape, *cod_shape). Index
            # selecting the first len(dom_shape) axes by `idx` gives
            # (batch, *cod_shape). For a single-axis domain this is
            # m_tensor[idx]; for multi-axis we treat each component
            # of idx separately (and require dom_shape to be
            # 1-axis for the simple gather path).
            if len(dom_shape) == 1:
                return m_tensor[idx]
            # Multi-axis: contract by einsum below.
            raise RuntimeError(
                "deterministic V-Cat step: integer-indexed input "
                f"only supported for 1-axis domains; got {dom_shape}"
            )
        # General contraction via einsum: input has shape
        # (batch, *dom.shape), m has shape (*dom.shape, *cod.shape).
        in_letters = "".join(chr(ord("a") + i) for i in range(len(dom_shape)))
        out_letters = "".join(
            chr(ord("a") + len(dom_shape) + j) for j in range(len(cod_shape))
        )
        eq = f"...{in_letters},{in_letters}{out_letters}->...{out_letters}"
        return torch.einsum(eq, inp, m_tensor)

    def has_conditional_density(self) -> bool:
        """No: a program's density at its output marginalizes its draws.

        [`log_prob`][quivers.continuous.programs.MonadicProgram.log_prob]
        raises for exactly this reason, and a caller choosing between
        constructions reads the answer here rather than by catching
        that raise.
        [`log_joint`][quivers.continuous.programs.MonadicProgram.log_joint]
        is the exact route: it scores the draws themselves, given
        values for all of them.
        """
        return False

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability is not supported for monadic programs.

        Computing log p(y | x) for a monadic program requires
        marginalizing over all intermediate variables, which is
        intractable in general. Use ``rsample`` for forward sampling
        and condition via score function estimators or variational
        methods.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "log_prob is not supported for monadic programs; "
            "computing p(y | x) requires marginalizing over all "
            "intermediate draws, which is intractable in general. "
            "use rsample() for forward sampling."
        )

    def log_joint(
        self,
        x: torch.Tensor,
        intermediates: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Joint log-density given all intermediate values.

        When all intermediate variables are observed (e.g. during
        inference with HMC/NUTS), computes the joint log-density:

            log p(x_1, ..., x_n | input) = sum_i log p(x_i | pa(x_i))

        where pa(x_i) is the parent variable of step i (either the
        program input or a previously drawn variable). The program
        runs on the reference machine with every draw conditioned on
        its given value, under the active handler stack, so the joint
        is the one [`trace`][quivers.inference.trace.trace] reports.

        Parameters
        ----------
        x : torch.Tensor
            Program input.
        intermediates : dict[str, torch.Tensor]
            Values for ALL bound variables (keyed by variable name
            or by return label if labels are set), together with any
            host data the steps read by name.

        Returns
        -------
        torch.Tensor
            Joint log-density. Shape (batch,).

        Raises
        ------
        KeyError
            If a draw the program makes is given no value.
        """
        return _evaluator().log_joint(self, x, intermediates)

    def __repr__(self) -> str:
        parts = []

        for s in self._step_specs:
            if isinstance(s, _ScoreSpec):
                parts.append(f"score {s.var} = {s.score}")
                continue

            if isinstance(s, _LetSpec):
                parts.append(f"let {s.var} = {s.value}")

            else:
                assert isinstance(s, _StepSpec)
                keyword = "observe" if s.is_observed else "draw"
                lhs = f"({','.join(s.vars)})" if len(s.vars) > 1 else s.vars[0]  # type: ignore[index]
                rhs = s.morphism_name.removeprefix("_step_")

                if s.args:
                    rhs += f"({', '.join(s.args)})"

                parts.append(f"{keyword} {lhs} ~ {rhs}")

        steps = ", ".join(parts)
        if self._return_labels and not self._return_is_single:
            ret_parts = [
                f"{l}: {v}" for l, v in zip(self._return_labels, self._return_vars)
            ]
            ret = f"({', '.join(ret_parts)})"

        elif not self._return_is_single:
            ret = f"({', '.join(self._return_vars)})"

        else:
            ret = self._return_vars[0]
        params = f"({', '.join(self._params)})" if self._params else ""
        return (
            f"MonadicProgram{params}({self.domain!r} -> {self.codomain!r}, "
            f"[{steps}] -> {ret})"
        )
