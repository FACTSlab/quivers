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


import collections.abc

import torch
from typing import cast

from quivers.continuous.morphisms import AnySpace, ContinuousMorphism


class _StepSpec:
    """Metadata for a single draw step (not a module, just a record).

    Parameters
    ----------
    vars : tuple[str, ...]
        Bound variable name(s). Single-element for simple binding,
        multi-element for destructuring.
    morphism_name : str
        Key into the program's morphism module dict.
    args : tuple[str, ...] or None
        Names of bound variables to use as input (stacked along
        feature dim), or None for the program input.
    """

    __slots__ = ("vars", "morphism_name", "args", "is_observed")

    def __init__(
        self,
        vars: tuple[str, ...],
        morphism_name: str,
        args: tuple[str, ...] | None,
        is_observed: bool = False,
    ) -> None:
        self.vars: tuple[str, ...] = vars
        self.morphism_name: str = morphism_name
        self.args: tuple[str, ...] | None = args
        self.is_observed: bool = is_observed


class _LetSpec:
    """Metadata for a deterministic let binding (no morphism).

    Parameters
    ----------
    var : str
        Variable name to bind.
    value : float, str, or callable
        Constant literal (float), name of a bound variable to
        alias (str), or a callable that computes the value from
        the environment dict.
    """

    __slots__ = ("var", "value")

    def __init__(
        self, var: str, value: float | str | collections.abc.Callable[..., torch.Tensor]
    ) -> None:
        self.var = var
        self.value = value


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
    ) -> None:
        super().__init__(domain, codomain)
        self._return_vars = return_vars
        self._return_is_single = len(return_vars) == 1
        self._params = params
        self._return_labels = return_labels
        self._step_specs: list[_StepSpec | _LetSpec] = []

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
            # support both 3-element (backward compat) and 4-element tuples
            if len(step) == 4:
                var_names, morph, arg_or_value, is_observed = step

            else:
                var_names, morph, arg_or_value = step
                is_observed = False

            if morph is None:
                # let binding: arg_or_value is float | str
                self._step_specs.append(_LetSpec(var_names[0], arg_or_value))

            else:
                key = f"_step_{var_names[0]}"
                self.add_module(key, morph)
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
            return env[spec.args[0]]

        # multiple args: stack along feature dimension
        parts = [env[a] for a in spec.args]
        return self._stack_tensors(parts)

    @staticmethod
    def _stack_tensors(parts: list[torch.Tensor]) -> torch.Tensor:
        """Stack tensors along feature dimension.

        Handles both 1D (discrete, shape (batch,)) and 2D
        (continuous, shape (batch, d)) tensors by unsqueezing
        1D tensors before concatenation.

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

        return torch.cat(expanded, dim=-1)

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

        Each draw step is executed in order. Steps that reference
        the program input use ``x`` directly; steps that reference
        bound variables use those variables' sampled values.

        Parameters
        ----------
        x : torch.Tensor
            Program input.
        sample_shape : torch.Size
            Additional leading sample dimensions (applied to the
            first draw only; subsequent draws inherit the shape).
        observations : dict[str, torch.Tensor] or None
            Values to clamp observed variables to. Keys are variable
            names, values are tensors of the appropriate shape.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            The value of the return variable(s). Returns a tensor
            for single-variable returns, or a dict keyed by variable
            name for tuple returns.
        """
        if observations is None:
            observations = {}

        env: dict[str, torch.Tensor] = {}

        # pre-populate env with named params (split product input)
        if self._params is not None and self._param_dims is not None:
            splits = torch.split(x, self._param_dims, dim=-1)

            assert self._param_is_continuous is not None
            for pname, chunk, is_cont in zip(
                self._params, splits, self._param_is_continuous
            ):
                # only squeeze discrete components (continuous dim=1 should stay 2D)
                if not is_cont and chunk.shape[-1] == 1:
                    env[pname] = chunk.squeeze(-1)

                else:
                    env[pname] = chunk

        for i, spec in enumerate(self._step_specs):
            if isinstance(spec, _LetSpec):
                # deterministic binding: constant, alias, or expression
                if isinstance(spec.value, str):
                    env[spec.var] = env[spec.value]

                elif callable(spec.value):
                    env[spec.var] = cast(torch.Tensor, spec.value(env))

                else:
                    env[spec.var] = torch.full(
                        (x.shape[0],),
                        spec.value,
                        device=x.device,
                    )

                continue

            assert self._modules[spec.morphism_name] is not None
            morph = cast(ContinuousMorphism, self._modules[spec.morphism_name])
            inp = self._resolve_input(spec, x, env)

            # check if any vars in this step are observed
            if len(spec.vars) == 1:
                var_name = spec.vars[0]

                if spec.is_observed and var_name in observations:
                    # clamp to observed value
                    env[var_name] = observations[var_name]
                    continue

            else:
                # destructuring: if observed vars are present, clamp them
                any_clamped = False

                for v in spec.vars:
                    if spec.is_observed and v in observations:
                        env[v] = observations[v]
                        any_clamped = True

                if any_clamped:
                    # for partially observed destructuring, sample the rest
                    all_clamped = all(v in observations for v in spec.vars)

                    if not all_clamped:
                        result = morph.rsample(inp)
                        # only bind un-clamped vars
                        if isinstance(result, dict):
                            result_dict = cast(dict[str, torch.Tensor], result)
                            for v in spec.vars:
                                if v not in observations:
                                    env[v] = result_dict[v]

                        else:
                            dims = self._compute_component_dims(morph.codomain)
                            splits = torch.split(result, dims, dim=-1)

                            for v, chunk in zip(spec.vars, splits):
                                if v not in observations:
                                    env[v] = (
                                        chunk.squeeze(-1)
                                        if chunk.shape[-1] == 1
                                        else chunk
                                    )

                    continue

            # only apply sample_shape to the first draw from input
            if i == 0 and spec.args is None and len(sample_shape) > 0:
                result = morph.rsample(inp, sample_shape)

            else:
                result = morph.rsample(inp)

            self._bind_result(spec, result, env)

        # return
        if self._return_is_single:
            return env[self._return_vars[0]]

        # use labels as keys if available, otherwise variable names
        keys = self._return_labels if self._return_labels else self._return_vars
        return {k: env[v] for k, v in zip(keys, self._return_vars)}

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
        program input or a previously drawn variable).

        For destructuring draw steps (tuple-returning sub-programs),
        the intermediates dict should contain entries for each
        individual variable name.

        Parameters
        ----------
        x : torch.Tensor
            Program input.
        intermediates : dict[str, torch.Tensor]
            Values for ALL bound variables (keyed by variable name
            or by return label if labels are set).

        Returns
        -------
        torch.Tensor
            Joint log-density. Shape (batch,).
        """
        total = torch.zeros(x.shape[0], device=x.device)

        # if labels are used, map label keys back to variable names
        env = dict(intermediates)

        if self._return_labels:
            for label, var in zip(self._return_labels, self._return_vars):
                if label in env and var not in env:
                    env[var] = env[label]

        if self._params is not None and self._param_dims is not None:
            splits = torch.split(x, self._param_dims, dim=-1)

            assert self._param_is_continuous is not None
            for pname, chunk, is_cont in zip(
                self._params, splits, self._param_is_continuous
            ):
                if pname not in env:
                    if not is_cont and chunk.shape[-1] == 1:
                        env[pname] = chunk.squeeze(-1)

                    else:
                        env[pname] = chunk

        for spec in self._step_specs:
            if isinstance(spec, _LetSpec):
                # deterministic binding: populate env, contribute 0
                if spec.var not in env:
                    if isinstance(spec.value, str):
                        env[spec.var] = env[spec.value]

                    elif callable(spec.value):
                        env[spec.var] = cast(torch.Tensor, spec.value(env))

                    else:
                        env[spec.var] = torch.full(
                            (x.shape[0],),
                            spec.value,
                            device=x.device,
                        )

                continue

            assert self._modules[spec.morphism_name] is not None
            morph = cast(ContinuousMorphism, self._modules[spec.morphism_name])
            inp = self._resolve_input(spec, x, env)

            if len(spec.vars) == 1:
                val = env[spec.vars[0]]
                total = total + morph.log_prob(inp, val)

            else:
                # destructuring step: if sub-program, call its log_joint
                # with the individual intermediate values
                if hasattr(morph, "log_joint") and hasattr(morph, "_return_vars"):
                    # reconstruct the sub-program's intermediates from
                    # the overall intermediates dict
                    sub_morph = cast(MonadicProgram, morph)
                    sub_intermediates = {}

                    for sub_spec in sub_morph._step_specs:
                        if isinstance(sub_spec, _LetSpec):
                            continue

                        for sv in sub_spec.vars:
                            if sv in env:
                                sub_intermediates[sv] = env[sv]

                    total = total + sub_morph.log_joint(inp, sub_intermediates)

                else:
                    # product-codomain morphism: reconstruct stacked output
                    parts = [env[v] for v in spec.vars]
                    val = self._stack_tensors(parts)
                    total = total + morph.log_prob(inp, val)

        return total

    def __repr__(self) -> str:
        parts = []

        for s in self._step_specs:
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
