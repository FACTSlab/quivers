"""Running a `MonadicProgram` on the reference machine under a handler stack.

`run_program` encodes the program as a kernel computation, installs the
handlers on the thread-local stack as lexical handlers of the program's
``random`` and ``score`` instances, serves the program's host steps
through ``Compute`` handlers over torch, and evaluates the whole on the
reference machine. The value it returns is the program's output;
handlers accumulate state on the side. A caller that wants a `Trace`
stacks a `TraceHandler` before `run_program` and reads its ``trace``
attribute afterwards; the thin `quivers.inference.trace.trace` wrapper
does exactly this.

The nesting of the installed handlers, outermost first, is fixed by the
kernel's rules rather than by the order of the stack alone: the run's
score accumulator, then every stack handler of the ``score`` instance in
stack order, then the default scoring draw and the run's observations,
then every stack handler of the ``random`` instance in stack order, then
the parameter store with every stack handler of the ``param`` instance
inside it, then the host steps. A density is a ``Score.add`` the answering
handler emits outward, so a score transformer anywhere on the stack
reaches every site, and an inner ``random`` handler sees the answer an
outer one gives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import weakref
from collections.abc import Callable, Mapping

import torch

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.core.morphisms import extract_morphism
from quivers.continuous.program_steps import _LetSpec, _ScoreSpec, _StepSpec
from quivers.continuous.programs import install_evaluator

if TYPE_CHECKING:
    from quivers.continuous.programs import MonadicProgram
from quivers.effects.base import (
    Contribution,
    EffectHandler,
    Installation,
    RunContext,
    _handler_stack,
)
from quivers.effects.program_module import (
    HostStep,
    ProgramKernel,
    program_kernel,
)
from quivers.effects.sites import TorchSampleable
from quivers.effects.trace_handler import TraceHandler
from quivers.qiec.builtins import (
    PARAM_GET,
    ExtraValuePolicy,
    MissingValuePolicy,
    compute_handler,
    condition_handler,
    draw_scoring_handler,
    param_handler,
    score_handler,
)
from quivers.qiec.effects import EffectRequest, EffectRow, HandlerDef
from quivers.qiec.evaluator import (
    ClauseContext,
    Evaluator,
    RuntimeAttachments,
    RuntimeRequest,
)
from quivers.qiec.identifiers import SiteProvenance, SourceOrigin
from quivers.qiec.kinds import NAT
from quivers.qiec.module import NamedComputation, QiecModule
from quivers.qiec.canonical import tensor_type
from quivers.qiec.terms import (
    Call,
    Computation,
    Handle,
    LiteralValue,
    Local,
    Perform,
    Var,
)
from quivers.qiec.types import REAL, STRING, IndexLiteral
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.effects import ComputationType
from quivers.qiec.module import KernelRegistry, validate_module
from quivers.effects.program_module import SOURCE_PROTOCOL

#: The name of the wrapper computation a run evaluates.
RUN = "__run"


def host_value(value: object) -> bool:
    """Whether a value is one a host step may produce or consume.

    Parameters
    ----------
    value : object
        The value.

    Returns
    -------
    bool
        True for tensors, numbers, strings, sampleables, ``None``, and
        tuples of such values. The kernel checks the terms statically;
        the host convention is that every tensor carries the batch axis
        in front of the shape its type names.
    """
    if isinstance(value, tuple):
        return all(host_value(item) for item in value)
    return value is None or isinstance(
        value, torch.Tensor | int | float | bool | str | TorchSampleable
    )


class _Method(torch.nn.Module):
    """A module whose forward calls a named method of an inner module.

    ``torch.func.functional_call`` substitutes parameters for a forward
    pass only; routing a method through a forward lets a morphism's
    ``rsample`` or ``log_prob`` run under substituted parameters.

    Parameters
    ----------
    inner : ContinuousMorphism
        The morphism whose methods are called.
    """

    def __init__(self, inner: ContinuousMorphism) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, method: str, *arguments: torch.Tensor) -> torch.Tensor:
        """Call a method of the inner module.

        Parameters
        ----------
        method : str
            The method's name.
        *arguments : torch.Tensor
            Its arguments.

        Returns
        -------
        torch.Tensor
            What the method returned.
        """
        return getattr(self.inner, method)(*arguments)


class _ParameterizedMorphism(ContinuousMorphism):
    """A morphism run under parameters a ``Param`` handler supplied.

    Parameters
    ----------
    inner : ContinuousMorphism
        The morphism.
    parameters : Mapping[str, torch.Tensor]
        The values to run it under, by the morphism's own parameter
        names.
    """

    def __init__(
        self, inner: ContinuousMorphism, parameters: Mapping[str, torch.Tensor]
    ) -> None:
        super().__init__(inner.domain, inner.codomain)
        self._inner = inner
        self._method = _Method(inner)
        self._values = {f"inner.{name}": value for name, value in parameters.items()}

    def _call(self, method: str, *arguments: torch.Tensor) -> torch.Tensor:
        """Run one method of the inner morphism under the supplied parameters.

        Parameters
        ----------
        method : str
            The method's name.
        *arguments : torch.Tensor
            Its arguments.

        Returns
        -------
        torch.Tensor
            What the method returned.
        """
        return torch.func.functional_call(
            self._method, self._values, (method, *arguments)
        )

    def rsample(  # type: ignore[override]
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Draw under the supplied parameters.

        Parameters
        ----------
        x : torch.Tensor
            The input.
        sample_shape : torch.Size
            Extra sample dimensions, which the inner morphism interprets.

        Returns
        -------
        torch.Tensor
            The draw.
        """
        if sample_shape:
            return self._call("rsample", x, torch.as_tensor(tuple(sample_shape)))
        return self._call("rsample", x)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Score under the supplied parameters.

        Parameters
        ----------
        x : torch.Tensor
            The input.
        y : torch.Tensor
            The value scored.

        Returns
        -------
        torch.Tensor
            The log density.
        """
        return self._call("log_prob", x, y)

    def __getattr__(self, name: str) -> torch.Tensor | torch.nn.Module:
        """Read a submodule, or any other attribute from the inner morphism.

        Parameters
        ----------
        name : str
            The attribute.

        Returns
        -------
        torch.Tensor | torch.nn.Module
            The registered submodule or parameter of that name, else the
            inner morphism's attribute, so a strategy reading its
            parameter methods sees them.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "_inner":
                raise
            return getattr(super().__getattr__("_inner"), name)


def _split_observations(
    program: MonadicProgram, observations: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split observations into site values and host data.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    observations : Mapping[str, torch.Tensor]
        Values keyed by name.

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]
        The values at declared sites, keyed by site label; the values
        standing in for let bindings, by name, which their steps read
        rather than compute; and the remaining entries, which are host
        data the program's steps read by name.

    Raises
    ------
    ValueError
        If an observation names one variable of a destructuring draw.
    """
    declared: dict[str, str] = {}
    computed: set[str] = set()
    for spec in program._step_specs:
        if isinstance(spec, _LetSpec | _ScoreSpec):
            computed.add(spec.var)
        else:
            label = ",".join(spec.vars)
            for name in spec.vars:
                declared[name] = label
    sites: dict[str, torch.Tensor] = {}
    supplied: dict[str, torch.Tensor] = {}
    data: dict[str, torch.Tensor] = {}
    for name, value in observations.items():
        if name in computed:
            supplied[name] = value
            continue
        label = declared.get(name)
        if label is None:
            data[name] = value
        elif label == name:
            sites[name] = value
        else:
            raise ValueError(
                f"observation {name!r} names one variable of the destructuring "
                f"draw {label!r}; observe the draw as a whole"
            )
    return sites, supplied, data


def _environment(step: HostStep, argument: object) -> dict[str, torch.Tensor]:
    """Rebuild the host environment a step reads from its argument tuple.

    Parameters
    ----------
    step : HostStep
        The step, naming the locals in order.
    argument : object
        The environment tuple the request carried.

    Returns
    -------
    dict[str, torch.Tensor]
        The locals by name.

    Raises
    ------
    TypeError
        If the argument is not a tuple of the step's arity.
    """
    if not isinstance(argument, tuple) or len(argument) != len(step.environment):
        raise TypeError(
            f"host step expects an environment of {len(step.environment)} values"
        )
    return dict(zip(step.environment, argument, strict=True))


def _canonical(morphism: ContinuousMorphism, value: torch.Tensor) -> torch.Tensor:
    """Restore a clamped plate latent's structured shape.

    Parameters
    ----------
    morphism : ContinuousMorphism
        The site's morphism.
    value : torch.Tensor
        The clamped value, which may arrive flattened.

    Returns
    -------
    torch.Tensor
        The value in the shape the morphism scores, when it has a
        canonical latent shape; the value unchanged otherwise.
    """
    canonical = getattr(morphism, "canonical_latent", None)
    if canonical is not None:
        return canonical(value)
    return value


class _HostSteps:
    """The host functions serving one program's ``Compute`` instances.

    Parameters
    ----------
    run : RunContext
        The run.
    x : torch.Tensor
        The program input.
    """

    def __init__(self, run: RunContext, x: torch.Tensor) -> None:
        self.run = run
        self.program = run.kernel.program
        self.x = x

    def function(self, step: HostStep) -> Callable[[object, ClauseContext], object]:
        """The host function of a step.

        Parameters
        ----------
        step : HostStep
            The step.

        Returns
        -------
        Callable[[object, ClauseContext], object]
            The function the step's ``Compute`` handler calls with the
            environment tuple and the clause context.

        Raises
        ------
        ValueError
            If the step is of an unknown kind.
        """
        if step.kind == "input":
            return self._split_input
        if step.kind == "draw":
            return lambda argument, context: self._draw(step, argument, context)
        if step.kind == "apply":
            return lambda argument, _context: self._apply(step, argument)
        if step.kind == "bind":
            return lambda argument, _context: self._bind(step, argument)
        if step.kind == "let":
            return lambda argument, _context: self._let(step, argument)
        if step.kind == "score":
            return lambda argument, _context: self._score(step, argument)
        raise ValueError(f"unknown host step kind {step.kind!r}")

    def _split_input(self, argument: object, _context: ClauseContext) -> object:
        """Split the program input into its named parameters.

        Parameters
        ----------
        argument : object
            The environment tuple, holding the input first.
        _context : ClauseContext
            Unused.

        Returns
        -------
        object
            The parameters as a tuple, an index squeezed to a vector.
        """
        program = self.program
        assert program._params is not None
        assert program._param_dims is not None
        assert program._param_is_continuous is not None
        splits = torch.split(self.x, program._param_dims, dim=-1)
        parts: list[torch.Tensor] = []
        for chunk, continuous in zip(splits, program._param_is_continuous, strict=True):
            if not continuous and chunk.shape[-1] == 1:
                parts.append(chunk.squeeze(-1))
            else:
                parts.append(chunk)
        del argument
        return tuple(parts)

    def _draw(self, step: HostStep, argument: object, context: ClauseContext) -> object:
        """Build the sampleable of a draw step.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : object
            The environment tuple.
        context : ClauseContext
            The clause context, through which the morphism's parameters
            are read from the ``param`` instance.

        Returns
        -------
        object
            The morphism at its resolved input, under the parameters the
            ``param`` handlers answered.
        """
        spec = step.spec
        assert isinstance(spec, _StepSpec)
        env = _environment(step, argument)
        morphism = self.program._modules[spec.morphism_name]
        assert isinstance(morphism, ContinuousMorphism)
        inp = self.program._resolve_input(spec, self.x, env)
        assert step.site is not None
        values = self._parameters(step.site, morphism, context)
        if values:
            return TorchSampleable(
                _ParameterizedMorphism(morphism, values), inp, declared=morphism
            )
        return TorchSampleable(morphism, inp)

    def _parameters(
        self, site: str, morphism: ContinuousMorphism, context: ClauseContext
    ) -> dict[str, torch.Tensor]:
        """Read a morphism's parameters through the ``param`` instance.

        Parameters
        ----------
        site : str
            The site the morphism draws at, which qualifies the names.
        morphism : ContinuousMorphism
            The morphism.
        context : ClauseContext
            The clause context the requests are performed through.

        Returns
        -------
        dict[str, torch.Tensor]
            The parameter values, by the morphism's own names.
        """
        kernel = self.run.kernel
        values: dict[str, torch.Tensor] = {}
        for name, parameter in morphism.named_parameters():
            qualified = f"{site}.{name}"
            shape = tuple(int(extent) for extent in parameter.shape)
            type_ = tensor_type(
                REAL, tuple(IndexLiteral(extent, NAT) for extent in shape)
            )
            request = EffectRequest(
                kernel.param.entry.instance,
                kernel.param.entry.effect,
                PARAM_GET,
                (type_,),
                (LiteralValue(qualified, STRING),),
                type_,
                SiteProvenance(
                    SourceOrigin(
                        kernel.module.module,
                        ("sites", site, "parameters", name),
                        "effect-request",
                        SOURCE_PROTOCOL,
                    )
                ),
            )
            value = context.evaluate(Perform(request))
            assert isinstance(value, torch.Tensor)
            values[name] = value
        return values

    def _apply(self, step: HostStep, argument: object) -> object:
        """Contract a categorical morphism's tensor against its input.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : object
            The environment tuple.

        Returns
        -------
        object
            The morphism's action on the batched input.
        """
        spec = step.spec
        assert isinstance(spec, _StepSpec)
        env = _environment(step, argument)
        module = self.program._modules[spec.morphism_name]
        assert module is not None
        categorical = extract_morphism(module)
        assert categorical is not None
        inp = self.program._resolve_input(spec, self.x, env)
        return self.program._apply_categorical_morphism(
            categorical, inp, self.x.shape[0]
        )

    def _bind(self, step: HostStep, argument: object) -> object:
        """Split a destructuring draw into its variables.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : object
            The environment tuple, holding the joint draw last.

        Returns
        -------
        object
            One tensor per bound variable.
        """
        spec = step.spec
        assert isinstance(spec, _StepSpec)
        env = _environment(step, argument)
        result = env[",".join(spec.vars)]
        scratch: dict[str, torch.Tensor] = {}
        self.program._bind_result(spec, result, scratch)
        return tuple(scratch[name] for name in spec.vars)

    def _let(self, step: HostStep, argument: object) -> object:
        """Compute a let binding.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : object
            The environment tuple.

        Returns
        -------
        object
            The bound value: an alias, a closure's result, or a constant
            broadcast over the batch.
        """
        spec = step.spec
        assert isinstance(spec, _LetSpec)
        supplied = self.run.supplied.get(spec.var)
        if supplied is not None:
            return supplied
        env = _environment(step, argument)
        if isinstance(spec.value, str):
            return env[spec.value]
        if callable(spec.value):
            return spec.value(env)
        return torch.full((self.x.shape[0],), float(spec.value), device=self.x.device)

    def _score(self, step: HostStep, argument: object) -> object:
        """Compute a score step's contribution.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : object
            The environment tuple.

        Returns
        -------
        object
            The contribution.
        """
        spec = step.spec
        assert isinstance(spec, _ScoreSpec)
        return spec.score(_environment(step, argument))


def _handle_chain(installations: list[Installation], inner: Computation) -> Computation:
    """Nest handler installations around a computation, outermost first.

    Parameters
    ----------
    installations : list[Installation]
        The installations, outermost first.
    inner : Computation
        The computation handled.

    Returns
    -------
    Computation
        The nested ``Handle`` terms.
    """
    body = inner
    for installation in reversed(installations):
        body = Handle(
            installation.instance.entry.instance,
            installation.runtime.definition.id,
            body,
            (),
        )
    return body


#: The kernel encodings and validated run modules of each program, keyed
#: by the program. An encoding depends on the program's steps and the
#: data names a run supplies; a run module also depends on the handlers
#: the run installs, whose definitions the module's identity carries. A
#: run rebuilds neither, so a program traced many times pays for its
#: encoding and its module's validation once.
_KERNELS: weakref.WeakKeyDictionary[
    MonadicProgram, dict[tuple[str, ...], ProgramKernel]
] = weakref.WeakKeyDictionary()
_REGISTRIES: weakref.WeakKeyDictionary[
    MonadicProgram, dict[tuple[tuple[str, ...], tuple[HandlerDef, ...]], KernelRegistry]
] = weakref.WeakKeyDictionary()


def _cached_kernel(program: MonadicProgram, data: tuple[str, ...]) -> ProgramKernel:
    """The kernel encoding of a program at the given data names.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    data : tuple[str, ...]
        The names of the host data a run supplies, sorted.

    Returns
    -------
    ProgramKernel
        The encoding, built once per program and data-name set.
    """
    kernels = _KERNELS.setdefault(program, {})
    kernel = kernels.get(data)
    if kernel is None:
        kernel = program_kernel(program, data)
        kernels[data] = kernel
    return kernel


def _checked_registry(
    program: MonadicProgram, kernel: ProgramKernel, module: QiecModule
) -> KernelRegistry:
    """The registry of a validated run module.

    Parameters
    ----------
    program : MonadicProgram
        The program the run module was built for.
    kernel : ProgramKernel
        The encoding the run module extends.
    module : QiecModule
        The run module: the kernel's declarations, the run's handlers,
        and the run's wrapper.

    Returns
    -------
    KernelRegistry
        The registry `validate_module` resolved, once per encoding and
        handler set: the module's declarations and wrapper are fixed by
        those, so two such runs build the same module.
    """
    registries = _REGISTRIES.setdefault(program, {})
    key = (kernel.data, tuple(module.handlers))
    registry = registries.get(key)
    if registry is None:
        registry = validate_module(module)
        registries[key] = registry
    return registry


def run_program(
    program: MonadicProgram,
    x: torch.Tensor,
    observations: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Execute a program on the reference machine under the handler stack.

    Parameters
    ----------
    program : MonadicProgram
        Program to execute.
    x : torch.Tensor
        Program input. Shape ``(batch, ...)``.
    observations : Mapping[str, torch.Tensor] or None
        Values to condition observed sites on, keyed by site name.
        Undeclared keys are host data the program's steps read by name.

    Returns
    -------
    torch.Tensor or dict[str, torch.Tensor]
        The program's return value: the single returned variable, or the
        returned variables keyed by their labels.

    Raises
    ------
    ValueError
        If an observation names one variable of a destructuring draw.
    """
    sites, supplied, data = _split_observations(program, observations or {})
    kernel = _cached_kernel(program, tuple(sorted(data)))
    run = RunContext(kernel, host_value, observations=dict(sites), supplied=supplied)
    stack = list(_handler_stack())
    steps = _HostSteps(run, x)
    result_type = kernel.result_type

    def record(request: RuntimeRequest, weight: object) -> None:
        """Record a contribution reaching the accumulator.

        Parameters
        ----------
        request : RuntimeRequest
            The score request.
        weight : object
            The contribution.
        """
        assert isinstance(weight, torch.Tensor | float | int)
        run.contributions.append(
            Contribution(
                run.key_of(request),
                request.core.origin.origin.role,
                torch.as_tensor(weight),
                request.core.origin.origin.structural_path,
            )
        )

    accumulator, _ = score_handler(
        answer_type=result_type, observer=record, key="run-accumulator"
    )
    score_installations = [Installation(kernel.score, accumulator)]
    random_installations: list[Installation] = []
    param_installations: list[Installation] = []
    user_random: list[Installation] = []
    step_observers: list[Installation] = []
    step_instances = {item.entry.instance for item in kernel.module.instances}
    finishers: list[EffectHandler] = []
    for handler in stack:
        for installation in handler.install(run):
            if installation.instance is kernel.score:
                score_installations.append(installation)
            elif installation.instance is kernel.random:
                user_random.append(installation)
            elif installation.instance is kernel.param:
                param_installations.append(installation)
            elif installation.instance.entry.instance in step_instances:
                step_observers.append(installation)
            else:
                raise ValueError(
                    f"{type(handler).__name__} installs on an instance the run "
                    "does not serve"
                )
        finishers.append(handler)
    random_installations.append(
        Installation(
            kernel.random,
            draw_scoring_handler(
                score_instance=kernel.score.entry.instance,
                result_validator=host_value,
                answer_type=result_type,
                key="run-draw",
            ),
        )
    )
    if sites:
        clamped = {
            label: _canonical(
                steps.program._modules[
                    next(
                        spec.morphism_name
                        for spec in program._step_specs
                        if isinstance(spec, _StepSpec) and ",".join(spec.vars) == label
                    )
                ],  # type: ignore[arg-type]
                value,
            )
            for label, value in sites.items()
        }
        random_installations.append(
            Installation(
                kernel.random,
                condition_handler(
                    clamped,
                    score_instance=kernel.score.entry.instance,
                    result_validator=host_value,
                    missing=MissingValuePolicy.FORWARD,
                    extra=ExtraValuePolicy.IGNORE,
                    answer_type=result_type,
                    key="run-observations",
                ),
            )
        )
    random_installations.extend(user_random)
    store: dict[str, torch.Tensor] = {}
    for step in kernel.steps:
        if step.kind != "draw" or step.site is None:
            continue
        assert isinstance(step.spec, _StepSpec)
        morphism = program._modules[step.spec.morphism_name]
        assert isinstance(morphism, ContinuousMorphism)
        for name, parameter in morphism.named_parameters():
            store[f"{step.site}.{name}"] = parameter
    run.parameters = store
    param_installations.insert(
        0,
        Installation(
            kernel.param,
            param_handler(
                lambda name, _request: store[name],
                result_validator=host_value,
                answer_type=result_type,
                key="run-parameters",
            ),
        ),
    )
    step_installations: list[Installation] = []
    for step in kernel.steps:
        instance = next(
            item
            for item in kernel.module.instances
            if item.entry.instance == step.instance
        )
        introduced = (
            EffectRow((kernel.param.entry,)) if step.kind == "draw" else EffectRow()
        )
        step_installations.append(
            Installation(
                instance,
                compute_handler(
                    steps.function(step),
                    effect=step.effect,
                    result_validator=host_value,
                    introduced=introduced,
                    answer_type=result_type,
                    key=f"run-step-{instance.name}",
                ),
            )
        )
    entry = kernel.module.computations[0]
    arguments = [x, *(data[name] for name in kernel.data)]
    parameters = [Local(local.name, local.type) for local in entry.parameters]
    call = Call(
        entry.id,
        entry.name,
        (),
        tuple(Var(parameter) for parameter in parameters),
        result_type,
        entry.type.effects,
        SourceOrigin(kernel.module.module, ("run", "call"), "call", SOURCE_PROTOCOL),
    )
    # A handler is installed only over an effect the entry performs:
    # a program with no draw performs neither Random nor Param, and
    # one with no draw and no score step performs no Score either.
    has_draws = any(step.kind == "draw" for step in kernel.steps)
    performs_score = entry.type.effects.lookup(kernel.score.entry.instance) is not None
    if not has_draws:
        random_installations = []
        param_installations = []
        if not performs_score:
            score_installations = []
    installations = [
        installation.named(f"{installation.runtime.definition.name}#{position}")
        for position, installation in enumerate(
            [
                *score_installations,
                *random_installations,
                *param_installations,
                *step_installations,
                *step_observers,
            ]
        )
    ]
    body = _handle_chain(installations, call)
    wrapper = NamedComputation(
        ComputationId.derive(kernel.module.module, "computation", RUN),
        RUN,
        (),
        tuple(parameters),
        body,
        ComputationType(EffectRow(), result_type),
        SourceOrigin(kernel.module.module, ("run",), "computation", SOURCE_PROTOCOL),
    )
    module = QiecModule(
        kernel.module.module,
        kernel.module.source_protocol,
        effects=kernel.module.effects,
        instances=kernel.module.instances,
        handlers=tuple(
            installation.runtime.definition for installation in installations
        ),
        computations=(entry, wrapper),
    )
    registry = _checked_registry(program, kernel, module)
    attachments = RuntimeAttachments()
    for installation in installations:
        attachments.bind_handler(installation.runtime)
    evaluator = Evaluator(
        attachments, module=module, type_validator=lambda _type: host_value
    )
    # The run owns its handlers: a program a host step runs in turn (a
    # score closing over another program's joint, a sub-program drawn as
    # a step) is a separate invocation, whose sites this run's handlers
    # neither observe nor condition. The stack is cleared for the
    # evaluation and restored after it.
    active = _handler_stack()
    entered = active[:]
    active.clear()
    try:
        output = evaluator.evaluate_validated(
            body, registry, dict(zip(parameters, arguments, strict=True))
        )
    finally:
        active.extend(entered)
    for handler in finishers:
        handler.finish(run, output)
    if program._return_is_single:
        assert isinstance(output, torch.Tensor)
        return output
    assert isinstance(output, tuple)
    keys = program._return_labels if program._return_labels else program._return_vars
    return {key: value for key, value in zip(keys, output, strict=True)}


def sample_program(
    program: MonadicProgram,
    x: torch.Tensor,
    sample_shape: torch.Size = torch.Size(),
    observations: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Run a program forward under the handler stack, with sample dimensions.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    x : torch.Tensor
        Program input. Shape ``(batch, ...)``.
    sample_shape : torch.Size
        Leading sample dimensions. Each element of the shape is one
        independent run of the program; the returned values stack the
        runs ahead of their batch axis.
    observations : Mapping[str, torch.Tensor] or None
        Values to condition observed sites on, keyed by site name, and
        host data the steps read by name.

    Returns
    -------
    torch.Tensor or dict[str, torch.Tensor]
        The program's return value, with ``sample_shape`` in front.
    """
    if not sample_shape:
        return run_program(program, x, observations)
    count = 1
    for extent in sample_shape:
        count *= int(extent)
    runs = [run_program(program, x, observations) for _ in range(count)]
    first = runs[0]
    if isinstance(first, torch.Tensor):
        stacked = torch.stack([run for run in runs if isinstance(run, torch.Tensor)])
        return stacked.reshape(*sample_shape, *first.shape)
    assert isinstance(first, dict)
    return {
        key: torch.stack([cast_dict(run)[key] for run in runs]).reshape(
            *sample_shape, *first[key].shape
        )
        for key in first
    }


def cast_dict(value: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Read a program's dictionary return.

    Parameters
    ----------
    value : torch.Tensor | dict[str, torch.Tensor]
        A program's return value.

    Returns
    -------
    dict[str, torch.Tensor]
        The value, which is a dictionary.

    Raises
    ------
    TypeError
        If the value is a single tensor.
    """
    if isinstance(value, dict):
        return value
    raise TypeError("the program returns a single tensor, not named values")


def log_joint(
    program: MonadicProgram,
    x: torch.Tensor,
    intermediates: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """The joint log density of a program at values for all of its draws.

    The program runs on the reference machine under the handler stack
    with every draw conditioned on its given value, and the joint is
    the sum of the site densities the run accumulated, one term per
    site, over the input's batch axis.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    x : torch.Tensor
        Program input. Shape ``(batch, ...)``.
    intermediates : Mapping[str, torch.Tensor]
        A value for every draw the program makes, keyed by variable
        name or by return label, together with any host data its steps
        read by name.

    Returns
    -------
    torch.Tensor
        The joint log density, shaped ``(batch,)``.

    Raises
    ------
    KeyError
        If a draw the program makes is given no value.
    """
    values = dict(intermediates)
    if program._return_labels:
        for label, name in zip(
            program._return_labels, program._return_vars, strict=True
        ):
            if label in values and name not in values:
                values[name] = values[label]
    with TraceHandler() as handler:
        run_program(program, x, values)
    total = torch.zeros(x.shape[0], device=x.device)
    for name, site in handler.trace.sites.items():
        if (
            site.sampleable is not None
            and not site.is_observed
            and not site.is_deterministic
        ):
            raise KeyError(name)
        total = total + site.log_prob
    return total


class _ReferenceEvaluator:
    """The reference machine as the evaluator programs run through."""

    def sample_program(
        self,
        program: MonadicProgram,
        x: torch.Tensor,
        sample_shape: torch.Size,
        observations: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run the program forward; see `sample_program`.

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
        return sample_program(program, x, sample_shape, observations)

    def log_joint(
        self,
        program: MonadicProgram,
        x: torch.Tensor,
        intermediates: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Score the program; see `log_joint`.

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
        return log_joint(program, x, intermediates)


install_evaluator(_ReferenceEvaluator())

__all__ = ["RUN", "host_value", "log_joint", "run_program", "sample_program"]
