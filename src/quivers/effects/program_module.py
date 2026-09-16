"""The kernel encoding of a `MonadicProgram`.

A `MonadicProgram` is a sequence of host steps: a draw from a torch
morphism at an input built from earlier bindings, a deterministic binding
computed by a host closure, or a score contributed by one. The kernel has
no term for a host function, and does not need one: each step's host part
is an instance of the prelude's ``Compute`` interface, performed on the
values it reads (the arguments of a draw, the names a closure declares,
or every value bound so far when it declares none), and each draw is the
canonical ``Random.sample`` request on the sampleable the host part
returns.

The computation this module builds therefore makes the program's data flow
and effect structure explicit while every tensor operation stays in the
host, behind a handler a runtime provider installs. The user's effect
handlers are lexical handlers of the same ``random`` and ``score``
instances, so conditioning, intervention, replay, reweighting, and tracing
compose by the kernel's own rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from collections.abc import Sequence
from dataclasses import dataclass, field

from quivers.continuous.program_steps import (
    _LetSpec,
    _ScoreSpec,
    _StepSpec,
    reads_of,
)

if TYPE_CHECKING:
    from quivers.continuous.programs import MonadicProgram
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.spaces import ContinuousSpace, ProductSpace
from quivers.core.morphisms import extract_morphism
from quivers.core.objects import SetObject
from quivers.qiec.builtins import (
    COMPUTE,
    COMPUTE_APPLY,
    COMPUTE_EFFECT,
    PARAM,
    PARAM_EFFECT,
    RANDOM,
    RANDOM_EFFECT,
    RANDOM_SAMPLE,
    SCORE,
    SCORE_ADD,
    SCORE_EFFECT,
)
from quivers.qiec.canonical import LOG_WEIGHT, sampleable_type, site_type, tensor_type
from quivers.qiec.effects import (
    ComputationType,
    EffectRequest,
    EffectRow,
    RowEntry,
    instantiate_effect,
)
from quivers.qiec.identifiers import (
    ComputationId,
    EffectInstanceId,
    SiteProvenance,
    SourceOrigin,
)
from quivers.qiec.module import NamedComputation, NamedEffectInstance, QiecModule
from quivers.qiec.terms import (
    Bind,
    Computation,
    Local,
    Perform,
    Projection,
    Return,
    SiteValue,
    TupleValue,
    Var,
)
from quivers.qiec.types import (
    INT,
    REAL,
    UNIT,
    EffectRef,
    IndexLiteral,
    TypeExpr,
    product_type,
)
from quivers.qiec.kinds import NAT

#: The source protocol the encoding claims; there is no source text.
SOURCE_PROTOCOL = "quivers-program/host"

#: The names of the canonical instances the encoding allocates.
RANDOM_INSTANCE = "random"
SCORE_INSTANCE = "score"
PARAM_INSTANCE = "param"

#: The name of the computation holding the program.
ENTRY = "program"

#: The local the program's input is bound to, which no step variable may
#: reuse; step callables read the input from the environment under it.
INPUT_NAME = "_x_input"


@dataclass(frozen=True, slots=True)
class HostStep:
    """One host part of the program, as the runtime provider serves it.

    Parameters
    ----------
    instance
        The ``Compute`` instance the step performs on.
    effect
        The applied ``Compute[X, A]`` interface of the instance.
    kind
        ``"input"`` for the program's parameter split, ``"draw"`` for a
        morphism applied to its input, ``"apply"`` for a categorical
        morphism's tensor contracted against its input, ``"bind"`` for
        a destructuring split, ``"let"`` for a deterministic binding,
        ``"score"`` for a scored binding.
    spec
        The program's step record, or ``None`` for the input split.
    environment
        The names of the locals the step reads, in the order the
        environment tuple carries them.
    answer_type
        The type the host part returns.
    site
        The site label of a draw step, else ``None``.
    """

    instance: EffectInstanceId
    effect: EffectRef
    kind: str
    spec: _StepSpec | _LetSpec | _ScoreSpec | None
    environment: tuple[str, ...]
    answer_type: TypeExpr
    site: str | None = None


@dataclass(frozen=True, slots=True)
class ProgramKernel:
    """A program's computation with the record the runtime needs to serve it.

    Parameters
    ----------
    module
        The kernel module: the interfaces, the instances, and the entry
        computation, without handlers; a run adds its own.
    program
        The program encoded.
    steps
        The host steps, in body order.
    sites
        The site label of every draw, in order.
    site_types
        The value type of each site, by label.
    data
        The names of the host data inputs after the program's input, in
        parameter order.
    random
        The canonical ``Random`` instance.
    score
        The canonical ``Score`` instance.
    param
        The canonical ``Param`` instance.
    result_type
        The entry computation's result type.
    """

    module: QiecModule
    program: MonadicProgram
    steps: tuple[HostStep, ...]
    sites: tuple[str, ...]
    site_types: dict[str, TypeExpr]
    data: tuple[str, ...]
    random: NamedEffectInstance
    score: NamedEffectInstance
    param: NamedEffectInstance
    result_type: TypeExpr
    locals: dict[str, Local] = field(default_factory=dict)


def space_type(space: SetObject | ContinuousSpace) -> TypeExpr:
    """The kernel type of a value of a space.

    Parameters
    ----------
    space : SetObject | ContinuousSpace
        A program's domain, codomain, or a morphism's codomain.

    Returns
    -------
    TypeExpr
        ``Int`` for a finite set, ``Tensor[Real]([dim])`` for a
        continuous space of that dimension, and the product of the
        component types for a product space.
    """
    if isinstance(space, ProductSpace):
        return product_type(*(space_type(component) for component in space.components))
    if isinstance(space, ContinuousSpace):
        return tensor_type(REAL, (IndexLiteral(int(space.dim), NAT),))
    return INT


def _origin(path: tuple[str | int, ...], role: str, module: str) -> SourceOrigin:
    """A source origin inside the encoding.

    Parameters
    ----------
    path : tuple[str | int, ...]
        The structural path.
    role : str
        The role of the node.
    module : str
        The module name.

    Returns
    -------
    SourceOrigin
        The origin, carrying no file position.
    """
    return SourceOrigin(module, path, role, SOURCE_PROTOCOL)


def _arg_reads(args: tuple[object, ...] | None) -> tuple[str, ...]:
    """The environment names a draw's arguments read.

    Parameters
    ----------
    args : tuple[object, ...] | None
        The draw's arguments: bound names, indexed references, or
        ``None`` for the program input.

    Returns
    -------
    tuple[str, ...]
        The names read, the program input's included when the draw
        takes it.
    """
    if args is None:
        return (INPUT_NAME,)
    names: list[str] = []
    for arg in args:
        if isinstance(arg, str):
            names.append(arg)
        elif getattr(arg, "kind", None) == "index":
            names.append(getattr(arg, "name"))
            names.extend(getattr(arg, "indices"))
        else:
            text = getattr(arg, "text", None)
            if isinstance(text, str):
                names.append(text)
    return tuple(names)


def _let_reads(spec: _LetSpec) -> tuple[str, ...] | None:
    """The environment names a let step reads.

    Parameters
    ----------
    spec : _LetSpec
        The step.

    Returns
    -------
    tuple[str, ...] | None
        The aliased name, nothing for a constant, a closure's declared
        names, or ``None`` when the closure declares none.
    """
    if isinstance(spec.value, str):
        return (spec.value,)
    if callable(spec.value):
        names = reads_of(spec.value)
        return None if names is None else tuple(sorted(names))
    return ()


def _step_sites(spec: _StepSpec) -> str:
    """The site label of a draw step.

    Parameters
    ----------
    spec : _StepSpec
        The step.

    Returns
    -------
    str
        The bound name, or the bound names joined by commas for a
        destructuring draw, which is one draw of the joint.
    """
    return ",".join(spec.vars)


def _module_name(program: MonadicProgram) -> str:
    """The module name the encoding of a program takes.

    Parameters
    ----------
    program : MonadicProgram
        The program.

    Returns
    -------
    str
        A name derived from the program's identity, so two programs
        encode to distinct modules.
    """
    return f"program_{id(program):x}"


class _Builder:
    """Accumulates the encoding of one program.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    data : Sequence[str]
        The names of the host data inputs a run supplies beside the
        program's input.
    """

    def __init__(self, program: MonadicProgram, data: Sequence[str]) -> None:
        self.program = program
        self.module = _module_name(program)
        self.data = tuple(data)
        self.instances: list[NamedEffectInstance] = []
        self.steps: list[HostStep] = []
        self.sites: list[str] = []
        self.site_types: dict[str, TypeExpr] = {}
        self.locals: dict[str, Local] = {}
        # How many names have been rebound, which keeps rebound locals apart.
        self.rebinds = 0
        self.order: list[str] = []
        self.binds: list[tuple[Local, Computation]] = []
        self.random = self._instance(RANDOM_INSTANCE, RANDOM)
        self.score = self._instance(SCORE_INSTANCE, SCORE)
        self.param = self._instance(PARAM_INSTANCE, PARAM)

    def _instance(self, name: str, effect: EffectRef) -> NamedEffectInstance:
        """Allocate a named lexical instance.

        Parameters
        ----------
        name : str
            The instance's name.
        effect : EffectRef
            The applied interface.

        Returns
        -------
        NamedEffectInstance
            The instance, added to the module.
        """
        entry = instantiate_effect(
            effect, module=self.module, lexical_path=("instances", name)
        )
        instance = NamedEffectInstance(
            name, entry, _origin(("instances", name), "effect-instance", self.module)
        )
        self.instances.append(instance)
        return instance

    def _bind(self, name: str, type_: TypeExpr, computation: Computation) -> Local:
        """Bind a local to a computation's result.

        Parameters
        ----------
        name : str
            The local's name.
        type_ : TypeExpr
            Its type.
        computation : Computation
            The computation whose result it takes.

        Returns
        -------
        Local
            The local.
        """
        # A step may rebind a name the program bound before, as a
        # nested marginalize block does for its placeholder lets; the
        # kernel local is then fresh, and the host environment reads
        # the name's latest binding.
        rebound = name in self.locals
        local = Local(f"{name}#{self.rebinds}" if rebound else name, type_)
        if rebound:
            self.rebinds += 1
        else:
            self.order.append(name)
        self.locals[name] = local
        self.binds.append((local, computation))
        return local

    def _environment(
        self, reads: Sequence[str] | None
    ) -> tuple[tuple[str, ...], TupleValue, TypeExpr]:
        """The tuple of the locals a step reads.

        Parameters
        ----------
        reads : Sequence[str] | None
            The names the step reads, of which those bound so far are
            passed in binding order; ``None`` passes every local bound
            so far.

        Returns
        -------
        tuple[tuple[str, ...], TupleValue, TypeExpr]
            The local names in order, the tuple value, and its type.
        """
        if reads is None:
            names = tuple(self.order)
        else:
            wanted = set(reads)
            names = tuple(name for name in self.order if name in wanted)
        locals_ = [self.locals[name] for name in names]
        value = TupleValue(
            tuple(Var(local) for local in locals_),
            product_type(*(local.type for local in locals_)),
        )
        return names, value, product_type(*(local.type for local in locals_))

    def _host(
        self,
        name: str,
        kind: str,
        spec: _StepSpec | _LetSpec | _ScoreSpec | None,
        answer_type: TypeExpr,
        site: str | None = None,
        reads: Sequence[str] | None = None,
    ) -> Local:
        """Add a host step and bind its result.

        Parameters
        ----------
        name : str
            The name of the local the result binds.
        kind : str
            The step's kind.
        spec : _StepSpec | _LetSpec | _ScoreSpec | None
            The program's step record.
        answer_type : TypeExpr
            The type the host part returns.
        site : str | None
            The site label of a draw step.
        reads : Sequence[str] | None
            The names the step reads, or ``None`` for every local bound
            so far.

        Returns
        -------
        Local
            The bound result.
        """
        environment_names, environment, environment_type = self._environment(reads)
        effect = EffectRef(COMPUTE.id, COMPUTE.name, (environment_type, answer_type))
        position = len(self.steps)
        instance = self._instance(f"step_{position}", effect)
        self.steps.append(
            HostStep(
                instance.entry.instance,
                effect,
                kind,
                spec,
                environment_names,
                answer_type,
                site,
            )
        )
        request = EffectRequest(
            instance.entry.instance,
            effect,
            COMPUTE_APPLY,
            (),
            (environment,),
            answer_type,
            SiteProvenance(
                _origin(("steps", position, "host"), "effect-request", self.module)
            ),
        )
        return self._bind(name, answer_type, Perform(request))

    def _sample(self, label: str, sampleable: Local, type_: TypeExpr) -> Local:
        """Add a ``Random.sample`` request at a site.

        Parameters
        ----------
        label : str
            The site label.
        sampleable : Local
            The local holding the sampleable.
        type_ : TypeExpr
            The sampled value's type.

        Returns
        -------
        Local
            The bound draw.
        """
        request = EffectRequest(
            self.random.entry.instance,
            self.random.entry.effect,
            RANDOM_SAMPLE,
            (type_,),
            (SiteValue(label, site_type(type_)), Var(sampleable)),
            type_,
            SiteProvenance(_origin(("sites", label), "effect-request", self.module)),
        )
        self.sites.append(label)
        self.site_types[label] = type_
        return self._bind(label, type_, Perform(request))

    def _add_score(self, weight: Local, position: int) -> None:
        """Add a ``Score.add`` of a bound weight.

        Parameters
        ----------
        weight : Local
            The local holding the weight.
        position : int
            The step's position, for the request's origin.
        """
        request = EffectRequest(
            self.score.entry.instance,
            self.score.entry.effect,
            SCORE_ADD,
            (),
            (Var(weight),),
            UNIT,
            SiteProvenance(
                _origin(("steps", position, "score"), "effect-request", self.module)
            ),
        )
        unit = Local(f"__score_{position}", UNIT)
        self.binds.append((unit, Perform(request)))

    def build(self) -> ProgramKernel:
        """Encode the program.

        Returns
        -------
        ProgramKernel
            The kernel module and the record the runtime serves it by.
        """
        program = self.program
        parameters: list[Local] = []
        x = Local(INPUT_NAME, space_type(program.domain))
        self.locals[INPUT_NAME] = x
        self.order.append(INPUT_NAME)
        parameters.append(x)
        for name in self.data:
            local = Local(name, REAL)
            self.locals[name] = local
            self.order.append(name)
            parameters.append(local)
        if program._params is not None and program._param_dims is not None:
            components = tuple(
                self._component_type(index) for index in range(len(program._params))
            )
            split = self._host(
                "__params", "input", None, product_type(*components), reads=()
            )
            for index, name in enumerate(program._params):
                self._bind(
                    name,
                    components[index],
                    Return(Projection(Var(split), index, components[index])),
                )
        for position, spec in enumerate(program._step_specs):
            if isinstance(spec, _LetSpec):
                self._host(spec.var, "let", spec, REAL, reads=_let_reads(spec))
            elif isinstance(spec, _ScoreSpec):
                weight = self._host(
                    spec.var, "score", spec, LOG_WEIGHT, reads=reads_of(spec.score)
                )
                self._add_score(weight, position)
            else:
                self._draw(spec)
        result, result_type = self._result()
        body: Computation = Return(result)
        performed: list[RowEntry] = []
        for local, computation in reversed(self.binds):
            if isinstance(computation, Perform):
                entry = RowEntry(
                    computation.request.instance, computation.request.effect
                )
                if entry not in performed:
                    performed.append(entry)
            body = Bind(local, computation, body)
        computation = NamedComputation(
            ComputationId.derive(self.module, "computation", ENTRY),
            ENTRY,
            (),
            tuple(parameters),
            body,
            ComputationType(EffectRow(tuple(performed)), result_type),
            _origin(("computations", ENTRY), "computation", self.module),
        )
        module = QiecModule(
            self.module,
            SOURCE_PROTOCOL,
            effects=(RANDOM_EFFECT, SCORE_EFFECT, PARAM_EFFECT, COMPUTE_EFFECT),
            instances=tuple(self.instances),
            computations=(computation,),
        )
        return ProgramKernel(
            module,
            program,
            tuple(self.steps),
            tuple(self.sites),
            dict(self.site_types),
            self.data,
            self.random,
            self.score,
            self.param,
            result_type,
            dict(self.locals),
        )

    def _component_type(self, index: int) -> TypeExpr:
        """The type of one named parameter of a product-domain program.

        Parameters
        ----------
        index : int
            The parameter's position.

        Returns
        -------
        TypeExpr
            ``Int`` for a finite factor, else the real vector of the
            factor's width.
        """
        program = self.program
        assert program._param_dims is not None
        assert program._param_is_continuous is not None
        if program._param_is_continuous[index]:
            return tensor_type(REAL, (IndexLiteral(program._param_dims[index], NAT),))
        return INT

    def _draw(self, spec: _StepSpec) -> None:
        """Encode a draw step: its host sampleable and its sample request.

        Parameters
        ----------
        spec : _StepSpec
            The step.
        """
        morphism = self.program._modules[spec.morphism_name]
        assert morphism is not None
        label = _step_sites(spec)
        reads = _arg_reads(spec.args)
        categorical = extract_morphism(morphism)
        if categorical is not None and not isinstance(morphism, ContinuousMorphism):
            # A categorical morphism is deterministic: its tensor is
            # contracted against the input, and the result is bound
            # like a let.
            shape = tuple(
                IndexLiteral(int(extent), NAT) for extent in categorical.codomain.shape
            )
            self._host(label, "apply", spec, tensor_type(REAL, shape), reads=reads)
            return
        codomain = getattr(morphism, "codomain")
        type_ = space_type(codomain)
        sampleable = self._host(
            f"__sampleable_{label}",
            "draw",
            spec,
            sampleable_type(type_),
            site=label,
            reads=reads,
        )
        drawn = self._sample(label, sampleable, type_)
        if len(spec.vars) > 1:
            parts = self._host(
                f"__parts_{label}",
                "bind",
                spec,
                product_type(*(REAL for _ in spec.vars)),
                reads=(label,),
            )
            for index, name in enumerate(spec.vars):
                self._bind(name, REAL, Return(Projection(Var(parts), index, REAL)))
        del drawn

    def _result(self) -> tuple[TupleValue | Var, TypeExpr]:
        """The value the computation returns.

        Returns
        -------
        tuple[TupleValue | Var, TypeExpr]
            The returned value and its type: the single return variable,
            or the tuple of the return variables.
        """
        program = self.program
        if program._return_is_single:
            local = self.locals[program._return_vars[0]]
            return Var(local), local.type
        locals_ = [self.locals[name] for name in program._return_vars]
        type_ = product_type(*(local.type for local in locals_))
        return TupleValue(tuple(Var(local) for local in locals_), type_), type_


def program_kernel(program: MonadicProgram, data: Sequence[str] = ()) -> ProgramKernel:
    """Encode a program as a kernel module.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    data : Sequence[str]
        The names of the host data inputs a run supplies beside the
        program's input, in parameter order.

    Returns
    -------
    ProgramKernel
        The module and the record the runtime serves it by.
    """
    return _Builder(program, data).build()


__all__ = [
    "INPUT_NAME",
    "ENTRY",
    "PARAM_INSTANCE",
    "RANDOM_INSTANCE",
    "SCORE_INSTANCE",
    "SOURCE_PROTOCOL",
    "HostStep",
    "ProgramKernel",
    "program_kernel",
    "space_type",
]
