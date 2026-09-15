"""Elaboration of ``program`` declarations into QIEC computations.

A program is domain-specific notation for a named computation: its data,
observations, and fibrations are the computation's typed parameters, its
``sample`` steps perform ``Random.sample`` on the module's canonical
``random`` instance, its ``observe`` and ``score`` steps perform
``Score.add`` on the canonical ``score`` instance, its ``let`` steps bind
pure expressions, its ``marginalize`` blocks call a helper computation
that enumerates a finite latent under the prelude's enumeration handler,
and its ``return`` builds the result tuple. The elaboration is a mixin
over the QIEC elaborator, since it shares the value lowering, the
registry, and the diagnostics of every other declaration.

Plates are typed data. A step's ``over`` and ``iid_over`` axes, its
``: Axis`` annotation, and a morphism's ``Real N`` codomain fix a
``PlateShape`` on the step's distribution, whose sample type is the
tensor over those axes; a ``via`` fibration re-indexes a grouped latent's
arguments to the observation rows and segments the observation's weights
back to the groups.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from quivers.dsl.ast_nodes import (
    CallStep,
    ContinuousConstructor,
    DiscreteConstructor,
    DrawArg,
    DrawArgDist,
    DrawArgIndex,
    DrawArgList,
    DrawArgName,
    DrawArgScalar,
    ExprIdent,
    LetExprBinOp,
    LetExprCall,
    LetExprIndex,
    LetExprLambda,
    LetExprList,
    LetExprMethodCall,
    LetExprNode,
    LetExprUnaryOp,
    LetExprVar,
    LetStep,
    MarginalizeStep,
    MorphismDecl,
    ObjectDecl,
    ObjectExpr,
    ObjectProduct,
    ObserveStep,
    OptionEntry,
    OptionList,
    OptionName,
    OptionNumber,
    OptionString,
    ProgramDecl,
    ProgramStep,
    ReturnStep,
    SampleStep,
    ScalarParam,
    ScoreStep,
    TypeEnumSet,
    TypeFromExpr,
    TypeName,
)
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprTuple,
)
from quivers.qiec.canonical import (
    LOG_WEIGHT,
    sampleable_type,
    site_type,
    tensor_shape,
    tensor_type,
)
from quivers.qiec.checking import (
    CheckContext,
    ComputationSignature,
    KernelError,
    infer_computation,
)
from quivers.qiec.effects import (
    ComputationType,
    EffectRef,
    EffectRequest,
    EffectRow,
    HandlerClauseDef,
    HandlerDef,
    ResumptionGrade,
    RowEntry,
    instantiate_effect,
)
from quivers.qiec.families import FAMILIES, DistributionFamily, FamilyParameter
from quivers.qiec.identifiers import (
    ComputationId,
    HandlerId,
    SiteProvenance,
)
from quivers.qiec.kinds import NAT, TypeBinder
from quivers.qiec.module import NamedComputation, NamedEffectInstance
from quivers.qiec.programs import (
    ParameterRole,
    ProgramEntry,
    ProgramParameter,
    ProgramSite,
)
from quivers.qiec.substitution import instantiate_operation, instantiate_telescope
from quivers.qiec.terms import (
    AffineMap,
    Bind,
    Call,
    Computation,
    DistributionValue,
    Gather,
    Handle,
    KernelMatrix,
    LiteralValue,
    Local,
    LogDensity,
    NewInstance,
    Perform,
    PlateAxis,
    PlateShape,
    Return,
    SegmentSum,
    SiteValue,
    TensorValue,
    TupleValue,
    Value,
    Var,
    WeightSum,
)
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    UNIT,
    IndexLiteral,
    TypeApplication,
    TypeExpr,
    TypeVariable,
    product_type,
)
from quivers.dsl.composite_lets import expand_composite_lets
from quivers.dsl.pure_builtins import PURE_BUILTINS
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.step_resolution import (
    ResolvedDist,
    StepResolutionError,
    build_let_table,
    build_morphism_table,
    resolve_step_dist,
)

if TYPE_CHECKING:
    from quivers.dsl.qiec_lowering import _Elaborator


#: The name of the handler enumerating a marginalized latent.
ENUMERATE_HANDLER = "enumerate_marginal"
#: The name of the handler collecting a marginalization scope's weights.
COLLECT_HANDLER = "collect_marginal"
#: The canonical instance names a program's requests address.
RANDOM_INSTANCE = "random"
SCORE_INSTANCE = "score"

#: The diagnostic code of a program construct whose elaboration is not yet
#: defined: a parsing chart, a chart method, or a morphism parameterized by
#: a network. Such a program is reported rather than approximated, and a
#: caller may lower the module without its programs meanwhile.
GAP_CODE = "qiec-program-gap"

#: Let-expression builtins that build or read a deduction chart.
_CHART_BUILTINS = frozenset({"parse", "chart", "chart_fold", "op_apply"})

#: Families whose bare ``~ Family`` kernel morphism reads its arguments off
#: an affine parameter map, with each head's transform.
_CONDITIONAL_HEADS: dict[
    str, tuple[tuple[str, Literal["identity", "exp_floor"]], ...]
] = {
    "Normal": (("loc", "identity"), ("scale", "exp_floor")),
}

#: Families whose sample is an index into their own alphabet, with the
#: argument naming the alphabet.
_CLASS_INDEX_FAMILIES: dict[str, str] = {"Categorical": "probs"}


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    """What the elaborator reads off one ``object`` declaration.

    Parameters
    ----------
    extent
        The object's size as an axis: its cardinality when finite, its
        coordinate count when continuous.
    real_width
        The width of a ``Real N`` object, whose values are real vectors.
    finite
        Whether the object is a finite set.
    """

    extent: int | None = None
    real_width: int | None = None
    finite: bool = False


@dataclass(slots=True)
class _Scope:
    """Bindings of one program body or marginalization scope.

    Parameters
    ----------
    locals
        Every bound name and its local, in binding order.
    steps
        The binds accumulated so far, each a binder and the computation
        it binds.
    referenced
        Names of enclosing bindings the scope has read, which a helper
        computation takes as parameters.
    group
        The grouping axis of a grouped marginalization scope, else
        ``None``.
    weight_instance
        The local ``Weight`` instance a marginalization scope adds its
        weights to, else ``None`` for the program body.
    weight_type
        The type that instance accumulates.
    outer
        The enclosing scope, else ``None``.
    """

    locals: dict[str, Local] = field(default_factory=dict)
    steps: list[tuple[Local, Computation]] = field(default_factory=list)
    referenced: set[str] = field(default_factory=set)
    group: PlateAxis | None = None
    weight_instance: NamedEffectInstance | None = None
    weight_type: TypeExpr | None = None
    outer: _Scope | None = None

    def lookup(self, name: str) -> Local | None:
        """Find a binding in this scope or an enclosing one.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        Local | None
            The innermost binding, or ``None``.
        """
        scope: _Scope | None = self
        while scope is not None:
            local = scope.locals.get(name)
            if local is not None:
                if scope is not self:
                    self.referenced.add(name)
                return local
            scope = scope.outer
        return None

    def context(self) -> CheckContext:
        """The kernel context of every binding visible here.

        Returns
        -------
        CheckContext
            Outer bindings first, then this scope's.
        """
        chain: list[_Scope] = []
        scope: _Scope | None = self
        while scope is not None:
            chain.append(scope)
            scope = scope.outer
        locals_: list[Local] = []
        for item in reversed(chain):
            locals_.extend(item.locals.values())
        return CheckContext(tuple(locals_))


def _option_value(options: tuple[OptionEntry, ...], key: str) -> str | float | None:
    """Read a scalar option off an option block.

    Parameters
    ----------
    options : tuple[OptionEntry, ...]
        The block.
    key : str
        The option's key.

    Returns
    -------
    str | float | None
        A name, a string, or a number, or ``None`` when absent or not
        scalar.
    """
    for entry in options:
        if entry.key != key:
            continue
        value = entry.value
        if isinstance(value, OptionName | OptionString):
            return value.value
        if isinstance(value, OptionNumber):
            return value.value
    return None


def _option_axes(options: tuple[OptionEntry, ...], key: str) -> tuple[str, ...]:
    """Read an axis list off an option block.

    Parameters
    ----------
    options : tuple[OptionEntry, ...]
        The block.
    key : str
        The option's key, ``over`` or ``iid_over``.

    Returns
    -------
    tuple[str, ...]
        The axis names; one name for a bare identifier.
    """
    for entry in options:
        if entry.key != key:
            continue
        value = entry.value
        if isinstance(value, OptionName):
            return (value.value,)
        if isinstance(value, OptionList):
            return tuple(
                item.value
                for item in value.items
                if isinstance(item, OptionName | OptionString)
            )
    return ()


def _factors(expr: ObjectExpr) -> tuple[ObjectExpr, ...]:
    """Flatten a product object expression into its factors.

    Parameters
    ----------
    expr : ObjectExpr
        The expression.

    Returns
    -------
    tuple[ObjectExpr, ...]
        The factors in source order; a non-product is its own factor.
    """
    if isinstance(expr, ObjectProduct):
        return tuple(
            item for component in expr.components for item in _factors(component)
        )
    return (expr,)


def _free_let_names(
    expr: LetExprNode, bound: frozenset[str] = frozenset()
) -> list[str]:
    """The variables a let expression reads, in first-use order.

    Parameters
    ----------
    expr : LetExprNode
        The expression.
    bound : frozenset[str]
        Names bound inside the expression.

    Returns
    -------
    list[str]
        Free variable names, without repeats.
    """
    found: list[str] = []

    def visit(node: LetExprNode) -> None:
        """Record the variables one node reads.

        Parameters
        ----------
        node : LetExprNode
            The node.
        """
        if isinstance(node, LetExprVar):
            if node.name not in bound and node.name not in found:
                found.append(node.name)
        elif isinstance(node, LetExprBinOp):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, LetExprUnaryOp):
            visit(node.operand)
        elif isinstance(node, LetExprCall):
            for argument in node.args:
                visit(argument)
        elif isinstance(node, LetExprIndex):
            visit(node.array)
            for index in node.indices:
                visit(index)
        elif isinstance(node, LetExprList | LetExprTuple):
            for item in node.items:
                visit(item)

    visit(expr)
    return found


def _is_gap(kinds: Sequence[str], elaborator: _Elaborator | None = None) -> bool:
    """Whether a resolution failure names a construct outside the calculus.

    Parameters
    ----------
    kinds : Sequence[str]
        The failure's structured kinds.
    elaborator : _Elaborator | None
        The elaborator, whose source says which names are program
        templates.

    Returns
    -------
    bool
        ``True`` for a network-parameterized morphism, a ``scan``
        recurrence, or a reference to a program template, whose
        elaborations are not yet defined.
    """
    for kind in kinds:
        if kind.startswith(("param-source:", "scan:")):
            return True
        if kind.startswith("family:") and elaborator is not None:
            name = kind.split(":", 1)[1].split(":", 1)[0]
            if any(
                isinstance(statement, ProgramDecl)
                and statement.name == name
                and statement.type_params is not None
                for statement in elaborator.source.syntax.statements
            ):
                return True
    return False


def _unknown_call(expr: LetExprNode, macros: Mapping[str, object]) -> str | None:
    """Name the first call to something the pure layer does not define.

    Parameters
    ----------
    expr : LetExprNode
        The expression.
    macros : Mapping[str, object]
        The lambda-bound names in scope.

    Returns
    -------
    str | None
        The called name, or ``None`` when every call is a builtin, a
        reduction, a last-axis operation, a family, or a macro.
    """
    if isinstance(expr, LetExprCall):
        if (
            expr.func not in PURE_BUILTINS
            and expr.func not in FAMILIES
            and expr.func not in macros
            and expr.func not in _CHART_BUILTINS
            and expr.func not in ("site", "log_prob")
        ):
            return expr.func
        for argument in expr.args:
            found = _unknown_call(argument, macros)
            if found is not None:
                return found
        return None
    if isinstance(expr, LetExprBinOp):
        return _unknown_call(expr.left, macros) or _unknown_call(expr.right, macros)
    if isinstance(expr, LetExprUnaryOp):
        return _unknown_call(expr.operand, macros)
    if isinstance(expr, LetExprIndex):
        found = _unknown_call(expr.array, macros)
        if found is not None:
            return found
        for index in expr.indices:
            found = _unknown_call(index, macros)
            if found is not None:
                return found
        return None
    if isinstance(expr, LetExprList | LetExprTuple):
        for item in expr.items:
            found = _unknown_call(item, macros)
            if found is not None:
                return found
    return None


def _chart_construct(expr: LetExprNode) -> str | None:
    """Name a chart construct inside a let expression, if any.

    Parameters
    ----------
    expr : LetExprNode
        The expression.

    Returns
    -------
    str | None
        A description of the first method call or chart builtin found,
        or ``None``.
    """
    if isinstance(expr, LetExprMethodCall):
        return f"the method call `.{expr.method}(...)`"
    if isinstance(expr, LetExprCall):
        if expr.func in _CHART_BUILTINS:
            return f"the builtin `{expr.func}(...)`"
        for argument in expr.args:
            found = _chart_construct(argument)
            if found is not None:
                return found
        return None
    if isinstance(expr, LetExprBinOp):
        return _chart_construct(expr.left) or _chart_construct(expr.right)
    if isinstance(expr, LetExprUnaryOp):
        return _chart_construct(expr.operand)
    if isinstance(expr, LetExprIndex):
        return _chart_construct(expr.array)
    if isinstance(expr, LetExprList | LetExprTuple):
        for item in expr.items:
            found = _chart_construct(item)
            if found is not None:
                return found
    return None


def _index_names(expr: LetExprNode) -> list[str]:
    """The variables an expression uses as tensor indices.

    Parameters
    ----------
    expr : LetExprNode
        The expression.

    Returns
    -------
    list[str]
        Names appearing directly as an index, in order.
    """
    found: list[str] = []

    def visit(node: LetExprNode) -> None:
        """Record index variables under one node.

        Parameters
        ----------
        node : LetExprNode
            The node.
        """
        if isinstance(node, LetExprIndex):
            visit(node.array)
            for index in node.indices:
                if isinstance(index, LetExprVar):
                    if index.name not in found:
                        found.append(index.name)
                else:
                    visit(index)
        elif isinstance(node, LetExprBinOp):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, LetExprUnaryOp):
            visit(node.operand)
        elif isinstance(node, LetExprCall):
            for argument in node.args:
                visit(argument)
        elif isinstance(node, LetExprList | LetExprTuple):
            for item in node.items:
                visit(item)

    visit(expr)
    return found


def _argument_names(argument: DrawArg) -> list[str]:
    """The names a draw argument reads.

    Parameters
    ----------
    argument : DrawArg
        The argument.

    Returns
    -------
    list[str]
        Bare names, indexed bases and their indices, and list entries.
    """
    if isinstance(argument, DrawArgName):
        return [argument.text]
    if isinstance(argument, DrawArgIndex):
        return [argument.name, *argument.indices]
    if isinstance(argument, DrawArgList):
        return [name for item in argument.items for name in _argument_names(item)]
    if isinstance(argument, DrawArgDist):
        return [name for item in argument.args for name in _argument_names(item)]
    return []


def _is_number_text(text: str) -> bool:
    """Whether wire text spells a number.

    Parameters
    ----------
    text : str
        The text.

    Returns
    -------
    bool
        ``True`` when ``float`` accepts it.
    """
    try:
        float(text)
    except ValueError:
        return False
    return True


def _literal(value: float, integral: bool) -> LiteralValue:
    """A numeric literal at the type a parameter expects.

    Parameters
    ----------
    value : float
        The number.
    integral : bool
        Whether the position takes an ``Int``.

    Returns
    -------
    LiteralValue
        An ``Int`` literal when integral and whole, else a ``Real``.
    """
    if integral and float(value).is_integer():
        return LiteralValue(int(value), INT)
    return LiteralValue(float(value), REAL)


def _element_of(parameter: FamilyParameter | _ScalarParameter) -> TypeApplication:
    """The scalar element type a family parameter is made of.

    Parameters
    ----------
    parameter : FamilyParameter | _ScalarParameter
        The parameter.

    Returns
    -------
    TypeApplication
        ``Bool`` for a Boolean constraint, ``Int`` for an integer one,
        else ``Real``.
    """
    if parameter.constraint == "boolean":
        return BOOL
    if "integer" in parameter.constraint:
        return INT
    return REAL


class _ProgramElaboration:
    """Program elaboration, mixed into the QIEC elaborator.

    The methods read and extend the elaborator's declaration tables, so
    they run after the module's instances, handlers, and computation
    signatures are declared and before its computation bodies are
    lowered, which lets a program call a declared computation and a
    computation call a program.
    """

    def _declare_programs(self: _Elaborator) -> tuple[NamedComputation, ...]:
        """Elaborate every non-parametric program in the module.

        Returns
        -------
        tuple[NamedComputation, ...]
            The programs' computations followed by their marginalization
            helpers, in source order.

        Raises
        ------
        QiecDiagnosticError
            If a program uses a form the elaboration does not admit, or
            its body fails to check.
        """
        declarations = [
            statement
            for statement in self.source.syntax.statements
            if isinstance(statement, ProgramDecl) and _is_entry_point(statement)
        ]
        self._program_objects = _object_table(self.source.syntax.statements)
        if not declarations or not self.source.elaborate_programs:
            return ()
        try:
            expanded = expand_composite_lets(self.source.syntax)
            self._program_morphisms = build_morphism_table(expanded)
        except StepResolutionError as error:
            self._fail(
                declarations[0],
                "; ".join(error.kinds),
                code=GAP_CODE if _is_gap(error.kinds) else "qiec-program",
            )
        self._program_lets = build_let_table(expanded)
        computations: list[NamedComputation] = []
        for declaration in expanded.statements:
            if isinstance(declaration, ProgramDecl) and _is_entry_point(declaration):
                try:
                    computations.extend(self._elaborate_program(declaration))
                except QiecDiagnosticError as error:
                    error.program = declaration.name
                    raise
        return tuple(computations)

    # ------------------------------------------------------------------
    # module-level scaffolding

    def _declare_program_instances(self: _Elaborator) -> None:
        """Allocate the canonical instances before any signature is read.

        A computation may name ``random`` or ``score`` in its row to call
        a program, so the instances exist as soon as the module has a
        program to elaborate.
        """
        if not self.source.elaborate_programs:
            return
        if not any(
            isinstance(statement, ProgramDecl) and _is_entry_point(statement)
            for statement in self.source.syntax.statements
        ):
            return
        self._program_instance(RANDOM_INSTANCE, "Random")
        self._program_instance(SCORE_INSTANCE, "Score")

    def _program_instance(
        self: _Elaborator, name: str, interface: str
    ) -> NamedEffectInstance:
        """The canonical module instance a program's requests address.

        Parameters
        ----------
        name : str
            The instance's name.
        interface : str
            The prelude interface it must be an instance of.

        Returns
        -------
        NamedEffectInstance
            The module's instance of that name, declared or synthesized.

        Raises
        ------
        QiecDiagnosticError
            If the module declares the name as an instance of another
            interface.
        """
        existing = self.instances.get(name)
        if existing is not None:
            if existing.entry.effect.name != interface:
                self._fail(
                    self.source.syntax.statements[0],
                    f"instance {name!r} is reserved for the program's {interface} "
                    f"requests but is an instance of {existing.entry.effect.name!r}",
                    code="qiec-program",
                )
            return existing
        definition = self._effect(interface)
        assert definition is not None
        entry = instantiate_effect(
            definition.ref,
            module=self.source.module_name,
            lexical_path=("instances", name),
        )
        instance = NamedEffectInstance(
            name,
            entry,
            _origin_at(self, ("instances", name), "effect-instance"),
        )
        self.instances[name] = instance
        return instance

    def _program_handlers(self: _Elaborator) -> tuple[HandlerDef, HandlerDef]:
        """The enumeration and collection handlers marginalization uses.

        Returns
        -------
        tuple[HandlerDef, HandlerDef]
            The ``enumerate_marginal[w]`` handler over ``Random``, whose
            input is a unit answer paired with the collected weight of
            type ``w`` and whose output is the marginal ``LogWeight``,
            and the ``collect_marginal[w]`` handler over ``Weight[w]``,
            which pairs the scope's unit answer with its total.

        Raises
        ------
        QiecDiagnosticError
            If the module declares a handler under one of the reserved
            names.
        """
        for name in (ENUMERATE_HANDLER, COLLECT_HANDLER):
            declared = self.handlers.get(name)
            if declared is not None and declared.implementation != "foreign":
                self._fail(
                    self.source.syntax.statements[0],
                    f"handler {name!r} is reserved for marginalization",
                    code="qiec-program",
                )
        enumerate = self.handlers.get(ENUMERATE_HANDLER)
        collect = self.handlers.get(COLLECT_HANDLER)
        if enumerate is not None and collect is not None:
            return enumerate, collect
        weight_binder = TypeBinder("w")
        weight = TypeVariable("w")
        random = self._effect("Random")
        weight_effect = self._effect("Weight")
        assert random is not None and weight_effect is not None
        sample = next(item for item in random.operations if item.name == "sample")
        add = next(item for item in weight_effect.operations if item.name == "add")
        enumerate = HandlerDef(
            HandlerId.derive(self.source.module_name, "handler", ENUMERATE_HANDLER),
            ENUMERATE_HANDLER,
            random.ref,
            (HandlerClauseDef(sample.id, ResumptionGrade.UNRESTRICTED),),
            product_type(UNIT, weight),
            LOG_WEIGHT,
            EffectRow(),
            total=True,
            telescope=(weight_binder,),
            implementation="foreign",
        )
        collect = HandlerDef(
            HandlerId.derive(self.source.module_name, "handler", COLLECT_HANDLER),
            COLLECT_HANDLER,
            weight_effect.apply((weight,)),
            (HandlerClauseDef(add.id, ResumptionGrade.LINEAR),),
            UNIT,
            product_type(UNIT, weight),
            EffectRow(),
            total=True,
            telescope=(weight_binder,),
            implementation="foreign",
        )
        self.handlers[ENUMERATE_HANDLER] = enumerate
        self.handlers[COLLECT_HANDLER] = collect
        self.registry.register_handler(enumerate)
        self.registry.register_handler(collect)
        return enumerate, collect

    # ------------------------------------------------------------------
    # one program

    def _elaborate_program(
        self: _Elaborator, declaration: ProgramDecl
    ) -> tuple[NamedComputation, ...]:
        """Elaborate one program declaration.

        Parameters
        ----------
        declaration : ProgramDecl
            The program.

        Returns
        -------
        tuple[NamedComputation, ...]
            The program's computation and its marginalization helpers.

        Raises
        ------
        QiecDiagnosticError
            If a step cannot be elaborated or the body fails to check.
        """
        if declaration.name in self.computation_signatures:
            self._fail(
                declaration,
                f"program {declaration.name!r} shares its name with a computation",
                code="qiec-program",
            )
        state = _ProgramState(declaration)
        self._program_state = state
        random = self._program_instance(RANDOM_INSTANCE, "Random")
        score = self._program_instance(SCORE_INSTANCE, "Score")
        state.random = random
        state.score = score
        scope = _Scope()
        self._program_parameters(declaration, scope, state)
        state.input_shapes = self._infer_input_shapes(declaration)
        self._elaborate_steps(declaration.draws, scope, state)
        result, result_type = self._program_return(declaration, scope, state)
        body = self._fold_steps(scope, Return(result))
        row_entries: list[RowEntry] = []
        if state.uses_random:
            row_entries.append(random.entry)
        if state.uses_score:
            row_entries.append(score.entry)
        row = EffectRow(tuple(row_entries))
        parameters = tuple(scope.locals[name] for name, _ in state.parameters)
        identity = ComputationId.derive(
            self.source.module_name, "computation", declaration.name
        )
        signature = ComputationSignature(
            identity,
            declaration.name,
            (),
            tuple(parameter.type for parameter in parameters),
            result_type,
            row,
        )
        try:
            self.registry.register_computation(signature)
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(declaration, error, fallback="qiec-program")
        self.computation_signatures[declaration.name] = signature
        computation = NamedComputation(
            identity,
            declaration.name,
            (),
            parameters,
            body,
            ComputationType(row, result_type),
            _origin_at(self, ("programs", declaration.name), "computation"),
        )
        context = CheckContext(parameters)
        try:
            actual = infer_computation(body, self.registry, context)
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(declaration, error, fallback="qiec-program")
        if actual.result != result_type:
            self._fail(
                declaration,
                f"program body returns {actual.result!r}, expected {result_type!r}",
                code="qiec-program",
            )
        self.entries.append(
            ProgramEntry(
                declaration.name,
                identity,
                tuple(
                    ProgramParameter(name, role, scope.locals[name].type)
                    for name, role in state.parameters
                ),
                tuple(declaration.return_vars),
                declaration.return_labels,
                tuple(state.sites),
                random.entry.instance,
                score.entry.instance,
            )
        )
        self._program_state = None
        return (computation, *state.helpers)

    def _program_parameters(
        self: _Elaborator, declaration: ProgramDecl, scope: _Scope, state: _ProgramState
    ) -> None:
        """Declare the parameters the program's signature names.

        A named parameter is typed by its domain factor: a real vector
        for a ``Real N`` factor, an index for a finite one. Without
        names, every ``Real N`` factor becomes a parameter named after
        its object in lowercase. Scalar parameters of the option block
        are reals or naturals.

        Parameters
        ----------
        declaration : ProgramDecl
            The program.
        scope : _Scope
            The body's scope, which the parameters are bound in.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If the parameter count does not fit the domain, or a factor
            has no readable shape.
        """
        factors = _factors(declaration.domain)
        if declaration.params is not None:
            if len(declaration.params) != len(factors):
                self._fail(
                    declaration,
                    f"program {declaration.name!r} names {len(declaration.params)} "
                    f"parameter(s) for a domain of {len(factors)} factor(s)",
                    code="qiec-program",
                )
            for name, factor in zip(declaration.params, factors, strict=True):
                info = self._object_info(factor, declaration)
                if info.real_width is not None:
                    type_: TypeExpr = tensor_type(REAL, (_extent(info.real_width),))
                elif info.finite:
                    type_ = INT
                else:
                    type_ = REAL
                self._declare_parameter(
                    name, type_, "domain", scope, state, declaration
                )
        else:
            for factor in factors:
                if not isinstance(factor, TypeName):
                    continue
                info = self._program_objects.get(factor.name)
                if info is None or info.real_width is None:
                    continue
                self._declare_parameter(
                    factor.name.lower(),
                    tensor_type(REAL, (_extent(info.real_width),)),
                    "domain",
                    scope,
                    state,
                    declaration,
                )
        if declaration.type_params is not None:
            for parameter in declaration.type_params:
                if isinstance(parameter, ScalarParam):
                    self._declare_parameter(
                        parameter.name,
                        REAL if parameter.scalar_kind == "Real" else INT,
                        "scalar",
                        scope,
                        state,
                        declaration,
                    )

    def _declare_parameter(
        self: _Elaborator,
        name: str,
        type_: TypeExpr,
        role: ParameterRole,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> Local:
        """Bind one program parameter in the body's scope.

        Parameters
        ----------
        name : str
            The parameter's name.
        type_ : TypeExpr
            Its type.
        role : ParameterRole
            How it is supplied.
        scope : _Scope
            The scope it is bound in; a nested scope binds it in the
            program body's scope beneath it.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node the use is reported at.

        Returns
        -------
        Local
            The parameter's local.

        Raises
        ------
        QiecDiagnosticError
            If the name is already a parameter at another type or a
            bound step.
        """
        root = scope
        while root.outer is not None:
            root = root.outer
        existing = root.locals.get(name)
        if existing is not None:
            if existing.type != type_:
                self._fail(
                    node,
                    f"program input {name!r} is read as {self._render(type_)} "
                    f"here but as {self._render(existing.type)} elsewhere",
                    code="qiec-program",
                )
            if scope is not root:
                scope.referenced.add(name)
            return existing
        if scope.lookup(name) is not None:
            self._fail(
                node,
                f"{name!r} is bound by a step and cannot also be a program input",
                code="qiec-program",
            )
        local = Local(name, type_)
        root.locals[name] = local
        state.parameters.append((name, role))
        if scope is not root:
            scope.referenced.add(name)
        return local

    # ------------------------------------------------------------------
    # steps

    def _elaborate_steps(
        self: _Elaborator,
        steps: Sequence[ProgramStep],
        scope: _Scope,
        state: _ProgramState,
    ) -> None:
        """Elaborate a sequence of steps into the scope's binds.

        Parameters
        ----------
        steps : Sequence[ProgramStep]
            The steps, in order.
        scope : _Scope
            The scope they bind into.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If a step is of a kind the elaboration does not admit.
        """
        for step in steps:
            if isinstance(step, SampleStep):
                self._elaborate_sample(step, scope, state)
            elif isinstance(step, ObserveStep):
                self._elaborate_observe(step, scope, state)
            elif isinstance(step, LetStep):
                self._elaborate_let(step, scope, state)
            elif isinstance(step, CallStep):
                self._elaborate_call(step, scope, state)
            elif isinstance(step, ScoreStep):
                self._elaborate_score(step, scope, state)
            elif isinstance(step, MarginalizeStep):
                self._elaborate_marginalize(step, scope, state)
            elif isinstance(step, ReturnStep):
                continue
            else:
                self._fail(
                    step,
                    f"program step {step.kind!r} has no QIEC elaboration",
                    code="qiec-program",
                )

    def _bind_step(
        self: _Elaborator,
        scope: _Scope,
        name: str | None,
        type_: TypeExpr,
        computation: Computation,
        node: object,
    ) -> Local:
        """Append one bind to a scope.

        Parameters
        ----------
        scope : _Scope
            The scope.
        name : str | None
            The bound name, or ``None`` for a step whose value is unused.
        type_ : TypeExpr
            The bound value's type.
        computation : Computation
            The computation bound.
        node : object
            The source node, for diagnostics.

        Returns
        -------
        Local
            The binder.

        Raises
        ------
        QiecDiagnosticError
            If the name is already bound.
        """
        if name is None:
            binder = Local(f"_step{len(scope.steps)}_{id(scope) & 0xFFFF:x}", type_)
        else:
            if scope.lookup(name) is not None:
                self._fail(
                    node,
                    f"variable {name!r} already bound in program",
                    code="qiec-program",
                )
            binder = Local(name, type_)
            scope.locals[name] = binder
        scope.steps.append((binder, computation))
        return binder

    def _fold_steps(self, scope: _Scope, tail: Computation) -> Computation:
        """Nest a scope's binds around a final computation.

        Parameters
        ----------
        scope : _Scope
            The scope.
        tail : Computation
            What runs after the last step.

        Returns
        -------
        Computation
            The nested binds.
        """
        body = tail
        for binder, computation in reversed(scope.steps):
            body = Bind(binder, computation, body)
        return body

    def _elaborate_sample(
        self: _Elaborator, step: SampleStep, scope: _Scope, state: _ProgramState
    ) -> None:
        """Elaborate ``sample x <- Family(args)`` to a ``Random.sample``.

        Parameters
        ----------
        step : SampleStep
            The step.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If the step destructures a tuple, or its distribution cannot
            be elaborated.
        """
        if len(step.vars) != 1:
            self._fail(
                step,
                "a sample step binds one name; tuple destructuring has no elaboration",
                code="qiec-program",
            )
        name = step.vars[0]
        state.current_site = name
        state.pending_alphabet = None
        distribution = self._step_distribution(step, scope, state, name)
        if state.pending_alphabet is not None:
            state.alphabets[name] = state.pending_alphabet
        sampled = self._sampled_type(distribution)
        state.uses_random = True
        request = self._sample_request(
            state.random, name, sampled, distribution, step, state
        )
        state.sites.append(
            ProgramSite(
                name,
                "sample",
                distribution.name,
                distribution.plate.batch,
                distribution.plate.event,
            )
        )
        self._bind_step(scope, name, sampled, Perform(request), step)

    def _sample_request(
        self: _Elaborator,
        instance: NamedEffectInstance,
        label: str,
        sampled: TypeExpr,
        distribution: DistributionValue,
        node: object,
        state: _ProgramState,
    ) -> EffectRequest:
        """Build a ``Random.sample`` request on an instance.

        Parameters
        ----------
        instance : NamedEffectInstance
            The ``Random`` instance addressed.
        label : str
            The site's label.
        sampled : TypeExpr
            The sampled value's type.
        distribution : DistributionValue
            The distribution.
        node : object
            The source node.
        state : _ProgramState
            The program's accumulating state, for the site path.

        Returns
        -------
        EffectRequest
            The request, instantiated at the sampled type.
        """
        random = self._effect("Random")
        assert random is not None
        operation = next(item for item in random.operations if item.name == "sample")
        outer = instantiate_telescope(random.telescope, instance.entry.effect.arguments)
        parameter_types, result_type = instantiate_operation(
            operation, (sampled,), outer
        )
        del parameter_types
        path = ("programs", state.declaration.name, "sites", label)
        return EffectRequest(
            instance.entry.instance,
            instance.entry.effect,
            operation.id,
            (sampled,),
            (SiteValue(label, site_type(sampled)), distribution),
            result_type,
            SiteProvenance(_origin_at(self, path, "effect-request", node)),
        )

    def _elaborate_observe(
        self: _Elaborator, step: ObserveStep, scope: _Scope, state: _ProgramState
    ) -> None:
        """Elaborate ``observe y <- Family(args)`` to a scored log density.

        Outside a marginalization scope the density's total goes to the
        canonical ``score`` instance. Inside one it goes to the scope's
        weight instance: as the weight vector over the group axis when
        the observation is plated over it, segmented by the ``via``
        fibration when the observation rows map to groups, or as its
        total in an ungrouped scope.

        Parameters
        ----------
        step : ObserveStep
            The step.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If the step destructures a tuple, uses ``via`` outside a
            grouped scope, or its weights cannot be brought to the
            scope's group axis.
        """
        if len(step.vars) != 1:
            self._fail(
                step,
                "an observe step names one observation; tuple destructuring has "
                "no elaboration",
                code="qiec-program",
            )
        name = step.vars[0]
        group = _enclosing_group(scope)
        if step.via is not None and group is None:
            self._fail(
                step,
                f"observation {name!r} names the fibration {step.via!r} outside a "
                "grouped marginalization",
                code="qiec-program",
            )
        state.current_site = name
        state.pending_alphabet = None
        distribution = self._step_distribution(
            step, scope, state, name, via=step.via, group=group
        )
        observed_type = self._sampled_type(distribution)
        observed = self._declare_parameter(
            name, observed_type, "observation", scope, state, step
        )
        state.sites.append(
            ProgramSite(
                name,
                "observe",
                distribution.name,
                distribution.plate.batch,
                distribution.plate.event,
            )
        )
        origin = _origin_at(
            self,
            ("programs", state.declaration.name, "sites", name),
            "log-density",
            step,
        )
        weight_scope = _weight_scope(scope)
        if weight_scope is None:
            weight: Value = WeightSum(
                LogDensity(
                    distribution, Var(observed), origin, distribution.plate.batch
                )
            )
            self._add_score(state.score, weight, scope, state, step)
            return
        assert weight_scope.weight_type is not None
        if weight_scope.group is None:
            weight = WeightSum(
                LogDensity(
                    distribution, Var(observed), origin, distribution.plate.batch
                )
            )
        else:
            batch = distribution.plate.batch
            density = LogDensity(distribution, Var(observed), origin, batch)
            if step.via is not None:
                if len(batch) != 1:
                    self._fail(
                        step,
                        f"observation {name!r} is fibred by {step.via!r} but is "
                        "not plated over one row axis",
                        code="qiec-program",
                    )
                fibration = self._declare_parameter(
                    step.via,
                    tensor_type(INT, (batch[0].size,)),
                    "fibration",
                    scope,
                    state,
                    step,
                )
                weight = SegmentSum(
                    density,
                    Var(fibration),
                    weight_scope.group.size,
                    tensor_type(LOG_WEIGHT, (weight_scope.group.size,)),
                )
            elif len(batch) == 1 and batch[0].size == weight_scope.group.size:
                weight = density
            elif not batch:
                weight = density
            else:
                self._fail(
                    step,
                    f"observation {name!r} inside the marginalization over "
                    f"{weight_scope.group.name!r} is plated over neither that "
                    "axis nor a fibration into it",
                    code="qiec-program",
                )
        assert weight_scope.weight_instance is not None
        self._add_weight(weight_scope.weight_instance, weight, scope, state, step)

    def _add_score(
        self: _Elaborator,
        instance: NamedEffectInstance,
        weight: Value,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> None:
        """Append a ``Score.add`` of a weight to a scope.

        Parameters
        ----------
        instance : NamedEffectInstance
            The ``Score`` instance.
        weight : Value
            The ``LogWeight`` added.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.
        """
        state.uses_score = True
        score = self._effect("Score")
        assert score is not None
        operation = next(item for item in score.operations if item.name == "add")
        request = EffectRequest(
            instance.entry.instance,
            instance.entry.effect,
            operation.id,
            (),
            (weight,),
            UNIT,
            SiteProvenance(
                _origin_at(
                    self,
                    ("programs", state.declaration.name, "scores", len(scope.steps)),
                    "effect-request",
                    node,
                )
            ),
        )
        self._bind_step(scope, None, UNIT, Perform(request), node)

    def _add_weight(
        self: _Elaborator,
        instance: NamedEffectInstance,
        weight: Value,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> None:
        """Append a ``Weight.add`` of a weight to a marginalization scope.

        Parameters
        ----------
        instance : NamedEffectInstance
            The scope's ``Weight`` instance.
        weight : Value
            The weight added, of the instance's weight type.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.
        """
        weight_effect = self._effect("Weight")
        assert weight_effect is not None
        operation = next(
            item for item in weight_effect.operations if item.name == "add"
        )
        outer = instantiate_telescope(
            weight_effect.telescope, instance.entry.effect.arguments
        )
        _parameter_types, result_type = instantiate_operation(operation, (), outer)
        request = EffectRequest(
            instance.entry.instance,
            instance.entry.effect,
            operation.id,
            (),
            (weight,),
            result_type,
            SiteProvenance(
                _origin_at(
                    self,
                    ("programs", state.declaration.name, "weights", len(scope.steps)),
                    "effect-request",
                    node,
                )
            ),
        )
        self._bind_step(scope, None, UNIT, Perform(request), node)

    def _elaborate_let(
        self: _Elaborator, step: LetStep, scope: _Scope, state: _ProgramState
    ) -> None:
        """Elaborate ``let name = expr`` to a pure binding.

        Parameters
        ----------
        step : LetStep
            The step.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        """
        if isinstance(step.value, LetExprLambda):
            if scope.lookup(step.name) is not None or step.name in self._lambda_macros:
                self._fail(
                    step,
                    f"variable {step.name!r} already bound in program",
                    code="qiec-program",
                )
            self._lambda_macros[step.name] = step.value
            return
        value = self._let_value(step.value, scope, state, step)
        type_ = self._value_type(value, scope.context(), step.value)
        self._bind_step(scope, step.name, type_, Return(value), step)

    def _elaborate_call(
        self: _Elaborator, step: CallStep, scope: _Scope, state: _ProgramState
    ) -> None:
        """Elaborate ``let name <- computation(args)`` to a bound call.

        Parameters
        ----------
        step : CallStep
            The step.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If the callee is unknown or the arguments do not fit, or the
            callee's row names an instance other than the program's.
        """
        for argument in step.call.arguments:
            for name in _free_let_names(argument):
                if scope.lookup(name) is None and name not in self._lambda_macros:
                    self._declare_parameter(
                        name, self._input_type(name, state), "data", scope, state, step
                    )
        call = self._lower_call(
            step.call,
            (),
            scope.context(),
            ("programs", state.declaration.name, "calls", step.name),
            None,
        )
        for entry in call.effects.entries:
            if entry.instance == state.random.entry.instance:
                state.uses_random = True
            elif entry.instance == state.score.entry.instance:
                state.uses_score = True
            else:
                self._fail(
                    step,
                    f"call to {step.call.callee!r} performs on instance "
                    f"{entry.instance!r}, which the program's row cannot carry",
                    code="qiec-program",
                )
        self._bind_step(scope, step.name, call.result_type, call, step)

    def _elaborate_score(
        self: _Elaborator, step: ScoreStep, scope: _Scope, state: _ProgramState
    ) -> None:
        """Elaborate ``score name = expr`` to a binding and a ``Score.add``.

        Parameters
        ----------
        step : ScoreStep
            The step.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If the expression is not real-valued, or the step lies in a
            grouped marginalization scope.
        """
        value = self._let_value(step.value, scope, state, step)
        type_ = self._value_type(value, scope.context(), step.value)
        if type_ != REAL:
            self._fail(
                step,
                f"score {step.name!r} has type {self._render(type_)}; a score is a "
                "Real",
                code="qiec-program",
            )
        binder = self._bind_step(scope, step.name, REAL, Return(value), step)
        weight = self._primitive(
            "as_weight",
            (Var(binder),),
            step,
            ("programs", state.declaration.name, "scores", step.name),
        )
        weight_scope = _weight_scope(scope)
        if weight_scope is None:
            self._add_score(state.score, weight, scope, state, step)
            return
        if weight_scope.group is not None:
            self._fail(
                step,
                f"score {step.name!r} lies inside a grouped marginalization; a "
                "scalar factor has no group to belong to",
                code="qiec-program",
            )
        assert weight_scope.weight_instance is not None
        self._add_weight(weight_scope.weight_instance, weight, scope, state, step)

    def _let_value(
        self: _Elaborator,
        expr: LetExprNode,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> Value:
        """Lower a let expression, declaring its free names as inputs.

        Parameters
        ----------
        expr : LetExprNode
            The expression.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        Value
            The lowered value.
        """
        chart = _chart_construct(expr)
        if chart is not None:
            self._fail(
                node,
                f"{chart} has no elaboration yet; deduction charts are a "
                "logic-programming construct outside the program calculus",
                code=GAP_CODE,
            )
        unknown = _unknown_call(expr, self._lambda_macros)
        if unknown is not None:
            self._fail(
                node,
                f"the call `{unknown}(...)` names no builtin, lambda, or "
                "computation; a host function reached by name has no "
                "elaboration",
                code=GAP_CODE,
            )
        for name in _free_let_names(expr):
            if scope.lookup(name) is None and name not in self._lambda_macros:
                self._declare_parameter(
                    name, self._input_type(name, state), "data", scope, state, node
                )
        return self._lower_value(
            expr,
            (),
            scope.context(),
            None,
            ("programs", state.declaration.name, "lets"),
        )

    def _input_type(self: _Elaborator, name: str, state: _ProgramState) -> TypeExpr:
        """The type a free name read by a let expression is declared at.

        Parameters
        ----------
        name : str
            The name.
        state : _ProgramState
            The program's accumulating state, whose inferred shapes say
            which plate the name's values range over.

        Returns
        -------
        TypeExpr
            ``Int`` or ``Real`` when the name reaches no plated step, else
            the tensor of that element over the plate's first batch axis.
        """
        element, axis = state.input_shapes.get(name, (REAL, None))
        if axis is None:
            return element
        return tensor_type(element, (axis.size,))

    def _infer_input_shapes(
        self: _Elaborator, declaration: ProgramDecl
    ) -> dict[str, tuple[TypeApplication, PlateAxis | None]]:
        """Infer the shapes of the free names let expressions read.

        A free name is a scalar unless a step plated over a batch axis
        reads a let expression that depends on it, in which case its
        values range over that axis; a name that indexes a tensor is
        integral. The inference walks the steps in order, so the first
        plated consumer of a name fixes its axis.

        Parameters
        ----------
        declaration : ProgramDecl
            The program.

        Returns
        -------
        dict[str, tuple[TypeApplication, PlateAxis | None]]
            Each free name's element type and batch axis, if any.
        """
        bound: set[str] = set()
        if declaration.params is not None:
            bound.update(declaration.params)
        if declaration.type_params is not None:
            bound.update(
                parameter.name
                for parameter in declaration.type_params
                if isinstance(parameter, ScalarParam)
            )
        depends: dict[str, set[str]] = {}
        elements: dict[str, TypeApplication] = {}
        axes: dict[str, PlateAxis] = {}

        def free_of(expr: LetExprNode) -> set[str]:
            """Every free name an expression depends on, through lets.

            Parameters
            ----------
            expr : LetExprNode
                The expression.

            Returns
            -------
            set[str]
                Free names, transitively through let bindings.
            """
            found: set[str] = set()
            for name in _free_let_names(expr):
                if name in depends:
                    found |= depends[name]
                elif name not in bound and name not in self._lambda_macros:
                    found.add(name)
            return found

        def visit(steps: Sequence[ProgramStep], group: PlateAxis | None) -> None:
            """Walk steps, recording dependencies and plated consumers.

            Parameters
            ----------
            steps : Sequence[ProgramStep]
                The steps.
            group : PlateAxis | None
                The grouping axis of the enclosing marginalization.
            """
            for step in steps:
                if isinstance(step, LetStep | ScoreStep):
                    if isinstance(step.value, LetExprLambda):
                        continue
                    for index_name in _index_names(step.value):
                        if index_name not in bound and index_name not in depends:
                            elements[index_name] = INT
                    depends[step.name] = free_of(step.value)
                    bound.add(step.name)
                elif isinstance(step, CallStep):
                    depends[step.name] = set().union(
                        *(free_of(argument) for argument in step.call.arguments)
                    )
                    bound.add(step.name)
                elif isinstance(step, SampleStep | ObserveStep | MarginalizeStep):
                    names = (
                        [step.var]
                        if isinstance(step, MarginalizeStep)
                        else list(step.vars)
                    )
                    try:
                        resolved = resolve_step_dist(
                            step.morphism,
                            step.args,
                            morphisms=self._program_morphisms,
                            lets=self._program_lets,
                            family_registry=frozenset(FAMILIES),
                            target="qvr-qiec",
                        )
                    except StepResolutionError:
                        bound.update(names)
                        continue
                    record = FAMILIES.get(resolved.family)
                    if record is None:
                        bound.update(names)
                        continue
                    morphism = self._program_morphisms.get(step.morphism)
                    plate = self._step_plate(step, record, morphism, group)
                    batch = plate.batch[0] if plate.batch else None
                    if batch is not None:
                        for argument in step.args or ():
                            for name in _argument_names(argument):
                                for free in depends.get(name, set()):
                                    axes.setdefault(free, batch)
                    bound.update(names)
                    if isinstance(step, MarginalizeStep):
                        inner_group = group
                        if step.over is not None:
                            inner_group = self._axis(step.over, step)
                        elif step.over_objs:
                            inner_group = self._axis(step.over_objs[0], step)
                        visit(step.scope, inner_group)

        visit(declaration.draws, None)
        return {
            name: (elements.get(name, REAL), axes.get(name))
            for name in set(elements) | set(axes)
        }

    def _program_return(
        self: _Elaborator, declaration: ProgramDecl, scope: _Scope, state: _ProgramState
    ) -> tuple[Value, TypeExpr]:
        """The program's result value and type.

        Parameters
        ----------
        declaration : ProgramDecl
            The program.
        scope : _Scope
            The body's scope.
        state : _ProgramState
            The program's accumulating state.

        Returns
        -------
        tuple[Value, TypeExpr]
            The returned value, a tuple for several names, and its type.

        Raises
        ------
        QiecDiagnosticError
            If a returned name is unbound.
        """
        del state
        values: list[Value] = []
        types: list[TypeExpr] = []
        for name in declaration.return_vars:
            local = scope.lookup(name)
            if local is None:
                self._fail(
                    declaration,
                    f"program {declaration.name!r} returns unbound name {name!r}",
                    code="qiec-program",
                )
            values.append(Var(local))
            types.append(local.type)
        if not values:
            return LiteralValue(None, UNIT), UNIT
        if len(values) == 1:
            return values[0], types[0]
        result_type = product_type(*types)
        return TupleValue(tuple(values), result_type), result_type

    # ------------------------------------------------------------------
    # marginalization

    def _elaborate_marginalize(
        self: _Elaborator, step: MarginalizeStep, scope: _Scope, state: _ProgramState
    ) -> None:
        """Elaborate a marginalization block to a helper computation call.

        The helper allocates a ``Random`` instance for the latent, handles
        it with the enumeration handler, samples the latent, allocates a
        ``Weight`` instance for the scope, handles it with the collecting
        handler, runs the scope's steps, and answers ``unit``; the
        enumeration handler's answer is the block's log marginal, which
        the enclosing scope scores.

        Parameters
        ----------
        step : MarginalizeStep
            The block.
        scope : _Scope
            The enclosing scope.
        state : _ProgramState
            The program's accumulating state.

        Raises
        ------
        QiecDiagnosticError
            If the reduction is not ``logsumexp``, or the latent's family
            has no finite support.
        """
        if step.reduction not in (None, "logsumexp"):
            self._fail(
                step,
                f"marginalize reduction {step.reduction!r} has no elaboration; "
                "the reduction is logsumexp",
                code="qiec-program",
            )
        enumerate_handler, collect_handler = self._program_handlers()
        group: PlateAxis | None = None
        if step.over is not None:
            group = self._axis(step.over, step)
        elif step.over_objs:
            if len(step.over_objs) != 1:
                self._fail(
                    step,
                    "a grouped marginalization names one grouping axis",
                    code="qiec-program",
                )
            group = self._axis(step.over_objs[0], step)
        inner = _Scope(outer=scope, group=group)
        latent_name = step.var
        state.current_site = latent_name
        state.pending_alphabet = None
        distribution = self._step_distribution(
            step, inner, state, latent_name, group=group
        )
        if state.pending_alphabet is not None:
            state.alphabets[latent_name] = state.pending_alphabet
        record = FAMILIES[distribution.name]
        if (
            record.finite_support is None
            and distribution.name not in _CLASS_INDEX_FAMILIES
        ):
            # Nothing finite to sum over: the latent is drawn once per
            # position and the scope runs in the enclosing scope, which
            # is what the runtime does with such a block.
            self._elaborate_continuous_marginalize(
                step, distribution, scope, state, latent_name
            )
            return
        sampled = self._sampled_type(distribution)
        weight_type: TypeExpr = (
            LOG_WEIGHT if group is None else tensor_type(LOG_WEIGHT, (group.size,))
        )
        path = ("programs", state.declaration.name, "marginals", latent_name)
        choice = self._local_instance(
            f"{latent_name}_choice", "Random", (), (*path, "choice")
        )
        weight = self._local_instance(
            f"{latent_name}_weight", "Weight", (weight_type,), (*path, "weight")
        )
        inner.weight_instance = weight
        inner.weight_type = weight_type
        request = self._sample_request(
            choice, latent_name, sampled, distribution, step, state
        )
        state.sites.append(
            ProgramSite(
                latent_name,
                "marginal",
                distribution.name,
                distribution.plate.batch,
                distribution.plate.event,
            )
        )
        latent = Local(latent_name, sampled)
        inner.locals[latent_name] = latent
        self._elaborate_steps(step.scope, inner, state)
        collected = Handle(
            weight.entry.instance,
            collect_handler.id,
            self._fold_steps(inner, Return(LiteralValue(None, UNIT))),
            (weight_type,),
        )
        del self.instances[weight.name]
        del self.instances[choice.name]
        enumerated = Handle(
            choice.entry.instance,
            enumerate_handler.id,
            Bind(
                latent,
                Perform(request),
                NewInstance(
                    weight.entry.instance,
                    weight.entry.effect,
                    collected,
                    _origin_at(self, (*path, "weight"), "local-instance", step),
                ),
            ),
            (weight_type,),
        )
        body = NewInstance(
            choice.entry.instance,
            choice.entry.effect,
            enumerated,
            _origin_at(self, (*path, "choice"), "local-instance", step),
        )
        helper = self._marginal_helper(step, inner, body, scope, state)
        weight_local = self._bind_step(scope, None, LOG_WEIGHT, helper, step)
        weight_scope = _weight_scope(scope)
        if weight_scope is None:
            self._add_score(state.score, Var(weight_local), scope, state, step)
        elif weight_scope.group is not None:
            self._fail(
                step,
                f"marginalization of {latent_name!r} lies inside a grouped "
                "marginalization; its scalar marginal has no group to belong to",
                code="qiec-program",
            )
        else:
            assert weight_scope.weight_instance is not None
            self._add_weight(
                weight_scope.weight_instance, Var(weight_local), scope, state, step
            )

    def _elaborate_continuous_marginalize(
        self: _Elaborator,
        step: MarginalizeStep,
        distribution: DistributionValue,
        scope: _Scope,
        state: _ProgramState,
        latent_name: str,
    ) -> None:
        """Elaborate a marginalization over a family with no finite support.

        The latent is sampled on the canonical ``random`` instance, plated
        as the block says, and the scope's steps run in the enclosing
        scope with the latent bound.

        Parameters
        ----------
        step : MarginalizeStep
            The block.
        distribution : DistributionValue
            The latent's construction.
        scope : _Scope
            The enclosing scope.
        state : _ProgramState
            The program's accumulating state.
        latent_name : str
            The latent's name.
        """
        sampled = self._sampled_type(distribution)
        state.uses_random = True
        request = self._sample_request(
            state.random, latent_name, sampled, distribution, step, state
        )
        state.sites.append(
            ProgramSite(
                latent_name,
                "sample",
                distribution.name,
                distribution.plate.batch,
                distribution.plate.event,
            )
        )
        self._bind_step(scope, latent_name, sampled, Perform(request), step)
        self._elaborate_steps(step.scope, scope, state)

    def _local_instance(
        self: _Elaborator,
        name: str,
        interface: str,
        arguments: tuple[TypeExpr, ...],
        path: tuple[str | int, ...],
    ) -> NamedEffectInstance:
        """Allocate a lexical instance for a marginalization block.

        Parameters
        ----------
        name : str
            The instance's name.
        interface : str
            The prelude interface.
        arguments : tuple[TypeExpr, ...]
            The interface's type arguments.
        path : tuple[str | int, ...]
            The lexical path, which the instance's identity derives from.

        Returns
        -------
        NamedEffectInstance
            The instance, registered under its name until the block ends.

        Raises
        ------
        QiecDiagnosticError
            If the name shadows an instance in scope.
        """
        definition = self._effect(interface)
        assert definition is not None
        effect: EffectRef = definition.apply(arguments) if arguments else definition.ref
        if name in self.instances:
            self._fail(
                self.source.syntax.statements[0],
                f"marginalization instance {name!r} shadows an instance in scope",
                code="qiec-program",
            )
        entry = instantiate_effect(
            effect, module=self.source.module_name, lexical_path=path
        )
        instance = NamedEffectInstance(
            name, entry, _origin_at(self, path, "effect-instance")
        )
        self.instances[name] = instance
        return instance

    def _marginal_helper(
        self: _Elaborator,
        step: MarginalizeStep,
        inner: _Scope,
        body: Computation,
        scope: _Scope,
        state: _ProgramState,
    ) -> Computation:
        """Declare the helper computation holding a marginalization block.

        The helper takes every enclosing binding the block read, so the
        block is a closed computation the enclosing scope calls.

        Parameters
        ----------
        step : MarginalizeStep
            The block.
        inner : _Scope
            The block's scope, whose ``referenced`` names are the
            helper's parameters.
        body : Computation
            The block's body.
        scope : _Scope
            The enclosing scope.
        state : _ProgramState
            The program's accumulating state.

        Returns
        -------
        Computation
            The call of the helper from the enclosing scope.

        Raises
        ------
        QiecDiagnosticError
            If the block's body fails to check.
        """
        captured = tuple(
            scope.lookup(name)
            for name in sorted(
                inner.referenced, key=lambda item: _binding_order(scope, item)
            )
        )
        parameters = tuple(local for local in captured if local is not None)
        name = f"{state.declaration.name}__{step.var}_marginal"
        if name in self.computation_signatures:
            self._fail(
                step,
                f"marginalization helper {name!r} is declared twice",
                code="qiec-program",
            )
        identity = ComputationId.derive(self.source.module_name, "computation", name)
        signature = ComputationSignature(
            identity,
            name,
            (),
            tuple(parameter.type for parameter in parameters),
            LOG_WEIGHT,
            EffectRow(),
        )
        try:
            self.registry.register_computation(signature)
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(step, error, fallback="qiec-program")
        self.computation_signatures[name] = signature
        context = CheckContext(parameters)
        try:
            actual = infer_computation(body, self.registry, context)
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(step, error, fallback="qiec-program")
        if actual.result != LOG_WEIGHT or actual.effects.entries:
            self._fail(
                step,
                f"marginalization of {step.var!r} does not close to a LogWeight",
                code="qiec-program",
            )
        origin = _origin_at(
            self,
            ("programs", state.declaration.name, "marginals", step.var),
            "computation",
            step,
        )
        state.helpers.append(
            NamedComputation(
                identity,
                name,
                (),
                parameters,
                body,
                ComputationType(EffectRow(), LOG_WEIGHT),
                origin,
            )
        )
        return Call(
            identity,
            name,
            (),
            tuple(Var(parameter) for parameter in parameters),
            LOG_WEIGHT,
            EffectRow(),
            _origin_at(
                self,
                ("programs", state.declaration.name, "marginals", step.var, "call"),
                "call",
                step,
            ),
        )

    # ------------------------------------------------------------------
    # distributions and plates

    def _step_distribution(
        self: _Elaborator,
        step: SampleStep | ObserveStep | MarginalizeStep,
        scope: _Scope,
        state: _ProgramState,
        site: str,
        *,
        via: str | None = None,
        group: PlateAxis | None = None,
    ) -> DistributionValue:
        """The distribution a step draws from, plated as the step says.

        Parameters
        ----------
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        scope : _Scope
            The scope its arguments are read in.
        state : _ProgramState
            The program's accumulating state.
        site : str
            The site's name, for diagnostics and identities.
        via : str | None
            The fibration re-indexing grouped arguments, for an observe.
        group : PlateAxis | None
            The grouping axis of the enclosing or this marginalization.

        Returns
        -------
        DistributionValue
            The typed construction.

        Raises
        ------
        QiecDiagnosticError
            If the morphism cannot be resolved to a family, or the
            arguments and plate do not type.
        """
        try:
            resolved = resolve_step_dist(
                step.morphism,
                step.args,
                morphisms=self._program_morphisms,
                lets=self._program_lets,
                family_registry=frozenset(FAMILIES),
                target="qvr-qiec",
            )
        except StepResolutionError as error:
            self._fail(
                step,
                "; ".join(error.kinds),
                code=GAP_CODE if _is_gap(error.kinds, self) else "qiec-program",
            )
        record = FAMILIES.get(resolved.family)
        if record is None:
            self._fail(
                step,
                f"family:{resolved.family}: not in the semantic family registry",
                code="qiec-program",
            )
        morphism = self._program_morphisms.get(step.morphism)
        plate = self._step_plate(step, record, morphism, group)
        if morphism is not None and _structured(record, morphism, step):
            arguments = self._structured_arguments(
                record, morphism, plate, scope, state, step
            )
        elif morphism is not None and self._conditional_map(morphism, record.name):
            arguments = self._mapped_arguments(
                record, morphism, step, plate, scope, state
            )
        else:
            arguments = self._family_arguments(
                record, resolved, step, plate, scope, state, via, group, morphism
            )
        arguments = self._widen_alphabet(record, morphism, arguments, scope, step)
        path = ("programs", state.declaration.name, "sites", site, "distribution")
        value = DistributionValue(
            record.id,
            record.name,
            tuple(arguments),
            sampleable_type(REAL),
            _origin_at(self, path, "distribution", step),
            plate,
        )
        value = DistributionValue(
            record.id,
            record.name,
            tuple(arguments),
            self._plated_type(value, scope, step),
            value.origin,
            plate,
        )
        self._value_type(value, scope.context(), step)
        return value

    def _plated_type(
        self: _Elaborator, value: DistributionValue, scope: _Scope, node: object
    ) -> TypeExpr:
        """The ``Sampleable`` type of a plated construction.

        Parameters
        ----------
        value : DistributionValue
            The construction, whose claimed type is provisional.
        scope : _Scope
            The scope.
        node : object
            The source node.

        Returns
        -------
        TypeExpr
            The type the kernel assigns from the family, its arguments,
            and its plate.

        Raises
        ------
        QiecDiagnosticError
            If the kernel rejects the construction.
        """
        from quivers.qiec.checking import _plated_sample_type

        record = FAMILIES[value.name]
        context = scope.context()
        argument_types = {
            name: self._value_type(argument, context, node)
            for name, argument in value.arguments
        }
        try:
            return _plated_sample_type(value, record, argument_types)
        except KernelError as error:
            self._fail_kernel(node, error, fallback="qiec-program")

    def _sampled_type(self, distribution: DistributionValue) -> TypeExpr:
        """The type a distribution's draws have.

        Parameters
        ----------
        distribution : DistributionValue
            The construction.

        Returns
        -------
        TypeExpr
            The element of its ``Sampleable`` type.
        """
        assert isinstance(distribution.result_type, TypeApplication)
        return cast(TypeExpr, distribution.result_type.arguments[0])

    def _step_plate(
        self: _Elaborator,
        step: SampleStep | ObserveStep | MarginalizeStep,
        record: DistributionFamily,
        morphism: MorphismDecl | None,
        group: PlateAxis | None,
    ) -> PlateShape:
        """The plate a step's distribution ranges over.

        Event axes come from the step's ``over``, else the morphism's
        ``[over=...]``, else a multivariate family's codomain factors;
        batch axes from ``iid_over``, else the ``: Axis`` annotation and
        a ``Real N`` codomain's width for a scalar family. A grouped
        marginalization's latent is batched over its group.

        Parameters
        ----------
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        record : DistributionFamily
            The family.
        morphism : MorphismDecl | None
            The declared morphism the step draws through, if any.
        group : PlateAxis | None
            The grouping axis of this marginalization.

        Returns
        -------
        PlateShape
            The plate.
        """
        if isinstance(step, MarginalizeStep):
            batch: list[PlateAxis] = [group] if group is not None else []
            event: list[PlateAxis] = []
            if step.index is not None:
                axis = self._object_axis(step.index, step)
                if record.event_rank > 0:
                    event.append(axis)
                elif (
                    record.finite_support is None
                    and record.name not in _CLASS_INDEX_FAMILIES
                ):
                    batch.append(axis)
            return PlateShape(tuple(batch), tuple(event))
        axes = step.axes
        if axes is not None:
            return PlateShape(
                tuple(self._axis(name, step) for name in axes.iid_over),
                tuple(self._axis(name, step) for name in axes.over),
            )
        over = _option_axes(step.options, "over")
        iid_over = _option_axes(step.options, "iid_over")
        if over or iid_over:
            return PlateShape(
                tuple(self._axis(name, step) for name in iid_over),
                tuple(self._axis(name, step) for name in over),
            )
        if morphism is not None:
            morphism_over = _option_axes(morphism.options, "over")
            if morphism_over:
                return PlateShape(
                    (), tuple(self._axis(name, step) for name in morphism_over)
                )
        event_axes: tuple[PlateAxis, ...] = ()
        if record.name == "GP" and morphism is not None:
            return PlateShape((), (self._gp_grid(morphism, step),))
        if record.event_rank > 0 and morphism is not None:
            factors = [
                factor
                for factor in _factors(morphism.codomain)
                if isinstance(factor, TypeName)
            ]
            finite = [
                self._axis(factor.name, step)
                for factor in factors
                if self._program_objects.get(factor.name) is not None
                and self._program_objects[factor.name].extent is not None
            ]
            if len(finite) == record.event_rank:
                event_axes = tuple(finite)
            elif finite and record.event_rank == 1:
                event_axes = (finite[0],)
        if event_axes:
            return PlateShape((), event_axes)
        width: PlateAxis | None = None
        if (
            record.event_rank == 0
            and morphism is not None
            and isinstance(morphism.codomain, TypeName)
        ):
            info = self._program_objects.get(morphism.codomain.name)
            if info is not None and info.real_width is not None:
                width = PlateAxis(morphism.codomain.name, _extent(info.real_width))
        batch = []
        if step.index is not None:
            axis = self._object_axis(step.index, step)
            if record.event_rank > 0:
                # A matrix family draws one square variate whose side is
                # the annotated axis; a vector family one vector over it.
                return PlateShape((), tuple(axis for _ in range(record.event_rank)))
            batch.append(axis)
        if width is not None:
            batch.append(width)
        return PlateShape(tuple(batch), ())

    def _axis(self: _Elaborator, name: str, node: object) -> PlateAxis:
        """A plate axis named by an object with a known extent.

        Parameters
        ----------
        name : str
            The object's name.
        node : object
            The source node.

        Returns
        -------
        PlateAxis
            The axis.

        Raises
        ------
        QiecDiagnosticError
            If the name is no object with a static extent.
        """
        info = self._program_objects.get(name)
        if info is None or info.extent is None:
            self._fail(
                node,
                f"axis {name!r} names no object with a static extent",
                code="qiec-program",
            )
        return PlateAxis(name, _extent(info.extent))

    def _object_axis(self: _Elaborator, expr: ObjectExpr, node: object) -> PlateAxis:
        """A plate axis from an index annotation.

        Parameters
        ----------
        expr : ObjectExpr
            The annotation: an object name, a ``FinSet N``, or a product
            of objects, which flattens to one axis.
        node : object
            The source node.

        Returns
        -------
        PlateAxis
            The axis.

        Raises
        ------
        QiecDiagnosticError
            If the expression has no static extent.
        """
        if isinstance(expr, TypeName):
            if expr.name.isdigit():
                return PlateAxis(expr.name, _extent(int(expr.name)))
            return self._axis(expr.name, node)
        info = self._object_info(expr, node)
        if info.extent is None:
            self._fail(
                node, "an index annotation needs a static extent", code="qiec-program"
            )
        return PlateAxis(_axis_name(expr), _extent(info.extent))

    def _object_info(self: _Elaborator, expr: ObjectExpr, node: object) -> ObjectInfo:
        """The shape an object expression denotes.

        Parameters
        ----------
        expr : ObjectExpr
            The expression.
        node : object
            The source node.

        Returns
        -------
        ObjectInfo
            Its extent, real width, and finiteness.

        Raises
        ------
        QiecDiagnosticError
            If the expression names an unknown object or one with no
            static shape.
        """
        info = _object_expr_info(expr, self._program_objects)
        if info is None:
            self._fail(
                node,
                f"object expression {_axis_name(expr)!r} has no static shape",
                code="qiec-program",
            )
        return info

    # ------------------------------------------------------------------
    # arguments

    def _family_arguments(
        self: _Elaborator,
        record: DistributionFamily,
        resolved: ResolvedDist,
        step: SampleStep | ObserveStep | MarginalizeStep,
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
        via: str | None,
        group: PlateAxis | None,
        morphism: MorphismDecl | None,
    ) -> list[tuple[str, Value]]:
        """Lower a step's arguments to the family's named parameters.

        The step's own arguments keep their structure; positions the
        resolver filled from a morphism's declaration or options come in
        wire form. Each is lowered against the parameter it fills, and a
        grouped argument reaching a fibred observation is re-indexed by
        the fibration.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        resolved : ResolvedDist
            The resolver's family and wire arguments.
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        plate : PlateShape
            The distribution's plate.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        via : str | None
            The fibration of a grouped observation.
        group : PlateAxis | None
            The grouping axis in force.
        morphism : MorphismDecl | None
            The morphism drawn through, whose codomain widens a class
            alphabet.

        Returns
        -------
        list[tuple[str, Value]]
            The named arguments in registry order.

        Raises
        ------
        QiecDiagnosticError
            If more arguments are supplied than the family has
            parameters.
        """
        del morphism
        structural: tuple[DrawArg, ...] = step.args or ()
        wire = tuple(resolved.args or ())
        raw: list[DrawArg | str | float] = list(structural)
        raw.extend(wire[len(structural) :])
        if len(raw) > len(record.parameters):
            self._fail(
                step,
                f"family {record.name!r} takes at most {len(record.parameters)} "
                f"parameters ({', '.join(record.parameter_names)})",
                code="qiec-program",
            )
        arguments: list[tuple[str, Value]] = []
        for parameter, argument in zip(record.parameters, raw, strict=False):
            value = self._draw_argument(argument, parameter, plate, scope, state, step)
            if via is not None and group is not None:
                value = self._fibred(value, via, group, plate, scope, state, step)
            arguments.append((parameter.name, value))
        return arguments

    def _draw_argument(
        self: _Elaborator,
        argument: DrawArg | str | float,
        parameter: FamilyParameter | _ScalarParameter,
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> Value:
        """Lower one draw argument against the parameter it fills.

        Parameters
        ----------
        argument : DrawArg | str | float
            The argument: a tagged node, or wire text or a number.
        parameter : FamilyParameter | _ScalarParameter
            The family parameter, which fixes the element type and rank.
        plate : PlateShape
            The distribution's plate.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        Value
            The lowered value.

        Raises
        ------
        QiecDiagnosticError
            If the argument is a nested distribution where a value is
            expected, or names something unreadable.
        """
        element = _element_of(parameter)
        integral = element == INT
        rank = parameter.rank
        constraint = parameter.constraint
        if isinstance(argument, float | int):
            return _literal(float(argument), integral)
        if isinstance(argument, DrawArgScalar):
            return _literal(argument.value, integral)
        if isinstance(argument, DrawArgName):
            return self._named_argument(
                argument.text, element, rank, constraint, plate, scope, state, node
            )
        if isinstance(argument, DrawArgIndex):
            return self._indexed_argument(
                argument.name, argument.indices, element, scope, state, node
            )
        if isinstance(argument, DrawArgList):
            return self._list_argument(argument, element, scope, state, node)
        if isinstance(argument, DrawArgDist):
            if constraint != "sampleable":
                self._fail(
                    node,
                    f"a nested distribution {argument.family!r} fills a parameter "
                    "that takes a value",
                    code="qiec-program",
                )
            return self._nested_distribution(argument, scope, state, node)
        text = str(argument)
        if _is_number_text(text):
            return _literal(float(text), integral)
        if "[" in text:
            base, _, rest = text.partition("[")
            indices = tuple(item.rstrip("]") for item in rest.split("["))
            return self._indexed_argument(base, indices, element, scope, state, node)
        return self._named_argument(
            text, element, rank, constraint, plate, scope, state, node
        )

    def _named_argument(
        self: _Elaborator,
        name: str,
        element: TypeApplication,
        rank: int,
        constraint: str,
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> Value:
        """Lower a bare name in argument position.

        A bound name is read; a morphism with a distribution initializer
        is that distribution when a sampleable is expected; any other
        name is a data input of the parameter's element type, which a
        plate or a tensor parameter broadcasts.

        Parameters
        ----------
        name : str
            The name.
        element : TypeApplication
            The parameter's element type.
        rank : int
            The parameter's rank.
        constraint : str
            The parameter's constraint.
        plate : PlateShape
            The distribution's plate.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        Value
            The lowered value.
        """
        local = scope.lookup(name)
        if local is not None:
            return Var(local)
        morphism = self._program_morphisms.get(name)
        if morphism is not None and constraint == "sampleable":
            return self._morphism_distribution(morphism, name, scope, state, node)
        if rank > 0 and plate.empty:
            self._fail(
                node,
                f"input {name!r} fills a rank-{rank} parameter without a plate to "
                "fix its shape; write the step's `over` axes",
                code="qiec-program",
            )
        return Var(self._declare_parameter(name, element, "data", scope, state, node))

    def _indexed_argument(
        self: _Elaborator,
        base: str,
        indices: Sequence[str],
        element: TypeApplication,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> Value:
        """Lower ``base[i][j]`` to nested gathers.

        Parameters
        ----------
        base : str
            The indexed name.
        indices : Sequence[str]
            The index names, outermost first.
        element : TypeApplication
            The parameter's element type, which types an unbound base.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        Value
            The selection.

        Raises
        ------
        QiecDiagnosticError
            If an index is unbound, or the base is unbound and its
            alphabet cannot be read off the index.
        """
        index_values: list[Value] = []
        for index in indices:
            local = scope.lookup(index)
            if local is None:
                if _is_number_text(index):
                    index_values.append(LiteralValue(int(float(index)), INT))
                    continue
                self._fail(
                    node,
                    f"index {index!r} of {base!r} is not bound by a step",
                    code="qiec-program",
                )
            index_values.append(Var(local))
        source = scope.lookup(base)
        if source is None:
            alphabet = state.alphabets.get(indices[0])
            if alphabet is None:
                self._fail(
                    node,
                    f"input {base!r} is indexed by {indices[0]!r}, whose alphabet is "
                    "unknown; index it by a class-valued step",
                    code="qiec-program",
                )
            source = self._declare_parameter(
                base, tensor_type(element, (alphabet.size,)), "data", scope, state, node
            )
        value: Value = Var(source)
        context = scope.context()
        for index_value in index_values:
            source_type = self._value_type(value, context, node)
            shape = tensor_shape(source_type)
            if shape is None:
                self._fail(
                    node,
                    f"{base!r} is indexed but is not a Tensor",
                    code="qiec-program",
                )
            index_type = self._value_type(index_value, context, node)
            rest = shape[1][1:]
            if index_type == INT:
                result: TypeExpr = shape[0] if not rest else tensor_type(shape[0], rest)
            else:
                index_shape = tensor_shape(index_type)
                if index_shape is None or index_shape[0] != INT:
                    self._fail(
                        node,
                        f"index into {base!r} is not an Int or a Tensor of them",
                        code="qiec-program",
                    )
                result = tensor_type(shape[0], (*index_shape[1], *rest))
            value = Gather(value, index_value, result)
        return value

    def _list_argument(
        self: _Elaborator,
        argument: DrawArgList,
        element: TypeApplication,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> Value:
        """Lower a list argument to a tensor literal.

        Parameters
        ----------
        argument : DrawArgList
            The list.
        element : TypeApplication
            The entry type.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        Value
            The tensor.

        Raises
        ------
        QiecDiagnosticError
            If the list is empty or its entries differ in type.
        """
        if not argument.items:
            self._fail(
                node,
                "an empty list fills no distribution parameter",
                code="qiec-program",
            )
        items = [
            self._list_argument(item, element, scope, state, node)
            if isinstance(item, DrawArgList)
            else self._draw_argument(
                item, _ScalarParameter(element), PlateShape(), scope, state, node
            )
            for item in argument.items
        ]
        context = scope.context()
        types = [self._value_type(item, context, node) for item in items]
        if any(item != types[0] for item in types):
            self._fail(node, "list entries have differing types", code="qiec-program")
        inner = tensor_shape(types[0])
        if inner is None:
            result = tensor_type(types[0], (_extent(len(items)),))
        else:
            result = tensor_type(inner[0], (_extent(len(items)), *inner[1]))
        return TensorValue(tuple(items), result)

    def _nested_distribution(
        self: _Elaborator,
        argument: DrawArgDist,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> DistributionValue:
        """Lower a nested family application in argument position.

        Parameters
        ----------
        argument : DrawArgDist
            The application.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        DistributionValue
            The unplated construction.

        Raises
        ------
        QiecDiagnosticError
            If the family is unknown.
        """
        record = FAMILIES.get(argument.family)
        if record is None:
            self._fail(
                node,
                f"family:{argument.family}: not in the semantic family registry",
                code="qiec-program",
            )
        arguments = [
            (
                parameter.name,
                self._draw_argument(item, parameter, PlateShape(), scope, state, node),
            )
            for parameter, item in zip(record.parameters, argument.args, strict=False)
        ]
        provisional = DistributionValue(
            record.id,
            record.name,
            tuple(arguments),
            sampleable_type(REAL),
            _origin_at(
                self,
                ("programs", state.declaration.name, "nested", argument.family),
                "distribution",
                node,
            ),
        )
        return DistributionValue(
            record.id,
            record.name,
            tuple(arguments),
            self._plated_type(provisional, scope, node),
            provisional.origin,
        )

    def _morphism_distribution(
        self: _Elaborator,
        morphism: MorphismDecl,
        name: str,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> DistributionValue:
        """The distribution a morphism's initializer names, as a value.

        Parameters
        ----------
        morphism : MorphismDecl
            The morphism.
        name : str
            Its name.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        DistributionValue
            The construction from the initializer's family and arguments.

        Raises
        ------
        QiecDiagnosticError
            If the morphism has no distribution initializer.
        """
        try:
            resolved = resolve_step_dist(
                name,
                None,
                morphisms=self._program_morphisms,
                lets=self._program_lets,
                family_registry=frozenset(FAMILIES),
                target="qvr-qiec",
            )
        except StepResolutionError as error:
            self._fail(node, "; ".join(error.kinds), code="qiec-program")
        record = FAMILIES.get(resolved.family)
        if record is None:
            self._fail(
                node,
                f"family:{resolved.family}: not in the semantic family registry",
                code="qiec-program",
            )
        arguments = [
            (
                parameter.name,
                self._draw_argument(item, parameter, PlateShape(), scope, state, node),
            )
            for parameter, item in zip(
                record.parameters, resolved.args or (), strict=False
            )
        ]
        provisional = DistributionValue(
            record.id,
            record.name,
            tuple(arguments),
            sampleable_type(REAL),
            _origin_at(
                self,
                ("programs", state.declaration.name, "morphisms", name),
                "distribution",
                node,
            ),
        )
        return DistributionValue(
            record.id,
            record.name,
            tuple(arguments),
            self._plated_type(provisional, scope, node),
            provisional.origin,
        )

    def _fibred(
        self: _Elaborator,
        value: Value,
        via: str,
        group: PlateAxis,
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> Value:
        """Re-index a grouped argument to the observation rows.

        Parameters
        ----------
        value : Value
            The argument.
        via : str
            The fibration's name.
        group : PlateAxis
            The grouping axis.
        plate : PlateShape
            The observation's plate, whose one batch axis is the rows.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        Value
            The argument gathered by the fibration when its leading
            dimension is the group axis, else unchanged.

        Raises
        ------
        QiecDiagnosticError
            If the observation is not plated over one row axis.
        """
        shape = tensor_shape(self._value_type(value, scope.context(), node))
        if shape is None or not shape[1] or shape[1][0] != group.size:
            return value
        if len(plate.batch) != 1:
            self._fail(
                node,
                f"fibration {via!r} needs an observation plated over one row axis",
                code="qiec-program",
            )
        rows = plate.batch[0].size
        fibration = self._declare_parameter(
            via, tensor_type(INT, (rows,)), "fibration", scope, state, node
        )
        return Gather(
            value, Var(fibration), tensor_type(shape[0], (rows, *shape[1][1:]))
        )

    def _widen_alphabet(
        self: _Elaborator,
        record: DistributionFamily,
        morphism: MorphismDecl | None,
        arguments: list[tuple[str, Value]],
        scope: _Scope,
        step: object,
    ) -> list[tuple[str, Value]]:
        """Record a class-index draw's alphabet for the steps indexing by it.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        morphism : MorphismDecl | None
            The morphism drawn through, whose finite codomain names the
            alphabet.
        arguments : list[tuple[str, Value]]
            The lowered arguments, whose probability vector's length
            names it otherwise.
        scope : _Scope
            The scope.
        step : object
            The step.

        Returns
        -------
        list[tuple[str, Value]]
            The arguments, unchanged.
        """
        del step
        alphabet = _CLASS_INDEX_FAMILIES.get(record.name)
        if alphabet is None:
            return arguments
        state = self._program_state
        assert state is not None
        for name, value in arguments:
            if name != alphabet:
                continue
            shape = tensor_shape(self._value_type(value, scope.context(), None))
            if shape is not None and shape[1]:
                state.pending_alphabet = PlateAxis("classes", shape[1][-1])
        if morphism is not None and isinstance(morphism.codomain, TypeName):
            info = self._program_objects.get(morphism.codomain.name)
            if info is not None and info.finite and info.extent is not None:
                state.pending_alphabet = PlateAxis(
                    morphism.codomain.name, _extent(info.extent)
                )
        return arguments

    # ------------------------------------------------------------------
    # structured and mapped morphisms

    def _gp_grid(self: _Elaborator, morphism: MorphismDecl, step: object) -> PlateAxis:
        """The grid axis a GP morphism's draws range over.

        Parameters
        ----------
        morphism : MorphismDecl
            The ``~ GP`` morphism.
        step : object
            The step, for diagnostics.

        Returns
        -------
        PlateAxis
            The first finite factor of the morphism's domain.

        Raises
        ------
        QiecDiagnosticError
            If the domain has no finite factor.
        """
        for factor in _factors(morphism.domain):
            if not isinstance(factor, TypeName):
                continue
            info = self._program_objects.get(factor.name)
            if info is not None and info.finite and info.extent is not None:
                return PlateAxis(factor.name, _extent(info.extent))
        self._fail(
            step,
            f"GP morphism {morphism.names[0]!r} has no finite domain axis for its grid",
            code="qiec-program",
        )

    def _conditional_map(
        self: _Elaborator, morphism: MorphismDecl, family: str
    ) -> bool:
        """Whether a kernel morphism reads its arguments off a parameter map.

        Parameters
        ----------
        morphism : MorphismDecl
            The morphism.
        family : str
            The family it draws from.

        Returns
        -------
        bool
            ``True`` for a kernel-role morphism with a bare ``~ Family``
            initializer of a family with conditional heads, none of whose
            heads the option block writes.
        """
        role = _option_value(morphism.options, "role")
        if role is not None and role != "kernel":
            return False
        heads = _CONDITIONAL_HEADS.get(family)
        if heads is None:
            return False
        init = morphism.init_family
        bare = (init is not None and init.family == family and not init.args) or (
            init is None
            and isinstance(morphism.init_expr, ExprIdent)
            and morphism.init_expr.name == family
        )
        if not bare:
            return False
        names = {head for head, _ in heads}
        return not any(
            entry.key in names and isinstance(entry.value, OptionNumber | OptionString)
            for entry in morphism.options
        )

    def _mapped_arguments(
        self: _Elaborator,
        record: DistributionFamily,
        morphism: MorphismDecl,
        step: SampleStep | ObserveStep | MarginalizeStep,
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
    ) -> list[tuple[str, Value]]:
        """The arguments of a draw through a conditional kernel morphism.

        The morphism's weight and bias are program inputs; each head of
        the family reads its row block of the map applied to the
        conditioning row, which is the step's arguments or, absent any,
        the program's domain inputs.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        morphism : MorphismDecl
            The morphism.
        step : SampleStep | ObserveStep | MarginalizeStep
            The step.
        plate : PlateShape
            The distribution's plate.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.

        Returns
        -------
        list[tuple[str, Value]]
            One affine head per family parameter the map supplies.

        Raises
        ------
        QiecDiagnosticError
            If the codomain has no real width, the conditioning row is
            not made of bindings, or its width differs from the domain's.
        """
        del plate
        codomain = morphism.codomain
        info = (
            self._program_objects.get(codomain.name)
            if isinstance(codomain, TypeName)
            else None
        )
        if info is None or info.real_width is None:
            self._fail(
                step,
                f"morphism {step.morphism!r} maps its parameters onto a codomain with no "
                "real width",
                code="qiec-program",
            )
        width = info.real_width
        heads = _CONDITIONAL_HEADS[record.name]
        sources: list[Value] = []
        widths: list[int] = []
        if step.args:
            for argument in step.args:
                if not isinstance(argument, DrawArgName):
                    self._fail(
                        step,
                        f"morphism {step.morphism!r} conditions on a value, not a {argument.kind} argument",
                        code="qiec-program",
                    )
                local = scope.lookup(argument.text)
                if local is None:
                    self._fail(
                        step,
                        f"conditioning binding {argument.text!r} is not bound",
                        code="qiec-program",
                    )
                shape = tensor_shape(local.type)
                if local.type == REAL:
                    widths.append(1)
                elif (
                    shape is not None
                    and len(shape[1]) == 1
                    and isinstance(shape[1][0], IndexLiteral)
                ):
                    widths.append(int(shape[1][0].value))
                else:
                    self._fail(
                        step,
                        f"conditioning binding {argument.text!r} has no static width",
                        code="qiec-program",
                    )
                sources.append(Var(local))
        else:
            for name, role in state.parameters:
                if role != "domain":
                    continue
                local = scope.lookup(name)
                assert local is not None
                shape = tensor_shape(local.type)
                if (
                    shape is None
                    or len(shape[1]) != 1
                    or not isinstance(shape[1][0], IndexLiteral)
                ):
                    continue
                widths.append(int(shape[1][0].value))
                sources.append(Var(local))
        declared = 0
        for factor in _factors(morphism.domain):
            factor_info = (
                self._program_objects.get(factor.name)
                if isinstance(factor, TypeName)
                else None
            )
            if factor_info is None or factor_info.real_width is None:
                self._fail(
                    step,
                    f"morphism {step.morphism!r} has a domain factor with no real width",
                    code="qiec-program",
                )
            declared += factor_info.real_width
        if declared != sum(widths):
            self._fail(
                step,
                f"morphism {step.morphism!r} reads a {declared}-wide row but is "
                f"conditioned on {sum(widths)} coordinates",
                code="qiec-program",
            )
        rows = width * len(heads)
        weight = self._declare_parameter(
            f"{step.morphism}_param_weight",
            tensor_type(REAL, (_extent(rows), _extent(declared))),
            "weight",
            scope,
            state,
            step,
        )
        bias = self._declare_parameter(
            f"{step.morphism}_param_bias",
            tensor_type(REAL, (_extent(rows),)),
            "bias",
            scope,
            state,
            step,
        )
        arguments: list[tuple[str, Value]] = []
        for index, (name, transform) in enumerate(heads):
            result: TypeExpr = (
                REAL if width == 1 else tensor_type(REAL, (_extent(width),))
            )
            arguments.append(
                (
                    name,
                    AffineMap(
                        Var(weight),
                        Var(bias),
                        tuple(sources),
                        index * width,
                        width,
                        transform,
                        result,
                    ),
                )
            )
        return arguments

    def _structured_arguments(
        self: _Elaborator,
        record: DistributionFamily,
        morphism: MorphismDecl,
        plate: PlateShape,
        scope: _Scope,
        state: _ProgramState,
        step: object,
    ) -> list[tuple[str, Value]]:
        """The arguments of a bare structured family morphism.

        A ``~ MultivariateNormal`` or ``~ MatrixNormal`` morphism reads
        its location and covariances from data inputs shaped by the
        plate's event axes; a ``~ GP`` morphism's mean is zero and its
        covariance a kernel matrix over the input locations.

        Parameters
        ----------
        record : DistributionFamily
            The family.
        morphism : MorphismDecl
            The morphism.
        plate : PlateShape
            The distribution's plate.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        step : object
            The step.

        Returns
        -------
        list[tuple[str, Value]]
            The named arguments.

        Raises
        ------
        QiecDiagnosticError
            If the plate names no event axes for the family's rank, or a
            kernel option is unreadable.
        """
        event = tuple(axis.size for axis in plate.event)
        if record.name == "GP":
            factors = [f for f in _factors(morphism.domain) if isinstance(f, TypeName)]
            grid = None
            for factor in factors:
                info = self._program_objects.get(factor.name)
                if info is not None and info.finite and info.extent is not None:
                    grid = PlateAxis(factor.name, _extent(info.extent))
                    break
            if grid is None:
                self._fail(
                    step,
                    f"GP morphism {morphism.names[0]!r} has no finite domain axis for its grid",
                    code="qiec-program",
                )
            kernel = _option_value(morphism.options, "kernel")
            length_scale = _option_value(morphism.options, "length_scale")
            if kernel is None:
                kernel = "rbf"
            if not isinstance(kernel, str) or kernel != "rbf":
                self._fail(
                    step,
                    f"GP kernel {kernel!r} has no elaboration; the kernels are rbf",
                    code="qiec-program",
                )
            if length_scale is None:
                length_scale = 1.0
            if not isinstance(length_scale, float | int):
                self._fail(
                    step, "GP length_scale must be a number", code="qiec-program"
                )
            inputs = self._declare_parameter(
                "x", tensor_type(REAL, (grid.size,)), "kernel-input", scope, state, step
            )
            zeros = TensorValue(
                tuple(
                    LiteralValue(0.0, REAL)
                    for _ in range(int(cast(IndexLiteral, grid.size).value))
                ),
                tensor_type(REAL, (grid.size,)),
            )
            return [
                ("mean", zeros),
                (
                    "kernel",
                    KernelMatrix(
                        Var(inputs),
                        "rbf",
                        float(length_scale),
                        1e-8,
                        tensor_type(REAL, (grid.size, grid.size)),
                    ),
                ),
            ]
        if len(event) != record.event_rank:
            self._fail(
                step,
                f"morphism {morphism.names[0]!r} draws a rank-{record.event_rank} "
                f"{record.name} but its plate names {len(event)} event axes",
                code="qiec-program",
            )
        site = state.current_site
        if record.name == "MultivariateNormal":
            (n,) = event
            return [
                (
                    "loc",
                    Var(
                        self._declare_parameter(
                            f"{site}_loc",
                            tensor_type(REAL, (n,)),
                            "data",
                            scope,
                            state,
                            step,
                        )
                    ),
                ),
                (
                    "covariance_matrix",
                    Var(
                        self._declare_parameter(
                            f"{site}_covariance_matrix",
                            tensor_type(REAL, (n, n)),
                            "data",
                            scope,
                            state,
                            step,
                        )
                    ),
                ),
            ]
        if record.name == "MatrixNormal":
            rows, columns = event
            return [
                (
                    "loc",
                    Var(
                        self._declare_parameter(
                            f"{site}_loc",
                            tensor_type(REAL, (rows, columns)),
                            "data",
                            scope,
                            state,
                            step,
                        )
                    ),
                ),
                (
                    "row_covariance",
                    Var(
                        self._declare_parameter(
                            f"{site}_row_covariance",
                            tensor_type(REAL, (rows, rows)),
                            "data",
                            scope,
                            state,
                            step,
                        )
                    ),
                ),
                (
                    "col_covariance",
                    Var(
                        self._declare_parameter(
                            f"{site}_col_covariance",
                            tensor_type(REAL, (columns, columns)),
                            "data",
                            scope,
                            state,
                            step,
                        )
                    ),
                ),
            ]
        self._fail(
            step,
            f"family {record.name!r} has no structured elaboration",
            code="qiec-program",
        )


def _structured(
    record: DistributionFamily, morphism: MorphismDecl, step: object
) -> bool:
    """Whether a step draws a structured family through a bare morphism.

    Parameters
    ----------
    record : DistributionFamily
        The family.
    morphism : MorphismDecl
        The morphism.
    step : object
        The step, whose own arguments override the structured reading.

    Returns
    -------
    bool
        ``True`` for ``GP`` always, and for ``MultivariateNormal`` and
        ``MatrixNormal`` when neither the step nor the initializer
        supplies arguments.
    """
    if record.name == "GP":
        return True
    if record.name not in ("MultivariateNormal", "MatrixNormal"):
        return False
    if getattr(step, "args", None):
        return False
    init = morphism.init_family
    return init is None or not init.args


class _ScalarParameter:
    """A stand-in family parameter for a list entry.

    Parameters
    ----------
    element : TypeApplication
        The entry's element type.
    """

    def __init__(self, element: TypeApplication) -> None:
        self.rank = 0
        self.constraint = (
            "boolean"
            if element == BOOL
            else "nonnegative_integer"
            if element == INT
            else "real"
        )


@dataclass(slots=True)
class _ProgramState:
    """What accumulates while one program elaborates.

    Parameters
    ----------
    declaration
        The program.
    parameters
        The computation's parameters with their roles, in order.
    sites
        The probabilistic steps seen so far.
    helpers
        The marginalization helper computations declared so far.
    uses_random
        Whether any step performs on the canonical ``random`` instance.
    uses_score
        Whether any step performs on the canonical ``score`` instance.
    alphabets
        The class alphabet of each class-valued step, by step name.
    pending_alphabet
        The alphabet the step being elaborated draws from, if any.
    current_site
        The name of the step being elaborated.
    input_shapes
        The element type and batch axis inferred for each free name a
        let expression reads.
    random
        The canonical ``Random`` instance.
    score
        The canonical ``Score`` instance.
    """

    declaration: ProgramDecl
    parameters: list[tuple[str, ParameterRole]] = field(default_factory=list)
    sites: list[ProgramSite] = field(default_factory=list)
    helpers: list[NamedComputation] = field(default_factory=list)
    uses_random: bool = False
    uses_score: bool = False
    alphabets: dict[str, PlateAxis] = field(default_factory=dict)
    pending_alphabet: PlateAxis | None = None
    current_site: str = ""
    input_shapes: dict[str, tuple[TypeApplication, PlateAxis | None]] = field(
        default_factory=dict
    )
    random: NamedEffectInstance | None = None  # type: ignore[assignment]
    score: NamedEffectInstance | None = None  # type: ignore[assignment]


def _is_entry_point(declaration: ProgramDecl) -> bool:
    """Whether a program elaborates to a computation of its own.

    Parameters
    ----------
    declaration : ProgramDecl
        The program.

    Returns
    -------
    bool
        ``True`` unless the program is a template over objects or
        morphisms, which only its call sites instantiate; scalar
        parameters alone make an ordinary computation with real or
        natural parameters.
    """
    return declaration.type_params is None or all(
        isinstance(parameter, ScalarParam) for parameter in declaration.type_params
    )


def _extent(size: int) -> IndexLiteral:
    """A literal axis extent.

    Parameters
    ----------
    size : int
        The extent.

    Returns
    -------
    IndexLiteral
        The literal of the nat sort.
    """
    return IndexLiteral(size, NAT)


def _axis_name(expr: ObjectExpr) -> str:
    """A readable name for an object expression used as an axis.

    Parameters
    ----------
    expr : ObjectExpr
        The expression.

    Returns
    -------
    str
        The object's name, the factor names joined by ``x``, or the
        constructor's spelling.
    """
    if isinstance(expr, TypeName):
        return expr.name
    if isinstance(expr, ObjectProduct):
        return "x".join(_axis_name(item) for item in expr.components)
    if isinstance(expr, DiscreteConstructor | ContinuousConstructor):
        return f"{expr.constructor}{''.join(expr.args)}"
    return expr.kind


def _enclosing_group(scope: _Scope) -> PlateAxis | None:
    """The grouping axis of the nearest grouped marginalization.

    Parameters
    ----------
    scope : _Scope
        The scope.

    Returns
    -------
    PlateAxis | None
        The axis, or ``None`` outside any grouped scope.
    """
    current: _Scope | None = scope
    while current is not None:
        if current.weight_instance is not None:
            return current.group
        current = current.outer
    return None


def _weight_scope(scope: _Scope) -> _Scope | None:
    """The nearest marginalization scope collecting weights.

    Parameters
    ----------
    scope : _Scope
        The scope.

    Returns
    -------
    _Scope | None
        The scope owning a ``Weight`` instance, or ``None`` in the program
        body.
    """
    current: _Scope | None = scope
    while current is not None:
        if current.weight_instance is not None:
            return current
        current = current.outer
    return None


def _binding_order(scope: _Scope, name: str) -> int:
    """Where a name was bound among a scope chain's bindings.

    Parameters
    ----------
    scope : _Scope
        The innermost scope.
    name : str
        The name.

    Returns
    -------
    int
        The binding's position, outermost scope first.
    """
    context = scope.context()
    for position, local in enumerate(context.locals):
        if local.name == name:
            return position
    return len(context.locals)


def _object_expr_info(
    expr: ObjectExpr, table: Mapping[str, ObjectInfo]
) -> ObjectInfo | None:
    """The shape an object expression denotes.

    Parameters
    ----------
    expr : ObjectExpr
        The expression.
    table : Mapping[str, ObjectInfo]
        The objects declared so far.

    Returns
    -------
    ObjectInfo | None
        The shape, or ``None`` for an expression with no static one.
    """
    if isinstance(expr, TypeName):
        if expr.name.isdigit():
            return ObjectInfo(extent=int(expr.name), finite=True)
        return table.get(expr.name)
    if isinstance(expr, DiscreteConstructor):
        if not expr.args:
            return None
        size = _size_argument(expr.args[0], table)
        return None if size is None else ObjectInfo(extent=size, finite=True)
    if isinstance(expr, ContinuousConstructor):
        if not expr.args:
            return None
        sizes = [_size_argument(item, table) for item in expr.args]
        if any(size is None for size in sizes):
            return None
        extents = [cast(int, size) for size in sizes]
        if expr.constructor == "Real":
            width = 1
            for size in extents:
                width *= size
            return ObjectInfo(extent=width, real_width=width)
        return ObjectInfo(extent=extents[0])
    if isinstance(expr, ObjectProduct):
        factors = [_object_expr_info(item, table) for item in _factors(expr)]
        if any(item is None or item.extent is None for item in factors):
            return None
        total = 1
        finite = True
        for item in factors:
            assert item is not None and item.extent is not None
            total *= item.extent
            finite = finite and item.finite
        return ObjectInfo(extent=total, finite=finite)
    return None


def _size_argument(argument: str, table: Mapping[str, ObjectInfo]) -> int | None:
    """Resolve a constructor size argument.

    Parameters
    ----------
    argument : str
        An integer literal or an object name.
    table : Mapping[str, ObjectInfo]
        The objects declared so far.

    Returns
    -------
    int | None
        The size, or ``None`` when the name has no extent.
    """
    if argument.isdigit():
        return int(argument)
    prior = table.get(argument)
    return None if prior is None else prior.extent


def _object_table(statements: Sequence[object]) -> dict[str, ObjectInfo]:
    """The shapes of every object declaration in a module.

    Parameters
    ----------
    statements : Sequence[object]
        The module's statements.

    Returns
    -------
    dict[str, ObjectInfo]
        Name to shape, in declaration order, so a later declaration can
        size itself by an earlier one.
    """
    table: dict[str, ObjectInfo] = {}
    for statement in statements:
        if not isinstance(statement, ObjectDecl):
            continue
        info: ObjectInfo | None = None
        if isinstance(statement.init, TypeEnumSet):
            info = ObjectInfo(extent=len(statement.init.elements), finite=True)
        elif isinstance(statement.init, TypeFromExpr):
            info = _object_expr_info(statement.init.expr, table)
        if info is None:
            continue
        for name in statement.names:
            table[name] = info
    return table


def _origin_at(
    elaborator: _Elaborator,
    path: tuple[str | int, ...],
    role: str,
    node: object | None = None,
) -> object:
    """A source origin at a structural path.

    Parameters
    ----------
    elaborator : _Elaborator
        The elaborator, for the module name.
    path : tuple[str | int, ...]
        The structural path the identity derives from.
    role : str
        What the position is.
    node : object | None
        The source node whose line and column are recorded.

    Returns
    -------
    object
        The origin.
    """
    return elaborator._origin(
        node if node is not None else elaborator.source.syntax.statements[0], path, role
    )


__all__ = [
    "COLLECT_HANDLER",
    "GAP_CODE",
    "ENUMERATE_HANDLER",
    "ObjectInfo",
    "RANDOM_INSTANCE",
    "SCORE_INSTANCE",
]
