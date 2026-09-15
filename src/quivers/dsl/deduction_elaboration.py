"""Elaborate ``deduction`` declarations into kernel computations.

A weighted deductive system is a closed data family of items, a
computation enumerating its derivations, and a pair of handlers
aggregating them. The family declares one nullary constructor per atom
and one constructor per applied symbol the rules and lexicon mention,
with a slot typed ``Int`` where the system uses it as a position and
``Item`` otherwise. Rules become branches of a recursive computation
that performs ``choose`` on the deduction's ``Search`` instance to pick
an axiom or a rule, derives each premise by recursion, matches the
derived item against the premise pattern by ``case`` analysis (a shared
pattern variable is checked by the family's structural equality
computation), instantiates the conclusion, and adds the rule's weight
on the deduction's ``Weight[K]`` instance, where ``K`` is the carrier of
the declared semiring. A failed match performs an empty choice, which
the search handler answers by resuming nobody.

The entry computation handles the ``Weight`` instance with a collecting
handler and the ``Search`` instance with a handler that resumes once per
alternative and combines the shots by the semiring's addition, so the
answer is the semiring sum over derivations of the goal of the product
of the weights along each: the inside weight of the goal. Learned
weights are read through the module's ``params`` instance of ``Param``,
under names the classic engine keys its parameters by, so a program
that calls the deduction and scores its answer leaves ``params`` in its
row for the run to serve.

A system whose rules all relate ``span`` items enumerates by span: the
derivation of ``span(i, j, ...)`` chooses split positions for the
premise positions the conclusion leaves free and derives each premise
over its own span, so the search is bounded by the sentence, and only a
unary rule, which keeps the span, needs the declaration's ``depth``
option to bound it. Any other system takes its axioms and their weights
as input and enumerates derivations of any item to the depth the
option declares, or to the number of axioms and rules together without
one. The agenda engine tabulates the same derivations; the two agree on
every system whose derivations of the goal fit the bound.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Literal

from quivers.dsl.ast_nodes import (
    DeductionDecl,
    LetExprCall,
    LetExprLiteral,
    LetExprNode,
    LetExprVar,
    LexiconCategoryFixed,
    LexiconCategoryRestricted,
    LexiconCategoryWildcard,
    ObjectEffectApply,
    ObjectExpr,
    ObjectProduct,
    ObjectSlash,
    OptionName,
    OptionNumber,
    SequentRule,
    TypeName,
)
from quivers.dsl.compiler._options import (
    find_option,
    get_option_flag,
    get_option_name,
)
from quivers.dsl.compiler.deductions import load_lexicon_tsv
from quivers.qiec.canonical import LOG_WEIGHT, tensor_type
from quivers.qiec.declarations import ConstructorDecl, FamilyDecl, FieldDef
from quivers.qiec.effects import (
    ArgumentDef,
    EffectDef,
    EffectRef,
    EffectRequest,
    EffectRow,
    HandlerClauseDef,
    HandlerDef,
    OperationDef,
    ResumptionGrade,
    SiteProvenance,
    instantiate_effect,
)
from quivers.qiec.identifiers import (
    ComputationId,
    ConstructorId,
    EffectId,
    FamilyId,
    HandlerId,
    OperationId,
    StaticScopeId,
)
from quivers.qiec.kinds import NAT
from quivers.qiec.module import NamedComputation, NamedEffectInstance
from quivers.qiec.checking import ComputationSignature
from quivers.qiec.terms import (
    Bind,
    Call,
    Case,
    CaseBranch,
    CaseMotive,
    Computation,
    Comprehension,
    ConstructorValue,
    Gather,
    Handle,
    If,
    LiteralValue,
    Local,
    Perform,
    Reduction,
    Return,
    TensorValue,
    Value,
    Var,
)
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    STRING,
    UNIT,
    IndexBinder,
    IndexLiteral,
    IndexVariable,
    TypeApplication,
    TypeBinder,
    TypeExpr,
    TypeVariable,
    product_type,
)

if TYPE_CHECKING:
    from quivers.dsl.qiec_lowering import _Elaborator

#: The module effect a deduction's search performs on: a choice among
#: the entries of a tensor, resumed once per entry by the search handler.
SEARCH_EFFECT = "Search"
#: The operation of that effect.
SEARCH_CHOOSE = "choose"
#: The module instance of ``Param`` every deduction reads its learned
#: weights through.
PARAMS_INSTANCE = "params"
#: The semirings a deduction may declare, each with the kernel type of
#: its carrier.
SEMIRING_CARRIERS: dict[str, TypeExpr] = {
    "LogProb": LOG_WEIGHT,
    "Viterbi": LOG_WEIGHT,
    "Boolean": BOOL,
    "Counting": INT,
}
#: The constructor name of the ``span`` items a lexicon injects.
SPAN = "span"
#: The constructor holding a canonically numbered bound variable of a
#: lexicon's logical forms.
BOUND = "bound"

type Sort = Literal["int", "item"]


@dataclass(frozen=True, slots=True)
class _Atom:
    """A declared atom in a pattern.

    Parameters
    ----------
    name
        The atom's name.
    """

    name: str


@dataclass(frozen=True, slots=True)
class _Wild:
    """A pattern variable.

    Parameters
    ----------
    name
        The variable's name.
    """

    name: str


@dataclass(frozen=True, slots=True)
class _Lit:
    """An integer literal in a pattern.

    Parameters
    ----------
    value
        The integer.
    """

    value: int


@dataclass(frozen=True, slots=True)
class _Bound:
    """A bound variable of a logical form, canonically numbered.

    Parameters
    ----------
    index
        The variable's number within its term, from one.
    """

    index: int


@dataclass(frozen=True, slots=True)
class _App:
    """An applied constructor symbol in a pattern.

    Parameters
    ----------
    symbol
        The symbol.
    args
        The argument patterns.
    """

    symbol: str
    args: tuple[Pattern, ...]


type Pattern = _Atom | _Wild | _Lit | _Bound | _App


@dataclass(slots=True)
class _Rule:
    """One rule of a deduction, in pattern form.

    Parameters
    ----------
    source
        The declaration.
    premises
        The premise patterns.
    conclusion
        The conclusion pattern.
    learnable
        Whether the rule reads a learned weight keyed by its bindings.
    bounded
        Whether that weight is bounded below one per firing.
    parent
        The rule whose weight on the same bindings is added, if any.
    """

    source: SequentRule
    premises: tuple[Pattern, ...]
    conclusion: Pattern
    learnable: bool
    bounded: bool
    parent: str | None


@dataclass(slots=True)
class _Entry:
    """One expanded lexicon entry.

    Parameters
    ----------
    word
        The token the entry fires on.
    category
        The category pattern, ground.
    form
        The logical form pattern, ground, or ``None`` when the system
        carries no forms.
    learnable
        Whether the entry reads a learned weight.
    """

    word: str
    category: Pattern
    form: Pattern | None
    learnable: bool


@dataclass(slots=True)
class _Group:
    """The rules whose premises share one shape.

    Parameters
    ----------
    arity
        The number of premises.
    spans
        For a span-shaped system, the span each premise covers as a
        pair of normalized position names.
    splits
        The normalized position names beyond the conclusion's, which a
        derivation chooses.
    rules
        The rules with their indices.
    names
        For each rule index, its position variables' normalized names.
    """

    arity: int
    spans: tuple[tuple[str, str], ...]
    splits: tuple[str, ...]
    rules: list[tuple[int, _Rule]]
    names: dict[int, dict[str, str]]


@dataclass(slots=True)
class _System:
    """Everything the elaboration reads off one deduction.

    Parameters
    ----------
    decl
        The declaration.
    semiring
        The declared semiring's name.
    start
        The start symbol, if any.
    depth
        The declared derivation depth bound, if any.
    rules
        The rules in pattern form.
    entries
        The expanded lexicon, empty for a system that takes its axioms
        as input.
    uses_forms
        Whether span items carry a logical form.
    symbols
        Every constructor symbol with its slot sorts, atoms included as
        symbols of no slots.
    bare
        The atoms used as items on their own, which cannot also be
        applied.
    span_shaped
        Whether every rule relates span items, so derivations may be
        enumerated by span.
    """

    decl: DeductionDecl
    semiring: str
    start: str | None
    depth: int | None
    rules: list[_Rule] = field(default_factory=list)
    entries: list[_Entry] = field(default_factory=list)
    uses_forms: bool = False
    symbols: dict[str, list[Sort]] = field(default_factory=dict)
    bare: set[str] = field(default_factory=set)
    span_shaped: bool = False
    family: FamilyDecl | None = None
    constructors: dict[str, ConstructorDecl] = field(default_factory=dict)
    choice: NamedEffectInstance | None = None
    weight: NamedEffectInstance | None = None
    fresh: int = 0

    @property
    def carrier(self) -> TypeExpr:
        """The kernel type of the semiring's weights.

        Returns
        -------
        TypeExpr
            ``LogWeight`` for the log semirings, ``Bool`` for the Boolean
            one, ``Int`` for the counting one.
        """
        return SEMIRING_CARRIERS[self.semiring]

    @property
    def item(self) -> TypeApplication:
        """The item family applied.

        Returns
        -------
        TypeApplication
            The type of every item.
        """
        assert self.family is not None
        return TypeApplication(self.family.type_constructor, ())

    def name(self, suffix: str) -> str:
        """A module-level name derived from the deduction's.

        Parameters
        ----------
        suffix : str
            What the name is for.

        Returns
        -------
        str
            ``<deduction>__<suffix>``.
        """
        return f"{self.decl.name}__{suffix}"

    def local(self, hint: str, type_: TypeExpr) -> Local:
        """A fresh local.

        Parameters
        ----------
        hint : str
            The name's stem.
        type_ : TypeExpr
            The local's type.

        Returns
        -------
        Local
            A local named uniquely within the deduction's computations.
        """
        self.fresh += 1
        return Local(f"{hint}_{self.fresh}", type_)


def _int(value: int) -> LiteralValue:
    """An ``Int`` literal.

    Parameters
    ----------
    value : int
        The integer.

    Returns
    -------
    LiteralValue
        The literal.
    """
    return LiteralValue(value, INT)


def _string(value: str) -> LiteralValue:
    """A ``String`` literal.

    Parameters
    ----------
    value : str
        The text.

    Returns
    -------
    LiteralValue
        The literal.
    """
    return LiteralValue(value, STRING)


def _pattern_variables(pattern: Pattern) -> list[str]:
    """The variables of a pattern in order of first occurrence.

    Parameters
    ----------
    pattern : Pattern
        The pattern.

    Returns
    -------
    list[str]
        The variable names, each once.
    """
    found: list[str] = []

    def visit(node: Pattern) -> None:
        """Collect one pattern's variables.

        Parameters
        ----------
        node : Pattern
            The pattern.
        """
        if isinstance(node, _Wild):
            if node.name not in found:
                found.append(node.name)
        elif isinstance(node, _App):
            for argument in node.args:
                visit(argument)

    visit(pattern)
    return found


def _is_span(pattern: Pattern) -> bool:
    """Whether a pattern is a span with variable positions.

    Parameters
    ----------
    pattern : Pattern
        The pattern.

    Returns
    -------
    bool
        ``True`` for ``span(I, J, ...)`` with ``I`` and ``J`` variables.
    """
    return (
        isinstance(pattern, _App)
        and pattern.symbol == SPAN
        and len(pattern.args) >= 3
        and isinstance(pattern.args[0], _Wild)
        and isinstance(pattern.args[1], _Wild)
    )


class _DeductionElaboration:
    """Deduction elaboration, mixed into the QIEC elaborator.

    The declaration pass runs once the module's own families, effects,
    and instances are declared and registered, so a deduction's item
    family and instances join the registry; the computation pass runs
    after the module's handlers are registered and before programs are
    elaborated, so a program body can call a deduction's entry.
    """

    def _declare_deduction_declarations(self: _Elaborator) -> None:
        """Declare each deduction's item family, effect, and instances.

        Raises
        ------
        QiecDiagnosticError
            If a deduction declares an unknown semiring, an option
            outside the elaboration, or a rule the pattern language does
            not cover.
        """
        self._deductions: dict[str, _System] = {}
        declarations = [
            statement
            for statement in self.source.syntax.statements
            if isinstance(statement, DeductionDecl)
        ]
        if not declarations:
            return
        self._declare_search_effect()
        self._program_instance(PARAMS_INSTANCE, "Param")
        for declaration in declarations:
            system = self._read_deduction(declaration)
            self._declare_item_family(system)
            self._declare_deduction_instances(system)
            self._deductions[declaration.name] = system

    def _declare_search_effect(self: _Elaborator) -> None:
        """Declare the ``Search`` effect and register it.

        Its one operation, ``choose[a : Type, n : Nat] : Tensor[a]([n])
        -> a``, is answered by resuming once per entry of the tensor.

        Raises
        ------
        QiecDiagnosticError
            If the module declares an effect of that name itself.
        """
        if SEARCH_EFFECT in self.effects:
            self._fail(
                self.source.syntax.statements[0],
                f"effect {SEARCH_EFFECT!r} is reserved for deductions",
                code="qiec-program",
            )
        effect_id = EffectId.derive(self.source.module_name, "effect", SEARCH_EFFECT)
        ref = EffectRef(effect_id, SEARCH_EFFECT)
        element = TypeVariable("a")
        extent = IndexVariable("n", NAT)
        operation = OperationDef(
            OperationId.derive(effect_id, "operation", SEARCH_CHOOSE),
            SEARCH_CHOOSE,
            (TypeBinder("a"), IndexBinder("n", NAT)),
            (ArgumentDef("alternatives", tensor_type(element, (extent,))),),
            element,
        )
        effect = EffectDef(ref, (), (operation,))
        self.effects[SEARCH_EFFECT] = effect
        self.registry.register_effect(effect)

    def _read_deduction(self: _Elaborator, declaration: DeductionDecl) -> _System:
        """Read a declaration's options, rules, and lexicon into patterns.

        Parameters
        ----------
        declaration : DeductionDecl
            The declaration.

        Returns
        -------
        _System
            The system, with its symbols' slot sorts inferred.

        Raises
        ------
        QiecDiagnosticError
            If the semiring is unknown, an option has no elaboration, or
            a rule's conclusion mentions a variable no premise binds.
        """
        semiring = get_option_name(
            declaration.options, "semiring", line=declaration.line, col=declaration.col
        )
        if semiring is None:
            semiring = "LogProb"
        if semiring not in SEMIRING_CARRIERS:
            self._fail(
                declaration,
                f"deduction {declaration.name!r} names the semiring {semiring!r}; "
                f"the semirings are {', '.join(SEMIRING_CARRIERS)}",
                code="qiec-program",
            )
        # An ``axioms`` morphism produces the axioms the agenda engine
        # starts from; on the reference machine the caller supplies them,
        # as for any system without a lexicon. A ``signature`` or an
        # ``encoder`` attaches item embeddings for structural losses and
        # leaves the weights alone, so neither concerns the elaboration.
        depth_entry = find_option(declaration.options, "depth")
        depth: int | None = None
        if depth_entry is not None:
            if not isinstance(depth_entry.value, OptionNumber):
                self._fail(
                    declaration,
                    f"deduction {declaration.name!r} option depth must be a number",
                    code="qiec-program",
                )
            depth = int(depth_entry.value.value)
        start = get_option_name(
            declaration.options, "start", line=declaration.line, col=declaration.col
        )
        system = _System(declaration, semiring, start, depth)
        atoms = set(declaration.atoms)
        for atom in declaration.atoms:
            system.symbols.setdefault(atom, [])
        system.uses_forms = bool(declaration.binders) or any(
            isinstance(pattern, ObjectEffectApply)
            and pattern.effect == SPAN
            and len(pattern.args) >= 4
            for rule in declaration.rules
            for pattern in (*rule.premises, rule.conclusion)
        )
        counter = [0]
        for rule in declaration.rules:
            premises = tuple(
                self._span_form(
                    self._object_pattern(item, atoms, system), system, counter
                )
                for item in rule.premises
            )
            conclusion = self._span_form(
                self._object_pattern(rule.conclusion, atoms, system), system, counter
            )
            bound = {
                name for premise in premises for name in _pattern_variables(premise)
            }
            unbound = [
                name for name in _pattern_variables(conclusion) if name not in bound
            ]
            if unbound:
                self._fail(
                    rule,
                    f"rule {rule.name!r} concludes with {', '.join(unbound)!r}, "
                    "which no premise binds",
                    code="qiec-program",
                )
            parent = find_option(rule.options, "parent")
            parent_name: str | None = None
            if parent is not None:
                if not isinstance(parent.value, OptionName):
                    self._fail(
                        rule,
                        f"rule {rule.name!r} option parent must name a rule",
                        code="qiec-program",
                    )
                parent_name = parent.value.value
            if find_option(rule.options, "weight") is not None:
                self._fail(
                    rule,
                    f"rule {rule.name!r} option weight is an expression over its "
                    "bindings; a computed rule weight has no elaboration yet",
                    code="qiec-program-gap",
                )
            system.rules.append(
                _Rule(
                    rule,
                    premises,
                    conclusion,
                    get_option_flag(rule.options, "learnable"),
                    get_option_flag(rule.options, "bounded"),
                    parent_name,
                )
            )
        for rule in system.rules:
            if rule.parent is not None and rule.parent not in {
                item.source.name for item in system.rules
            }:
                self._fail(
                    rule.source,
                    f"rule {rule.source.name!r} names the parent {rule.parent!r}, "
                    "which the deduction does not declare",
                    code="qiec-program",
                )
        self._read_lexicon(declaration, atoms, system)
        self._infer_sorts(system)
        system.span_shaped = bool(system.entries) and all(
            _is_span(pattern)
            for rule in system.rules
            for pattern in (*rule.premises, rule.conclusion)
        )
        unary = any(len(rule.premises) == 1 for rule in system.rules)
        if system.depth is None and unary and system.span_shaped:
            self._fail(
                declaration,
                f"deduction {declaration.name!r} has a unary rule, so its "
                "derivations over one span need a depth option to bound them",
                code="qiec-program",
            )
        return system

    def _object_pattern(
        self: _Elaborator, expr: ObjectExpr, atoms: set[str], system: _System
    ) -> Pattern:
        """Convert a rule's object expression into a pattern.

        Parameters
        ----------
        expr : ObjectExpr
            The expression.
        atoms : set[str]
            The declared atoms, whose names are ground; every other
            identifier is a variable.
        system : _System
            The system, whose symbol table the applied heads join.

        Returns
        -------
        Pattern
            The pattern.

        Raises
        ------
        QiecDiagnosticError
            If the expression has a shape the pattern language does not
            cover.
        """
        if isinstance(expr, TypeName):
            if expr.name in atoms:
                self._note_atom(system, expr.name)
                return _Atom(expr.name)
            if expr.name.isdigit():
                return _Lit(int(expr.name))
            return _Wild(expr.name)
        if isinstance(expr, ObjectEffectApply):
            args = tuple(
                self._object_pattern(item, atoms, system) for item in expr.args
            )
            self._note_symbol(system, expr.effect, len(args))
            return _App(expr.effect, args)
        if isinstance(expr, ObjectSlash):
            symbol = "fwd_slash" if expr.direction == "/" else "bwd_slash"
            args = (
                self._object_pattern(expr.result, atoms, system),
                self._object_pattern(expr.argument, atoms, system),
            )
            self._note_symbol(system, symbol, 2)
            return _App(symbol, args)
        if isinstance(expr, ObjectProduct):
            args = tuple(
                self._object_pattern(item, atoms, system) for item in expr.components
            )
            symbol = f"product{len(args)}"
            self._note_symbol(system, symbol, len(args))
            return _App(symbol, args)
        self._fail(
            system.decl,
            f"deduction {system.decl.name!r} has a rule pattern of kind "
            f"{expr.kind!r}, which the pattern language does not cover",
            code="qiec-program",
        )

    def _note_atom(self: _Elaborator, system: _System, atom: str) -> None:
        """Record an atom used as an item on its own.

        Parameters
        ----------
        system : _System
            The system.
        atom : str
            The atom.

        Raises
        ------
        QiecDiagnosticError
            If the atom is also applied to arguments.
        """
        if system.symbols.get(atom):
            self._fail(
                system.decl,
                f"deduction {system.decl.name!r} uses {atom!r} both bare and applied",
                code="qiec-program",
            )
        system.bare.add(atom)

    def _note_symbol(
        self: _Elaborator, system: _System, symbol: str, arity: int
    ) -> None:
        """Record an applied symbol and its arity.

        A declared atom applied to arguments is a constructor of that
        arity, as the classic engine reads it, unless it is also used
        bare.

        Parameters
        ----------
        system : _System
            The system.
        symbol : str
            The symbol.
        arity : int
            How many arguments it takes.

        Raises
        ------
        QiecDiagnosticError
            If the symbol is applied at two arities, or is an atom also
            used bare.
        """
        existing = system.symbols.get(symbol)
        if existing is None or (not existing and symbol not in system.bare):
            system.symbols[symbol] = ["item"] * arity
        elif len(existing) != arity:
            self._fail(
                system.decl,
                f"deduction {system.decl.name!r} applies {symbol!r} to both "
                f"{len(existing)} and {arity} arguments",
                code="qiec-program",
            )

    def _span_form(
        self: _Elaborator, pattern: Pattern, system: _System, counter: list[int]
    ) -> Pattern:
        """Give a three-slot span pattern its form slot when the system has one.

        Parameters
        ----------
        pattern : Pattern
            The pattern.
        system : _System
            The system.
        counter : list[int]
            The count of fresh form variables minted so far.

        Returns
        -------
        Pattern
            The pattern, every ``span(I, J, C)`` extended with a fresh
            variable in the form slot when the system carries forms.
        """
        if not system.uses_forms:
            return pattern
        if isinstance(pattern, _App):
            args = tuple(
                self._span_form(item, system, counter) for item in pattern.args
            )
            if pattern.symbol == SPAN and len(args) == 3:
                counter[0] += 1
                args = (*args, _Wild(f"_lf{counter[0]}"))
                system.symbols[SPAN] = ["item"] * 4
            return _App(pattern.symbol, args)
        return pattern

    def _read_lexicon(
        self: _Elaborator, declaration: DeductionDecl, atoms: set[str], system: _System
    ) -> None:
        """Expand the lexicon into entries with pattern-form categories and forms.

        Parameters
        ----------
        declaration : DeductionDecl
            The declaration.
        atoms : set[str]
            The declared atoms.
        system : _System
            The system the entries join.

        Raises
        ------
        QiecDiagnosticError
            If a restricted category names an undeclared atom, or a
            logical form uses an expression outside the constructor
            language.
        """
        raw: list[
            tuple[
                str,
                LexiconCategoryFixed | None,
                LetExprNode | None,
                bool,
                tuple[str, ...],
            ]
        ] = []
        for entry in declaration.lexicon:
            learnable = get_option_flag(entry.options, "learnable")
            if isinstance(entry.category, LexiconCategoryFixed):
                candidates: tuple[str, ...] = ()
            elif isinstance(entry.category, LexiconCategoryWildcard):
                candidates = tuple(declaration.atoms)
            elif isinstance(entry.category, LexiconCategoryRestricted):
                unknown = [name for name in entry.category.atoms if name not in atoms]
                if unknown:
                    self._fail(
                        declaration,
                        f"deduction {declaration.name!r}: lexicon entry restricts its "
                        f"category to {', '.join(unknown)!r}, not declared atoms",
                        code="qiec-program",
                    )
                candidates = tuple(entry.category.atoms)
            else:
                self._fail(
                    declaration,
                    f"deduction {declaration.name!r}: lexicon entry has an unknown "
                    "category kind",
                    code="qiec-program",
                )
            fixed = (
                entry.category
                if isinstance(entry.category, LexiconCategoryFixed)
                else None
            )
            for word in entry.words:
                raw.append((word, fixed, entry.lf, learnable, candidates))
        if declaration.lexicon_from_file is not None:
            learnable = get_option_flag(
                declaration.lexicon_from_file_options, "learnable"
            )
            for word, category, form in load_lexicon_tsv(
                declaration.lexicon_from_file, declaration
            ):
                raw.append(
                    (word, LexiconCategoryFixed(category=category), form, learnable, ())
                )
        binders = frozenset(declaration.binders)
        for word, fixed, form_expr, learnable, candidates in raw:
            form = (
                self._form_pattern(form_expr, atoms, binders, system, declaration)
                if system.uses_forms and form_expr is not None
                else None
            )
            if fixed is not None:
                category = self._object_pattern(fixed.category, atoms, system)
                system.entries.append(_Entry(word, category, form, learnable))
            else:
                for atom in candidates:
                    system.entries.append(_Entry(word, _Atom(atom), form, True))
        if system.entries:
            sorts: list[Sort] = ["int", "int", "item", "item"]
            system.symbols[SPAN] = sorts[: 4 if system.uses_forms else 3]

    def _form_pattern(
        self: _Elaborator,
        expr: LetExprNode,
        atoms: set[str],
        binders: frozenset[str],
        system: _System,
        declaration: DeductionDecl,
    ) -> Pattern:
        """Convert a logical form into a ground pattern with numbered binders.

        Parameters
        ----------
        expr : LetExprNode
            The form.
        atoms : set[str]
            The declared atoms.
        binders : frozenset[str]
            The constructors whose first argument binds a variable.
        system : _System
            The system, whose symbol table the heads join.
        declaration : DeductionDecl
            The declaration, for diagnostics.

        Returns
        -------
        Pattern
            The pattern, with bound variables numbered from one in
            binding order so alpha-equivalent forms are equal.

        Raises
        ------
        QiecDiagnosticError
            If the form uses an expression that is not a constructor
            application, an atom, a bound variable, or an integer.
        """
        counter = [0]

        def visit(node: LetExprNode, scope: dict[str, int]) -> Pattern:
            """Convert one subterm.

            Parameters
            ----------
            node : LetExprNode
                The subterm.
            scope : dict[str, int]
                The bound variables in scope, by number.

            Returns
            -------
            Pattern
                The pattern.
            """
            if isinstance(node, LetExprVar):
                if node.name in scope:
                    self._note_symbol(system, BOUND, 1)
                    return _Bound(scope[node.name])
                if node.name in atoms:
                    self._note_atom(system, node.name)
                    return _Atom(node.name)
                self._fail(
                    declaration,
                    f"deduction {declaration.name!r}: logical form names "
                    f"{node.name!r}, which is neither an atom nor bound",
                    code="qiec-program",
                )
            if isinstance(node, LetExprLiteral):
                if float(node.value).is_integer():
                    return _Lit(int(node.value))
                self._fail(
                    declaration,
                    f"deduction {declaration.name!r}: logical form holds the real "
                    f"{node.value!r}; items carry integers",
                    code="qiec-program",
                )
            if isinstance(node, LetExprCall):
                if node.func in binders and node.args:
                    first = node.args[0]
                    if not isinstance(first, LetExprVar):
                        self._fail(
                            declaration,
                            f"deduction {declaration.name!r}: binder {node.func!r} "
                            "binds a name in its first argument",
                            code="qiec-program",
                        )
                    counter[0] += 1
                    inner = {**scope, first.name: counter[0]}
                    self._note_symbol(system, BOUND, 1)
                    args = (
                        _Bound(counter[0]),
                        *(visit(item, inner) for item in node.args[1:]),
                    )
                else:
                    args = tuple(visit(item, scope) for item in node.args)
                self._note_symbol(system, node.func, len(args))
                return _App(node.func, args)
            self._fail(
                declaration,
                f"deduction {declaration.name!r}: logical form uses a "
                f"{type(node).__name__}, which is not a constructor term",
                code="qiec-program",
            )

        return visit(expr, {})

    def _infer_sorts(self: _Elaborator, system: _System) -> None:
        """Fix each constructor slot's sort from how the system uses it.

        A slot is an integer when a literal fills it or a variable shared
        with an integer slot does, and a position of ``span`` is an
        integer by the lexicon's convention; every other slot holds an
        item.

        Parameters
        ----------
        system : _System
            The system, whose symbol table is updated in place.
        """
        if BOUND in system.symbols:
            system.symbols[BOUND] = ["int"]
        changed = True
        while changed:
            changed = False
            for rule in system.rules:
                integral: set[str] = set()
                occurrences: dict[str, list[tuple[str, int]]] = {}

                def walk(pattern: Pattern) -> None:
                    """Collect the slots each variable fills.

                    Parameters
                    ----------
                    pattern : Pattern
                        The pattern.
                    """
                    if not isinstance(pattern, _App):
                        return
                    sorts = system.symbols[pattern.symbol]
                    for position, argument in enumerate(pattern.args):
                        if isinstance(argument, _Wild):
                            occurrences.setdefault(argument.name, []).append(
                                (pattern.symbol, position)
                            )
                            if sorts[position] == "int":
                                integral.add(argument.name)
                        elif isinstance(argument, _Lit):
                            if sorts[position] != "int":
                                sorts[position] = "int"
                        else:
                            walk(argument)

                for pattern in (*rule.premises, rule.conclusion):
                    walk(pattern)
                for name in integral:
                    for symbol, position in occurrences[name]:
                        if system.symbols[symbol][position] != "int":
                            system.symbols[symbol][position] = "int"
                            changed = True

    def _declare_item_family(self: _Elaborator, system: _System) -> None:
        """Declare the deduction's item family and register it.

        Parameters
        ----------
        system : _System
            The system.

        Raises
        ------
        QiecDiagnosticError
            If the kernel rejects a declaration.
        """
        family_name = system.name("Item")
        family_id = FamilyId.derive(self.source.module_name, "family", family_name)
        names = tuple(system.symbols)
        constructor_ids = tuple(
            ConstructorId.derive(family_id, "constructor", system.name(symbol))
            for symbol in names
        )
        family = FamilyDecl(family_id, family_name, (), (), constructor_ids)
        system.family = family
        item = TypeApplication(family.type_constructor, ())
        self.families[family_name] = family
        self.registry.register_family(family)
        for symbol, identity in zip(names, constructor_ids, strict=True):
            fields = tuple(
                FieldDef(f"arg{position}", INT if sort == "int" else item)
                for position, sort in enumerate(system.symbols[symbol])
            )
            constructor = ConstructorDecl(
                identity, family_id, system.name(symbol), (), fields, ()
            )
            system.constructors[symbol] = constructor
            self.constructors[constructor.name] = constructor
            self.registry.register_constructor(constructor)

    def _declare_deduction_instances(self: _Elaborator, system: _System) -> None:
        """Allocate the deduction's search and weight instances.

        Parameters
        ----------
        system : _System
            The system.
        """
        search = self.effects[SEARCH_EFFECT]
        choice_name = system.name("choice")
        entry = instantiate_effect(
            search.ref,
            module=self.source.module_name,
            lexical_path=("instances", choice_name),
        )
        system.choice = NamedEffectInstance(
            choice_name,
            entry,
            self._origin(system.decl, ("instances", choice_name), "effect-instance"),
        )
        self.instances[choice_name] = system.choice
        weight_effect = self._effect("Weight")
        assert weight_effect is not None
        weight_name = system.name("weight")
        entry = instantiate_effect(
            weight_effect.apply((system.carrier,)),
            module=self.source.module_name,
            lexical_path=("instances", weight_name),
        )
        system.weight = NamedEffectInstance(
            weight_name,
            entry,
            self._origin(system.decl, ("instances", weight_name), "effect-instance"),
        )
        self.instances[weight_name] = system.weight

    # ------------------------------------------------------------------
    # computations

    def _declare_deductions(self: _Elaborator) -> tuple[NamedComputation, ...]:
        """Elaborate every deduction into its handlers and computations.

        Returns
        -------
        tuple[NamedComputation, ...]
            The deductions' computations in source order: for each, the
            equality, rendering, and goal computations, the axiom and
            derivation computations, and the entry.

        Raises
        ------
        QiecDiagnosticError
            If a body fails to check.
        """
        computations: list[NamedComputation] = []
        for system in getattr(self, "_deductions", {}).values():
            computations.extend(self._elaborate_deduction(system))
        return tuple(computations)

    def _elaborate_deduction(
        self: _Elaborator, system: _System
    ) -> list[NamedComputation]:
        """Elaborate one deduction.

        Parameters
        ----------
        system : _System
            The system.

        Returns
        -------
        list[NamedComputation]
            Its computations.
        """
        params = self.instances[PARAMS_INSTANCE]
        assert system.choice is not None and system.weight is not None
        # Only a log semiring reads learned weights, and only the
        # computations whose entries or rules declare one perform on
        # the parameter store.
        logged = system.semiring in ("LogProb", "Viterbi")
        lexical_learned = logged and any(entry.learnable for entry in system.entries)
        learned = lexical_learned or (
            logged
            and any(rule.learnable or rule.parent is not None for rule in system.rules)
        )
        base = (system.choice.entry, system.weight.entry)
        axiom_row = EffectRow((*base, params.entry) if lexical_learned else base)
        row = EffectRow((*base, params.entry) if learned else base)
        outer = EffectRow((params.entry,) if learned else ())
        pure = EffectRow()
        extent = IndexBinder("n", NAT)
        n = IndexVariable("n", NAT)
        item = system.item
        carrier = system.carrier
        signatures: dict[str, ComputationSignature] = {}

        def declare(
            name: str,
            telescope: tuple[IndexBinder, ...],
            parameters: tuple[TypeExpr, ...],
            result: TypeExpr,
            effects: EffectRow,
        ) -> ComputationSignature:
            """Register one computation's signature before any body is built.

            Parameters
            ----------
            name : str
                The computation's name.
            telescope : tuple[IndexBinder, ...]
                Its static binders.
            parameters : tuple[TypeExpr, ...]
                Its parameter types.
            result : TypeExpr
                Its result type.
            effects : EffectRow
                Its row.

            Returns
            -------
            ComputationSignature
                The registered signature.
            """
            identity = ComputationId.derive(
                self.source.module_name, "computation", name
            )
            signature = ComputationSignature(
                identity, name, telescope, parameters, result, effects
            )
            self.registry.register_computation(signature)
            self.computation_signatures[name] = signature
            signatures[name] = signature
            return signature

        eq = declare(system.name("eq"), (), (item, item), BOOL, pure)
        show = declare(system.name("show"), (), (item,), STRING, pure)
        goal = declare(system.name("goal"), (), (item,), BOOL, pure)
        failing = declare(
            system.name("fail"), (), (), item, EffectRow((system.choice.entry,))
        )
        if system.entries:
            tokens_type = tensor_type(STRING, (n,))
            axiom = declare(
                system.name("axiom"), (extent,), (INT, tokens_type), item, axiom_row
            )
            derive = declare(
                system.name("derive"),
                (extent,),
                (INT, INT, INT, tokens_type)
                if system.span_shaped
                else (INT, tokens_type),
                item,
                row,
            )
            entry = declare(
                system.name("run"), (extent,), (tokens_type,), carrier, outer
            )
        else:
            axioms_type = tensor_type(item, (n,))
            weights_type = tensor_type(carrier, (n,))
            axiom = None
            derive = declare(
                system.name("derive"),
                (extent,),
                (INT, axioms_type, weights_type),
                item,
                row,
            )
            entry = declare(
                system.name("run"),
                (extent,),
                (axioms_type, weights_type),
                carrier,
                outer,
            )
        context = _Context(self, system, signatures, params, n)
        computations = [
            context.equality(eq),
            context.rendering(show),
            context.goal(goal),
            context.failure(failing),
        ]
        if axiom is not None:
            computations.append(context.axiom(axiom))
        computations.append(context.derivation(derive))
        computations.append(context.entry(entry, self._deduction_handlers(system)))
        return computations

    def _deduction_handlers(
        self: _Elaborator, system: _System
    ) -> tuple[HandlerDef, HandlerDef]:
        """Declare the search and collection handlers of one deduction.

        Parameters
        ----------
        system : _System
            The system.

        Returns
        -------
        tuple[HandlerDef, HandlerDef]
            The ``Search`` handler, answering the semiring sum of the
            collected weights of its shots, and the ``Weight`` handler,
            answering the scope's unit paired with its product.
        """
        assert system.choice is not None and system.weight is not None
        search = self.effects[SEARCH_EFFECT]
        weight_effect = self._effect("Weight")
        assert weight_effect is not None
        carrier = system.carrier
        shot = product_type(UNIT, carrier)
        search_name = system.name(f"search_{system.semiring.lower()}")
        search_handler = HandlerDef(
            HandlerId.derive(self.source.module_name, "handler", search_name),
            search_name,
            search.ref,
            (HandlerClauseDef(search.operations[0].id, ResumptionGrade.UNRESTRICTED),),
            shot,
            carrier,
            EffectRow(),
            total=True,
            implementation="foreign",
        )
        collect_name = system.name(f"collect_{system.semiring.lower()}")
        add = next(item for item in weight_effect.operations if item.name == "add")
        collect_handler = HandlerDef(
            HandlerId.derive(self.source.module_name, "handler", collect_name),
            collect_name,
            weight_effect.apply((carrier,)),
            (HandlerClauseDef(add.id, ResumptionGrade.LINEAR),),
            UNIT,
            shot,
            EffectRow(),
            total=True,
            implementation="foreign",
        )
        for handler in (search_handler, collect_handler):
            self.handlers[handler.name] = handler
            self.registry.register_handler(handler)
        return search_handler, collect_handler


type _Continuation = Callable[[dict[str, Local]], Computation]


class _Context:
    """The term builders of one deduction's computations.

    Parameters
    ----------
    elaborator : _Elaborator
        The elaborator, for primitives, origins, and diagnostics.
    system : _System
        The system.
    signatures : dict[str, ComputationSignature]
        The deduction's computations by name.
    params : NamedEffectInstance
        The module's ``Param`` instance.
    extent : IndexVariable
        The static extent of the entry's tensor argument.
    """

    def __init__(
        self,
        elaborator: _Elaborator,
        system: _System,
        signatures: dict[str, ComputationSignature],
        params: NamedEffectInstance,
        extent: IndexVariable,
    ) -> None:
        self.elaborator = elaborator
        self.system = system
        self.signatures = signatures
        self.params = params
        self.extent = extent
        self.module = elaborator.source.module_name
        self.path: tuple[str | int, ...] = ("deductions", system.decl.name)

    # -- term helpers ---------------------------------------------------

    def origin(self, *path: str | int) -> object:
        """A source origin beneath the deduction's path.

        Parameters
        ----------
        *path : str | int
            The path beneath it.

        Returns
        -------
        object
            The origin.
        """
        return self.elaborator._origin(
            self.system.decl, (*self.path, *path), "computation"
        )

    def primitive(self, name: str, *arguments: Value) -> Value:
        """Apply a registry primitive to scalar arguments.

        Parameters
        ----------
        name : str
            The primitive.
        *arguments : Value
            Its arguments.

        Returns
        -------
        Value
            The application.
        """
        return self.elaborator._primitive(
            name, tuple(arguments), self.system.decl, (*self.path, "primitives", name)
        )

    def call(self, name: str, arguments: tuple[Value, ...]) -> Call:
        """Call one of the deduction's computations.

        Parameters
        ----------
        name : str
            The computation's name.
        arguments : tuple[Value, ...]
            The arguments.

        Returns
        -------
        Call
            The call, with the extent passed along when the callee
            binds one.
        """
        signature = self.signatures[name]
        statics = tuple(
            IndexVariable(binder.name, binder.sort)
            for binder in signature.telescope
            if isinstance(binder, IndexBinder)
        )
        return Call(
            signature.id,
            name,
            statics,
            arguments,
            signature.result,
            signature.effects,
            self.elaborator._origin(
                self.system.decl, (*self.path, "calls", name), "call"
            ),
        )

    def request(
        self,
        instance: NamedEffectInstance,
        operation: str,
        statics: tuple[TypeExpr | IndexLiteral | IndexVariable, ...],
        arguments: tuple[Value, ...],
        result: TypeExpr,
        *path: str | int,
    ) -> Perform:
        """Perform an operation on one of the deduction's instances.

        Parameters
        ----------
        instance : NamedEffectInstance
            The instance.
        operation : str
            The operation's name.
        statics : tuple[TypeExpr | IndexLiteral | IndexVariable, ...]
            The operation's static arguments.
        arguments : tuple[Value, ...]
            Its value arguments.
        result : TypeExpr
            The request's result type.
        *path : str | int
            The request's structural path beneath the deduction's.

        Returns
        -------
        Perform
            The request.
        """
        effect = self.elaborator.registry.effects[instance.entry.effect.id]
        definition = next(item for item in effect.operations if item.name == operation)
        return Perform(
            EffectRequest(
                instance.entry.instance,
                instance.entry.effect,
                definition.id,
                statics,
                arguments,
                result,
                SiteProvenance(
                    self.elaborator._origin(
                        self.system.decl, (*self.path, *path), "effect-request"
                    )
                ),
            )
        )

    def choose(
        self,
        alternatives: Value,
        element: TypeExpr,
        count: IndexLiteral | IndexVariable,
        *path: str | int,
    ) -> Perform:
        """Choose among the entries of a tensor.

        Parameters
        ----------
        alternatives : Value
            A ``Tensor[element]([count])``.
        element : TypeExpr
            The entry type.
        count : IndexLiteral | IndexVariable
            The tensor's extent.
        *path : str | int
            The request's path.

        Returns
        -------
        Perform
            The request, answered with one entry per resumption.
        """
        assert self.system.choice is not None
        return self.request(
            self.system.choice,
            SEARCH_CHOOSE,
            (element, count),
            (alternatives,),
            element,
            *path,
        )

    def fail(self, result: TypeExpr, *path: str | int) -> Computation:
        """An empty choice, which no derivation continues from.

        Parameters
        ----------
        result : TypeExpr
            The type the failed computation would have produced.
        *path : str | int
            The request's path.

        Returns
        -------
        Computation
            A choice among no alternatives: a call of the deduction's
            failing computation when the type is its item type, which
            the matches reach at every constructor a pattern refuses,
            else the request itself.
        """
        if result == self.system.item and self.system.name("fail") in self.signatures:
            return self.call(self.system.name("fail"), ())
        empty = TensorValue((), tensor_type(result, (IndexLiteral(0, NAT),)))
        return self.choose(empty, result, IndexLiteral(0, NAT), *path)

    def failure(self, signature: ComputationSignature) -> NamedComputation:
        """The failing computation: an empty choice at the item type.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation.
        """
        empty = TensorValue((), tensor_type(self.system.item, (IndexLiteral(0, NAT),)))
        body = self.choose(empty, self.system.item, IndexLiteral(0, NAT), "fail")
        return self.computation(signature, (), body)

    def add_weight(self, weight: Value, *path: str | int) -> Perform:
        """Add a weight on the deduction's ``Weight`` instance.

        Parameters
        ----------
        weight : Value
            A value of the semiring's carrier.
        *path : str | int
            The request's path.

        Returns
        -------
        Perform
            The request.
        """
        assert self.system.weight is not None
        return self.request(self.system.weight, "add", (), (weight,), UNIT, *path)

    def parameter(self, name: Value, *path: str | int) -> Perform:
        """Read a learned real parameter by name.

        Parameters
        ----------
        name : Value
            The parameter's name, a ``String``.
        *path : str | int
            The request's path.

        Returns
        -------
        Perform
            The request, answered with a ``Real``.
        """
        return self.request(self.params, "get", (REAL,), (name,), REAL, *path)

    def constructor(self, symbol: str, fields: tuple[Value, ...]) -> ConstructorValue:
        """Apply one of the item family's constructors.

        Parameters
        ----------
        symbol : str
            The symbol.
        fields : tuple[Value, ...]
            The field values.

        Returns
        -------
        ConstructorValue
            The item.
        """
        return ConstructorValue(
            self.system.constructors[symbol].id, (), fields, self.system.item
        )

    def case(
        self,
        scrutinee: Value,
        result: TypeExpr,
        branches: dict[str, Callable[[tuple[Local, ...]], Computation]],
        otherwise: Callable[[], Computation],
        *path: str | int,
    ) -> Case:
        """Analyze an item by constructor.

        Parameters
        ----------
        scrutinee : Value
            The item.
        result : TypeExpr
            The type every branch produces.
        branches : dict[str, Callable[[tuple[Local, ...]], Computation]]
            The bodies of the constructors matched, each given the
            locals bound to the constructor's fields.
        otherwise : Callable[[], Computation]
            The body of every other constructor.
        *path : str | int
            The case's path, which the branch scopes derive from.

        Returns
        -------
        Case
            The complete case.
        """
        built: list[CaseBranch] = []
        for position, (symbol, constructor) in enumerate(
            self.system.constructors.items()
        ):
            fields = tuple(
                self.system.local(f"f{index}", definition.type)
                for index, definition in enumerate(constructor.fields)
            )
            body = branches[symbol](fields) if symbol in branches else otherwise()
            built.append(
                CaseBranch(
                    constructor.id,
                    (),
                    fields,
                    body,
                    StaticScopeId.derive(
                        self.module,
                        (*self.path, *path),
                        "branch",
                        position,
                        constructor.id,
                    ),
                )
            )
        return Case(scrutinee, CaseMotive((), result), tuple(built))

    def bind(
        self,
        hint: str,
        type_: TypeExpr,
        first: Computation,
        then: Callable[[Local], Computation],
    ) -> Bind:
        """Bind a computation's result to a fresh local.

        Parameters
        ----------
        hint : str
            The local's name stem.
        type_ : TypeExpr
            The result's type.
        first : Computation
            The computation.
        then : Callable[[Local], Computation]
            The continuation, given the local.

        Returns
        -------
        Bind
            The bind.
        """
        local = self.system.local(hint, type_)
        return Bind(local, first, then(local))

    def weight_of(self, real: Value | None) -> Value:
        """A weight of the semiring's carrier from a real log weight.

        Parameters
        ----------
        real : Value | None
            The log weight as a ``Real``, or ``None`` for the semiring's
            multiplicative identity.

        Returns
        -------
        Value
            The carrier value: the log weight itself for the log
            semirings, ``true`` for the Boolean one, ``1`` for the
            counting one, which ignore learned weights.
        """
        semiring = self.system.semiring
        if semiring in ("LogProb", "Viterbi"):
            return self.primitive(
                "as_weight", real if real is not None else LiteralValue(0.0, REAL)
            )
        if semiring == "Boolean":
            return LiteralValue(True, BOOL)
        return _int(1)

    # -- equality, rendering, goal --------------------------------------

    def equality(self, signature: ComputationSignature) -> NamedComputation:
        """Structural equality of two items.

        Two items are equal exactly when their renderings are: the
        rendering writes each constructor's name and its fields in
        order, integers in decimal, so it is injective on items, and
        comparing the two strings costs one case analysis per item
        rather than one per pair of constructors.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation.
        """
        item = self.system.item
        left = Local("left", item)
        right = Local("right", item)
        show = self.system.name("show")
        body = self.bind(
            "left_text",
            STRING,
            self.call(show, (Var(left),)),
            lambda left_text: self.bind(
                "right_text",
                STRING,
                self.call(show, (Var(right),)),
                lambda right_text: Return(
                    self.primitive("eq_string", Var(left_text), Var(right_text))
                ),
            ),
        )
        return self.computation(signature, (left, right), body)

    def rendering(self, signature: ComputationSignature) -> NamedComputation:
        """Render an item as the text the classic engine keys weights by.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation.
        """
        item = Local("item", self.system.item)

        def render(symbol: str) -> Callable[[tuple[Local, ...]], Computation]:
            """The branch rendering one constructor.

            Parameters
            ----------
            symbol : str
                The constructor's symbol.

            Returns
            -------
            Callable[[tuple[Local, ...]], Computation]
                The branch body builder.
            """

            def body(fields: tuple[Local, ...]) -> Computation:
                """Render the constructor applied to its fields.

                Parameters
                ----------
                fields : tuple[Local, ...]
                    The fields.

                Returns
                -------
                Computation
                    The text.
                """
                if not fields:
                    return Return(_string(symbol))

                def piece(position: int, rendered: list[Value]) -> Computation:
                    """Render the fields from a position on, then join.

                    Parameters
                    ----------
                    position : int
                        The next field.
                    rendered : list[Value]
                        The fields rendered so far.

                    Returns
                    -------
                    Computation
                        The text.
                    """
                    if position == len(fields):
                        text: Value = _string(f"{symbol}(")
                        for index, part in enumerate(rendered):
                            if index:
                                text = self.primitive("concat", text, _string(","))
                            text = self.primitive("concat", text, part)
                        return Return(self.primitive("concat", text, _string(")")))
                    local = fields[position]
                    if local.type == INT:
                        return piece(
                            position + 1,
                            [*rendered, self.primitive("int_to_string", Var(local))],
                        )
                    return self.bind(
                        "text",
                        STRING,
                        self.call(signature.name, (Var(local),)),
                        lambda text: piece(position + 1, [*rendered, Var(text)]),
                    )

                return piece(0, [])

            return body

        body = self.case(
            Var(item),
            STRING,
            {symbol: render(symbol) for symbol in self.system.constructors},
            lambda: Return(_string("")),
            "show",
        )
        return self.computation(signature, (item,), body)

    def goal(self, signature: ComputationSignature) -> NamedComputation:
        """Whether an item is a goal: the start symbol as the engine reads it.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation: every item is a goal without a start
            symbol; with one, the atom itself, an item headed by it, or
            a span whose category is that atom.
        """
        item = Local("item", self.system.item)
        start = self.system.start

        def yes() -> Computation:
            """Answer that the item is a goal.

            Returns
            -------
            Computation
                ``true``.
            """
            return Return(LiteralValue(True, BOOL))

        def no() -> Computation:
            """Answer that the item is not a goal.

            Returns
            -------
            Computation
                ``false``.
            """
            return Return(LiteralValue(False, BOOL))

        if start is None:
            return self.computation(signature, (item,), yes())
        branches: dict[str, Callable[[tuple[Local, ...]], Computation]] = {}
        if start in self.system.constructors:
            branches[start] = lambda fields: yes()
        if (
            SPAN in self.system.constructors
            and start in self.system.constructors
            and not self.system.constructors[start].fields
        ):
            branches[SPAN] = lambda fields: self.case(
                Var(fields[2]),
                BOOL,
                {start: lambda inner: yes()},
                no,
                "goal",
                "category",
            )
        body = self.case(Var(item), BOOL, branches, no, "goal")
        return self.computation(signature, (item,), body)

    # -- axioms ---------------------------------------------------------

    def axiom(self, signature: ComputationSignature) -> NamedComputation:
        """The lexical items a token yields at a position.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation, choosing among the entries whose word is
            the token and adding the entry's weight.
        """
        position = Local("position", INT)
        tokens = Local("tokens", signature.parameters[1])
        entries = self.system.entries
        token = self.system.local("token", STRING)
        alternatives = TensorValue(
            tuple(_int(index) for index in range(len(entries))),
            tensor_type(INT, (IndexLiteral(len(entries), NAT),)),
        )

        def entry_body(index: int, entry: _Entry) -> Computation:
            """The item one entry yields, once its word matches.

            Parameters
            ----------
            index : int
                The entry's index.
            entry : _Entry
                The entry.

            Returns
            -------
            Computation
                Add the entry's weight and return its span.
            """
            fields: list[Value] = [
                Var(position),
                self.primitive("add_int", Var(position), _int(1)),
                self.ground(entry.category),
            ]
            if self.system.uses_forms:
                assert entry.form is not None
                fields.append(self.ground(entry.form))
            span = self.constructor(SPAN, tuple(fields))
            result = Return(span)

            def weighted(weight: Value) -> Computation:
                """Add a weight, then answer the span.

                Parameters
                ----------
                weight : Value
                    The weight.

                Returns
                -------
                Computation
                    The sequence.
                """
                return Bind(
                    self.system.local("added", UNIT),
                    self.add_weight(weight, "lexicon", index, "weight"),
                    result,
                )

            if entry.learnable and self.system.semiring in ("LogProb", "Viterbi"):
                return self.bind(
                    "weight",
                    REAL,
                    self.parameter(
                        _string(f"{self.system.decl.name}.lex.{index}"),
                        "lexicon",
                        index,
                        "parameter",
                    ),
                    lambda real: weighted(self.weight_of(Var(real))),
                )
            return weighted(self.weight_of(None))

        def dispatch(chosen: Local, index: int) -> Computation:
            """Select the chosen entry by index.

            Parameters
            ----------
            chosen : Local
                The chosen index.
            index : int
                The entry tested here.

            Returns
            -------
            Computation
                The chain of tests from this entry on.
            """
            if index == len(entries):
                return self.fail(self.system.item, "lexicon", "none")
            entry = entries[index]
            return If(
                self.primitive("eq_int", Var(chosen), _int(index)),
                If(
                    self.primitive("eq_string", Var(token), _string(entry.word)),
                    entry_body(index, entry),
                    self.fail(self.system.item, "lexicon", index, "mismatch"),
                ),
                dispatch(chosen, index + 1),
            )

        body = Bind(
            token,
            Return(Gather(Var(tokens), Var(position), STRING)),
            self.bind(
                "entry",
                INT,
                self.choose(
                    alternatives,
                    INT,
                    IndexLiteral(len(entries), NAT),
                    "lexicon",
                    "choose",
                ),
                lambda chosen: dispatch(chosen, 0),
            ),
        )
        return self.computation(signature, (position, tokens), body)

    def ground(self, pattern: Pattern) -> Value:
        """The item a ground pattern denotes.

        Parameters
        ----------
        pattern : Pattern
            A pattern without variables.

        Returns
        -------
        Value
            The item, or the integer for a literal.
        """
        return self.instantiate(pattern, {})

    def instantiate(self, pattern: Pattern, bindings: dict[str, Local]) -> Value:
        """The value a pattern denotes under bindings.

        Parameters
        ----------
        pattern : Pattern
            The pattern.
        bindings : dict[str, Local]
            The variables' values.

        Returns
        -------
        Value
            The item, or the integer for a literal or an integer
            variable.
        """
        if isinstance(pattern, _Atom):
            return self.constructor(pattern.name, ())
        if isinstance(pattern, _Wild):
            return Var(bindings[pattern.name])
        if isinstance(pattern, _Lit):
            return _int(pattern.value)
        if isinstance(pattern, _Bound):
            return self.constructor(BOUND, (_int(pattern.index),))
        return self.constructor(
            pattern.symbol,
            tuple(self.instantiate(argument, bindings) for argument in pattern.args),
        )

    # -- matching -------------------------------------------------------

    def match(
        self,
        value: Local,
        pattern: Pattern,
        bindings: dict[str, Local],
        result: TypeExpr,
        continue_with: _Continuation,
        *path: str | int,
    ) -> Computation:
        """Match a value against a pattern, continuing with the bindings.

        Parameters
        ----------
        value : Local
            The value, an item or an integer as the pattern's slot says.
        pattern : Pattern
            The pattern.
        bindings : dict[str, Local]
            The variables bound so far.
        result : TypeExpr
            The type the continuation produces, which a failed match
            produces by an empty choice.
        continue_with : _Continuation
            What to run once the pattern matches, given the bindings.
        *path : str | int
            The match's path.

        Returns
        -------
        Computation
            The match.
        """
        if isinstance(pattern, _Wild):
            bound = bindings.get(pattern.name)
            if bound is None:
                return continue_with({**bindings, pattern.name: value})
            if value.type == INT:
                condition = self.primitive("eq_int", Var(bound), Var(value))
                return If(
                    condition,
                    continue_with(bindings),
                    self.fail(result, *path, "shared"),
                )
            return self.bind(
                "same",
                BOOL,
                self.call(self.system.name("eq"), (Var(bound), Var(value))),
                lambda same: If(
                    Var(same),
                    continue_with(bindings),
                    self.fail(result, *path, "shared"),
                ),
            )
        if isinstance(pattern, _Lit):
            return If(
                self.primitive("eq_int", Var(value), _int(pattern.value)),
                continue_with(bindings),
                self.fail(result, *path, "literal"),
            )
        if isinstance(pattern, _Bound):
            return self.case(
                Var(value),
                result,
                {
                    BOUND: lambda fields: If(
                        self.primitive("eq_int", Var(fields[0]), _int(pattern.index)),
                        continue_with(bindings),
                        self.fail(result, *path, "bound"),
                    )
                },
                lambda: self.fail(result, *path, "bound"),
                *path,
                "bound",
            )
        symbol = pattern.name if isinstance(pattern, _Atom) else pattern.symbol
        arguments = () if isinstance(pattern, _Atom) else pattern.args

        def fields_matched(fields: tuple[Local, ...]) -> Computation:
            """Match the constructor's fields in order.

            Parameters
            ----------
            fields : tuple[Local, ...]
                The fields.

            Returns
            -------
            Computation
                The matches, innermost the continuation.
            """

            def step(position: int, current: dict[str, Local]) -> Computation:
                """Match the fields from a position on.

                Parameters
                ----------
                position : int
                    The next field.
                current : dict[str, Local]
                    The bindings so far.

                Returns
                -------
                Computation
                    The match from that field on.
                """
                if position == len(fields):
                    return continue_with(current)
                return self.match(
                    fields[position],
                    arguments[position],
                    current,
                    result,
                    lambda extended: step(position + 1, extended),
                    *path,
                    position,
                )

            return step(0, bindings)

        return self.case(
            Var(value),
            result,
            {symbol: fields_matched},
            lambda: self.fail(result, *path, "constructor"),
            *path,
            symbol,
        )

    # -- derivation -----------------------------------------------------

    def derivation(self, signature: ComputationSignature) -> NamedComputation:
        """Derive an item by one rule application or one axiom.

        Rules are grouped by the shape of their premises: the spans the
        premises cover relative to the conclusion's, or, without spans,
        their number. A derivation chooses an axiom or a group, derives
        the group's premises once, then chooses a rule of the group and
        matches the premises against its patterns, so the sub-derivations
        are shared by every rule of the same shape.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation, which answers the conclusion of the chosen
            rule having added its weight.
        """
        system = self.system
        item = system.item
        fuel = Local("fuel", INT)
        start = Local("start", INT)
        stop = Local("stop", INT)
        spent = self.primitive("sub_int", Var(fuel), _int(1))
        if system.span_shaped:
            tokens = Local("tokens", signature.parameters[3])
            parameters: tuple[Local, ...] = (fuel, start, stop, tokens)

            def recursion(low: Value, high: Value) -> Call:
                """Derive an item over a sub-span with one less fuel.

                Parameters
                ----------
                low : Value
                    The sub-span's start.
                high : Value
                    Its end.

                Returns
                -------
                Call
                    The recursive call.
                """
                return self.call(signature.name, (spent, low, high, Var(tokens)))

            axiom_alternative: Computation = If(
                self.primitive(
                    "eq_int", Var(stop), self.primitive("add_int", Var(start), _int(1))
                ),
                self.call(system.name("axiom"), (Var(start), Var(tokens))),
                self.fail(item, "derive", "lexical", "span"),
            )
        elif system.entries:
            # A lexicon whose rules do not relate spans: every lexical
            # item at every position is an axiom, and the rules derive
            # their premises without positions.
            tokens = Local("tokens", signature.parameters[1])
            parameters = (fuel, tokens)

            def recursion(low: Value, high: Value) -> Call:
                """Derive any item with one less fuel.

                Parameters
                ----------
                low : Value
                    Unused; the system derives without spans.
                high : Value
                    Unused; the system derives without spans.

                Returns
                -------
                Call
                    The recursive call.
                """
                del low, high
                return self.call(signature.name, (spent, Var(tokens)))

            positions = self.system.local("k", INT)
            axiom_alternative = self.bind(
                "chosen",
                INT,
                self.choose(
                    Comprehension(
                        positions,
                        self.extent,
                        Var(positions),
                        tensor_type(INT, (self.extent,)),
                    ),
                    INT,
                    self.extent,
                    "derive",
                    "axiom",
                    "choose",
                ),
                lambda chosen: self.call(
                    system.name("axiom"), (Var(chosen), Var(tokens))
                ),
            )
        else:
            axioms = Local("axioms", signature.parameters[1])
            weights = Local("weights", signature.parameters[2])
            parameters = (fuel, axioms, weights)

            def recursion(low: Value, high: Value) -> Call:
                """Derive any item with one less fuel.

                Parameters
                ----------
                low : Value
                    Unused; the system has no spans.
                high : Value
                    Unused; the system has no spans.

                Returns
                -------
                Call
                    The recursive call.
                """
                del low, high
                return self.call(signature.name, (spent, Var(axioms), Var(weights)))

            positions = self.system.local("k", INT)
            axiom_alternative = self.bind(
                "chosen",
                INT,
                self.choose(
                    Comprehension(
                        positions,
                        self.extent,
                        Var(positions),
                        tensor_type(INT, (self.extent,)),
                    ),
                    INT,
                    self.extent,
                    "derive",
                    "axiom",
                    "choose",
                ),
                lambda chosen: Bind(
                    system.local("added", UNIT),
                    self.add_weight(
                        Gather(Var(weights), Var(chosen), system.carrier),
                        "derive",
                        "axiom",
                        "weight",
                    ),
                    Return(Gather(Var(axioms), Var(chosen), item)),
                ),
            )
        groups = self.rule_groups()
        alternatives = TensorValue(
            tuple(_int(index) for index in range(-1, len(groups))),
            tensor_type(INT, (IndexLiteral(len(groups) + 1, NAT),)),
        )

        def group_body(group_index: int, group: _Group) -> Computation:
            """Apply one rule of a group.

            Parameters
            ----------
            group_index : int
                The group's index.
            group : _Group
                The group.

            Returns
            -------
            Computation
                The application.
            """
            path: tuple[str | int, ...] = ("derive", "group", group_index)

            def choose_positions(
                remaining: list[str], chosen: dict[str, Local]
            ) -> Computation:
                """Choose a value for each split the group's premises need.

                Parameters
                ----------
                remaining : list[str]
                    The normalized position names still to choose.
                chosen : dict[str, Local]
                    The positions chosen so far, by normalized name.

                Returns
                -------
                Computation
                    The choices, innermost the premises.
                """
                if not remaining:
                    return premises(0, chosen, [])
                name = remaining[0]
                index_local = system.local("k", INT)
                return self.bind(
                    "split",
                    INT,
                    self.choose(
                        Comprehension(
                            index_local,
                            self.extent,
                            Var(index_local),
                            tensor_type(INT, (self.extent,)),
                        ),
                        INT,
                        self.extent,
                        *path,
                        "position",
                        name,
                    ),
                    lambda split: If(
                        self.primitive(
                            "and",
                            self.primitive("lt_int", Var(start), Var(split)),
                            self.primitive("lt_int", Var(split), Var(stop)),
                        ),
                        choose_positions(remaining[1:], {**chosen, name: split}),
                        self.fail(item, *path, "position", name),
                    ),
                )

            def premises(
                position: int, chosen: dict[str, Local], derived: list[Local]
            ) -> Computation:
                """Derive the group's premises from a position on.

                Parameters
                ----------
                position : int
                    The next premise.
                chosen : dict[str, Local]
                    The positions, by normalized name.
                derived : list[Local]
                    The premises derived so far.

                Returns
                -------
                Computation
                    The derivations, innermost the rule choice.
                """
                if position == group.arity:
                    return choose_rule(chosen, derived)
                if system.span_shaped:
                    low, high = group.spans[position]
                    call = recursion(Var(chosen[low]), Var(chosen[high]))
                else:
                    call = recursion(_int(0), _int(0))
                return self.bind(
                    "premise",
                    item,
                    call,
                    lambda value: premises(position + 1, chosen, [*derived, value]),
                )

            def choose_rule(
                chosen: dict[str, Local], derived: list[Local]
            ) -> Computation:
                """Choose a rule of the group and match the premises against it.

                Parameters
                ----------
                chosen : dict[str, Local]
                    The positions, by normalized name.
                derived : list[Local]
                    The derived premises.

                Returns
                -------
                Computation
                    The choice, each rule's match innermost.
                """
                if len(group.rules) == 1:
                    return match_rule(group.rules[0], chosen, derived)
                rule_alternatives = TensorValue(
                    tuple(_int(index) for index in range(len(group.rules))),
                    tensor_type(INT, (IndexLiteral(len(group.rules), NAT),)),
                )

                def dispatch(selected: Local, position: int) -> Computation:
                    """Select the chosen rule by index.

                    Parameters
                    ----------
                    selected : Local
                        The chosen index.
                    position : int
                        The rule tested here.

                    Returns
                    -------
                    Computation
                        The chain of tests from this rule on.
                    """
                    if position == len(group.rules):
                        return self.fail(item, *path, "rule", "none")
                    return If(
                        self.primitive("eq_int", Var(selected), _int(position)),
                        match_rule(group.rules[position], chosen, derived),
                        dispatch(selected, position + 1),
                    )

                return self.bind(
                    "rule",
                    INT,
                    self.choose(
                        rule_alternatives,
                        INT,
                        IndexLiteral(len(group.rules), NAT),
                        *path,
                        "rule",
                    ),
                    lambda selected: dispatch(selected, 0),
                )

            def match_rule(
                entry: tuple[int, _Rule], chosen: dict[str, Local], derived: list[Local]
            ) -> Computation:
                """Match the derived premises against one rule's patterns.

                Parameters
                ----------
                entry : tuple[int, _Rule]
                    The rule and its index.
                chosen : dict[str, Local]
                    The positions, by normalized name.
                derived : list[Local]
                    The derived premises.

                Returns
                -------
                Computation
                    The matches, innermost the conclusion.
                """
                index, rule = entry
                bindings: dict[str, Local] = {}
                if system.span_shaped:
                    for name, normalized in group.names[index].items():
                        bindings[name] = (
                            start
                            if normalized == "L"
                            else stop
                            if normalized == "R"
                            else chosen[normalized]
                        )

                def step(position: int, current: dict[str, Local]) -> Computation:
                    """Match the premises from a position on.

                    Parameters
                    ----------
                    position : int
                        The next premise.
                    current : dict[str, Local]
                        The bindings so far.

                    Returns
                    -------
                    Computation
                        The match from that premise on.
                    """
                    if position == len(rule.premises):
                        return conclude(index, rule, current)
                    return self.match(
                        derived[position],
                        rule.premises[position],
                        current,
                        item,
                        lambda extended: step(position + 1, extended),
                        *path,
                        "rule",
                        index,
                        "premise",
                        position,
                    )

                return step(0, bindings)

            return choose_positions(list(group.splits), {"L": start, "R": stop})

        def conclude(index: int, rule: _Rule, current: dict[str, Local]) -> Computation:
            """Instantiate a rule's conclusion, add its weight, answer.

            Parameters
            ----------
            index : int
                The rule's index.
            rule : _Rule
                The rule.
            current : dict[str, Local]
                The complete bindings.

            Returns
            -------
            Computation
                The conclusion.
            """
            answer = Return(self.instantiate(rule.conclusion, current))

            def weighted(weight: Value) -> Computation:
                """Add the weight, then answer.

                Parameters
                ----------
                weight : Value
                    The weight.

                Returns
                -------
                Computation
                    The sequence.
                """
                return Bind(
                    system.local("added", UNIT),
                    self.add_weight(weight, "derive", "rule", index, "weight"),
                    answer,
                )

            if system.semiring not in ("LogProb", "Viterbi") or not (
                rule.learnable or rule.parent is not None
            ):
                return weighted(self.weight_of(None))
            return self.rule_weight(rule, index, current, weighted)

        def dispatch(chosen: Local, index: int) -> Computation:
            """Select the chosen alternative by index.

            Parameters
            ----------
            chosen : Local
                The chosen index, ``-1`` for an axiom.
            index : int
                The group tested here.

            Returns
            -------
            Computation
                The chain of tests from this group on.
            """
            if index == len(groups):
                return self.fail(item, "derive", "none")
            return If(
                self.primitive("eq_int", Var(chosen), _int(index)),
                group_body(index, groups[index]),
                dispatch(chosen, index + 1),
            )

        body = If(
            self.primitive("le_int", Var(fuel), _int(0)),
            self.fail(item, "derive", "fuel"),
            self.bind(
                "alternative",
                INT,
                self.choose(
                    alternatives,
                    INT,
                    IndexLiteral(len(groups) + 1, NAT),
                    "derive",
                    "choose",
                ),
                lambda chosen: If(
                    self.primitive("eq_int", Var(chosen), _int(-1)),
                    axiom_alternative,
                    dispatch(chosen, 0),
                ),
            ),
        )
        return self.computation(signature, parameters, body)

    def rule_groups(self) -> list[_Group]:
        """Group the rules by the shape of their premises.

        Returns
        -------
        list[_Group]
            For a span-shaped system, one group per pattern of premise
            spans relative to the conclusion's, the conclusion's start
            and end normalized to ``L`` and ``R`` and each further
            position to ``k0``, ``k1``, and so on in order of first
            occurrence; otherwise one group per premise count.
        """
        groups: dict[tuple[object, ...], _Group] = {}
        for index, rule in enumerate(self.system.rules):
            if not self.system.span_shaped:
                key: tuple[object, ...] = (len(rule.premises),)
                spans: tuple[tuple[str, str], ...] = ()
                names: dict[str, str] = {}
            else:
                conclusion = rule.conclusion
                assert isinstance(conclusion, _App)
                first, second = conclusion.args[0], conclusion.args[1]
                assert isinstance(first, _Wild) and isinstance(second, _Wild)
                names = {first.name: "L", second.name: "R"}
                for name in self.position_variables(rule):
                    if name not in names:
                        names[name] = f"k{len(names) - 2}"
                spans_list: list[tuple[str, str]] = []
                for premise in rule.premises:
                    assert isinstance(premise, _App)
                    low, high = premise.args[0], premise.args[1]
                    assert isinstance(low, _Wild) and isinstance(high, _Wild)
                    spans_list.append((names[low.name], names[high.name]))
                spans = tuple(spans_list)
                key = spans
            group = groups.get(key)
            if group is None:
                splits = tuple(
                    name
                    for name in dict.fromkeys(
                        normalized for span in spans for normalized in span
                    )
                    if name not in ("L", "R")
                )
                group = _Group(len(rule.premises), spans, splits, [], {})
                groups[key] = group
            group.rules.append((index, rule))
            group.names[index] = names
        return list(groups.values())

    def position_variables(self, rule: _Rule) -> list[str]:
        """The variables a rule uses in integer slots, in first-occurrence order.

        Parameters
        ----------
        rule : _Rule
            The rule.

        Returns
        -------
        list[str]
            The names.
        """
        found: list[str] = []

        def walk(pattern: Pattern) -> None:
            """Collect the integer-slot variables of one pattern.

            Parameters
            ----------
            pattern : Pattern
                The pattern.
            """
            if not isinstance(pattern, _App):
                return
            sorts = self.system.symbols[pattern.symbol]
            for position, argument in enumerate(pattern.args):
                if isinstance(argument, _Wild):
                    if sorts[position] == "int" and argument.name not in found:
                        found.append(argument.name)
                else:
                    walk(argument)

        for pattern in (*rule.premises, rule.conclusion):
            walk(pattern)
        return found

    def rule_weight(
        self,
        rule: _Rule,
        index: int,
        bindings: dict[str, Local],
        continue_with: Callable[[Value], Computation],
    ) -> Computation:
        """Read a rule's learned weight on its bindings, then continue.

        The weight is keyed by the rule's name and its bindings rendered
        in name order, as the classic engine keys it; a bounded rule's
        raw parameter is mapped below zero, and a parented rule adds its
        parent's weight on the same bindings.

        Parameters
        ----------
        rule : _Rule
            The rule.
        index : int
            The rule's index.
        bindings : dict[str, Local]
            The complete bindings.
        continue_with : Callable[[Value], Computation]
            What to run with the weight, a carrier value.

        Returns
        -------
        Computation
            The reads, innermost the continuation.
        """
        names = sorted(bindings)

        def render(
            position: int, rendered: list[Value], then: Callable[[Value], Computation]
        ) -> Computation:
            """Render the bindings from a position on into one key suffix.

            Parameters
            ----------
            position : int
                The next binding.
            rendered : list[Value]
                The bindings rendered so far.
            then : Callable[[Value], Computation]
                What to run with the suffix.

            Returns
            -------
            Computation
                The renders, innermost the continuation.
            """
            if position == len(names):
                suffix: Value = _string("")
                for part_index, part in enumerate(rendered):
                    if part_index:
                        suffix = self.primitive("concat", suffix, _string("|"))
                    suffix = self.primitive("concat", suffix, part)
                return then(suffix)
            local = bindings[names[position]]
            if local.type == INT:
                return render(
                    position + 1,
                    [*rendered, self.primitive("int_to_string", Var(local))],
                    then,
                )
            return self.bind(
                "text",
                STRING,
                self.call(self.system.name("show"), (Var(local),)),
                lambda text: render(position + 1, [*rendered, Var(text)], then),
            )

        bounded = [
            item
            for item in self.system.rules
            if item.bounded and (item.learnable or item.parent is not None)
        ]
        cap = math.log(len(bounded)) if len(bounded) > 1 else 0.0

        def own(
            chain: _Rule, suffix: Value, then: Callable[[Value], Computation]
        ) -> Computation:
            """Read one rule's own weight on the bindings, then continue.

            Parameters
            ----------
            chain : _Rule
                The rule.
            suffix : Value
                The rendered bindings.
            then : Callable[[Value], Computation]
                What to run with the weight, a ``Real``.

            Returns
            -------
            Computation
                The read, or the identity when the rule learns nothing.
            """
            if not chain.learnable:
                return then(LiteralValue(0.0, REAL))
            key = self.primitive(
                "concat",
                _string(f"{self.system.decl.name}.rule.{chain.source.name}:"),
                suffix,
            )

            def bounded_or_raw(raw: Local) -> Value:
                """The rule's weight from its raw parameter.

                Parameters
                ----------
                raw : Local
                    The parameter.

                Returns
                -------
                Value
                    ``-softplus(raw) - log(n)`` for a bounded rule, the
                    parameter itself otherwise.
                """
                if not chain.bounded:
                    return Var(raw)
                return self.primitive(
                    "sub_real",
                    self.primitive("neg_real", self.primitive("softplus", Var(raw))),
                    LiteralValue(cap, REAL),
                )

            return self.bind(
                "raw",
                REAL,
                self.parameter(
                    key, "derive", "rule", index, "parameter", chain.source.name
                ),
                lambda raw: then(bounded_or_raw(raw)),
            )

        by_name = {item.source.name: item for item in self.system.rules}

        def chain(
            current: _Rule, suffix: Value, total: Value, seen: set[str]
        ) -> Computation:
            """Add the parent chain's weights to a running total.

            Parameters
            ----------
            current : _Rule
                The rule whose parent is read next.
            suffix : Value
                The rendered bindings.
            total : Value
                The weight so far.
            seen : set[str]
                The rules read, to refuse a cycle.

            Returns
            -------
            Computation
                The reads, innermost the continuation with the total.
            """
            if current.parent is None:
                return continue_with(self.weight_of(total))
            if current.parent in seen:
                self.elaborator._fail(
                    rule.source,
                    f"rule {rule.source.name!r}: parent chain cycles at "
                    f"{current.parent!r}",
                    code="qiec-program",
                )
            parent = by_name[current.parent]
            return own(
                parent,
                suffix,
                lambda weight: chain(
                    parent,
                    suffix,
                    self.primitive("add_real", total, weight),
                    seen | {parent.source.name},
                ),
            )

        return render(
            0,
            [],
            lambda suffix: own(
                rule,
                suffix,
                lambda weight: chain(rule, suffix, weight, {rule.source.name}),
            ),
        )

    # -- entry ------------------------------------------------------------

    def entry(
        self, signature: ComputationSignature, handlers: tuple[HandlerDef, HandlerDef]
    ) -> NamedComputation:
        """The deduction's entry: the inside weight of the goal.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.
        handlers : tuple[HandlerDef, HandlerDef]
            The search and collection handlers.

        Returns
        -------
        NamedComputation
            The computation, which handles the weight instance with the
            collector and the search instance with the searcher around a
            derivation of the goal.
        """
        system = self.system
        assert system.choice is not None and system.weight is not None
        search_handler, collect_handler = handlers
        item = system.item
        parameters = tuple(
            Local(name, type_)
            for name, type_ in zip(
                ("tokens",) if system.entries else ("axioms", "weights"),
                signature.parameters,
                strict=True,
            )
        )
        if system.span_shaped:
            # The sentence's length is the sum of a tensor of ones over
            # the extent; without a declared depth it bounds the depth of
            # a binary system's derivations of the whole sentence.
            length: Value = Reduction(
                "sum",
                Comprehension(
                    system.local("k", INT),
                    self.extent,
                    _int(1),
                    tensor_type(INT, (self.extent,)),
                ),
                INT,
            )
            fuel_value: Value = (
                _int(system.depth) if system.depth is not None else length
            )
            derived = self.call(
                system.name("derive"),
                (fuel_value, _int(0), length, Var(parameters[0])),
            )
        else:
            # Without a declared depth, a system deriving without spans
            # enumerates derivations at most as deep as it has tokens or
            # axioms and rules together, which every derivation without a
            # repeated rule application along a path fits within.
            if system.depth is not None:
                fuel_value = _int(system.depth)
            else:
                count: Value = Reduction(
                    "sum",
                    Comprehension(
                        system.local("k", INT),
                        self.extent,
                        _int(1),
                        tensor_type(INT, (self.extent,)),
                    ),
                    INT,
                )
                fuel_value = self.primitive("add_int", count, _int(len(system.rules)))
            derived = self.call(
                system.name("derive"),
                (fuel_value, *(Var(parameter) for parameter in parameters)),
            )
        goal_check = self.bind(
            "item",
            item,
            derived,
            lambda found: self.bind(
                "is_goal",
                BOOL,
                self.call(system.name("goal"), (Var(found),)),
                lambda is_goal: If(
                    Var(is_goal),
                    Return(LiteralValue(None, UNIT)),
                    self.fail(UNIT, "entry", "goal"),
                ),
            ),
        )
        body = Handle(
            system.choice.entry.instance,
            search_handler.id,
            Handle(system.weight.entry.instance, collect_handler.id, goal_check, ()),
            (),
        )
        return self.computation(signature, parameters, body)

    def computation(
        self,
        signature: ComputationSignature,
        parameters: tuple[Local, ...],
        body: Computation,
    ) -> NamedComputation:
        """Check a body against its signature and wrap it as a computation.

        Parameters
        ----------
        signature : ComputationSignature
            The registered signature.
        parameters : tuple[Local, ...]
            The parameter locals, in order.
        body : Computation
            The body.

        Returns
        -------
        NamedComputation
            The computation.

        Raises
        ------
        QiecDiagnosticError
            If the body fails to check or its type differs from the
            signature's.
        """
        return self.elaborator._checked_computation(
            signature, parameters, body, self.system.decl, self.path
        )


__all__ = [
    "PARAMS_INSTANCE",
    "SEARCH_CHOOSE",
    "SEARCH_EFFECT",
    "SEMIRING_CARRIERS",
]
