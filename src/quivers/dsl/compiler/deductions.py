"""Compiler mixin: deduction systems and lexicon loading."""

from __future__ import annotations
from collections.abc import Callable
import torch
import torch.nn as nn
from quivers.core.algebras import BOOLEAN
from quivers.dsl.ast_nodes import (
    DeductionDecl,
    TypeEffectApply,
    TypeName,
    TypeProduct,
    TypeSlash,
)
from quivers.stochastic.agenda import (
    DeductionSystem,
    InferenceRule,
    Wildcard,
    cky_agenda,
    depth_first_agenda,
    semi_naive_agenda,
)
from quivers.stochastic.semiring import (
    BOOLEAN as SEMIRING_BOOLEAN,
    COUNTING as SEMIRING_COUNTING,
    LOG_PROB as SEMIRING_LOG_PROB,
    VITERBI as SEMIRING_VITERBI,
)
from quivers.dsl.compiler._options import (
    get_option_flag,
    get_option_int,
    get_option_name,
)
from quivers.dsl.compiler._prelude import CompileError
from quivers.dsl.compiler.programs import _ProgramsMixin
from quivers.dsl.parser import parse as _parse_qvr


# Category-side pattern carried by a lexicon entry: either a
# wildcard variable (any identifier not declared as an atom) or a
# nested structural pattern over atoms (the result of
# ``_convert_pattern`` walking a TypeExpr).
type LexiconPattern = Wildcard | tuple[str | int | "LexiconPattern", ...]

# Logical-form payload attached to a lexicon entry: the result of
# evaluating the entry's ``lf`` template under the empty
# environment.  The let-expression evaluator can return any value
# the program theory admits at the LF position; the concrete
# union below enumerates the possibilities.
type LexiconLF = (
    torch.Tensor
    | int
    | float
    | bool
    | str
    | tuple["LexiconLF", ...]
    | Callable[[dict[str, "LexiconLF"]], "LexiconLF"]
)

# A lexicon entry quadruple ``(word, category_pattern, lf,
# learnable_flag)``.
type LexiconEntry = tuple[str, LexiconPattern, LexiconLF, bool]


class _DeductionsMixin:
    """Mixin: deduction-block compilation methods."""

    def _compile_deduction(self, decl: DeductionDecl) -> None:
        """Compile a ``deduction { … }`` block into an agenda-engine
        :class:`DeductionSystem` and register it under ``decl.name``.

        Translates the declarative sequent-style rules into the
        runtime's :class:`InferenceRule` form (with single-uppercase
        identifiers treated as wildcard variables). Resolves the
        semiring by name. Wires the axiom source — one of:

        * a ``lexicon { ... }`` block, compiled into a learnable
          dispatch table keyed on the input token at each position;
        * a ``lexicon from "path"`` declaration, loaded from a TSV
          file at compile time and treated identically to the
          inline form;
        * an ``axioms = source_name`` declaration, naming a
          previously-defined morphism whose callable returns a
          list of `(item, weight)` pairs given an input;
        * none of the above — the user supplies axioms directly at
          call time (identity axiom-injector).

        The result is callable as ``parse(NAME, input)`` from
        program bodies, producing a :class:`ChartView`.
        """
        if not hasattr(self, "_deductions"):
            self._deductions: dict[str, "DeductionSystem"] = {}

        globals_ = dict(getattr(self, "_deductions", {}))
        # The deduction's declared atomic + complex constructor
        # symbols form the user-controlled free term algebra used
        # by lexicon LF expressions, rule weights, and any other
        # let-expressions evaluated inside this deduction's scope.
        # No constructor symbol is privileged by the compiler — the
        # user states the entire algebra explicitly.
        globals_["__constructors__"] = frozenset(decl.atoms)

        if decl.name in self._deductions or decl.name in self._morphisms:
            raise CompileError(
                f"deduction {decl.name!r} already declared",
                decl.line,
                decl.col,
            )

        # Pattern-conversion: TypeExpr -> agenda-engine Pattern.
        # The conversion is fully general; users may use any
        # type-expression shape and the runtime pattern-matcher
        # walks it structurally. Identifiers that match a declared
        # atom name become ground atoms; identifiers NOT in the
        # atoms set are treated as wildcard variables (the standard
        # Prolog / Datalog convention: declared constants are
        # ground; undeclared identifiers in patterns are variables).
        atoms_set = set(decl.atoms)

        def _convert_pattern(texpr):
            if isinstance(texpr, TypeName):
                name = texpr.name
                if name in atoms_set:
                    return ("atom", name)
                # Variable convention: any identifier not in the
                # atoms list (and not a numeric literal) is a
                # wildcard. This permits arbitrary metavariable
                # names — X, Y, Foo, antecedent — without ad-hoc
                # capitalisation rules.
                if name.isdigit():
                    return ("literal", int(name))
                return Wildcard(name)
            if isinstance(texpr, TypeProduct):
                return (
                    "product",
                    tuple(_convert_pattern(c) for c in texpr.components),
                )
            if isinstance(texpr, TypeSlash):
                # Categorial-grammar slash types: X/Y, X\Y.
                return (
                    texpr.direction,
                    _convert_pattern(texpr.result),
                    _convert_pattern(texpr.argument),
                )
            if isinstance(texpr, TypeEffectApply):
                # T(X) = ("effect_apply", T_name, *X_args). The
                # constructor's args are recursively converted; this
                # encodes any structured term — proof witnesses, LF
                # constructors, dependent-type applications.
                args = tuple(_convert_pattern(a) for a in texpr.args)
                return (texpr.effect, *args)
            # Fallback: a structural-equality probe.
            return ("atom", repr(texpr))

        semiring_registry = {
            "LogProb": SEMIRING_LOG_PROB,
            "Boolean": SEMIRING_BOOLEAN,
            "Viterbi": SEMIRING_VITERBI,
            "Counting": SEMIRING_COUNTING,
        }
        semiring_name = get_option_name(
            decl.options, "semiring", line=decl.line, col=decl.col,
        )
        semiring = (
            semiring_registry.get(semiring_name, SEMIRING_LOG_PROB)
            if semiring_name is not None
            else SEMIRING_LOG_PROB
        )

        inference_rules: list = []
        # The application rule and other generic combinators
        # carry no learnable weight by default; users may wrap
        # the deduction with a `weight_fn` that consults
        # rule-weight parameters.
        for sr in decl.rules:
            premises = tuple(_convert_pattern(p) for p in sr.premises)
            conclusion = _convert_pattern(sr.conclusion)
            inference_rules.append(
                InferenceRule(
                    name=sr.name,
                    premises=premises,
                    conclusion=conclusion,
                )
            )

        # ---- Axiom source ----
        #
        # Resolve in priority order:
        #   1. `axioms = some_morphism` (most general).
        #   2. `lexicon { ... }` or `lexicon from "..."` (sugar
        #      for the label-indexed-lookup case).
        #   3. Identity — input itself is the axiom list.

        axioms_source = get_option_name(
            decl.options, "axioms", line=decl.line, col=decl.col,
        )
        if axioms_source is not None:
            # General axiom source — look up the named morphism and
            # invoke it on the input at call time. The morphism may
            # be any callable.
            src_name = axioms_source
            if src_name not in self._morphisms:
                raise CompileError(
                    f"deduction {decl.name!r}: axioms source "
                    f"{src_name!r} is not a declared morphism",
                    decl.line,
                    decl.col,
                )
            morph = self._morphisms[src_name]

            def _axiom_injector(input_value, _morph=morph):
                # The morphism is expected to be a callable that,
                # given the input, returns a list of (item, weight)
                # pairs.
                return list(_morph(input_value))

            axiom_module = morph if isinstance(morph, nn.Module) else None
        elif decl.lexicon or decl.lexicon_from_file is not None:
            # Lexicon-based axiom source. Build a learnable lookup
            # table keyed on the literal word string; emit one
            # axiom per matching entry per input position.
            entries: list[LexiconEntry] = []
            for entry in decl.lexicon:
                lf_fn = _ProgramsMixin._compile_let_expr(entry.lf, globals_=globals_)
                # Evaluate the LF eagerly under an empty environment;
                # LF templates in lexicons must be closed expressions.
                try:
                    lf_value = lf_fn({})
                except CompileError as e:
                    raise CompileError(
                        f"deduction {decl.name!r}: lexicon entry for "
                        f"{entry.word!r} has unresolved variable: {e}",
                        entry.line,
                        entry.col,
                    ) from e
                entries.append(
                    (
                        entry.word,
                        _convert_pattern(entry.category),
                        lf_value,
                        get_option_flag(entry.options, "learnable"),
                    )
                )
            # File-loaded lexicon: TSV with `word\tcategory\tlf` rows.
            if decl.lexicon_from_file is not None:
                file_entries = self._load_lexicon_tsv(
                    decl.lexicon_from_file,
                    get_option_flag(
                        decl.lexicon_from_file_options, "learnable",
                    ),
                    decl,
                )
                entries.extend(file_entries)
            # Allocate one learnable Parameter per learnable entry.
            # We keep the Parameter list on a small nn.Module so it
            # participates in `.parameters()` of any Program that
            # owns the deduction.
            axiom_module = nn.Module()
            param_list: list = []
            for idx, (_w, _cat, _lf, is_learnable) in enumerate(entries):
                if is_learnable:
                    p = nn.Parameter(torch.zeros(()))
                    axiom_module.register_parameter(f"lex_weight_{idx}", p)
                    param_list.append(p)
                else:
                    param_list.append(None)
            # Capture the axiom-injector as a closure over the
            # entries + parameter list.
            entries_local = tuple(entries)
            params_local = tuple(param_list)

            def _axiom_injector(
                input_value, _entries=entries_local, _params=params_local
            ):
                # `input_value` may be a list/tuple of token strings,
                # OR a list of `(token, position)` pairs. We accept
                # bare-string lists for the common case.
                tokens = list(input_value)
                out: list = []
                for pos, tok in enumerate(tokens):
                    if isinstance(tok, tuple) and len(tok) == 2:
                        tok = tok[0]
                    for idx, (word, cat_pat, lf_val, _learn) in enumerate(_entries):
                        if word != tok:
                            continue
                        weight_param = _params[idx]
                        if weight_param is not None:
                            weight_tensor = weight_param
                        else:
                            weight_tensor = torch.tensor(0.0)
                        # Emit a span axiom carrying the lexical
                        # category and LF; positions cover the
                        # single token at [pos, pos+1).
                        item = ("span", pos, pos + 1, cat_pat, lf_val)
                        out.append((item, weight_tensor))
                return out
        else:
            # Identity injector — input is already a list of axioms.
            def _axiom_injector(input_value):
                if isinstance(input_value, list):
                    return input_value
                return list(input_value)

            axiom_module = None

        # Goal: items matching the start symbol's atom form for
        # top-level spans. Users override by composing the parse
        # result with their own predicate.
        start = get_option_name(
            decl.options, "start", line=decl.line, col=decl.col,
        )

        def _goal(item) -> bool:
            if start is None:
                return True
            if not (isinstance(item, tuple) and len(item) > 0):
                return False
            # Three goal-item shapes the framework recognizes by
            # default; users can override via a custom goal
            # predicate (the `axioms = source_kernel` escape hatch
            # composes with an arbitrary `goal` field on the
            # underlying DeductionSystem).
            head = item[0]
            # 1. Bare atom: ("atom", "S").
            if head == "atom" and len(item) == 2 and item[1] == start:
                return True
            # 2. Head-keyed (Datalog-style): ("reach", ...).
            if isinstance(head, str) and head == start:
                return True
            # 3. CKY-shaped span: ("span", i, j, ("atom", "S"), lf).
            if head == "span" and len(item) >= 4:
                cat = item[3]
                if (
                    isinstance(cat, tuple)
                    and len(cat) == 2
                    and cat[0] == "atom"
                    and cat[1] == start
                ):
                    return True
            return False

        # Choose an agenda strategy. Default to CKY for
        # context-free-shaped systems; depth-first for proof
        # search (Boolean semiring with rule-arity-2-or-less);
        # semi-naive for Datalog-shaped (no aggregation needed).
        if semiring is BOOLEAN and any(len(r.premises) == 1 for r in inference_rules):
            agenda_factory = depth_first_agenda
        elif semiring is BOOLEAN:
            agenda_factory = semi_naive_agenda
        else:
            agenda_factory = cky_agenda

        system = DeductionSystem(
            rules=tuple(inference_rules),
            semiring=semiring,
            axiom_injector=_axiom_injector,
            goal=_goal,
            agenda_factory=agenda_factory,
            max_iterations=10_000,
        )
        # Stash the axiom-module on the system so Programs that
        # reach it (via `parse(NAME, …)`) can include its
        # parameters in their optimizer.
        if axiom_module is not None:
            system._axiom_module = axiom_module  # type: ignore[attr-defined]
        # Attach a signature / encoder pairing, if declared. The
        # chart-query operations (`chart.embedding(pattern)`) consult
        # this attached encoder to compute on-demand item
        # embeddings.
        item_signature = get_option_name(
            decl.options, "signature", line=decl.line, col=decl.col,
        )
        if item_signature is not None:
            sigs = getattr(self, "_signatures", {})
            if item_signature not in sigs:
                raise CompileError(
                    f"deduction {decl.name!r}: unknown item signature "
                    f"{item_signature!r}",
                    decl.line,
                    decl.col,
                )
            system._item_signature = sigs[item_signature]  # type: ignore[attr-defined]
        item_encoder = get_option_name(
            decl.options, "encoder", line=decl.line, col=decl.col,
        )
        if item_encoder is not None:
            comps = getattr(self, "_encoders", {})
            if item_encoder not in comps:
                raise CompileError(
                    f"deduction {decl.name!r}: unknown item encoder "
                    f"{item_encoder!r}",
                    decl.line,
                    decl.col,
                )
            system._item_encoder = comps[item_encoder]  # type: ignore[attr-defined]
        self._deductions[decl.name] = system

    def _load_lexicon_tsv(
        self,
        path: str,
        learnable: bool,
        decl: "DeductionDecl",
    ) -> list[LexiconEntry]:
        """Load a lexicon from a TSV file at compile time.

        Format: each row has three tab-separated columns:
        ``word``, ``category``, ``lf_template``. The category is
        parsed as a type expression; the LF template is parsed as
        a let-arithmetic expression. Multiple rows per word are
        allowed (latent disjunction).

        Resolved relative to the working directory; paths starting
        with ``/`` are absolute.
        """
        from pathlib import Path
        # Re-parse the category and LF text by feeding them to the
        # tree-sitter parser inside a synthetic dummy program.
        # This keeps the lexicon-file syntax aligned with the
        # main grammar.

        # For simplicity, we expect categories and LFs in a
        # restricted form: bare identifiers for categories
        # (atom names) and bare identifiers for LFs (let_var refs).
        # Richer TSV formats may be supported by adding a custom
        # parser; this is the minimum viable schema.
        p = Path(path)
        if not p.exists():
            raise CompileError(
                f"deduction {decl.name!r}: lexicon file {path!r} not found",
                decl.line,
                decl.col,
            )
        out: list[LexiconEntry] = []
        with p.open("r", encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                line = raw_line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    raise CompileError(
                        f"deduction {decl.name!r}: lexicon file "
                        f"{path!r}:{lineno}: expected 3 tab-separated "
                        f"columns (word, category, lf), got {len(parts)}",
                        decl.line,
                        decl.col,
                    )
                word, cat_text, lf_text = parts[0], parts[1], parts[2]
                # Build a TypeName for the category atom. (Richer
                # category-shape parsing happens on the live
                # grammar; here we accept atom identifiers as a
                # safe, broadly-useful starting point.)
                cat_pattern = ("atom", cat_text)
                # LF: treat as a constructor-application or atom.
                # If the text contains '(' it's a let-call shape;
                # otherwise it's a bare identifier. Building the
                # corresponding pattern directly:
                if "(" in lf_text:
                    # Parse the LF text as a let-arith expression by
                    # wrapping it in a synthetic program whose body
                    # contains a single let step bound to the LF.
                    syn_src = (
                        "type _DummyObj : 1\n"
                        "morphism _f : _DummyObj -> _DummyObj "
                        "[role=latent]\n"
                        "program _dummy_prog : _DummyObj -> _DummyObj:\n"
                        "    sample _x : _DummyObj <- _f\n"
                        f"    let _lex_lf = {lf_text}\n"
                        "    return _x\n"
                    )
                    syn_mod = _parse_qvr(syn_src.encode(), "<lex-lf>")
                    # The third statement is the program; its
                    # second step's value carries the parsed LF.
                    prog = next(
                        s
                        for s in syn_mod.statements
                        if hasattr(s, "draws")
                        and getattr(s, "name", None) == "_dummy_prog"
                    )
                    let_step = prog.draws[1]
                    lex_globals = self._lex_globals_for_structural()
                    lf_value = _ProgramsMixin._compile_let_expr(
                        let_step.value, globals_=lex_globals
                    )({})
                else:
                    lf_value = lf_text
                out.append((word, cat_pattern, lf_value, learnable))
        return out
