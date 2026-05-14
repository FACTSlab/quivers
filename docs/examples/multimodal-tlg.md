# Multimodal Type-Logical Grammar

## QVR Source

```qvr
object Term : 16

deduction MMTLG : Term -> Term {
    atoms {
        S, NP, N, VP, PP,
        Fwd, Bwd, Dia, Box,
        span
    }

    rule right_app
        : span(I, K, Fwd(A, B)), span(K, J, B)
        |- span(I, J, A)

    rule left_app
        : span(I, K, B), span(K, J, Bwd(A, B))
        |- span(I, J, A)

    rule dia_intro
        : span(I, J, A)
        |- span(I, J, Dia(A))

    rule dia_elim
        : span(I, J, Dia(A))
        |- span(I, J, A)

    semiring  LogProb
    start     S
    depth     6
}
```

## Overview

Multimodal type-logical grammar (Moortgat 1997) extends the Lambek calculus with unary modal type constructors `Dia` (`◇`) and `Box` (`□`) that form a residuated pair (`◇ ⊣ □`). The structural rules associated with each modality are controlled by the grammar's structural component; the deduction above licenses base right / left application together with modal introduction and elimination, leaving further structural postulates as user-added rules.

## Walkthrough

`atoms { … }` declares category atoms (`S`, `NP`, `N`, `VP`, `PP`), slash constructors (`Fwd`, `Bwd`), and modal constructors (`Dia`, `Box`). Chart items are `span(I, J, X)` triples.

- **`right_app` / `left_app`**: modus ponens for forward / backward slash, exactly as in the base Lambek calculus.
- **`dia_intro`**: modal introduction: a derivation of `A` lifts to a derivation of `Dia(A)` over the same span. This is the unit of the modality's monadic structure on the category of formulas.
- **`dia_elim`**: modal elimination: a derivation of `Dia(A)` projects back to a derivation of `A` over the same span. Combined with structural rules licensed under the modality, `dia_elim` is what permits controlled exchange / weakening / contraction within modal-marked subderivations.

A richer fragment would add explicit modal structural rules (modal exchange, modal contraction) and the `Box` introduction / elimination duals; both are sequent rules in the same style.

## DSL Features

- **Sequent rules with arbitrary arity**: rule premises can be unary or binary; the agenda dispatches each pattern to the appropriate chart-cell shape.
- **Modal constructors as atoms**: `Dia` and `Box` are user-declared atoms in the same vocabulary as the slash constructors. There is no built-in modal syntax.
- **Depth bounding**: `depth 6` keeps modal nesting finite, which is essential because the category space is otherwise infinite (every `A` admits `Dia(A)`, `Dia(Dia(A))`, …).

## Try it

```python
from quivers.dsl import load

prog = load("docs/examples/source/multimodal_tlg.qvr")
```

Adding a `lexicon { ... }` block of learnable per-entry log-weights plus an outer `program` that observes the chart's goal weight on a corpus closes the fit loop. The modal structural-rule licensing is purely additional sequent rules: extending the modality with controlled exchange, contraction, or weakening just adds more rules to the deduction body without changing the surface.

## Categorical Perspective

The diamond `◇` acts as a monad on the category of formulas: its unit `η : X → ◇X` (modal introduction) lifts any formula into the modal regime, and its multiplication `μ : ◇(◇X) → ◇X` (idempotence) collapses nested modals when the corresponding structural rule is licensed. The standard Lambek calculus is the internal language of a residuated monoidal category with no extra structure. Adding the diamond monad and licensing modal structural rules permits controlled relaxation of resource sensitivity: inside the monad, structural rules like exchange, weakening, and contraction become available without contaminating the surrounding linear context.

## Linguistic Applications

Multimodal type-logical grammar handles cases where strict linearity must be relaxed:

- **Extraction and long-range dependencies**: extracted elements marked as modal can permute with intermediate functors, threading through multiple clause boundaries (wh-questions, relative clauses).
- **Non-constituent coordination**: modal operators can license extracting a common context, letting two fragments coordinate even if they are not standard constituents.
- **Gapping**: a gapped functor with a modal type can be reused across a coordination boundary.
