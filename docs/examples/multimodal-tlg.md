# Multimodal Type-Logical Grammar

## QVR Source

```qvr
category S, NP, N, VP, PP
object Token : 256

let tlg = parser(
    rules=[evaluation, adjunction_units, modal_introduction, modal_elimination],
    terminal=Token,
    constructors=[slash, diamond],
    depth=1,
    start=S
)

output tlg
```

## Overview

Multimodal type-logical grammar extends the Lambek calculus with modal operators that allow controlled relaxation of resource sensitivity. This example demonstrates the `diamond` constructor, the `depth` parameter for bounding modal nesting, and the `modal_introduction`/`modal_elimination` rule schemas.

## Walkthrough

The category and token declarations are identical to the basic type-logical grammar example.

The parser declaration introduces two new parameters. `constructors=[slash, diamond]` specifies which type-level operations are available: `slash` permits forming $X/Y$ and $X \backslash Y$; `diamond` permits forming $\Diamond X$, marking $X$ as modally flexible. `depth=1` bounds nesting: $\Diamond(\mathrm{NP}/\mathrm{VP})$ is allowed, but $\Diamond(\Diamond \mathrm{NP})$ is not. Without a depth bound, the category space would be infinite and parsing intractable.

The rules `evaluation` and `adjunction_units` work as in the basic type-logical grammar. The two new rules:

- **`modal_introduction`** wraps any formula in a diamond: $X \vdash \Diamond X$. A resource of type $X$ can be used where $\Diamond X$ is expected. This is the unit of the diamond's monadic structure.
- **`modal_elimination`** encodes that diamond-wrapped formulas are exempt from certain structural constraints. In the strict Lambek calculus, exchange ($A/B, C$ reordering to $C, A/B$) is invalid. With diamond-marked resources, exchange becomes derivable. The diamond acts as a marker licensing structural rearrangements.

## DSL Features

- **`constructors` parameter**: Lists which type-level operations are available (`slash`, `diamond`, optionally `box`).
- **`depth` parameter**: Bounds constructor nesting. Depth 1 applies constructors to atoms and simple slashed formulas only. Depth 2 permits double nesting but risks exponential blowup.
- **`modal_introduction`**: The reflexive unit for diamond; $A$ can be used where $\Diamond A$ is expected.
- **`modal_elimination`**: Licenses structural rules (exchange, weakening, contraction) under the diamond regime.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

The diamond operator $\Diamond$ acts as a monad on the category of formulas. Its unit $\eta : X \to \Diamond X$ (modal introduction) lifts any formula into the modal, and its multiplication $\mu : \Diamond(\Diamond X) \to \Diamond X$ (idempotence) collapses nested modals. The standard Lambek calculus is the internal language of a symmetric monoidal closed category with no extra structure. Adding the diamond monad permits controlled relaxation of resource sensitivity: inside the monad, structural rules like exchange, weakening, and contraction become available. The parser stays tractable because the `depth` parameter keeps the category space finite.

## Linguistic Applications

Multimodal type-logical grammar handles cases where strict linearity must be relaxed:

- **Extraction and long-range dependencies**: Extracted elements marked as modal can be permuted with intermediate functors, threading through multiple clause boundaries (wh-questions, relative clauses).
- **Non-constituent coordination**: Modal operators can license extracting a common context, letting two fragments coordinate even if they are not standard constituents.
- **Gapping**: A gapped functor with a modal type can be reused across a coordination boundary.

## Connections to Other Modal Formalisms

Multimodal type-logical grammar differs from related approaches (display logic, separation logic) in that its modes are chosen for specific linguistic phenomena, and the structural rules each mode licenses are carefully restricted to avoid over-generation. It is more flexible than the strict Lambek calculus but more constrained than unrestricted phrase-structure grammars.
