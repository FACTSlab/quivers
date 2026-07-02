"""Algebraic effect handlers for probabilistic programs.

An `EffectHandler` sits on a thread-local handler stack. When a
`MonadicProgram` executes under one or more handlers, every
observable action (a `sample`, an `observe`, a `let`, a `score`) is
wrapped in a `Message` that walks the stack outer-to-inner, giving
each handler the chance to intercept, rewrite, or annotate the
action. The design is a direct port of the Pyro `poutine` /
NumPyro `handlers` `Messenger` shape ([Pyro poutine
docs](https://docs.pyro.ai/en/stable/poutine.html); [NumPyro
handlers docs](https://num.pyro.ai/en/stable/handlers.html))
grounded in the operational-semantics account of
[Plotkin and Pretnar (2009)](https://doi.org/10.1007/978-3-642-00590-9_7)
and the eff-language calculus of
[Bauer and Pretnar (2015)](https://doi.org/10.1016/j.jlamp.2014.02.001).
The application of algebraic effects to probabilistic programming
is developed by
[Scibior et al. (2018)](https://doi.org/10.1145/3236778)
and
[Nguyen et al. (2023)](https://doi.org/10.1145/3609026.3609729).

Distinction from
[`quivers.monadic.algebraic.Handler`][quivers.monadic.algebraic.Handler]:
`EffectHandler` is a mutable-message dispatcher intended for
runtime interception of a `MonadicProgram` executing in the
concrete torch semantics. `monadic.algebraic.Handler` is a
free-monad-over-signature interpreter that folds a bounded-depth
signature tree into a target monad by post-order recursion on the
flat-FinSet carrier. The two abstractions cover different
territories: use `EffectHandler` for Pyro-style handler stacks
over sample / observe / let / score sites, `monadic.algebraic.Handler`
for structural interpretations of a user-defined effect signature
into a target monad.

A `Message` carries the site name, the site kind (``sample``,
``observe``, ``let``, ``score``), a reference to the underlying
distribution morphism (when applicable), the current value, the
current log-density, and a ``stop`` flag that lets a handler
short-circuit further intervention. Handlers respond by mutating
the message: `clamp` clamps ``value``; `do` clamps ``value`` and
zeroes ``log_prob``; `mask` element-wise gates ``log_prob``;
`scale` multiplies ``log_prob``; and so on.

The stack is pushed outer-first. When the program interpreter emits
a message it calls `apply_stack(msg)`, which invokes each handler
outer-to-inner via `_process_message`, then runs the site
computation (unless a handler already supplied ``value``), then
invokes each handler inner-to-outer via `_postprocess_message`.
This is the standard `Messenger` protocol; the two-pass discipline
lets `trace` observe the final state after every rewrite.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
import threading

import torch

from quivers.continuous.morphisms import ContinuousMorphism


_LOCAL = threading.local()


def _handler_stack() -> "list[EffectHandler]":
    """Return the current thread's active handler stack.

    A fresh list is materialised on first access per thread; every
    handler's `__enter__` appends to this list and every
    `__exit__` pops it. The stack is ordered outer-first, matching
    Pyro's `PYRO_STACK` convention.
    """
    stack = getattr(_LOCAL, "stack", None)
    if stack is None:
        stack = []
        _LOCAL.stack = stack
    return stack


@dataclass
class Message:
    """Effect message flowing through the handler stack.

    Every observable action a program executes emits a `Message`
    that each active handler sees in turn. Handlers mutate the
    message in place: setting ``value`` clamps the site,
    multiplying ``log_prob`` reweights it, setting ``stop = True``
    prevents inner handlers from firing.

    Parameters
    ----------
    kind : str
        One of ``"sample"``, ``"observe"``, ``"let"``, ``"score"``.
    name : str
        Site name (the bound variable).
    morphism : ContinuousMorphism or None
        The site's distribution morphism. ``None`` for ``let`` and
        ``score`` sites.
    input : torch.Tensor or None
        Conditioning input tensor for the morphism. ``None`` for
        `let` and `score` sites.
    value : torch.Tensor or None
        Current value at the site. ``None`` until a handler or the
        default interpretation supplies it.
    log_prob : torch.Tensor or None
        Current log-density contribution. ``None`` until computed.
    is_observed : bool
        ``True`` when the site was clamped by ``condition`` or by
        the DSL's ``observe`` keyword.
    is_deterministic : bool
        ``True`` for `let` bindings and for sample sites that a
        handler has demoted to deterministic (e.g. after a `do`
        intervention).
    stop : bool
        When ``True``, inner handlers and the default site
        interpretation are skipped.
    metadata : dict[str, object]
        Free-form per-site annotations handlers may set.
    """

    kind: str
    name: str
    morphism: ContinuousMorphism | None = None
    input: torch.Tensor | None = None
    value: torch.Tensor | None = None
    log_prob: torch.Tensor | None = None
    is_observed: bool = False
    is_deterministic: bool = False
    stop: bool = False
    metadata: dict[str, torch.Tensor | str | bool | int | float] = field(
        default_factory=dict
    )


class EffectHandler(ABC):
    """Abstract base for effect handlers.

    Subclass and override any of the per-kind hooks
    (`_pyro_sample`, `_pyro_observe`, `_pyro_let`,
    `_pyro_score`) or the catch-all `_process_message` and
    `_postprocess_message`. The handler activates by being used as
    a context manager:

        with condition({"z": z_val}):
            samples = predictive.rsample(x)

    Nested `with` blocks stack handlers outer-first; this handler
    is closest to the effect on the inner side.
    """

    def __enter__(self) -> "EffectHandler":
        _handler_stack().append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        stack = _handler_stack()
        if not stack or stack[-1] is not self:
            raise RuntimeError(
                "EffectHandler.__exit__: handler stack corruption; "
                "handlers must be entered / exited in strict LIFO order."
            )
        stack.pop()

    def _process_message(self, msg: Message) -> None:
        """Dispatch to a per-kind hook before the site runs.

        The default routes to `_pyro_sample`, `_pyro_observe`,
        `_pyro_let`, or `_pyro_score` by ``msg.kind``. Subclasses
        override the specific hooks; override this method directly
        only when a handler must see every kind.
        """
        method = getattr(self, f"_pyro_{msg.kind}", None)
        if method is not None:
            method(msg)

    def _postprocess_message(self, msg: Message) -> None:
        """Dispatch to a per-kind post-hook after the site runs.

        Handlers that record state (e.g. `TraceHandler`) use this
        pass to snapshot the final message. Handlers that only
        rewrite the site typically leave this a no-op.
        """
        method = getattr(self, f"_pyro_post_{msg.kind}", None)
        if method is not None:
            method(msg)


def apply_stack(
    msg: Message,
    default: Callable[[Message], None] | None = None,
) -> Message:
    """Run a message through the active handler stack.

    Three phases, in order:

    1. **Pre-pass** (outer-to-inner). Each handler's
       `_process_message` is called until one sets
       ``msg.stop = True`` or the stack is exhausted. Innermost
       handler has the final say on the pre-pass rewrite.
    2. **Default computation.** If ``default`` was supplied, it
       runs after the pre-pass and installs the site's fallback
       ``value`` / ``log_prob``. The default is responsible for
       respecting whatever the pre-pass wrote (typically a
       sample-and-score against the site's underlying morphism
       when the pre-pass left ``value`` unset).
    3. **Post-pass** (outer-to-inner, same order as the pre-pass).
       Each handler that saw the pre-pass sees the now-populated
       message in `_postprocess_message`. The convention is that
       handlers pushed later see the final rewritten state last,
       so a `TraceHandler` pushed inside a `with condition(...)`
       block snapshots the conditioned value, and a
       `TraceHandler` pushed inside a `with mask(...)` block
       snapshots the mask-multiplied log-density.

    Splitting default computation out of `apply_stack` (rather than
    inlining it) matches the Pyro `Messenger` protocol: handlers
    get first-and-last look, defaults fire only when the site was
    not fully rewritten, and post-hooks always see the resolved
    site.
    """
    stack = _handler_stack()
    seen: list[EffectHandler] = []
    # Pre-pass: innermost-first (top of stack). Matches Pyro's
    # `_PYRO_STACK` iteration order. A `block` handler placed near
    # the top of the stack sets ``msg.stop`` and prevents outer
    # handlers from ever seeing the message.
    for handler in reversed(stack):
        handler._process_message(msg)
        seen.append(handler)
        if msg.stop:
            break
    if default is not None:
        default(msg)
    # Post-pass: the handlers we actually saw, in the same stack
    # order they occupy (outer-to-inner among the visited frames).
    # A `TraceHandler` pushed innermost thus fires post last and
    # snapshots the rewrites of every outer handler.
    for handler in reversed(seen):
        handler._postprocess_message(msg)
    return msg
