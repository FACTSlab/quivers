"""Block handler: hide sites from upstream handlers.

`BlockHandler` short-circuits `apply_stack` for named sites by
setting ``msg.stop = True``. Handlers stacked outside this one
will not see the blocked sites. The typical use is preventing an
outer `TraceHandler` from recording variational-family
intermediates that live inside a guide (see
[Pyro's `block`](https://docs.pyro.ai/en/stable/poutine.html#pyro.poutine.handlers.block)).
"""

from __future__ import annotations

from quivers.effects.base import EffectHandler, Message


class BlockHandler(EffectHandler):
    """Hide named sites (or every site) from outer handlers.

    Parameters
    ----------
    hide : list[str] or None
        Site names to hide. ``None`` hides every site.
    expose : list[str] or None
        Site names to expose. When set, only these sites reach
        outer handlers; every other site is hidden. Mutually
        exclusive with ``hide``.
    """

    def __init__(
        self,
        hide: list[str] | None = None,
        expose: list[str] | None = None,
    ) -> None:
        if hide is not None and expose is not None:
            raise ValueError(
                "BlockHandler: pass at most one of `hide` and `expose`, not both."
            )
        self.hide = set(hide) if hide is not None else None
        self.expose = set(expose) if expose is not None else None

    def _should_block(self, name: str) -> bool:
        if self.expose is not None:
            return name not in self.expose
        if self.hide is None:
            return True
        return name in self.hide

    def _process_message(self, msg: Message) -> None:
        if self._should_block(msg.name):
            msg.stop = True


def block(
    hide: list[str] | None = None,
    expose: list[str] | None = None,
) -> BlockHandler:
    """Return a `BlockHandler` that hides the named sites."""
    return BlockHandler(hide=hide, expose=expose)
