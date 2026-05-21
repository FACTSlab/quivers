"""Parser for the quivers DSL.

The lexer/parser pipeline is delegated to panproto via the `qvr`
tree-sitter grammar registered in `panproto-grammars-all`. The public
`parse` entry point consumes `.qvr` source bytes and returns a
`Module` of dataclass AST nodes.

This package's submodules group the walker logic by topic:

* `._registry` for the panproto registry singleton, ``ParseError``,
  and the ``_Tree`` view that every walker reads from.
* `._helpers` for the low-level helpers ``_required_text``,
  ``_walk_options``, and ``_walk_return_pattern``.
* `.expressions` for type / space / morphism-expression / let-arith walkers.
* `.axes` for axis-role and morphism-prior walkers.
* `.program_steps` for program-block step walkers.
* `.statements` for the top-level ``_walk_statement`` dispatcher and
  every per-declaration walker (object, morphism, kernel, deduction,
  contraction, signature, encoder, decoder, loss, ...).
* `.core` for the public `parse` / `parse_file` entry
  points and ``_attach_docs``.

Every public name is re-exported here so ``from quivers.dsl.parser
import X`` keeps working unchanged.
"""

from quivers.dsl.parser.core import parse, parse_file
from quivers.dsl.parser._registry import ParseError

__all__ = ["ParseError", "parse", "parse_file"]
