"""Let-step arithmetic expression AST nodes."""

from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes.objects import ObjectExpr

class LetExprNode(dx.TaggedUnion, discriminator="kind"):
    """Sum of let-step arithmetic expression nodes."""

class LetExprBinOp(LetExprNode):
    """Binary arithmetic operation in a let expression."""

    op: Literal["+", "-", "*", "/"]
    left: LetExprNode
    right: LetExprNode
    kind: Literal["let_expr_binop"] = "let_expr_binop"

class LetExprUnaryOp(LetExprNode):
    """Unary negation in a let expression."""

    operand: LetExprNode
    kind: Literal["let_expr_unary"] = "let_expr_unary"

class LetExprCall(LetExprNode):
    """Built-in function call in a let expression."""

    func: str
    args: tuple[LetExprNode, ...]
    kind: Literal["let_expr_call"] = "let_expr_call"

class LetExprLiteral(LetExprNode):
    """Numeric literal in a let expression."""

    value: float
    kind: Literal["let_expr_literal"] = "let_expr_literal"

class LetExprVar(LetExprNode):
    """Variable reference in a let expression."""

    name: str
    kind: Literal["let_expr_var"] = "let_expr_var"

class LetExprIndex(LetExprNode):
    """Indexed access into a finite-domain-indexed family ``a[i]``.

    Categorically the *pullback* morphism: given a finite-fibration
    ``index : N → A`` and a per-A morphism ``arr : A → B``, the
    indexed expression ``arr[index[n]]`` denotes
    ``arr ∘ index : N → B``, the natural Kleisli pullback of
    ``arr`` along ``index``.

    Attributes
    ----------
    array : LetExprNode
        The indexed-family expression (typically a `LetExprVar`
        naming a previously-drawn plate variable).
    indices : tuple of LetExprNode
        The index expressions; supports multi-dim indexing for
        nested plates (``coefs[subj[n], k]``).
    """

    array: LetExprNode
    indices: tuple[LetExprNode, ...]
    kind: Literal["let_expr_index"] = "let_expr_index"

class LetExprString(LetExprNode):
    """String literal in a let expression.

    Used for tokenisation, lexicon keys, and as ground-atom names
    in LF constructors like ``pred("dog")`` and
    ``forall("x", body)``. The runtime represents these as plain
    Python strings.
    """

    value: str
    kind: Literal["let_expr_string"] = "let_expr_string"

class LetExprList(LetExprNode):
    """List literal in a let expression: ``[a, b, c]``.

    Categorically a free-monoid element over the value sublanguage;
    the runtime represents it as a Python list (with autograd
    flowing through tensor-valued items).
    """

    items: tuple[LetExprNode, ...]
    kind: Literal["let_expr_list"] = "let_expr_list"

class LetExprLambda(LetExprNode):
    """Lambda expression ``param -> body`` in a let expression.

    Closes over the surrounding let-environment at instantiation
    time. Categorically a curried function in the Kleisli
    setting; used as the argument to fold / map / filter / reduce
    combinators.
    """

    param: str
    body: LetExprNode
    kind: Literal["let_expr_lambda"] = "let_expr_lambda"

class LetFactorBinder(dx.Model):
    """One ``<var> : <Index>`` binder in a multi-axis factor expression.

    The variable name binds to integer index values 0, 1, ...,
    |Index|-1 in the surrounding factor body.  The index type
    expression resolves to a finite-set object whose cardinality
    is the corresponding axis size of the constructed tensor.
    """

    var: str
    index: ObjectExpr
    line: int = 0
    col: int = 0

class LetFactorCase(dx.Model):
    """One ``<integer> -> <body>`` case in a factor pattern-match.

    The label is the integer index this case populates; the body
    is the value at that index.  The compiler verifies that the
    union of labels across all cases covers ``{0, ..., |Index|-1}``
    exactly.
    """

    label: int
    value: LetExprNode
    line: int = 0
    col: int = 0

class LetExprFactor(LetExprNode):
    """Multi-axis factor expression: assemble an indexed tensor.

    Surface forms:

    ``factor v1 : I1, v2 : I2, ..., vn : In in <body>`` denotes the
    tensor of shape ``(|I1|, ..., |In|, *body_shape)`` whose value
    at position ``(i1, ..., in)`` is ``body[v_k := i_k]``.

    ``factor v : I in { 0 -> e0, 1 -> e1, ... }`` denotes the
    single-axis case-structured form: the body at index `k` is the
    expression labelled `k`, and the labels must cover
    ``{0, ..., |I|-1}`` exactly.  Multi-axis case form is not
    accepted; the uniform body form (which can itself contain
    conditionals on the binders) is the general construction.

    Categorically the left adjoint of indexing.  Single-axis is a
    section of the trivial bundle ``I -> body_type``; multi-axis is
    a section over the product ``I1 x ... x In``.  The dual
    operation is the index pullback ``arr[i1, ..., in]``
    (`LetExprIndex`); together they realize the indexed-family
    colim / lim pair in the slice category over ``FinSet``.
    """

    binders: tuple[LetFactorBinder, ...]
    body: LetExprNode | None = None
    cases: tuple[LetFactorCase, ...] = ()
    kind: Literal["let_expr_factor"] = "let_expr_factor"

class LetExprMethodCall(LetExprNode):
    """Method call ``receiver.method(args)`` in a let expression.

    The receiver is itself a let-expression (typically a variable
    reference to a let-bound chart-valued, list-valued, or other
    object-valued value); the method is dispatched at runtime
    against the receiver's type. Used primarily for chart-view
    queries (``chart.weight(item)``, ``chart.enumerate(pattern)``,
    ``chart.goal_weight()``).
    """

    receiver: LetExprNode
    method: str
    args: tuple[LetExprNode, ...]
    kind: Literal["let_expr_method_call"] = "let_expr_method_call"

__all__ = [
    "LetExprNode",
    "LetExprBinOp",
    "LetExprUnaryOp",
    "LetExprCall",
    "LetExprLiteral",
    "LetExprVar",
    "LetExprIndex",
    "LetExprString",
    "LetExprList",
    "LetExprLambda",
    "LetFactorBinder",
    "LetFactorCase",
    "LetExprFactor",
    "LetExprMethodCall",
]
