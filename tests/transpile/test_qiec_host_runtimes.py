"""Calls, recursion, local instances, and authored handlers in every host runtime.

The four host runtimes share one free-computation ABI. These tests run the
same source through each of them and hold them to the reference machine's
observable behavior: recursion runs on a trampoline rather than the host
stack, tail calls retire their caller's frame, requests reached through
recursion or through two allocations of one instance address distinctly,
and an authored handler needs no attachment.
"""

from __future__ import annotations

import json
import math
import pathlib
import shutil
import subprocess
import sys
from typing import cast

import pytest

from quivers.dsl import parse
from quivers.transpile import Lower, transpile
from quivers.transpile.qiec_ir import IRQiecRowEntry

SCHEME_EXECUTABLE = next(
    (
        executable
        for name in ("scheme", "chez", "petite", "chezscheme")
        if (executable := shutil.which(name)) is not None
    ),
    None,
)

LIST = """\
family List[A : Type] : Type
    constructor Nil : List[A]
    constructor Cons : A * List[A] -> List[A]
"""

RECURSION = (
    LIST
    + """
define last[A : Type](xs : List[A], fallback : A) : A !{} =
    case xs motive => A
        Nil =>
            return fallback
        Cons(head, tail) =>
            last[A](tail, head)

define first[A : Type](xs : List[A], fallback : A) : A !{} =
    case xs motive => A
        Nil =>
            return fallback
        Cons(head, tail) =>
            let rest <- first[A](tail, head)
            return head

define even_length[A : Type](xs : List[A]) : Bool !{} =
    case xs motive => Bool
        Nil =>
            return true
        Cons(head, tail) =>
            odd_length[A](tail)

define odd_length[A : Type](xs : List[A]) : Bool !{} =
    case xs motive => Bool
        Nil =>
            return false
        Cons(head, tail) =>
            even_length[A](tail)
"""
)

STATE = """\
effect State[S : Type]
    get : Unit -> S
    put : S -> Unit

handler run_state for State[Int] : Int -> Int [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    get resumes 1 =>
        resume(0)
    put(s : Int) resumes 1 =>
        resume(unit)

define counted(seed : Int) : Int !{} =
    with instance cell : State[Int] in
        handle cell with run_state in
            perform cell.put(seed)
            let current <- perform cell.get()
            return current
"""

TICKING = (
    LIST
    + """
effect Tick
    tick : Unit -> Unit

instance clock : Tick

define ticking(xs : List[Int]) : Int !{clock} =
    case xs motive => Int
        Nil =>
            return 0
        Cons(head, tail) =>
            perform clock.tick()
            let rest <- ticking(tail)
            return rest

define nested(seed : Int) : Int !{clock} =
    with instance outer : Tick in
        with instance inner : Tick in
            perform clock.tick()
            return seed
"""
)

DEEP = 20_000
INT = {
    "kind": "type-application",
    "constructor": {"id": "x", "name": "Int", "telescope": []},
    "arguments": [],
}


def _constructors(source: str) -> dict[str, str]:
    """The stable constructor identities a source's lists are built from.

    Parameters
    ----------
    source : str
        QVR text declaring the ``List`` family.

    Returns
    -------
    dict[str, str]
        Constructor name to its ``qiec:constructor:...`` identity, as the
        generated code spells it.
    """
    ir = Lower().forward(parse(source)).qiec
    assert ir is not None
    return {item.name: item.id.text for item in ir.constructors}


def _request_key(source: str, instance: str, operation: str) -> str:
    """The operations-table key of requests on a module instance.

    Parameters
    ----------
    source : str
        QVR text declaring the instance and its interface.
    instance : str
        The instance's source name.
    operation : str
        The operation's source name.

    Returns
    -------
    str
        ``instance|operation`` in stable identities, the flat key the host
        runtimes look up.
    """
    ir = Lower().forward(parse(source)).qiec
    assert ir is not None
    named = next(item for item in ir.instances if item.name == instance)
    entry = cast(IRQiecRowEntry, named.entry)
    operation_id = next(
        item.id.text
        for effect in ir.effects
        for item in effect.operations
        if item.name == operation and effect.ref.id == entry.effect.id
    )
    return entry.instance.text + "|" + operation_id


def test_python_recursion_runs_on_a_trampoline() -> None:
    """Generated Python recurses tens of thousands deep under a tiny host limit."""
    namespace: dict[str, object] = {}
    exec(transpile(parse(RECURSION), target="pyro"), namespace)
    cons = _constructors(RECURSION)

    def lst(n: int) -> dict[str, object]:
        value: dict[str, object] = {
            "qiec": "constructor",
            "constructor": cons["Nil"],
            "static_arguments": (),
            "fields": (),
            "result_type": None,
        }
        for item in range(n):
            value = {
                "qiec": "constructor",
                "constructor": cons["Cons"],
                "static_arguments": (),
                "fields": (item, value),
                "result_type": None,
            }
        return value

    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(400)
    try:
        last = namespace["qiec_last"](lst(DEEP), -1, qiec_static_arguments=[INT])  # type: ignore[operator]
        first = namespace["qiec_first"](lst(DEEP), -1, qiec_static_arguments=[INT])  # type: ignore[operator]
        even = [
            namespace["qiec_even_length"](lst(n), qiec_static_arguments=[INT])  # type: ignore[operator]
            for n in (0, 1, 7, 8)
        ]
    finally:
        sys.setrecursionlimit(previous)
    assert last == 0
    assert first == DEEP - 1
    assert even == [True, False, False, True]


def test_python_authored_handler_needs_no_attachment() -> None:
    namespace: dict[str, object] = {}
    exec(transpile(parse(STATE), target="pyro"), namespace)
    assert namespace["qiec_counted"](5) == 0  # type: ignore[operator]


def test_python_requests_are_addressed_by_call_and_instance_frames() -> None:
    """Each recursive iteration and each allocation names its request distinctly."""
    namespace: dict[str, object] = {}
    exec(transpile(parse(TICKING), target="pyro"), namespace)
    cons = _constructors(TICKING)
    value: dict[str, object] = {
        "qiec": "constructor",
        "constructor": cons["Nil"],
        "static_arguments": (),
        "fields": (),
        "result_type": None,
    }
    for item in range(3):
        value = {
            "qiec": "constructor",
            "constructor": cons["Cons"],
            "static_arguments": (),
            "fields": (item, value),
            "result_type": None,
        }
    addresses: list[tuple[object, ...]] = []

    def tick(request: dict[str, object]) -> None:
        addresses.append(tuple(request["address"]))  # type: ignore[arg-type]
        return None

    key = tuple(_request_key(TICKING, "clock", "tick").split("|"))
    assert namespace["qiec_ticking"](value, qiec_operations={key: tick}) == 0  # type: ignore[operator]
    assert len(addresses) == 3
    assert len({address[0] for address in addresses}) == 1
    # The entry point itself is not a call, so the first request is unframed
    # and each recursive call adds one frame beneath the previous ones.
    assert [address[1] for address in addresses] == [
        (),
        (("call", "ticking#1"),),
        (("call", "ticking#1"), ("call", "ticking#2")),
    ]
    again: list[tuple[object, ...]] = []
    namespace["qiec_ticking"](  # type: ignore[operator]
        value,
        qiec_operations={key: lambda request: again.append(tuple(request["address"]))},
    )
    assert again == addresses

    nested: list[tuple[object, ...]] = []
    namespace["qiec_nested"](  # type: ignore[operator]
        7,
        qiec_operations={key: lambda request: nested.append(tuple(request["address"]))},
    )
    assert [address[1] for address in nested] == [(("instance", 1), ("instance", 2))]


_JS_LIST = """
var cons = %s;
var lst = function(n) {
  var v = { qiec: "constructor", constructor: cons.Nil, static_arguments: [], fields: [], result_type: null };
  for (var i = 0; i < n; i++) { v = { qiec: "constructor", constructor: cons.Cons, static_arguments: [], fields: [i, v], result_type: null }; }
  return v;
};
var INT = %s;
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_javascript_recursion_and_authored_handler(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "deep.js"
    script.write_bytes(
        transpile(parse(RECURSION + "\n" + STATE), target="webppl")
        + (
            _JS_LIST
            % (json.dumps(_constructors(RECURSION)), json.dumps(INT))
            + f"console.log(JSON.stringify([qiec_last(lst({DEEP}), -1, [INT], {{}}, {{}}, {{}}), "
            f"qiec_first(lst({DEEP}), -1, [INT], {{}}, {{}}, {{}}), "
            "qiec_even_length(lst(7), [INT], {}, {}, {}), "
            "qiec_even_length(lst(8), [INT], {}, {}, {}), qiec_counted(5)]));\n"
        ).encode()
    )
    completed = subprocess.run(
        ["node", "--stack-size=200", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == [0, DEEP - 1, False, True, 0]


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is unavailable")
def test_julia_recursion_and_authored_handler(tmp_path: pathlib.Path) -> None:
    cons = _constructors(RECURSION)
    script = tmp_path / "deep.jl"
    script.write_text(
        "macro model(expression)\n    esc(expression)\nend\n"
        + transpile(parse(RECURSION + "\n" + STATE), target="turing").decode()
        + f"""
function lst(n)
    v = Dict("qiec" => "constructor", "constructor" => "{cons["Nil"]}", "static_arguments" => Any[], "fields" => Any[], "result_type" => nothing)
    for i in 0:(n-1)
        v = Dict("qiec" => "constructor", "constructor" => "{cons["Cons"]}", "static_arguments" => Any[], "fields" => Any[i, v], "result_type" => nothing)
    end
    return v
end
INT = Dict("kind" => "type-application", "constructor" => Dict("id" => "x", "name" => "Int", "telescope" => Any[]), "arguments" => Any[])
println(qiec_last(lst({DEEP}), -1; qiec_static_arguments=Any[INT]), " ",
        qiec_first(lst({DEEP}), -1; qiec_static_arguments=Any[INT]), " ",
        qiec_even_length(lst(7); qiec_static_arguments=Any[INT]), " ",
        qiec_even_length(lst(8); qiec_static_arguments=Any[INT]), " ", qiec_counted(5))
"""
    )
    completed = subprocess.run(
        ["julia", "--startup-file=no", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.split() == ["0", str(DEEP - 1), "false", "true", "0"]


@pytest.mark.skipif(SCHEME_EXECUTABLE is None, reason="Chez Scheme is unavailable")
def test_scheme_recursion_and_authored_handler(tmp_path: pathlib.Path) -> None:
    cons = _constructors(RECURSION)
    script = tmp_path / "deep.scm"
    script.write_bytes(
        transpile(parse(RECURSION + "\n" + STATE), target="church")
        + f"""
(define (lst n)
  (let loop ((i 0) (v (list (cons "qiec" "constructor") (cons "constructor" "{cons["Nil"]}") (cons "static_arguments" '()) (cons "fields" '()) (cons "result_type" '()))))
    (if (= i n) v
        (loop (+ i 1) (list (cons "qiec" "constructor") (cons "constructor" "{cons["Cons"]}") (cons "static_arguments" '()) (cons "fields" (list i v)) (cons "result_type" '()))))))
(define INT (list (cons "kind" "type-application") (cons "constructor" (list (cons "id" "x") (cons "name" "Int") (cons "telescope" '()))) (cons "arguments" '())))
(display (qiec_last (lst {DEEP}) -1 (list INT))) (display " ")
(display (qiec_first (lst {DEEP}) -1 (list INT))) (display " ")
(display (qiec_even_length (lst 7) (list INT))) (display " ")
(display (qiec_even_length (lst 8) (list INT))) (display " ")
(display (qiec_counted 5))
(newline)
""".encode()
    )
    completed = subprocess.run(
        [SCHEME_EXECUTABLE, "--script", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.split() == ["0", str(DEEP - 1), "#f", "#t", "0"]


ARITHMETIC = """\
define triangle(n : Int) : Int !{} =
    if n <= 0 then
        return 0
    else
        let rest <- triangle(n - 1)
        return n + rest

define mixed(x : Int, y : Real) : Real !{} =
    let quotient = x / 2
    let remainder = x % 3
    let scaled = real(quotient) * y + real(remainder)
    let clipped = max(scaled, 0.5)
    let pair = (clipped, "v" + "1")
    let flag = pair[1] == "v1" && not (y == 1.0)
    if flag then
        return sqrt(pair[0]) - real(int(y))
    else
        return -1.0
"""


def _mixed_expected(x: int, y: float) -> float:
    """The reference value of ``mixed``, computed in Python.

    Parameters
    ----------
    x : int
        The integer argument.
    y : float
        The real argument.

    Returns
    -------
    float
        What every runtime must produce.
    """
    quotient = int(x / 2)
    remainder = x - 3 * int(x / 3)
    clipped = max(quotient * y + remainder, 0.5)
    if y != 1.0:
        return clipped**0.5 - float(int(y))
    return -1.0


def test_python_expressions_and_branches() -> None:
    namespace: dict[str, object] = {}
    exec(transpile(parse(ARITHMETIC), target="pyro"), namespace)
    assert namespace["qiec_triangle"](10) == 55  # type: ignore[operator]
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(400)
    try:
        assert namespace["qiec_triangle"](DEEP) == DEEP * (DEEP + 1) // 2  # type: ignore[operator]
    finally:
        sys.setrecursionlimit(previous)
    for x, y in ((-7, 2.5), (9, 1.0), (4, 0.1)):
        assert namespace["qiec_mixed"](x, y) == pytest.approx(_mixed_expected(x, y))  # type: ignore[operator]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_javascript_expressions_and_branches(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "expr.js"
    script.write_bytes(
        transpile(parse(ARITHMETIC), target="webppl")
        + (
            f"console.log(JSON.stringify([qiec_triangle(10), qiec_triangle({DEEP}), "
            "qiec_mixed(-7, 2.5), qiec_mixed(9, 1.0), qiec_mixed(4, 0.1)]));\n"
        ).encode()
    )
    completed = subprocess.run(
        ["node", "--stack-size=200", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    values = json.loads(completed.stdout)
    assert values[:2] == [55, DEEP * (DEEP + 1) // 2]
    assert values[2:] == pytest.approx(
        [_mixed_expected(-7, 2.5), _mixed_expected(9, 1.0), _mixed_expected(4, 0.1)]
    )


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is unavailable")
def test_julia_expressions_and_branches(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "expr.jl"
    script.write_text(
        "macro model(expression)\n    esc(expression)\nend\n"
        + transpile(parse(ARITHMETIC), target="turing").decode()
        + f'\nprintln(qiec_triangle(10), " ", qiec_triangle({DEEP}), " ", '
        'qiec_mixed(-7, 2.5), " ", qiec_mixed(9, 1.0), " ", qiec_mixed(4, 0.1))\n'
    )
    completed = subprocess.run(
        ["julia", "--startup-file=no", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    values = completed.stdout.split()
    assert values[:2] == ["55", str(DEEP * (DEEP + 1) // 2)]
    assert [float(item) for item in values[2:]] == pytest.approx(
        [_mixed_expected(-7, 2.5), _mixed_expected(9, 1.0), _mixed_expected(4, 0.1)]
    )


@pytest.mark.skipif(SCHEME_EXECUTABLE is None, reason="Chez Scheme is unavailable")
def test_scheme_expressions_and_branches(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "expr.scm"
    script.write_bytes(
        transpile(parse(ARITHMETIC), target="church")
        + (
            f'(display (qiec_triangle 10)) (display " ") (display (qiec_triangle {DEEP})) '
            '(display " ") (display (qiec_mixed -7 2.5)) (display " ") '
            '(display (qiec_mixed 9 1.0)) (display " ") (display (qiec_mixed 4 0.1)) '
            "(newline)\n"
        ).encode()
    )
    completed = subprocess.run(
        [SCHEME_EXECUTABLE, "--script", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    values = completed.stdout.split()
    assert values[:2] == ["55", str(DEEP * (DEEP + 1) // 2)]
    assert [float(item) for item in values[2:]] == pytest.approx(
        [_mixed_expected(-7, 2.5), _mixed_expected(9, 1.0), _mixed_expected(4, 0.1)]
    )


TENSORS = """\
object K : FinSet 3

define stats(v : Tensor[Real]([3])) : Real !{} =
    let scaled = tanh(v) * 2.0 + sigmoid(0.0)
    let weights = softmax(v)
    let squares = factor k : K in real(k) * real(k)
    return sum(scaled) + max(weights) + squares[2] + v[1] + erf(0.5) + lgamma(4.5)
"""


def _stats_expected(v: tuple[float, ...]) -> float:
    """The reference value of ``stats``, computed in Python.

    Parameters
    ----------
    v : tuple[float, ...]
        The vector argument.

    Returns
    -------
    float
        What every runtime must produce.
    """
    scaled = sum(math.tanh(x) * 2.0 + 0.5 for x in v)
    weights = [math.exp(x) for x in v]
    total = sum(weights)
    return (
        scaled
        + max(w / total for w in weights)
        + 4.0
        + v[1]
        + math.erf(0.5)
        + math.lgamma(4.5)
    )


_TENSOR_POINT = (0.3, -1.2, 2.0)


def test_python_tensor_expressions() -> None:
    namespace: dict[str, object] = {}
    exec(transpile(parse(TENSORS), target="pyro"), namespace)
    assert namespace["qiec_stats"](_TENSOR_POINT) == pytest.approx(  # type: ignore[operator]
        _stats_expected(_TENSOR_POINT)
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_javascript_tensor_expressions(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "tensors.js"
    script.write_bytes(
        transpile(parse(TENSORS), target="webppl")
        + (
            f"console.log(JSON.stringify(qiec_stats(Object.freeze({list(_TENSOR_POINT)}), "
            "[], {}, {}, {})));\n"
        ).encode()
    )
    completed = subprocess.run(
        ["node", str(script)], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == pytest.approx(_stats_expected(_TENSOR_POINT))


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is unavailable")
def test_julia_tensor_expressions(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "tensors.jl"
    script.write_text(
        "macro model(expression)\n    esc(expression)\nend\n"
        + transpile(parse(TENSORS), target="turing").decode()
        + f"\nprintln(qiec_stats({_TENSOR_POINT}))\n"
    )
    completed = subprocess.run(
        ["julia", "--startup-file=no", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(completed.stdout.strip()) == pytest.approx(_stats_expected(_TENSOR_POINT))


@pytest.mark.skipif(SCHEME_EXECUTABLE is None, reason="Chez Scheme is unavailable")
def test_scheme_tensor_expressions(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "tensors.scm"
    point = " ".join(str(x) for x in _TENSOR_POINT)
    script.write_bytes(
        transpile(parse(TENSORS), target="church")
        + f"(display (qiec_stats (_qvr-qiec-tuple (list {point})))) (newline)\n".encode()
    )
    completed = subprocess.run(
        [SCHEME_EXECUTABLE, "--script", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(completed.stdout.strip()) == pytest.approx(_stats_expected(_TENSOR_POINT))
