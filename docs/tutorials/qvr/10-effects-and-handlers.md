# Effects and handlers

This tutorial defines a robustification request separately from the policy
that interprets it. The separation is useful when a model should expose an
operation in its checked graph while allowing a caller to choose the policy.
We will call this split the **request-policy separation (RPS)**.

## 1. Declare the request

<!-- compile: qiec -->
```qvr
effect Robust
    shrink : Real -> Real

instance robust : Robust

define robust_request(x : Real) : Real !{robust | rho lacks robust} =
    let adjusted <- perform robust.shrink(x)
    return adjusted
```

`Robust` is the interface, and `robust` is one lexical instance. The open row
says that `robust_request` performs on this instance and preserves any caller
effects in `rho`; `lacks robust` prevents the tail from naming the instance a
second time.

## 2. Write an authored handler

<!-- compile: qiec -->
```qvr
effect Robust
    shrink : Real -> Real

instance robust : Robust

handler half_weight for Robust : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    shrink(x : Real) resumes 1 =>
        resume(0.5 * x)

define robustify(x : Real) : Real !{} =
    handle robust with half_weight in
        let adjusted <- perform robust.shrink(x)
        return adjusted
```

The handler is total, so handling `robust` removes that instance from the
residual row. Its operation clause has grade `1`: every request must resume
exactly once. The value passed to `resume` becomes the result of the suspended
`perform`, after which `robustify` continues beneath the same handler.

## 3. Check the row

Save the complete source as
[`authored-handler.qvr`](source/authored-handler.qvr), then ask the REPL for
both rows:

```text
$ qvr repl docs/tutorials/qvr/source/authored-handler.qvr
qvr> :effects robust_request
qvr> :effects robustify
```

The first entry retains `robust`; the second is pure. `qvr run` can thus invoke
`robustify` under the core runtime without a foreign provider:

```bash
qvr run docs/tutorials/qvr/source/authored-handler.qvr robustify 8.0 --trace
```

Run it from Python and inspect the trace:

```python
from quivers.dsl import load

effects = load("docs/tutorials/qvr/source/authored-handler.qvr")
run = effects.run("robustify", 8.0)
assert run.value == 4.0

events = [event.event for event in run.result.trace]
assert "operation.requested" in events
assert "operation.handled" in events
assert "resumption.invoked" in events
print(run.value, events)
```

The trace distinguishes the request, handler dispatch, clause entry,
resumption, and final answer. These event names are stable enough for runtime
tests; their attached payloads retain the source and stable IDs needed for
deeper inspection.

## 4. Choose a resumption grade

Use the weakest grade that states the policy:

- `0` for aborting or replacing the continuation;
- `aff` when the clause may discard the continuation or resume once;
- `1` for ordinary forwarding or state-like updates; and
- `omega` for search that may resume once per alternative.

The evaluator checks the dynamic count. Thus a handler cannot declare linear
use and silently resume twice. Multi-shot resumption additionally requires a
duplicable context; mutable accumulators used by generated deductions are
forked per shot.

## 5. Decide between authored and foreign handlers

An authored clause is portable kernel code and can be inspected, serialized,
and emitted when a target supports its operations. A foreign handler declares
only the typed contract and receives its implementation from a runtime
provider. Use a foreign handler for a sampler, tensor module, database, or
other process-local resource. Write `[implementation=foreign]` and omit clause
bodies; omission without that option is rejected.

This distinction limits what a serialized module can execute by itself, but it
also keeps host callables out of the stable format. The runtime configuration
can select a provider; it cannot inject code.

Next, [Parsing as an effectful computation](11-parsing-and-search.md) shows why
multi-shot handlers matter for statistical search. The [effect
reference](../../reference/qvr/effects-and-handlers.md) lists the prelude,
rows, options, and failure modes.
