# Docker images for the transpile numeric-equivalence tier

Tier 3 of the transpile test suite runs each backend's transpiled
output through the real target runtime to verify
[log-density equivalence][tests.transpile._equivalence.assert_log_density_match]
with the QVR-source reference. Each runtime lives in its own pinned
image; the test harness shells out to `docker run --rm -v
$tmp:/io <image> <script.py>` and reads the result back through a
JSON file under the bind-mounted scratch directory.

## Build

```sh
./build.sh
```

Builds every image. Idempotent: rebuilding with no changes is a
cache hit; first build takes 5–20 minutes depending on bandwidth
(cmdstan in particular bootstraps a C++ install).

## Images

| Image tag | Backends served | Runtime |
|---|---|---|
| `panproto-test-stan` | stan | cmdstanpy + stanc3 |
| `panproto-test-numpyro` | numpyro | numpyro + jax |
| `panproto-test-pyro` | pyro | pyro-ppl + torch |
| `panproto-test-pymc` | pymc | pymc + pytensor |
| `panproto-test-edward2` | edward2 | tensorflow_probability.edward2 |
| `panproto-test-julia` | turing, gen | julia + Turing.jl + Gen.jl |
| `panproto-test-node` | webppl | node + webppl CLI |
| `panproto-test-jags` | jags | jags + pyjags |
| `panproto-test-bugs` | bugs | jags + pyjags (BUGS-syntax) |

Church has no Tier-3 image: no maintained interpreter exposes a
programmable joint log-density for a `(sample, observe)` program.

## Test gating

Tests use `@pytest.mark.requires_image("panproto-test-<runtime>")`
(see `tests/transpile/conftest.py`); cells whose image is not
locally built skip with a clear reason. CI builds them once per
cache key.

## I/O contract (per-backend script)

Each `tests/transpile/probes/_scripts/<backend>.py` is copied into
the container's `/io/probe.py` at run time. It reads:

- `/io/source.<ext>` — transpiled source bytes
- `/io/points.json` — list of points (`[{params: {...}, data: {...}}, ...]`)

and writes:

- `/io/result.json` — `{"log_densities": [...]}` with one value per
  input point (same order).
