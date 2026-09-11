# NumPyro probe image: numpyro + jax.
# Tag: panproto-test-numpyro
FROM python:3.12-slim

# `Poisson(rate=0).log_prob(0)` returned `nan` through numpyro 0.18,
# where the density is `log 1 = 0`. A zero rate is exactly what a
# zero-inflated model's off atom carries, so a `marginalize` over one
# scored `nan` at every point on this image alone. 0.21 computes the
# `xlogy` form, which is defined there. jax is left to the resolver:
# pinning it against a numpyro this new is what forced the stale pin
# to persist.
RUN pip install --no-cache-dir \
    numpyro==0.21.0 \
    numpy==1.26.4

ENTRYPOINT ["python"]
