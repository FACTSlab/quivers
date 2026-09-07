# NumPyro probe image: numpyro + jax.
# Tag: panproto-test-numpyro
FROM python:3.12-slim

# `Poisson(rate=0).log_prob(0)` returns `nan` on numpyro 0.15.3, where
# the density is `log 1 = 0`: the zero rate is exactly what a
# zero-inflated model's off atom carries, so a `marginalize` over it
# scored `nan` on this image alone. Fixed upstream.
RUN pip install --no-cache-dir \
    numpyro==0.18.0 \
    jax==0.4.38 \
    jaxlib==0.4.38 \
    numpy==1.26.4

ENTRYPOINT ["python"]
