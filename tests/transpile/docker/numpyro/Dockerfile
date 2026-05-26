# NumPyro probe image: numpyro + jax.
# Tag: panproto-test-numpyro
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    numpyro==0.15.3 \
    jax==0.4.34 \
    jaxlib==0.4.34 \
    numpy==1.26.4

ENTRYPOINT ["python"]
