# Stan probe image: cmdstanpy + stanc3.
# Tag: panproto-test-stan
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir cmdstanpy==1.2.5 numpy==1.26.4

# Install cmdstan once at build time so the probe doesn't pay the
# install cost on every container start.
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan(progress=False)"

ENTRYPOINT ["python"]
