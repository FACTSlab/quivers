# JAGS probe image: jags + pyjags.
# Tag: panproto-test-jags
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    jags \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pyjags==1.3.8 \
    numpy==1.26.4

ENTRYPOINT ["python"]
