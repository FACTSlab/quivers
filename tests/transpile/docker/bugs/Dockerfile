# BUGS probe image: delegates to JAGS (BUGS-syntax compatible).
# Tag: panproto-test-bugs
#
# BUGS / WinBUGS / OpenBUGS / MultiBUGS do not have maintained
# python-bindings or a programmable log-density interface that runs
# on Linux. JAGS was designed as a free implementation of the BUGS
# language and accepts straight BUGS source verbatim; the probe
# script reuses pyjags + jags through the same code path as the
# JAGS probe.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    jags \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pyjags==2.4.0 \
    numpy==2.5.1

ENTRYPOINT ["python"]
