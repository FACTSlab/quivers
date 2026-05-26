# Julia probe image: Turing.jl + Gen.jl.
# Tag: panproto-test-julia
#
# The image carries both packages because Turing and Gen share the
# Julia runtime; the test harness picks the probe entry script per
# backend.
FROM julia:1.11

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN julia -e 'using Pkg; Pkg.add(["Turing", "Gen", "Distributions", "JSON3"]); \
              using Turing, Gen, Distributions, JSON3'

# Pre-compile to avoid the slow first-import cost on each container run.
RUN julia -e 'using Turing; using Gen; using Distributions; using JSON3'

ENTRYPOINT ["julia", "--startup-file=no", "--quiet"]
