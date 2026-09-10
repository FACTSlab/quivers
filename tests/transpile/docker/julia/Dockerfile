# Julia probe image: Turing.jl + Gen.jl.
# Tag: panproto-test-julia
#
# The image carries both packages because Turing and Gen share the
# Julia runtime; the test harness picks the probe entry script per
# backend.
FROM julia:1.11

# `g++` is here for PackageCompiler, which links the sysimage below.
# `gcc` alone gets as far as the link step and fails there.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip g++ \
    && rm -rf /var/lib/apt/lists/*

RUN julia -e 'using Pkg; Pkg.add(["Turing", "Gen", "Distributions", "JSON3"]); \
              using Turing, Gen, Distributions, JSON3'

# Bake the packages into a sysimage. Precompiling alone still leaves
# every container loading the modules into a fresh session, and that is
# what a probe actually pays for: the container starts in well under a
# second, while `using Turing` and `using Gen` together take five to
# seven. From the sysimage the same load takes well under one.
RUN julia -e 'using Pkg; Pkg.add("PackageCompiler")'
RUN julia -e 'using PackageCompiler; \
              create_sysimage([:Turing, :Gen, :Distributions, :JSON3]; \
                              sysimage_path="/opt/qvr-sysimage.so")'

ENTRYPOINT ["julia", "--sysimage=/opt/qvr-sysimage.so", \
            "--startup-file=no", "--quiet"]
