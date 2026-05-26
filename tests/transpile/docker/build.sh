#!/usr/bin/env bash
# Build every panproto-test-* image used by the Tier-3 numeric
# equivalence suite. Idempotent: rebuilding an image with no
# changes is a fast cache hit. Run once on a fresh checkout; the
# numeric tests will skip cells whose image is absent.
set -euo pipefail

cd "$(dirname "$0")"

images=(stan numpyro pyro pymc edward2 julia node jags bugs)

for img in "${images[@]}"; do
    echo "=== building panproto-test-${img} ==="
    docker build -t "panproto-test-${img}" "${img}/"
done

echo "=== all images built ==="
docker images | grep panproto-test-
