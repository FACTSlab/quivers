# WebPPL probe image: webppl CLI + node.
# Tag: panproto-test-node
FROM node:22-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g webppl

ENTRYPOINT ["python3"]
