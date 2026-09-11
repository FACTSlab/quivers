# Edward2 probe image: standalone `edward2` package (google/edward2)
# with TensorFlow + tensorflow-probability backends.
# Tag: panproto-test-edward2
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    tensorflow==2.18.0 \
    tensorflow-probability==0.25.0 \
    tf-keras==2.18.0 \
    numpy==1.26.4 \
    tqdm \
    tenacity \
    git+https://github.com/google/edward2.git@main

ENTRYPOINT ["python"]
