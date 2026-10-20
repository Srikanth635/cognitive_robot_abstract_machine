# %% dependency runtime
FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

LABEL org.opencontainers.image.source="https://github.com/Srikanth635/cognitive_robot_abstract_machine"

RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/ros/cram-env

COPY requirements.txt /tmp/runtime-requirements.txt
RUN /opt/ros/cram-env/bin/python -m pip install --no-cache-dir --require-hashes --no-deps -r /tmp/runtime-requirements.txt \
    && /opt/ros/cram-env/bin/python -m pip check

ENV PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ENTRYPOINT ["/opt/ros/cram-env/bin/python"]
