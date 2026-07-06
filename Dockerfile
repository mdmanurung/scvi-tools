# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim AS cpu

RUN pip install --no-cache-dir uv

WORKDIR /opt/scvi-tools
COPY . /opt/scvi-tools

ARG DEPENDENCIES=""
RUN if [ -n "$DEPENDENCIES" ]; then \
      uv pip install --system --no-cache ".[$DEPENDENCIES]"; \
    else \
      uv pip install --system --no-cache "."; \
    fi

ENV MPLCONFIGDIR=/tmp/matplotlib
ENV NUMBA_CACHE_DIR=/tmp/numba

CMD ["/bin/bash"]


FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS cuda

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      python3 \
      python3-pip \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir uv

WORKDIR /opt/scvi-tools
COPY . /opt/scvi-tools

ARG DEPENDENCIES="cuda,cytoanvi-hierarchy,cytoanvi-mapping-qc,cytoanvi-annbatch,cytoanvi-baselines"
RUN uv pip install --system --break-system-packages --no-cache ".[$DEPENDENCIES]"

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV NUMBA_CACHE_DIR=/tmp/numba

CMD ["/bin/bash"]
