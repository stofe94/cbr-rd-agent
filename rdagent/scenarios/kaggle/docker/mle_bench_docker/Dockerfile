FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    git \
    git-lfs \
    build-essential \
    strace \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/openai/mle-bench.git && \
    cd mle-bench && \
    git lfs fetch --all && \
    git lfs pull && \
    pip install --no-cache-dir -e .

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    lightgbm \
    catboost \
    pyarrow \
    coverage \
    fastparquet

WORKDIR /workspace
