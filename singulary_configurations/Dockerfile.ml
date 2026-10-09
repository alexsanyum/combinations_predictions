FROM python:3.11.15-slim

# Linux system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install the exact versions requested to run the Transformers with GPU acceleration
RUN pip install --no-cache-dir \
    numpy==2.4.6 \
    pandas==3.0.3 \
    scikit-learn==1.9.0 \
    imbalanced-learn==0.14.2 \
    scikit-optimize==0.10.2 \
    lightgbm==4.6.0 \
    xgboost==3.2.0

ENV LC_ALL=C