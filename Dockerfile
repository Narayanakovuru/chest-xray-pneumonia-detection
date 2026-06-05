# Use official PyTorch GPU runtime base image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set system environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory in the container
WORKDIR /workspace

# Install system dependencies (e.g. for image processing / utilities)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install the package in editable mode for absolute imports resolution
COPY pyproject.toml setup.py ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Copy remaining code (configs, scripts, tests)
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Default command to run training pipeline
CMD ["python", "-m", "src.pneumonia_detection.pipelines.train_pipeline", "--config", "configs/train.yaml"]
