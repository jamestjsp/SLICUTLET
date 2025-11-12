# Dev container for SLICUTLET - C11 translation of SLICOT
FROM ubuntu:24.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    gfortran \
    make \
    cmake \
    pkg-config \
    # Meson build system
    meson \
    ninja-build \
    # BLAS/LAPACK libraries
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    # Python development (deadsnakes PPA for Python 3.13)
    software-properties-common \
    # Git for submodules
    git \
    # Utilities
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA and install Python 3.13
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.13 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1

# Install uv (fast Python package manager) - use standalone installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    cp /root/.local/bin/uv /usr/local/bin/uv && \
    cp /root/.local/bin/uvx /usr/local/bin/uvx

# Set working directory
WORKDIR /workspace

# Verify installations
RUN gcc --version && \
    gfortran --version && \
    meson --version && \
    ninja --version && \
    python3 --version && \
    uv --version
