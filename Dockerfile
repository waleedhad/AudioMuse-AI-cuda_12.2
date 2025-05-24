# Use the specified base image for Python 3.10 on Ubuntu 22.04 (Jammy Jellyfish).
FROM ubuntu:22.04

# Set environment variables for non-interactive apt-get and Python buffering.
ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# --- Essential Setup ---

# 1) Set the initial working directory inside the container.
# All subsequent commands (like COPY) will be run from this directory by default.
WORKDIR /app

# 2) Install Python, pip, and all required system libraries.
# This single RUN command ensures all base dependencies are installed efficiently.
# It includes:
# - Core Python development tools (python3, python3-pip, python3-dev).
# - Essentia's C/C++ dependencies (libfftw3, libyaml, libtag, libsamplerate).
# - FFMPEG for robust audio processing.
# - General utilities (wget, git, vim, redis-tools, curl).
# - **CRUCIAL DEBUGGING TOOLS**: `strace` for system call tracing, `procps` for `ps` and `top`,
#   and `iputils-ping` for network diagnostics.
# - **OPTIMIZED LIBRARIES**: `libopenblas-dev` and `liblapack-dev` provide highly optimized
#   linear algebra routines that TensorFlow and Essentia can leverage, potentially resolving
#   performance issues or subtle hangs caused by fallback to unoptimized paths.
# `--no-install-recommends` helps keep the image size smaller by only installing direct dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libfftw3-3 libyaml-0-2 libtag1v5 libsamplerate0 \
    ffmpeg wget git vim \
    redis-tools curl \
    strace \
    procps \
    iputils-ping \
    libopenblas-dev \
    liblapack-dev \
    # Clean up apt caches to reduce image size after installation
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# --- Python Dependencies ---

# 3) Install Python packages with compatible NumPy for Essentia-TensorFlow.
# The order is critical for dependency resolution:
# - NumPy is installed first and **pinned to a compatible version (1.26.4)**. This is vital
#   because Essentia and TensorFlow can have strict requirements and conflicts with NumPy's versions.
# - Other core Python dependencies are installed next.
# - Essentia with TensorFlow support is installed last, ensuring it uses the already-pinned NumPy.
RUN pip3 install --no-cache-dir numpy==1.26.4

# Install other core Python dependencies that your application needs.
# Flask-Cors is included here as per your original setup.
RUN pip3 install --no-cache-dir \
    Flask \
    Flask-Cors \
    celery \
    redis \
    requests \
    scikit-learn \
    pyyaml \
    six

# Install Essentia with TensorFlow support.
# This will pick up the NumPy version installed above.
RUN pip3 install --no-cache-dir essentia-tensorflow

# --- Application Code & Runtime Configuration ---

# 4) Copy your entire application code into the container's working directory (/app).
# This assumes your Dockerfile is at the root of your project, and it will include all
# necessary files like app.py, config.py, and the 'models' directory.
COPY . /app

# 5) Set Essentia environment variables.
# ESSENTIA_MODELS_DIR tells Essentia where to find its model files.
# PYTHONPATH helps Python locate installed packages and modules, though pip often handles this automatically.
ENV ESSENTIA_MODELS_DIR=/app/models
ENV PYTHONPATH=/usr/local/lib/python3/dist-packages

# Create necessary directories for runtime if they don't exist.
# This ensures that `/app/temp_audio` is available for storing temporary audio files
# downloaded during the analysis process.
RUN mkdir -p /app/temp_audio

# Expose the port Flask will run on.
# This makes port 8000 accessible from outside the container, typically for your frontend service.
EXPOSE 8000

# --- Final Working Directory and Command ---

# Set the final working directory for the container at runtime.
# This means when the container starts, its current directory will be /workspace.
# Note: Your application code is still in /app from the COPY instruction.
WORKDIR /workspace

# 6) Define the command to run the application when the container starts.
# This `CMD` uses an 'if/else' condition based on the `SERVICE_TYPE` environment variable.
# Because the final `WORKDIR` is now `/workspace`, we must use the **absolute path**
# to `app.py` for both the Celery worker and the Flask application.
CMD ["sh", "-c", "if [ \"$SERVICE_TYPE\" = \"celery\" ]; then celery -A /app/app.celery worker --loglevel=info; else python3 /app/app.py; fi"]
