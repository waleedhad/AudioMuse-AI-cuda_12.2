# Use the specified base image
FROM ubuntu:22.04

# These environment variables are part of a multi-line definition.
# Comments must be on their own line or before the backslash.
ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1
# Ensures Python output is unbuffered

# 1) Create a directory for your application code
WORKDIR /app

# 2) Install Python, pip, and required system libraries
# Include `redis-tools` for optional Redis CLI for debugging, and `curl` for potential health checks or debugging.
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    libfftw3-3 libyaml-0-2 libtag1v5 libsamplerate0 \
    ffmpeg wget git vim \
    redis-tools curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 3) Install Python packages with compatible numpy for essentia-tensorflow
# Order matters: numpy first, then essentia-tensorflow, then others.
# Using a specific numpy version as per your original Dockerfile.
RUN pip3 install --no-cache-dir \
    numpy==1.21.6

# Install Essentia with tensorflow support.
# If essentia-tensorflow is not found on PyPI, you might need to install via pip wheel or directly from source.
# Assuming 'essentia-tensorflow' is the correct package name.
# If it's `essentia_extractor`, use that.
# pip3 install --no-cache-dir essentia_extractor[tensorflow] # modern way if essentia_extractor is on PyPI
RUN pip3 install --no-cache-dir \
    essentia-tensorflow \
    Flask \
    celery \
    redis \
    requests \
    scikit-learn \
    pyyaml \
    six

# 4) Copy your application code into the container
# This assumes your Dockerfile is in the root of your project
# IMPORTANT: This will copy the 'models' directory and its contents too,
# assuming they are present in your local git repo before building the image.
COPY . /app

# 5) Essentia environment variables
# These are less critical when models are directly specified by path in config.py,
# but good practice to keep for Essentia's internal mechanisms if it relies on them.
ENV ESSENTIA_MODELS_DIR=/app/models
# PYTHONPATH might not be strictly necessary if all dependencies are installed globally via pip3
ENV PYTHONPATH=/usr/local/lib/python3/dist-packages

# Create necessary directories for runtime if they don't exist
RUN mkdir -p /app/temp_audio

# Expose the port Flask will run on
EXPOSE 8000

# 6) Define the command to run the application
# This CMD allows the same image to be used for different services (Flask or Celery worker).
# In Kubernetes, you'll specify the `command` for each container in your Deployment.
# `sh -c` is used to allow conditional logic.
CMD ["sh", "-c", "if [ \"$SERVICE_TYPE\" = \"celery\" ]; then celery -A app.celery worker --loglevel=info; else python3 app.py; fi"]
