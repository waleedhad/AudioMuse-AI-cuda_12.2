FROM ubuntu:22.04

ENV LANG=C.UTF-8

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

RUN pip3 install --no-cache-dir numpy==1.26.4

# Install other core dependencies first, then essentia-tensorflow
# ADDED Flask-Cors here
RUN pip3 install --no-cache-dir \
    Flask \
    Flask-Cors \
    celery \
    redis \
    requests \
    scikit-learn \
    pyyaml \
    six

# Install Essentia with tensorflow support *after* core dependencies and pinned numpy.
# If essentia-tensorflow still tries to upgrade numpy, we might need a more specific pip command.
RUN pip3 install --no-cache-dir essentia-tensorflow

# 4) Copy your application code into the container
# This assumes your Dockerfile is in the root of your project
# IMPORTANT: This will copy the 'models' directory and its contents too,
# assuming they are present in your local git repo before building the image.
COPY . /app

# 5) Essentia environment variables
# PYTHONPATH might not be strictly necessary if all dependencies are installed globally via pip3
ENV PYTHONPATH=/usr/local/lib/python3/dist-packages

# Default working directory for runtime (so any relative writes go under /workspace)
WORKDIR /workspace

# Expose the port Flask will run on
EXPOSE 8000

# 6) Define the command to run the application
# This CMD allows the same image to be used for different services (Flask or Celery worker).
# In Kubernetes, you'll specify the `command` for each container in your Deployment.
# `sh -c` is used to allow conditional logic.
CMD ["sh", "-c", "if [ \"$SERVICE_TYPE\" = \"celery\" ]; then celery -A app.celery worker --loglevel=info; else python3 app.py; fi"]
