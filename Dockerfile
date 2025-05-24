FROM python:3.11-slim-bookworm

# These environment variables are part of a multi-line definition.
# Comments must be on their own line or before the backslash.
ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1
# Ensures Python output is unbuffered

# 1) Create a directory for your application code
WORKDIR /app

# 2) Install required system libraries (Python and pip are already there)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfftw3-3 libyaml-0-2 libtag1v5 libsamplerate0 \
    ffmpeg wget git vim \
    redis-tools curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 3) Install Python packages with compatible numpy for essentia-tensorflow
# Order matters: numpy first, then essentia-tensorflow, then others.
# Strictly pinning numpy to a version known to be 1.x and compatible with Essentia.
# Trying 1.26.4, one of the last 1.x versions before 2.x broke compatibility.
RUN pip install --no-cache-dir numpy==1.26.4

# Install other core dependencies first, then essentia-tensorflow
RUN pip install --no-cache-dir \
    Flask \
    Flask-Cors \
    celery \
    redis \
    requests \
    scikit-learn \
    pyyaml \
    six

# Install Essentia with tensorflow support *after* core dependencies and pinned numpy.
RUN pip install --no-cache-dir essentia-tensorflow

# 4) Copy your application code into the container
COPY . /app

# 5) Essentia environment variables
ENV ESSENTIA_MODELS_DIR=/app/models
# PYTHONPATH is usually not needed when using python:slim images as pip installs into the correct path.
# You can often remove this line.
# ENV PYTHONPATH=/usr/local/lib/python3/dist-packages

# Create necessary directories for runtime if they don't exist
RUN mkdir -p /app/temp_audio

# Expose the port Flask will run on
EXPOSE 8000

# 6) Define the command to run the application
CMD ["sh", "-c", "if [ \"$SERVICE_TYPE\" = \"celery\" ]; then celery -A app.celery worker --loglevel=info; else python3 app.py; fi"]
