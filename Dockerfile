# Start with the latest stable Debian base image (e.g., Bookworm, the current stable)
FROM debian:bookworm-slim

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install packages using apt-get (standard for Debian)
# debian:bookworm-slim is a very minimal image, so many base tools might be missing.
# We'll re-add build-essential and possibly other common tools if they were implicitly present in ubuntu:22.04
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
    # Added dependencies for psycopg2-binary
    libpq-dev \
    gcc \
    build-essential \
    # Clean up apt caches to keep the image small
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Using pip3, as it's common for Debian/Ubuntu
RUN pip3 install --no-cache-dir numpy==1.26.4

RUN pip3 install --no-cache-dir \
    Flask \
    Flask-Cors \
    redis \
    requests \
    scikit-learn \
    rq \
    pyyaml \
    six \
    psycopg2-binary

RUN pip3 install --no-cache-dir essentia-tensorflow

COPY . /app

# Default Python path for Debian/Ubuntu based systems
ENV PYTHONPATH=/usr/local/lib/python3/dist-packages:/app

EXPOSE 8000

WORKDIR /workspace
CMD ["sh", "-c", "if [ \"$SERVICE_TYPE\" = \"worker\" ]; then python3 /app/rq_worker.py; else python3 /app/app.py; fi"]