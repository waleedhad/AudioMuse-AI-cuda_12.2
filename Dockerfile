FROM debian:bookworm-slim

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Add 'contrib' and 'non-free' components to sources.list and update
RUN echo "deb http://deb.debian.org/debian bookworm main contrib non-free" > /etc/apt/sources.list.d/debian.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main contrib non-free" >> /etc/apt/sources.list.d/debian.list && \
    echo "deb http://deb.debian.org/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list.d/debian.list && \
    apt-get update -o Acquire::Retries=5 -o Acquire::Timeout=30 && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libfftw3-dev libyaml-0-2 libtag1v5 libsamplerate0 \
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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

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
    # Added psycopg2-binary for PostgreSQL connectivity
    psycopg2-binary

RUN pip3 install --no-cache-dir essentia-tensorflow

COPY . /app

ENV PYTHONPATH=/usr/local/lib/python3/dist-packages:/app


EXPOSE 8000

WORKDIR /workspace
CMD ["sh", "-c", "if [ \"$SERVICE_TYPE\" = \"worker\" ]; then python3 /app/rq_worker.py; else python3 /app/app.py; fi"]
