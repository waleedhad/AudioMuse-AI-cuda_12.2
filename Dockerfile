FROM ubuntu:22.04

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install --no-cache-dir numpy==1.26.4

RUN pip3 install --no-cache-dir \
    Flask \
    Flask-Cors \
    celery \
    redis \
    requests \
    scikit-learn \
    pyyaml \
    six

RUN pip3 install --no-cache-dir essentia-tensorflow

COPY . /app

ENV PYTHONPATH=/usr/local/lib/python3/dist-packages:/app


EXPOSE 8000

WORKDIR /workspace

CMD ["sh", "-c", "if [ \"$SERVICE_TYPE\" = \"celery\" ]; then celery -A app.celery worker --pool=solo --loglevel=debug; else python3 /app/app.py; fi"]
