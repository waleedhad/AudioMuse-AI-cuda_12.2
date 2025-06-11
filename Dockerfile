FROM ubuntu:22.04

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update -o Acquire::Retries=5 -o Acquire::Timeout=30 && \
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
    psycopg2-binary \
    ftfy \
    flasgger \
    google-generativeai

RUN pip3 install --no-cache-dir essentia-tensorflow

# Create the model directory
RUN mkdir -p /app/model

# Download models from the GitHub release
RUN wget -q -P /app/model \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/audioset-vggish-3.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/danceability-audioset-vggish-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/danceability-msd-musicnn-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_aggressive-audioset-vggish-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_aggressive-msd-musicnn-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_happy-audioset-vggish-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_happy-msd-musicnn-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_party-audioset-vggish-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_party-msd-musicnn-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_relaxed-audioset-vggish-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_relaxed-msd-musicnn-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_sad-audioset-vggish-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_sad-msd-musicnn-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/msd-msd-musicnn-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/msd-musicnn-1.pb

COPY . /app

ENV PYTHONPATH=/usr/local/lib/python3/dist-packages:/app


EXPOSE 8000

WORKDIR /workspace
CMD ["sh", "-c", "if [ \"$SERVICE_TYPE\" = \"worker\" ]; then python3 /app/rq_worker.py; else python3 /app/app.py; fi"]
