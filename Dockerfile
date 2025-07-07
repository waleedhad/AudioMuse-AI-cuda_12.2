FROM ubuntu:22.04

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Reverting mirror change, relying on default Ubuntu mirrors which are generally more stable.
# RUN sed -i 's|http://archive.ubuntu.com|http://ports.ubuntu.com|g' /etc/apt/sources.list && \
#     sed -i 's|http://security.ubuntu.com|http://ports.ubuntu.com|g' /etc/apt/sources.list

# Clean apt cache and update package lists before installing to avoid stale data issues
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get update -o Acquire::Retries=5 -o Acquire::Timeout=30 && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libfftw3-3 libyaml-0-2 libtag1v5 libsamplerate0 \
    ffmpeg wget git vim \
    redis-tools curl \
    supervisor \
    strace \
    procps \
    iputils-ping \
    libopenblas-dev \
    liblapack-dev \
    # Added dependencies for psycopg2-binary
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install --no-cache-dir numpy==1.26.4

RUN pip3 install --no-cache-dir --break-system-packages \
    Flask \
    Flask-Cors \
    redis \
    requests \
    scikit-learn \
    rq \
    pyyaml \
    six \
    annoy \
    # Added psycopg2-binary for PostgreSQL connectivity
    psycopg2-binary \
    ftfy \
    flasgger \
    sqlglot \
    google-generativeai \
    tensorflow==2.15.0 \
    librosa

# Removed essentia-tensorflow as it's no longer used

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

# Copy the application code
COPY . /app

# Copy the supervisor configuration
COPY deplyment/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

ENV PYTHONPATH=/usr/local/lib/python3/dist-packages:/app

EXPOSE 8000

WORKDIR /workspace
CMD ["bash", "-c", "if [ \"$SERVICE_TYPE\" = \"worker\" ]; then echo 'Starting worker processes via supervisord...' && /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf; else echo 'Starting web service...' && python3 /app/app.py; fi"]
