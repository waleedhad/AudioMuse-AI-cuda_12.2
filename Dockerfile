FROM ubuntu:22.04

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Clean apt cache and update package lists 
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get update -o Acquire::Retries=5 -o Acquire::Timeout=30

# Install system dependencies, removed libtag1v5 as a potential source of 404s
RUN apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libfftw3-3 libyaml-0-2 libsamplerate0 \
    ffmpeg wget git vim \
    redis-tools curl \
    supervisor \
    strace \
    procps \
    iputils-ping \
    libopenblas-dev \
    liblapack-dev \
    libpq-dev \
    gcc \
    g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip to a newer version that supports necessary flags and is more robust
RUN pip3 install --no-cache-dir --upgrade pip

# Install Python packages, with conditional TensorFlow installation for ARM
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      pip3 install --no-cache-dir \
        Flask \
        Flask-Cors \
        redis \
        requests \
        scikit-learn \
        rq \
        pyyaml \
        six \
        voyager \
        psycopg2-binary \
        ftfy \
        flasgger \
        sqlglot \
        google-generativeai \
        tensorflow-aarch64==2.15.0 \
        librosa; \
    else \
      pip3 install --no-cache-dir \
        Flask \
        Flask-Cors \
        redis \
        requests \
        scikit-learn \
        rq \
        pyyaml \
        six \
        voyager \
        psycopg2-binary \
        ftfy \
        flasgger \
        sqlglot \
        google-generativeai \
        tensorflow==2.15.0 \
        librosa; \
    fi

# Removed essentia-tensorflow as it's no longer used

# Create the model directory
RUN mkdir -p /app/model

# Download models from the GitHub release (corrected URL)
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
