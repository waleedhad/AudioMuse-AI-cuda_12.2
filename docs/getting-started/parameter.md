---
layout: default
title: Parameter
parent: Getting Started
nav_order: 2
---
# **Configuration Parameters**

These are the parameters accepted for this script. You can pass them as environment variables using, for example, /deployment/deployment.yaml in this repository.

The **mandatory** parameter that you need to change from the example are this:
| Parameter               | Description                                 | Default Value                       |
| ----------------------- | ------------------------------------------- | ----------------------------------- |
| `JELLYFIN_URL`          | (Required) Your Jellyfin server's full URL  | `http://YOUR_JELLYFIN_IP:8096`      |
| `JELLYFIN_USER_ID`      | (Required) Jellyfin User ID.| *(N/A - from Secret)*    |
| `JELLYFIN_TOKEN`        | (Required) Jellyfin API Token.| *(N/A - from Secret)*    |
| `POSTGRES_USER`         | (Required) PostgreSQL username.| *(N/A - from Secret)*    |
| `POSTGRES_PASSWORD`     | (Required) PostgreSQL password.| *(N/A - from Secret)*    |
| `POSTGRES_DB`           | (Required) PostgreSQL database name.| *(N/A - from Secret)*    |
| `POSTGRES_HOST`         | (Required) PostgreSQL host.| `postgres-service.playlist` |
| `POSTGRES_PORT`         | (Required) PostgreSQL port.| `5432`                      |
| `REDIS_URL`             | (Required) URL for Redis.| `redis://redis-service.playlist:6379/0` |
| `GEMINI_API_KEY`        | (Required if `AI_MODEL_PROVIDER` is GEMINI) Your Google Gemini API Key. | *(N/A - from Secret)* |

These parameter can be leave as it is:

| Parameter               | Description                                 | Default Value                       |
| ----------------------- | ------------------------------------------- | ----------------------------------- |
| `TEMP_DIR`              | Temp directory for audio files              | `/app/temp_audio`                   |


This are the default parameters on wich the analysis or clustering task will be lunched. You will be able to change them to another value directly in the front-end:

| Parameter               | Description                                 | Default Value                       |
| ----------------------- | ------------------------------------------- | ----------------------------------- |
| **Analysis General** |                                                                             |                    |
| `NUM_RECENT_ALBUMS`       | Number of recent albums to scan (0 for all).                                | `2000`   |
| `TOP_N_MOODS`             | Number of top moods per track for feature vector.                           | `5`      |
| **Clustering General** |                                                                             |                    |
| `CLUSTER_ALGORITHM`       | Default clustering: `kmeans`, `dbscan`, `gmm`.                              | `kmeans` |
| `MAX_SONGS_PER_CLUSTER`   | Max songs per generated playlist segment.                                   | `40`     |
| `MAX_SONGS_PER_ARTIST`    | Max songs from one artist per cluster.                                      | `3`      |
| `MAX_DISTANCE`            | Normalized distance threshold for tracks in a cluster.                      | `0.5`    |
| `CLUSTERING_RUNS`         | Iterations for Monte Carlo evolutionary search.                             | `1000`   |
| **K-Means Ranges** |                                                                             |                    |
| `NUM_CLUSTERS_MIN`        | Min $K$ for K-Means.                                                        | `20`     |
| `NUM_CLUSTERS_MAX`        | Max $K$ for K-Means.                                                        | `60`     |
| **DBSCAN Ranges** |                                                                             |                    |
| `DBSCAN_EPS_MIN`          | Min epsilon for DBSCAN.                                                     | `0.1`    |
| `DBSCAN_EPS_MAX`          | Max epsilon for DBSCAN.                                                     | `0.5`    |
| `DBSCAN_MIN_SAMPLES_MIN`  | Min `min_samples` for DBSCAN.                                               | `5`      |
| `DBSCAN_MIN_SAMPLES_MAX`  | Max `min_samples` for DBSCAN.                                               | `20`     |
| **GMM Ranges** |                                                                             |                    |
| `GMM_N_COMPONENTS_MIN`    | Min components for GMM.                                                     | `20`     |
| `GMM_N_COMPONENTS_MAX`    | Max components for GMM.                                                     | `60`     |
| `GMM_COVARIANCE_TYPE`     | Covariance type for GMM (task uses `'full'`).                               | `full`   |
| **PCA Ranges** |                                                                             |                    |
| `PCA_COMPONENTS_MIN`      | Min PCA components (0 to disable).                                          | `0`      |
| `PCA_COMPONENTS_MAX`      | Max PCA components.                                                         | `5`     |
| **AI Naming (*)** |                                                                             |                    |
| `AI_MODEL_PROVIDER`       | AI provider: `OLLAMA`, `GEMINI`, or `NONE`.                                 | `GEMINI` |
| `OLLAMA_SERVER_URL`       | URL for your Ollama instance (if `AI_MODEL_PROVIDER` is OLLAMA).            | `http://<your-ip>11434/api/generate` |
| `OLLAMA_MODEL_NAME`       | Ollama model to use (if `AI_MODEL_PROVIDER` is OLLAMA).                     | `mistral:7b` |
| `GEMINI_MODEL_NAME`       | Gemini model to use (if `AI_MODEL_PROVIDER` is GEMINI).                     | `gemini-1.5-flash-latest` |
| `GEMINI_API_CALL_DELAY_SECONDS` | Seconds to wait between Gemini API calls to respect rate limits.          | `7`      |
| `PCA_COMPONENTS_MIN`      | Min PCA components (0 to disable).                                          | `0`      |
| `PCA_COMPONENTS_MAX`      | Max PCA components.                                                         | `5`     |

**(*)** For using GEMINI API you need to have a Google account, a free account can be used if needed. Instead if you want to self-host Ollama here you can find a deployment example:

* https://github.com/NeptuneHub/k3s-supreme-waffle/tree/main/ollama

