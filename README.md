# **AudioMuse-AI**

AudioMuse-AI is a Dockerized environment that brings smart playlist generation to [Jellyfin](https://jellyfin.org) using deep audio analysis via [Essentia](https://essentia.upf.edu/) with TensorFlow. All you need is in a container that you can deploy locally or on your Kubernetes cluster (tested on K3S). In this repo, you also have a /deployment/deployment.yaml example that you need to configure following the configuration parameter chapter.

The main scope of this application is testing the clustering algorithm. A front-end is provided for easy use. You can also find the stand-alone Python script in the Jellyfin-Essentia-Playlist repo.

**IMPORTANT:** This is an ALPHA open-source project I’m developing just for fun. All the source code is fully open and visible. It’s intended only for testing purposes, not for production environments. Please use it at your own risk. I cannot be held responsible for any issues or damages that may occur.

## **Table of Contents**

- [Workflow Overview](#workflow-overview)
- [Clustering Algorithm Deep Dive](#clustering-algorithm-deep-dive)
  - [1. K-Means](#1-k-means)
  - [2. DBSCAN](#2-dbscan)
  - [3. GMM (Gaussian Mixture Models)](#3-gmm-gaussian-mixture-models)
  - [AI Playlist Naming](#ai-playlist-naming)
- [Configuration Parameters](#configuration-parameters)
- [Kubernetes Deployment (K3S Example)](#kubernetes-deployment-k3s-example)
- [Local Deployment with Docker Compose](#local-deployment-with-docker-compose)
- [Docker Image Tagging Strategy](#docker-image-tagging-strategy)
- [Screenshots](#screenshots)
  - [Analysis task](#analysis-task)
  - [Clustering task](#clustering-task)
- [Key Technologies](#key-technologies)
- [Additional Documentation](#additional-documentation)
- [Future Possibilities](#future-possibilities)
- [Contributing](#contributing)

## **Workflow Overview**

This is the main workflow of how this algorithm works. For an easy way to use it, you will have a front-end reachable at **your\_ip:8000** with buttons to start/cancel the analysis (it can take hours depending on the number of songs in your Jellyfin library) and a button to create the playlist.

* **Initiate from Frontend:** Start an analysis via the Flask web UI.  
* **Job Queued on Redis:** Task is sent to Redis Queue.  
* **RQ Worker Processes:**  
  * Multiple worker containers are supported for parallel processing. It is suggested to have at least 2 workers (even on the same machine) because one runs the main process and the other runs subprocesses (so with only one worker, it can get stuck).  
  * Fetch metadata and download audio from Jellyfin.  
  * Analyze tracks using Essentia and TensorFlow.  
  * Store results in PostgreSQL.  
* **Flexible Clustering:** Re-cluster tracks anytime using stored analysis data.  
* **Jellyfin Playlist Creation:** Create playlists based on clustering results directly in Jellyfin.

**Persistence:** PostgreSQL database is used for persisting analyzed track metadata, generated playlist structures, and task status.

## **Clustering Algorithm Deep Dive**

AudioMuse-AI offers three algorithms, each suited for different scenarios when clustering music based on tempo and 5 mood scores. This clustering algorithm is executed multiple times (default 1000\) following an Evolutionary Monte Carlo approach: in this way, multiple configurations of parameters are tested, and the best ones are kept.

Here's an explanation of the pros and cons of the different algorithms:

### **1\. K-Means**

* **Best For:** Speed, simplicity, when clusters are roughly spherical and of similar size.  
* **Pros:** Very fast, scalable, clear "average" cluster profiles.  
* **Cons:** Requires knowing cluster count (K), struggles with irregular shapes, sensitive to outliers.

### **2\. DBSCAN**

* **Best For:** Discovering clusters of arbitrary shapes, handling outliers well, when the number of clusters is unknown.  
* **Pros:** No need to set K, finds varied shapes, robust to noise.  
* **Cons:** Sensitive to eps and min\_samples parameters, can struggle with varying cluster densities, no direct "centroids."

### **3\. GMM (Gaussian Mixture Models)**

* **Best For:** Modeling more complex, elliptical cluster shapes and when a probabilistic assignment of tracks to clusters is beneficial.  
* **Pros:** Flexible cluster shapes, "soft" assignments, model-based insights.  
* **Cons:** Requires setting number of components, computationally intensive (can be slow), sensitive to initialization.

**Recommendation:** Start with **K-Means** for general use due to its speed in the evolutionary search. Experiment with **GMM** for more nuanced results. Use **DBSCAN** if you suspect many outliers or highly irregular cluster shapes. Using a high number of runs (default 1000\) helps the integrated evolutionary algorithm to find a good solution.

### **AI Playlist Naming**

After the clustering algorithm has identified groups of similar songs, AudioMuse-AI can optionally use an AI model to generate creative, human-readable names for the resulting playlists. This replaces the default "Mood_Tempo" naming scheme with something more evocative.

1.  **Input to AI:** For each cluster, the system extracts key characteristics derived from the cluster's centroid (like predominant moods and tempo) and provides a sample list of songs from that cluster.
2.  **AI Model Interaction:** This information is sent to a configured AI model (either a self-hosted **Ollama** instance or **Google Gemini**) along with a carefully crafted prompt.
3.  **Prompt Engineering:** The prompt guides the AI to act as a music curator and generate a concise playlist name (15-35 characters) that reflects the mood, tempo, and overall vibe of the songs, while adhering to strict formatting rules (standard ASCII characters only, no extra text).
4.  **Output Processing:** The AI's response is cleaned to ensure it meets the formatting and length constraints before being used as the final playlist name (with the `_automatic` suffix appended later by the task runner).

This step adds a layer of creativity to the purely data-driven clustering process, making the generated playlists more appealing and easier to understand at a glance. The choice of AI provider and model is configurable via environment variables and the frontend.


## **Configuration Parameters**

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

## **Kubernetes Deployment (K3S Example)**

An example K8s deployment is provided in **deployments/deployment.yaml**. Start from it as a template.

The provided deployment will deploy on your cluster **pod:**
* **audiomuse-ai-worker:** The worker at the **minimum number of 2**, you can put more if you have more than 1 node;
* **audiomuse-ai-flask:** The api server plus the front-end;
* **postgres-deployment:** The database for saving the analysis of score, playlist created and status of the stad.;
* **redis-master:** Needed for Redis Queue, is hwere the worker get the task to run.

**service:**
* **audiomuse-ai-flask-service:** is a LoadBalancer service to direct access to your fronte-end. You can convert to ClusterIP if you want to add an ingress on top;
* **postgres-service:** is a ClusterIP service needed toreach PostgreSQL form the worker internally to the cluster;
* **redis-service:** is a ClusterIP service needed to reach Redis from the Worker internally to the cluster;

**secret:**
* **jellyfin-credentials:** which contain `api_token` and `user_id` of Jellyfin
* **postgres-credentials:** which contain `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`

**ConfigMap:**
* **audiomuse-ai-config:** which will contain your non-sensitive parameters. By default, it includes: `JELLYFIN_URL`, `POSTGRES_HOST`, `POSTGRES_PORT`, and `REDIS_URL`.

**PVC:**
* **postgres-pvc:** this is very important because will keep all the data of the database.


it will also create the **namespace** playlist and everything will be deployed in it.
**Before Deploying:** 
1. Edit the `jellyfin-credentials` and `postgres-credentials` Secrets with your actual values.
2. Edit the `JELLYFIN_URL` in the `audiomuse-ai-config` ConfigMap.
3. Review and ensure your `postgres-pvc` PersistentVolumeClaim configuration is appropriate for your environment to prevent data loss. Regularly backing up the database is also recommended.

Then you can easily deploy it with:
```
kubectl apply -f deployments/deployment.yaml
```

When deployed you can access to:
* **Front-end:** through the port exposed by your LoadBalancer. For exmaple in this templae case is **http://your-ip:8000**;
* **Flask service and swagger:** You will also have the swagger at **http://your-iè:8000/apidocs** with all the API example.

For a more stable use, I suggest editing the deployment container image to use the alpha tags, for example, ghcr.io/neptunehub/audiomuse-ai:0.2.1-alpha.

## **Local Deployment with Docker Compose**

For a quick local setup or for users not using Kubernetes, a `docker-compose.yaml` file is provided in the `deployment/` directory.

**Prerequisites:**
*   Docker and Docker Compose installed.

**Steps:**
1.  **Navigate to the `deployment` directory:**
    ```bash
    cd deployment
    ```
2.  **Review and Customize (Optional):**
    The `docker-compose.yaml` file is pre-configured with default credentials and settings suitable for local testing. You can edit environment variables within this file directly if needed (e.g., `JELLYFIN_URL`, `JELLYFIN_USER_ID`, `JELLYFIN_TOKEN`).
3.  **Start the Services:**
    ```bash
    docker compose up -d --scale audiomuse-ai-worker=2
    ```
    This command starts all services (Flask app, RQ workers, Redis, PostgreSQL) in detached mode (`-d`). The `--scale audiomuse-ai-worker=2` ensures at least two worker instances are running, which is recommended for the task processing architecture.
4.  **Access the Application:**
    Once the containers are up, you can access the web UI at `http://localhost:8000`.
5.  **Stopping the Services:**
    ```bash
    docker compose down
    ```
## **Docker Image Tagging Strategy**

Our GitHub Actions workflow automatically builds and pushes Docker images. Here's how our tags work:

* :**latest**  
  * Builds from the **main branch**.  
  * Represents the latest stable release.  
  * **Recommended for most users.**  
* :**devel**  
  * Builds from the **devel branch**.  
  * Contains features still in development, not fully tested, and they could not work.  
  * **Use only for development.**  
* :**vX.Y.Z** (e.g., :v0.1.4-alpha, :v1.0.0)  
  * Immutable tags created from **specific Git releases/tags**.  
  * Ensures you're running a precise, versioned build.  
  * **Use for reproducible deployments or locking to a specific version.**

**Important version**
* **0.1.5-alpha** - Last version with the use of Sqlite and Celery. Only 1 worker admitted.
  * To deploy you can download the source code from here and use the appropriate deployment.yaml example: https://github.com/NeptuneHub/AudioMuse-AI/releases/tag/v0.1.5-alpha

## Screenshots

Here are a few glimpses of AudioMuse AI in action (more can be found in /screnshot):

### Analysis task

![Screenshot of AudioMuse AI's web interface showing the progress of music analysis tasks.](screenshot/analysis_task.png "Audio analysis and task status.")

### Clustering task

**TBD**

## **Key Technologies**

AudioMuse AI is built upon a robust stack of open-source technologies:

* **Flask:** Provides the lightweight web interface for user interaction and API endpoints.  
* **Redis Queue (RQ):** A simple Python library for queueing jobs and processing them in the background with Redis. It handles the computationally intensive audio analysis and playlist generation, ensuring the web UI remains responsive.  
* [**Essentia-tensorflow**](https://essentia.upf.edu/) An open-source library for audio analysis, feature extraction, and music information retrieval. It's used here for:  
  * MonoLoader: Loading and resampling audio files.  
  * RhythmExtractor2013: Extracting tempo information.  
  * KeyExtractor: Determining the musical key and scale.  
  * TensorflowPredictMusiCNN & TensorflowPredict2D: Leveraging pre-trained TensorFlow models (like MusiCNN) for generating rich audio embeddings and predicting mood tags.  
* [**scikit-learn**](https://scikit-learn.org/) Utilized for machine learning algorithms:  
  * KMeans / DBSCAN: For clustering tracks based on their extracted features (tempo and mood vectors).  
  * PCA (Principal Component Analysis): Optionally used for dimensionality reduction before clustering, to improve performance or cluster quality.  
* **PostgreSQL:** A powerful, open-source relational database used for persisting:  
  * Analyzed track metadata (tempo, key, mood vectors).  
  * Generated playlist structures.  
  * Task status for the web interface.  
* [**Jellyfin API**](https://jellyfin.org/) Integrates directly with your Jellyfin server to fetch media, download audio, and create/manage playlists.  
* **MusiCNN embedding model** – Developed as part of the [AcousticBrainz project](https://acousticbrainz.org/), based on a convolutional neural network trained for music tagging and embedding.  
* **Mood prediction model** – A TensorFlow-based model trained to map MusiCNN embeddings to mood probabilities (you must provide or train your own compatible model).  
* **Docker / OCI-compatible Containers** – The entire application is packaged as a container, ensuring consistent and portable deployment across environments.

## **Additional Documentation**

In **audiomuse-ai/docs** you can find additional documentation for this project, in details:
* **api_doc.pdf** - Is a minimal documentation of the API in case you want to integrate it in your front-end (still work in progress)
* **docker_docs.pdf** - Is a step-by-step documentation if you want to deploy everything on your local machine (debian) becuase you don't have a K3S (or other kubernetes) cluster avaiable.
  * **audiomuse-ai/deployment/docker-compose.yaml** - is the docker compose file used in the docker_docs.pdf
  

## **Future Possibilities**

This MVP lays the groundwork for further development:

* 💡 **Integration into Music Clients:** Directly used in Music Player that interacts with Jellyfin media server, for playlist creation OR instant mix;  
* 🖥️ **Jellyfin Plugin:** Integration as a Jellyfin plugin to have only one and easy-to-use front-end.  
* 🔁 **Cross-Platform Sync** Export playlists to .m3u or sync to external platforms.

## **Contributing**

Contributions, issues, and feature requests are welcome\!  
This is an ALPHA early release, so expect bugs or functions that are still not implemented.
