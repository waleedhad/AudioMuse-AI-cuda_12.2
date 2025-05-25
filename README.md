# AudioMuse-AI
AutomAudioMuse-AI is a Dockerized environment that brings smart playlist generation to [Jellyfin](https://jellyfin.org) using deep audio analysis via [Essentia](https://essentia.upf.edu/) with TensorFlow. All you need is in a container that you can deploy locally or on your kubernetes cluster (tested on K3S). In this repo you also have a /deployment/deployment.yaml example that you need to configure followuing the configuration parameter chapter.

The main scope of this application is testing the clustering algorithm. A front-end is provided for an eaxy to use. You can also found the stand-alone python script in [`Jellyfin-Essentia-Playlist`](https://github.com/NeptuneHub/Jellyfin-Essentia-Playlist) repo.

**IMPORTANT:** This is an ALPHA open-source project I’m developing just for fun. All the source code is fully open and visible. It’s intended only for testing purpose, not for production environments. Please use it at your own risk. I cannot be held responsible for any issues or damages that may occur.


## 🔄 Workflow Overview

This is the main workflow on how this algorithm work. To have an easy way to use it you will have a front-end reachable to **yourip:8000** with the button to start/cancel the analysis (it can take hours depending from the number of songs in hour jellyfin library) and a button to create the playlist.

* Initiate from Frontend: Start an analysis via the Flask web UI.
* Job Queued on Redis: Task is sent to Redis.
* Celery Worker Processes:
  * Fetch metadata and download audio from Jellyfin
  * Analyze tracks using Essentia and TensorFlow
  * Store results in db.sqlite
* Flexible Clustering: Re-cluster tracks anytime using stored analysis data.
* Jellyfin Playlist Creation: Create playlists based on clustering results directly in Jellyfin.

**Persistence:** SQLite databases (db.sqlite, status_db.sqlite) are stored in mounted volumes to retain data between restarts.

## Configuration parameter

This are the parameter accepted for thi script, you can pass them as env value using as an example **/depoyment/deployment.yaml** in this repository.

The **mandatory** parameter that you need to change from the example are this:
| Parameter               | Description                                 | Default Value                       |
| ----------------------- | ------------------------------------------- | ----------------------------------- |
| `JELLYFIN_URL`          | (Required) Your Jellyfin server's full URL  | `http://YOUR_JELLYFIN_IP:8096`      |
| `JELLYFIN_USER_ID`      | (Required) Jellyfin User ID (K8s Secret)    | *(N/A - from Secret)*               |
| `JELLYFIN_TOKEN`        | (Required) Jellyfin API Token (K8s Secret)  | *(N/A - from Secret)*               |
| `CELERY_BROKER_URL`     | URL for Celery broker (your redis endpoint) | `redis://redis-service.playlist:6379/0`|
| `CELERY_RESULT_BACKEND` | Celery result backend (your redit endpoint) | `redis://redis-service.playlist:6379/0`|

This are parameter that you can leave as it is, the important things is that you mount **/workspace** container folder in a PVC or in an hostPath. This because the db with all the song analysis will be saved there.
| Parameter               | Description                                 | Default Value                       |
| ----------------------- | ------------------------------------------- | ----------------------------------- |
| `TEMP_DIR`              | Temp directory for audio files              | `/workspace/temp_audio`                   |
| `DB_PATH`               | SQLite DB path for analysis data            | `/workspace/db.sqlite`              |
| `STATUS_DB_PATH`        | SQLite DB path for task status              | `/workspace/status_db.sqlite`       |

This are the default parameters on wich the analysis or clustering task will be lunched. You will be able to change them to another value directly in the front-end
| Parameter               | Description                                 | Default Value                       |
| ----------------------- | ------------------------------------------- | ----------------------------------- |
| `NUM_RECENT_ALBUMS`     | Albums to fetch from Jellyfin               | `500`                               |
| `TOP_N_MOODS`           | Number of top moods for naming playlists    | `3`                                 |
| `NUM_CLUSTERS`          | KMeans: Number of clusters                  | `10`                                |
| `MAX_DISTANCE`          | Max normalized distance for track inclusion | `0.5`                               |
| `MAX_SONGS_PER_CLUSTER` | Max tracks in a playlist                    | `50`                                |
| `MAX_SONGS_PER_ARTIST`  | Max songs per artist per playlist           | `3`                                 |
| `CLUSTER_ALGORITHM`     | Clustering algorithm (`kmeans`, `dbscan`)   | `kmeans`                            |
| `PCA_ENABLED`           | Enable PCA (True/False)                     | `False`                             |
| `DBSCAN_EPS`            | DBSCAN epsilon param                        | `0.5`                               |
| `DBSCAN_MIN_SAMPLES`    | DBSCAN min samples param                    | `5`                                 |

## ☸️ Kubernetes Deployment (K3S Example)
An example K8s deployment is provided in **deployments/deployment.yaml** start from it as a template.

**Persistence Note:** SQLite databases are stored in /workspace and mounted to a persistent volume:

```
volumeMounts:
  - name: workspace-hostpath
    mountPath: /workspace
volumes:
  - name: workspace-hostpath
    hostPath:
      path: /mnt/workspace
```

**Before Deploying:**
* update the **mandatory** requirements like jellyfin-credentials Secret with real Jellyfin user_id and api_token. Set JELLYFIN_URL in the audiomuse-ai-config ConfigMap.
* Remember to configure the storage: Adjust /mnt/workspace to a persistent path or switch to a PVC.

then you can easy deploy it with:
```
kubectl apply -f deployments/deployment.yaml
Access the Flask service through the port exposed by your LoadBalancer or service type.
```
## 🛠️ Key Technologies
AudioMuse AI is built upon a robust stack of open-source technologies:

* **Flask:** Provides the lightweight web interface for user interaction and API endpoints.
* **Celery:** A distributed task queue that handles the computationally intensive audio analysis and playlist generation in the background, ensuring the web UI remains responsive. It uses Redis as its broker and result backend.
* [**Essentia-tensorflow**](https://essentia.upf.edu/) An open-source library for audio analysis, feature extraction, and music information retrieval. It's used here for:
  * MonoLoader: Loading and resampling audio files.
  * RhythmExtractor2013: Extracting tempo information.
  * KeyExtractor: Determining the musical key and scale.
  * TensorflowPredictMusiCNN & TensorflowPredict2D: Leveraging pre-trained TensorFlow models (like MusiCNN) for generating rich audio embeddings and predicting mood tags.
* [**scikit-learn**](https://scikit-learn.org/)  Utilized for machine learning algorithms:
  * KMeans / DBSCAN: For clustering tracks based on their extracted features (tempo and mood vectors).
  * PCA (Principal Component Analysis): Optionally used for dimensionality reduction before clustering, to improve performance or cluster quality.
* **SQLite:** A lightweight, file-based database used for persisting:
  * Analyzed track metadata (tempo, key, mood vectors).
  * Generated playlist structures.
  * Task status for the web interface.
* [**Jellyfin API**](https://jellyfin.org/) Integrates directly with your Jellyfin server to fetch media, download audio, and create/manage playlists.
* **MusiCNN embedding model** – Developed as part of the [AcousticBrainz project](https://acousticbrainz.org/), based on a convolutional neural network trained for music tagging and embedding.
* **Mood prediction model** – A TensorFlow-based model trained to map MusiCNN embeddings to mood probabilities (you must provide or train your own compatible model).
* **Docker / OCI-compatible Containers** – The entire application is packaged as a container, ensuring consistent and portable deployment across environments.

## 🚀 Future Possibilities
This MVP lays the groundwork for further development:

* 💡 **Integration into Music Clients:** Directly used in Music Player that interact with Jellyfin media server, for playlist creation OR istant mix;
* 🖥️ **Jellyfing Plugin:** Integration as a Jellyfin plugin to have only one and easy-to-use front-end
* 🔁 **Cross-Platform Sync** Export playlists to .m3u or sync to external platforms

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!
This is an ALPHA early release, so expect bug or function that are still not implemented.
