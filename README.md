# AudioMuse-AI
AutomAudioMuse AI: Leverages Essentia for deep audio analysis and AI-powered clustering to create smart, genre-based playlists within Jellyfinatic APlaylist creation container for Jellyfin


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
