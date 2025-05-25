# AudioMuse-AI
AutomAudioMuse AI: Leverages Essentia for deep audio analysis and AI-powered clustering to create smart, genre-based playlists within Jellyfinatic APlaylist creation container for Jellyfin

# Configuration parameter
| Parameter               | Description                                 | Default Value                       |
| ----------------------- | ------------------------------------------- | ----------------------------------- |
| `JELLYFIN_URL`          | (Required) Your Jellyfin server's full URL  | `http://YOUR_JELLYFIN_IP:8096`      |
| `JELLYFIN_USER_ID`      | (Required) Jellyfin User ID (K8s Secret)    | *(N/A - from Secret)*               |
| `JELLYFIN_TOKEN`        | (Required) Jellyfin API Token (K8s Secret)  | *(N/A - from Secret)*               |
| `CELERY_BROKER_URL`     | URL for Celery broker                       | `redis://localhost:6379/0`          |
| `CELERY_RESULT_BACKEND` | Celery result backend                       | `redis://localhost:6379/0`          |
| `TEMP_DIR`              | Temp directory for audio files              | `/app/temp_audio`                   |
| `DB_PATH`               | SQLite DB path for analysis data            | `/workspace/db.sqlite`              |
| `STATUS_DB_PATH`        | SQLite DB path for task status              | `/workspace/status_db.sqlite`       |
| `EMBEDDING_MODEL_PATH`  | Path to MusiCNN model                       | `models/msd-musicnn-1.pb`           |
| `PREDICTION_MODEL_PATH` | Path to mood prediction model               | `models/mood_acoustic.pb`           |
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
| `MOOD_LABELS`           | Mood labels to predict                      | `"energetic,calm,happy,sad,...etc"` |
