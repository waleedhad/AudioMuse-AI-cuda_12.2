import os
import shutil
import sqlite3
import requests
from collections import defaultdict
import numpy as np
from flask import Flask, jsonify, request, render_template, g
from flask_cors import CORS
from celery import Celery
from celery.result import AsyncResult
from contextlib import closing
import logging

# Import your existing analysis functions
from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor, TensorflowPredictMusiCNN, TensorflowPredict2D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Your existing config (e.g., JELLYFIN_URL, TEMP_DIR, DB_PATH, etc.)
# This should be a separate config.py file with constants.
from config import (
    CELERY_BROKER_URL, CELERY_RESULT_BACKEND, TEMP_DIR, DB_PATH,
    JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, # HEADERS will be generated per request/task
    NUM_RECENT_ALBUMS, TOP_N_MOODS, MOOD_LABELS, EMBEDDING_MODEL_PATH,
    PREDICTION_MODEL_PATH, MAX_SONGS_PER_ARTIST, MAX_SONGS_PER_CLUSTER,
    PCA_ENABLED, CLUSTER_ALGORITHM, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, NUM_CLUSTERS, MAX_DISTANCE
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for cross-origin requests

# --- Celery App Setup ---
celery = Celery(
    app.name,
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

celery.conf.update(
    result_serializer='pickle',
    accept_content=['json', 'pickle'],
    # You can add task routing here if you want specific tasks to go to specific queues/workers
    # task_routes = {
    #     'app.run_analysis_task': {'queue': 'analysis_queue'},
    #     'app.run_clustering_task': {'queue': 'clustering_queue'},
    # }
)

# Set Flask application context for Celery tasks.
# This ensures that Celery tasks can properly interact with Flask-specific
# constructs like `g` for database connections.
class ContextTask(celery.Task):
    def __call__(self, *args, **kwargs):
        with app.app_context():
            return self.run(*args, **kwargs)

celery.Task = ContextTask

# --- Status DB Setup ---
STATUS_DB_PATH = "status_db.sqlite"

def init_status_db():
    try:
        with closing(sqlite3.connect(STATUS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute('''CREATE TABLE IF NOT EXISTS analysis_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE,
                    task_type TEXT, -- NEW: To distinguish between analysis and clustering tasks
                    status TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
                conn.commit()
        logger.info(f"Status database initialized at {STATUS_DB_PATH}")
    except sqlite3.Error as e:
        logger.error(f"Error initializing status database: {e}")

# Ensure status DB is initialized on app startup
with app.app_context():
    init_status_db()

def get_status_db():
    if 'status_db' not in g:
        try:
            g.status_db = sqlite3.connect(STATUS_DB_PATH)
            g.status_db.row_factory = sqlite3.Row # Return rows as dictionaries
        except sqlite3.Error as e:
            logger.error(f"Error connecting to status database: {e}")
            raise
    return g.status_db

@app.teardown_appcontext
def close_status_db(exception):
    status_db = g.pop('status_db', None)
    if status_db is not None:
        try:
            status_db.close()
        except sqlite3.Error as e:
            logger.error(f"Error closing status database connection: {e}")

def save_analysis_task_id(task_id, task_type, status="PENDING"):
    conn = get_status_db()
    cur = conn.cursor()
    # Using 'task_id' as the unique key, allowing multiple task entries.
    # We might want to track the *last* task of each type, or just log all.
    # For simplicity, let's keep it to replace the single entry for the last task.
    # If you need to track a history of tasks, change 'id' to be a primary key that auto-increments
    # and remove 'INSERT OR REPLACE' to just 'INSERT'
    cur.execute("INSERT OR REPLACE INTO analysis_status (id, task_id, task_type, status) VALUES (1, ?, ?, ?)", (task_id, task_type, status))
    conn.commit()
    logger.info(f"Saved {task_type} task ID {task_id} with status {status} to status DB.")


def get_last_analysis_task_id(): # Renamed to be more general for any last task
    conn = get_status_db()
    cur = conn.cursor()
    cur.execute("SELECT task_id, task_type, status FROM analysis_status ORDER BY timestamp DESC LIMIT 1")
    row = cur.fetchone()
    return dict(row) if row else None

# --- Core Utility Functions (Modified to accept headers) ---

def clean_temp():
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            logger.info(f"Cleaned temporary directory: {TEMP_DIR}")
        os.makedirs(TEMP_DIR, exist_ok=True)
        logger.info(f"Ensured temporary directory exists: {TEMP_DIR}")
    except OSError as e:
        logger.error(f"Error cleaning/creating temp directory {TEMP_DIR}: {e}")

def init_db():
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute('''CREATE TABLE IF NOT EXISTS score (
                    item_id TEXT PRIMARY KEY, title TEXT, author TEXT,
                    tempo REAL, key TEXT, scale TEXT, mood_vector TEXT
                )''')
                cur.execute('''CREATE TABLE IF NOT EXISTS playlist (
                    playlist TEXT, item_id TEXT, title TEXT, author TEXT
                )''')
                conn.commit()
        logger.info(f"Main database initialized at {DB_PATH}")
    except sqlite3.Error as e:
        logger.error(f"Error initializing main database at {DB_PATH}: {e}")

def get_recent_albums(jellyfin_url, jellyfin_user_id, headers, limit=NUM_RECENT_ALBUMS):
    url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {
        "IncludeItemTypes": "MusicAlbum",
        "SortBy": "DateCreated",
        "SortOrder": "Descending",
        "Limit": limit,
        "Recursive": True,
    }
    try:
        logger.info(f"Fetching recent albums from Jellyfin: {url}")
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        albums = r.json().get("Items", [])
        logger.info(f"Found {len(albums)} recent albums.")
        return albums
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get albums from Jellyfin: {e}")
        return []

def get_tracks_from_album(jellyfin_url, jellyfin_user_id, album_id, headers):
    url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {"ParentId": album_id, "IncludeItemTypes": "Audio"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        tracks = r.json().get("Items", [])
        logger.info(f"Found {len(tracks)} tracks for album {album_id}")
        return tracks
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get tracks from album {album_id}: {e}")
        return []

def download_track(jellyfin_url, item, headers):
    filename = f"{item['Name'].replace('/', '_').replace(':', '_')}-{item.get('AlbumArtist', 'Unknown')}.mp3"
    path = os.path.join(TEMP_DIR, filename)
    try:
        logger.info(f"Downloading track: {item['Name']} to {path}")
        r = requests.get(f"{jellyfin_url}/Items/{item['Id']}/Download", headers=headers, stream=True, timeout=300)
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Successfully downloaded track: {item['Name']}")
        return path
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed for {item['Name']}: {e}")
        return None
    except IOError as e:
        logger.error(f"File system error downloading {item['Name']}: {e}")
        return None

def predict_moods(file_path, top_n_moods):
    try:
        logger.info(f"Predicting moods for {file_path}")
        audio = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
        embedding_model = TensorflowPredictMusiCNN(
            graphFilename=EMBEDDING_MODEL_PATH, output="model/dense/BiasAdd"
        )
        embeddings = embedding_model(audio)
        model = TensorflowPredict2D(
            graphFilename=PREDICTION_MODEL_PATH,
            input="serving_default_model_Placeholder",
            output="PartitionedCall"
        )
        predictions = model(embeddings)[0]
        results = dict(zip(MOOD_LABELS, predictions))
        moods = {label: float(score) for label, score in sorted(results.items(), key=lambda x: -x[1])[:top_n_moods]}
        logger.info(f"Mood prediction complete for {file_path}")
        return moods
    except Exception as e:
        logger.error(f"Error predicting moods for {file_path}: {e}")
        raise

def analyze_track_features(file_path, top_n_moods): # Renamed to avoid confusion with task
    logger.info(f"Analyzing track features: {file_path}")
    loader = MonoLoader(filename=file_path)
    audio = loader()
    tempo, _, _, _, _ = RhythmExtractor2013()(audio)
    key, scale, _ = KeyExtractor()(audio)
    moods = predict_moods(file_path, top_n_moods)
    logger.info(f"Track feature analysis complete for {file_path}. Tempo: {tempo}, Key: {key} {scale}")
    return tempo, key, scale, moods

def track_exists(item_id):
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute("SELECT 1 FROM score WHERE item_id=?", (item_id,))
                return cur.fetchone() is not None
    except sqlite3.Error as e:
        logger.error(f"Error checking if track {item_id} exists: {e}")
        return False

def save_track_analysis(item_id, title, author, tempo, key, scale, moods):
    mood_str = ','.join(f"{k}:{v:.3f}" for k, v in moods.items())
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute("INSERT OR IGNORE INTO score VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (item_id, title, author, tempo, key, scale, mood_str))
                conn.commit()
        logger.info(f"Saved analysis for {title} (ID: {item_id})")
    except sqlite3.Error as e:
        logger.error(f"Error saving analysis for {item_id}: {e}")

def score_vector(row):
    tempo = float(row[3]) if row[3] is not None else 0.0
    mood_str = row[6] or ""

    tempo_norm = (tempo - 40) / (200 - 40)
    tempo_norm = np.clip(tempo_norm, 0.0, 1.0)

    mood_scores = np.zeros(len(MOOD_LABELS))
    if mood_str:
        for pair in mood_str.split(","):
            if ":" not in pair:
                continue
            label, score = pair.split(":")
            if label in MOOD_LABELS:
                try:
                    mood_scores[MOOD_LABELS.index(label)] = float(score)
                except ValueError:
                    logger.warning(f"Could not convert mood score '{score}' to float for label '{label}'. Skipping.")
                    continue

    full_vector = [tempo_norm] + list(mood_scores)
    return full_vector

def get_all_tracks():
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute("SELECT * FROM score")
                rows = cur.fetchall()
        logger.info(f"Retrieved {len(rows)} tracks from database.")
        return rows
    except sqlite3.Error as e:
        logger.error(f"Error getting all tracks from database: {e}")
        return []

def name_cluster(centroid_scaled_vector, pca_model=None):
    if pca_model is not None:
        scaled_vector = pca_model.inverse_transform([centroid_scaled_vector])[0]
    else:
        scaled_vector = centroid_scaled_vector

    tempo_norm = scaled_vector[0]
    mood_values = scaled_vector[1:]

    tempo = tempo_norm * (200 - 40) + 40

    if tempo < 80:
        tempo_label = "Slow"
    elif tempo < 130:
        tempo_label = "Medium"
    else:
        tempo_label = "Fast"

    if len(mood_values) == 0 or np.all(mood_values == 0):
        mood_names = ["Unknown"]
    else:
        top_indices = np.argsort(mood_values)[::-1][:3]
        mood_names = [MOOD_LABELS[i] for i in top_indices if i < len(MOOD_LABELS)]

    mood_part = "_".join(mood_names).title()
    full_name = f"{mood_part}_{tempo_label}"

    top_mood_scores = {MOOD_LABELS[i]: mood_values[i] for i in top_indices if i < len(MOOD_LABELS)}
    extra_info = {"tempo_bpm": round(tempo, 2)}

    return full_name, {**top_mood_scores, **extra_info}

def update_playlist_table(playlists):
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute("DELETE FROM playlist")
                for name, cluster in playlists.items():
                    for item_id, title, author in cluster:
                        cur.execute("INSERT INTO playlist VALUES (?, ?, ?, ?)", (name, item_id, title, author))
                conn.commit()
        logger.info(f"Updated local playlist database with {len(playlists)} playlists.")
    except sqlite3.Error as e:
        logger.error(f"Error updating local playlist database: {e}")

def delete_old_automatic_playlists(jellyfin_url, jellyfin_user_id, headers):
    url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        logger.info("Attempting to delete old automatic Jellyfin playlists.")
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        for item in r.json().get("Items", []):
            if "_automatic" in item.get("Name", ""):
                del_url = f"{jellyfin_url}/Items/{item['Id']}"
                del_resp = requests.delete(del_url, headers=headers, timeout=10)
                if del_resp.ok:
                    logger.info(f"🗑️ Deleted old Jellyfin playlist: {item['Name']}")
                else:
                    logger.warning(f"Failed to delete old Jellyfin playlist {item['Name']}: {del_resp.status_code} {del_resp.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to clean old Jellyfin playlists: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during old playlist cleanup: {e}")

def create_or_update_playlists_on_jellyfin(jellyfin_url, jellyfin_user_id, headers, playlists, cluster_centers, pca_model=None):
    delete_old_automatic_playlists(jellyfin_url, jellyfin_user_id, headers)
    if not playlists:
        logger.info("No playlists to create on Jellyfin.")
        return

    for base_name, cluster in playlists.items():
        if not cluster:
            logger.warning(f"Skipping empty cluster: {base_name}")
            continue

        chunks = [cluster[i:i+MAX_SONGS_PER_CLUSTER] for i in range(0, len(cluster), MAX_SONGS_PER_CLUSTER)]
        
        for idx, chunk in enumerate(chunks, 1):
            playlist_name = f"{base_name}_automatic_{idx}" if len(chunks) > 1 else f"{base_name}_automatic"
            item_ids = [item_id for item_id, _, _ in chunk]
            
            if not item_ids:
                logger.warning(f"Skipping creation of empty playlist chunk for {playlist_name}")
                continue
            
            body = {"Name": playlist_name, "Ids": item_ids, "UserId": jellyfin_user_id}
            try:
                logger.info(f"Creating/updating Jellyfin playlist: {playlist_name} with {len(item_ids)} tracks.")
                r = requests.post(f"{jellyfin_url}/Playlists", headers=headers, json=body, timeout=30)
                r.raise_for_status()
                
                centroid_info = cluster_centers.get(base_name, {})
                top_moods = {k: v for k, v in centroid_info.items() if k in MOOD_LABELS}
                extra_info = {k: v for k, v in centroid_info.items() if k not in MOOD_LABELS}

                centroid_str = ", ".join(f"{k}: {v:.2f}" for k, v in top_moods.items())
                extras_str = ", ".join(f"{k}: {v:.2f}" for k, v in extra_info.items())

                logger.info(f"✅ Created playlist {playlist_name} with {len(item_ids)} tracks (Centroid: {centroid_str} | {extras_str})")

            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Failed to create/update Jellyfin playlist {playlist_name}: {e}. Response: {e.response.text if e.response else 'N/A'}")
            except Exception as e:
                logger.error(f"❌ An unexpected exception occurred creating {playlist_name}: {e}")

# --- Celery Tasks (Separated) ---

@celery.task(bind=True, name='app.run_analysis_task')
def run_analysis_task(self, jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods):
    """
    Celery task to download new albums/tracks from Jellyfin and analyze their features.
    It does NOT perform clustering or playlist generation.
    """
    task_headers = {"X-Emby-Token": jellyfin_token}

    self.update_state(state='PENDING', meta={'progress': 0, 'status': 'Starting analysis...', 'log_output': [], 'current_album': 'N/A', 'current_album_idx': 0, 'total_albums': 0})
    log_messages = []
    
    def log_and_update(message, progress, current_album=None, current_album_idx=0, total_albums=0):
        logger.info(message)
        log_messages.append(message)
        if len(log_messages) > 100: # Keep log messages manageable
            log_messages.pop(0)
        self.update_state(state='PROGRESS', meta={
            'progress': progress,
            'status': message,
            'log_output': log_messages,
            'current_album': current_album,
            'current_album_idx': current_album_idx,
            'total_albums': total_albums
        })

    try:
        log_and_update("🚀 Starting audio analysis and feature extraction...", 0)
        clean_temp()
        init_db() # Ensure main DB is ready for saving analysis

        albums = get_recent_albums(jellyfin_url, jellyfin_user_id, task_headers, limit=num_recent_albums)
        if not albums:
            log_and_update("⚠️ No new albums to analyze found.", 10)
        else:
            total_albums = len(albums)
            for idx, album in enumerate(albums, 1):
                album_progress_weight = 0.9 # Allocate 90% of progress to analysis
                current_album_base_progress = 10 + int(album_progress_weight * 80 * ((idx - 1) / total_albums))
                
                log_and_update(f"🎵 Album: {album.get('Name', 'Unknown Album')} ({idx}/{total_albums})",
                               current_album_base_progress,
                               current_album=album.get('Name', 'Unknown Album'), current_album_idx=idx, total_albums=total_albums)
                
                tracks = get_tracks_from_album(jellyfin_url, jellyfin_user_id, album['Id'], task_headers)
                for track_idx, item in enumerate(tracks, 1):
                    track_progress_in_album = (track_idx / len(tracks)) if len(tracks) > 0 else 0
                    overall_progress = current_album_base_progress + int(album_progress_weight * 80 * (1 / total_albums) * track_progress_in_album)

                    log_and_update(f"   🎶 Analyzing track: {item.get('Name', 'Unknown Track')} by {item.get('AlbumArtist', 'Unknown')}",
                                   overall_progress,
                                   current_album=album.get('Name', 'Unknown Album'), current_album_idx=idx, total_albums=total_albums)
                    
                    if track_exists(item['Id']):
                        log_and_update(f"     ⏭️ Skipping (already analyzed)",
                                       overall_progress,
                                       current_album=album.get('Name', 'Unknown Album'), current_album_idx=idx, total_albums=total_albums)
                        continue
                    
                    path = None
                    try:
                        path = download_track(jellyfin_url, item, task_headers)
                        if not path:
                            log_and_update(f"     ❌ Skipping track {item.get('Name', 'Unknown')} due to download failure.",
                                           overall_progress,
                                           current_album=album.get('Name', 'Unknown Album'), current_album_idx=idx, total_albums=total_albums)
                            continue
                        
                        tempo, key, scale, moods = analyze_track_features(path, top_n_moods) # Pass top_n_moods
                        save_track_analysis(item['Id'], item['Name'], item.get('AlbumArtist', 'Unknown'), tempo, key, scale, moods)
                        log_and_update(f"     ✅ Moods: {', '.join(f'{k}:{v:.2f}' for k,v in moods.items())}",
                                       overall_progress,
                                       current_album=album.get('Name', 'Unknown Album'), current_album_idx=idx, total_albums=total_albums)
                    except Exception as e:
                        logger.error(f"Error processing track {item.get('Name', 'Unknown')}: {e}", exc_info=True)
                        log_and_update(f"     ❌ Error analyzing track {item.get('Name', 'Unknown')}: {e}",
                                       overall_progress,
                                       current_album=album.get('Name', 'Unknown Album'), current_album_idx=idx, total_albums=total_albums)
                    finally:
                        if path and os.path.exists(path):
                            try:
                                os.remove(path)
                                logger.info(f"Deleted temporary track file: {path}")
                            except OSError as e:
                                logger.warning(f"Could not delete temporary track file {path}: {e}")
            clean_temp() # Final cleanup
        
        log_and_update("✅ Audio analysis and feature extraction complete!", 100)
        return {"status": "SUCCESS", "message": "Audio analysis and feature extraction complete!"}

    except Exception as e:
        logger.exception("❌ Audio analysis failed with an unhandled exception.")
        log_and_update(f"❌ Audio analysis failed: {str(e)}", 100)
        self.update_state(state='FAILURE', meta={'progress': 100, 'status': f'Analysis failed: {str(e)}', 'log_output': log_messages})
        return {"status": "FAILURE", "message": f"Audio analysis failed: {str(e)}"}

@celery.task(bind=True, name='app.run_clustering_task')
def run_clustering_task(self, jellyfin_url, jellyfin_user_id, jellyfin_token,
                        cluster_algorithm, num_clusters, dbscan_eps, dbscan_min_samples,
                        pca_components, max_distance):
    """
    Celery task to perform clustering on existing analyzed data and generate Jellyfin playlists.
    It does NOT perform audio analysis or downloading.
    """
    task_headers = {"X-Emby-Token": jellyfin_token}

    self.update_state(state='PENDING', meta={'progress': 0, 'status': 'Starting clustering...', 'log_output': []})
    log_messages = []

    def log_and_update_clustering(message, progress):
        logger.info(message)
        log_messages.append(message)
        if len(log_messages) > 100:
            log_messages.pop(0)
        self.update_state(state='PROGRESS', meta={
            'progress': progress,
            'status': message,
            'log_output': log_messages
        })

    try:
        log_and_update_clustering("📊 Starting mood-based clustering and playlist generation...", 0)
        
        rows = get_all_tracks()
        if len(rows) < 2:
            log_and_update_clustering("⚠️ Not enough analyzed tracks for clustering (minimum 2 required).", 100)
            return {"status": "SUCCESS", "message": "Clustering complete, but not enough data."}

        X_original = [score_vector(row) for row in rows]
        if not X_original or any(len(vec) != len(X_original[0]) for vec in X_original):
            log_and_update_clustering("⚠️ Inconsistent feature vector lengths in analyzed data. Skipping clustering.", 100)
            return {"status": "SUCCESS", "message": "Clustering complete, but inconsistent data."}

        X_scaled = np.array(X_original) # No explicit scaling needed beyond score_vector's internal normalization

        pca_model = None
        if pca_components > 0 and X_scaled.shape[1] > 1:
            try:
                n_components_pca = min(pca_components, X_scaled.shape[1])
                pca_model = PCA(n_components=n_components_pca)
                data_for_clustering = pca_model.fit_transform(X_scaled)
                log_and_update_clustering(f"PCA applied, reduced to {n_components_pca} components.", 20)
            except Exception as e:
                logger.error(f"Error during PCA: {e}")
                log_and_update_clustering(f"❌ Error during PCA: {e}. Proceeding without PCA.", 20)
                data_for_clustering = X_scaled
        else:
            data_for_clustering = X_scaled
        
        log_and_update_clustering(f"Using clustering algorithm: {cluster_algorithm}", 30)

        labels = []
        cluster_centers = {}
        raw_distances = np.zeros(len(data_for_clustering))

        if cluster_algorithm == "kmeans":
            k = num_clusters if num_clusters > 0 else max(1, len(rows) // MAX_SONGS_PER_CLUSTER)
            k = min(k, len(rows))
            if k == 0:
                log_and_update_clustering("⚠️ Not enough tracks for K-Means clustering.", 100)
                return {"status": "SUCCESS", "message": "Clustering complete, not enough tracks for K-Means."}
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(data_for_clustering)
            cluster_centers = {i: kmeans.cluster_centers_[i] for i in range(k)}
            centers_for_points = kmeans.cluster_centers_[labels]
            raw_distances = np.linalg.norm(data_for_clustering - centers_for_points, axis=1)
            log_and_update_clustering(f"K-Means clustering completed with {k} clusters.", 50)
        elif cluster_algorithm == "dbscan":
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            labels = dbscan.fit_predict(data_for_clustering)
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue
                indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                cluster_points = np.array([data_for_clustering[i] for i in indices])
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    for i in indices:
                        raw_distances[i] = np.linalg.norm(data_for_clustering[i] - center)
                    cluster_centers[cluster_id] = center
            log_and_update_clustering(f"DBSCAN clustering completed. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters.", 50)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {cluster_algorithm}")

        max_dist = raw_distances.max()
        normalized_distances = raw_distances / max_dist if max_dist > 0 else raw_distances

        track_info = []
        for row, label, vec, dist in zip(rows, labels, data_for_clustering, normalized_distances):
            if label == -1:
                continue
            track_info.append({"row": row, "label": label, "vector": vec, "distance": dist})

        filtered_clusters = defaultdict(list)
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            cluster_tracks = [t for t in track_info if t["label"] == cluster_id and t["distance"] <= max_distance]
            if not cluster_tracks:
                logger.info(f"Cluster {cluster_id} is empty after MAX_DISTANCE filtering. Skipping.")
                continue

            cluster_tracks.sort(key=lambda x: x["distance"])

            count_per_artist = defaultdict(int)
            selected = []
            for t in cluster_tracks:
                author = t["row"][2]
                if count_per_artist[author] < MAX_SONGS_PER_ARTIST:
                    selected.append(t)
                    count_per_artist[author] += 1
                if len(selected) >= MAX_SONGS_PER_CLUSTER:
                    break

            for t in selected:
                item_id, title, author = t["row"][0], t["row"][1], t["row"][2]
                filtered_clusters[cluster_id].append((item_id, title, author))

        named_playlists = defaultdict(list)
        playlist_centroids = {}
        for label, songs in filtered_clusters.items():
            if not songs:
                continue
            center = cluster_centers[label]
            name, top_scores = name_cluster(center, pca_model)
            named_playlists[name].extend(songs)
            playlist_centroids[name] = top_scores
        
        log_and_update_clustering("Clustering complete. Updating playlists on Jellyfin...", 80)
        update_playlist_table(named_playlists) # Update local DB with new playlists
        create_or_update_playlists_on_jellyfin(jellyfin_url, jellyfin_user_id, task_headers, named_playlists, playlist_centroids, pca_model)
        
        log_and_update_clustering("✅ All clustering and playlist generation done!", 100)
        return {"status": "SUCCESS", "message": "Clustering and playlist generation complete!"}

    except Exception as e:
        logger.exception("❌ Clustering failed with an unhandled exception.")
        log_and_update_clustering(f"❌ Clustering failed: {str(e)}", 100)
        self.update_state(state='FAILURE', meta={'progress': 100, 'status': f'Clustering failed: {str(e)}', 'log_output': log_messages})
        return {"status": "FAILURE", "message": f"Clustering failed: {str(e)}"}


# --- API Endpoints ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analysis/start', methods=['POST'])
def start_analysis():
    """
    Triggers the Celery task to perform audio analysis on recent albums.
    """
    data = request.json
    jellyfin_url = data.get('jellyfin_url', JELLYFIN_URL)
    jellyfin_user_id = data.get('jellyfin_user_id', JELLYFIN_USER_ID)
    jellyfin_token = data.get('jellyfin_token', JELLYFIN_TOKEN)
    num_recent_albums = int(data.get('num_recent_albums', NUM_RECENT_ALBUMS))
    top_n_moods = int(data.get('top_n_moods', TOP_N_MOODS))

    task = run_analysis_task.delay(
        jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods
    )
    save_analysis_task_id(task.id, "analysis", "PENDING")
    logger.info(f"Analysis task {task.id} started.")
    return jsonify({"task_id": task.id, "status": "PENDING", "task_type": "analysis"}), 202

@app.route('/api/clustering', methods=['POST'])
def trigger_clustering():
    """
    Triggers the Celery task to perform clustering on already analyzed data
    and update Jellyfin playlists.
    """
    data = request.json
    # Jellyfin credentials are still needed by the clustering task to create playlists
    jellyfin_url = data.get('jellyfin_url', JELLYFIN_URL)
    jellyfin_user_id = data.get('jellyfin_user_id', JELLYFIN_USER_ID)
    jellyfin_token = data.get('jellyfin_token', JELLYFIN_TOKEN)

    cluster_algorithm = data.get('clustering_method', CLUSTER_ALGORITHM)
    num_clusters = int(data.get('num_clusters', NUM_CLUSTERS))
    dbscan_eps = float(data.get('dbscan_eps', DBSCAN_EPS))
    dbscan_min_samples = int(data.get('dbscan_min_samples', DBSCAN_MIN_SAMPLES))
    pca_components = int(data.get('pca_components', 0))
    max_distance = float(data.get('max_distance', MAX_DISTANCE))

    task = run_clustering_task.delay(
        jellyfin_url, jellyfin_user_id, jellyfin_token,
        cluster_algorithm, num_clusters, dbscan_eps, dbscan_min_samples,
        pca_components, max_distance
    )
    save_analysis_task_id(task.id, "clustering", "PENDING")
    logger.info(f"Clustering task {task.id} started.")
    return jsonify({"task_id": task.id, "status": "PENDING", "task_type": "clustering"}), 202


@app.route('/api/analysis/status/<task_id>', methods=['GET'])
def analysis_status(task_id):
    """
    Fetches the status of any Celery task (analysis or clustering).
    """
    task = AsyncResult(task_id, app=celery)
    response = {
        'task_id': task.id,
        'state': task.state,
        'status': 'Processing...',
        'task_type': 'unknown' # Default, will be updated from DB if found
    }
    
    # Get info safely, providing defaults if task.info is not a dict or missing keys
    info = task.info if isinstance(task.info, dict) else {}
    response['progress'] = info.get('progress', 0)
    response['status'] = info.get('status', 'Initializing...')
    response['log_output'] = info.get('log_output', [])
    response['current_album'] = info.get('current_album', 'N/A')
    response['current_album_idx'] = info.get('current_album_idx', 0)
    response['total_albums'] = info.get('total_albums', 0)

    # Attempt to retrieve task type from status DB if available
    conn = get_status_db()
    cur = conn.cursor()
    cur.execute("SELECT task_type FROM analysis_status WHERE task_id = ?", (task_id,))
    db_row = cur.fetchone()
    if db_row:
        response['task_type'] = db_row['task_type']
    else: # If not in DB, try to infer from task name if possible
        if task.name == 'app.run_analysis_task':
            response['task_type'] = 'analysis'
        elif task.name == 'app.run_clustering_task':
            response['task_type'] = 'clustering'

    if task.state == 'PENDING':
        response['status'] = info.get('status', 'Task is pending or not yet started.')
    elif task.state == 'PROGRESS':
        pass
    elif task.state == 'SUCCESS':
        response['status'] = info.get('status', f'{response["task_type"].capitalize()} complete!')
        save_analysis_task_id(task_id, response['task_type'], "SUCCESS")
    elif task.state == 'FAILURE':
        response['status'] = info.get('status', f'{response["task_type"].capitalize()} failed: {str(task.info)}')
        save_analysis_task_id(task_id, response['task_type'], "FAILURE")
    elif task.state == 'REVOKED':
        response['status'] = info.get('status', f'{response["task_type"].capitalize()} revoked.')
        save_analysis_task_id(task_id, response['task_type'], "REVOKED")
    else:
        response['status'] = f'Unknown state: {task.state}'
    
    return jsonify(response)


@app.route('/api/analysis/cancel/<task_id>', methods=['POST'])
def cancel_analysis(task_id):
    task = AsyncResult(task_id, app=celery)
    if task.state in ['PENDING', 'STARTED', 'PROGRESS']:
        task.revoke(terminate=True)
        # Determine task type to save status correctly
        task_type = 'unknown'
        conn = get_status_db()
        cur = conn.cursor()
        cur.execute("SELECT task_type FROM analysis_status WHERE task_id = ?", (task_id,))
        db_row = cur.fetchone()
        if db_row:
            task_type = db_row['task_type']
        else:
            if task.name == 'app.run_analysis_task': task_type = 'analysis'
            elif task.name == 'app.run_clustering_task': task_type = 'clustering'

        save_analysis_task_id(task_id, task_type, "REVOKED")
        logger.info(f"Task {task_id} (type: {task_type}) cancelled.")
        return jsonify({"message": "Task cancelled.", "task_id": task_id, "task_type": task_type}), 200
    else:
        logger.warning(f"Attempted to cancel task {task_id} which is in state {task.state}. Cannot cancel.")
        return jsonify({"message": "Task cannot be cancelled in its current state.", "state": task.state}), 400

@app.route('/api/analysis/last_task', methods=['GET'])
def get_last_analysis_status():
    last_task = get_last_analysis_task_id()
    if last_task:
        task_id = last_task['task_id']
        task = AsyncResult(task_id, app=celery)
        response = {
            'task_id': task_id,
            'state': task.state,
            'status': task.info.get('status', last_task['status']),
            'task_type': last_task['task_type']
        }
        return jsonify(response), 200
    return jsonify({"task_id": None, "status": "NO_PREVIOUS_TASK"}), 200


@app.route('/api/config', methods=['GET'])
def get_config():
    # Expose current config values to the frontend (excluding sensitive tokens)
    return jsonify({
        "jellyfin_url": JELLYFIN_URL,
        "jellyfin_user_id": JELLYFIN_USER_ID,
        "num_recent_albums": NUM_RECENT_ALBUMS,
        "max_distance": MAX_DISTANCE,
        "max_songs_per_cluster": MAX_SONGS_PER_CLUSTER,
        "max_songs_per_artist": MAX_SONGS_PER_ARTIST,
        "cluster_algorithm": CLUSTER_ALGORITHM,
        "pca_enabled": PCA_ENABLED, # Will be determined by pca_components > 0
        "dbscan_eps": DBSCAN_EPS,
        "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
        "num_clusters": NUM_CLUSTERS,
        "top_n_moods": TOP_N_MOODS,
        "mood_labels": MOOD_LABELS,
    })

@app.route('/api/playlists', methods=['GET'])
def get_playlists():
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            with closing(conn.cursor()) as cur:
                cur.execute("SELECT playlist, item_id, title, author FROM playlist ORDER BY playlist, title")
                rows = cur.fetchall()

        playlists = defaultdict(list)
        for row in rows:
            playlists[row['playlist']].append({
                "item_id": row['item_id'],
                "title": row['title'],
                "author": row['author']
            })
        
        logger.info(f"Retrieved {len(playlists)} playlists from local DB.")
        return jsonify(dict(playlists)), 200
    except sqlite3.Error as e:
        logger.error(f"Error retrieving playlists from database: {e}")
        return jsonify({"status": "error", "message": f"Failed to retrieve playlists: {e}"}), 500

if __name__ == '__main__':
    init_db()
    os.makedirs(TEMP_DIR, exist_ok=True)
    logger.info(f"Flask app starting. Main DB at {DB_PATH}, Temp dir at {TEMP_DIR}")
    
    app.run(host='0.0.0.0', port=8000)
