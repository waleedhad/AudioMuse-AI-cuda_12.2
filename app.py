import os
import shutil
import sqlite3
import requests
from collections import defaultdict
import numpy as np
from flask import Flask, jsonify, request, render_template, g
from celery import Celery
from celery.result import AsyncResult
from contextlib import closing # For proper SQLite connection handling
from flask_cors import CORS # Import CORS for cross-origin requests

# Import your existing analysis functions
# Ensure these paths are correct relative to the container's /app directory
from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor, TensorflowPredictMusiCNN, TensorflowPredict2D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Your existing config.py, which now reads from environment variables
from config import (
    JELLYFIN_USER_ID, JELLYFIN_URL, JELLYFIN_TOKEN, TEMP_DIR, DB_PATH, STATUS_DB_PATH,
    HEADERS, MAX_DISTANCE, MAX_SONGS_PER_CLUSTER, MAX_SONGS_PER_ARTIST, NUM_RECENT_ALBUMS,
    CLUSTER_ALGORITHM, PCA_ENABLED, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, NUM_CLUSTERS,
    MOOD_LABELS, TOP_N_MOODS, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH,
    CELERY_BROKER_URL, CELERY_RESULT_BACKEND # Ensure these are imported from config
)

# --- Flask App Setup ---
# static_folder='static' and static_url_path='' are important if Flask serves index.html
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app) # Enable CORS for all routes

# --- Celery Configuration ---
# Use values from config.py which are loaded from environment variables
app.config['CELERY_BROKER_URL'] = CELERY_BROKER_URL
app.config['CELERY_RESULT_BACKEND'] = CELERY_RESULT_BACKEND

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# --- CRITICAL CELERY SERIALIZATION FIX ---
# Set result serializer to pickle to handle complex exception objects
celery.conf.result_serializer = 'pickle'
celery.conf.accept_content = ['json', 'pickle'] # Allow both for flexibility
# Optional: if task arguments themselves are complex, also set task_serializer
# celery.conf.task_serializer = 'pickle'

# --- Debugging Celery Configuration ---
print(f"DEBUG: Celery Broker URL set to: {celery.conf.broker_url}")
print(f"DEBUG: Celery Result Backend set to: {celery.conf.result_backend}")
print(f"DEBUG: Celery Result Serializer set to: {celery.conf.result_serializer}")


# --- Status DB Setup ---
# Use the STATUS_DB_PATH from config.py (which is from environment variables)
# No need to re-assign STATUS_DB_PATH here if it's imported from config.
# The init_status_db function will use the global STATUS_DB_PATH.

def init_status_db():
    print(f"DEBUG: Initializing Status DB at: {STATUS_DB_PATH}") # Debug print
    try:
        with closing(sqlite3.connect(STATUS_DB_PATH)) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute('''CREATE TABLE IF NOT EXISTS analysis_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE,
                    status TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
                conn.commit()
        print(f"DEBUG: Status DB initialized successfully at: {STATUS_DB_PATH}")
    except sqlite3.OperationalError as e:
        print(f"ERROR: Could not initialize Status DB at {STATUS_DB_PATH}: {e}")
        # Depending on severity, you might want to exit or raise here
        raise

# Ensure status DB is initialized on app startup
with app.app_context():
    init_status_db()

def get_status_db():
    if 'status_db' not in g:
        try:
            g.status_db = sqlite3.connect(STATUS_DB_PATH)
            g.status_db.row_factory = sqlite3.Row # Return rows as dictionaries
        except sqlite3.OperationalError as e:
            print(f"ERROR: Could not connect to Status DB at {STATUS_DB_PATH}: {e}")
            raise
    return g.status_db

@app.teardown_appcontext
def close_status_db(exception):
    status_db = g.pop('status_db', None)
    if status_db is not None:
        status_db.close()

def save_analysis_task_id(task_id, status="PENDING"):
    try:
        conn = get_status_db()
        cur = conn.cursor()
        # Using INSERT OR REPLACE to always have only one entry for the last task
        cur.execute("INSERT OR REPLACE INTO analysis_status (id, task_id, status) VALUES (1, ?, ?)", (task_id, status))
        conn.commit()
        print(f"DEBUG: Saved task ID {task_id} with status {status} to status DB.")
    except Exception as e:
        print(f"ERROR: Failed to save analysis task ID {task_id} to status DB: {e}")

def get_last_analysis_task_id():
    try:
        conn = get_status_db()
        cur = conn.cursor()
        cur.execute("SELECT task_id, status FROM analysis_status ORDER BY timestamp DESC LIMIT 1")
        row = cur.fetchone()
        result = dict(row) if row else None
        print(f"DEBUG: Retrieved last analysis task from DB: {result}")
        return result
    except Exception as e:
        print(f"ERROR: Failed to retrieve last analysis task from status DB: {e}")
        return None

# --- Existing Script Functions (Minimum Changes) ---
# Ensure these functions use DB_PATH and TEMP_DIR from config.py

def clean_temp():
    print(f"DEBUG: Cleaning temp directory: {TEMP_DIR}")
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True) # Ensure it's created even if it didn't exist
    print(f"DEBUG: Temp directory cleaned/created.")

def init_db():
    print(f"DEBUG: Initializing Main DB at: {DB_PATH}") # Debug print
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS score (
            item_id TEXT PRIMARY KEY, title TEXT, author TEXT,
            tempo REAL, key TEXT, scale TEXT, mood_vector TEXT
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS playlist (
            playlist TEXT, item_id TEXT, title TEXT, author TEXT
        )''')
        conn.commit()
        conn.close()
        print(f"DEBUG: Main DB initialized successfully at: {DB_PATH}")
    except sqlite3.OperationalError as e:
        print(f"ERROR: Could not initialize Main DB at {DB_PATH}: {e}")
        raise

def get_recent_albums(limit=NUM_RECENT_ALBUMS):
    url = f"{JELLYFIN_URL}/Users/{JELLYFIN_USER_ID}/Items"
    params = {
        "IncludeItemTypes": "MusicAlbum",
        "SortBy": "DateCreated",
        "SortOrder": "Descending",
        "Limit": limit,
        "Recursive": True,
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params)
        r.raise_for_status()
        albums = r.json().get("Items", [])
        print(f"DEBUG: Fetched {len(albums)} recent albums from Jellyfin.")
        return albums
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to get albums from Jellyfin: {e}")
        return []


def get_tracks_from_album(album_id):
    url = f"{JELLYFIN_URL}/Users/{JELLYFIN_USER_ID}/Items"
    params = {"ParentId": album_id, "IncludeItemTypes": "Audio"}
    try:
        r = requests.get(url, headers=HEADERS, params=params)
        r.raise_for_status()
        tracks = r.json().get("Items", [])
        print(f"DEBUG: Fetched {len(tracks)} tracks for album {album_id}.")
        return tracks
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to get tracks from album {album_id}: {e}")
        return []


def download_track(item):
    filename = f"{item['Name'].replace('/', '_')}-{item.get('AlbumArtist', 'Unknown')}.mp3"
    path = os.path.join(TEMP_DIR, filename)
    print(f"DEBUG: Attempting to download track {item['Name']} to {path}")
    try:
        r = requests.get(f"{JELLYFIN_URL}/Items/{item['Id']}/Download", headers=HEADERS)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        print(f"DEBUG: Successfully downloaded {filename}")
        return path
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Download failed for {item['Name']}: {e}")
        return None
    except IOError as e:
        print(f"ERROR: File system error saving {filename} to {path}: {e}")
        return None


def predict_moods(file_path):
    # Essentia models are loaded from the paths defined in config.py
    # which are set via environment variables in the Dockerfile/Kubernetes
    print(f"DEBUG: Predicting moods for {file_path} using models: {EMBEDDING_MODEL_PATH}, {PREDICTION_MODEL_PATH}")
    try:
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
        top_moods = {label: float(score) for label, score in sorted(results.items(), key=lambda x: -x[1])[:TOP_N_MOODS]}
        print(f"DEBUG: Predicted moods: {top_moods}")
        return top_moods
    except Exception as e:
        print(f"ERROR: Failed to predict moods for {file_path}: {e}")
        raise # Re-raise to be caught by Celery task


def analyze_track(file_path):
    print(f"DEBUG: Analyzing track: {file_path}")
    try:
        loader = MonoLoader(filename=file_path)
        audio = loader()
        tempo, _, _, _, _ = RhythmExtractor2013()(audio)
        key, scale, _ = KeyExtractor()(audio)
        moods = predict_moods(file_path)
        print(f"DEBUG: Analysis complete for {file_path}: Tempo={tempo}, Key={key}, Scale={scale}")
        return tempo, key, scale, moods
    except Exception as e:
        print(f"ERROR: Failed to analyze track {file_path}: {e}")
        raise # Re-raise to be caught by Celery task


def track_exists(item_id):
    print(f"DEBUG: Checking if track {item_id} exists in DB at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM score WHERE item_id=?", (item_id,))
    row = cur.fetchone()
    conn.close()
    exists = bool(row)
    print(f"DEBUG: Track {item_id} exists: {exists}")
    return exists


def save_track_analysis(item_id, title, author, tempo, key, scale, moods):
    print(f"DEBUG: Saving analysis for {title} by {author} to DB at {DB_PATH}")
    mood_str = ','.join(f"{k}:{v:.3f}" for k, v in moods.items())
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO score VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (item_id, title, author, tempo, key, scale, mood_str))
        conn.commit()
        conn.close()
        print(f"DEBUG: Analysis saved successfully for {title}.")
    except sqlite3.Error as e:
        print(f"ERROR: Failed to save analysis for {title} to DB: {e}")
        raise


def score_vector(row):
    tempo = float(row[3]) if row[3] is not None else 0.0
    mood_str = row[6] or ""

    # Normalize tempo (assumed typical BPM range: 40–200 BPM)
    tempo_norm = (tempo - 40) / (200 - 40)
    tempo_norm = np.clip(tempo_norm, 0.0, 1.0)

    # Process mood string into vector
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
                    continue

    # Final vector: [tempo_norm] + list(mood_scores)
    full_vector = [tempo_norm] + list(mood_scores)
    return full_vector



def get_all_tracks():
    print(f"DEBUG: Fetching all tracks from DB at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM score")
    rows = cur.fetchall()
    conn.close()
    print(f"DEBUG: Fetched {len(rows)} tracks from DB.")
    return rows


def limit_songs_per_artist(cluster):
    count = defaultdict(int)
    result = []
    for item_id, title, author in cluster:
        if count[author] < MAX_SONGS_PER_ARTIST:
            result.append((item_id, title, author))
            count[author] += 1
    return result


def name_cluster(centroid_scaled_vector, pca_model=None): # Renamed pca to pca_model to avoid shadowing
    if PCA_ENABLED and pca_model is not None:
        scaled_vector = pca_model.inverse_transform(centroid_scaled_vector)
    else:
        scaled_vector = centroid_scaled_vector

    tempo_norm = scaled_vector[0]
    mood_values = scaled_vector[1:]

    # Denormalize tempo
    tempo = tempo_norm * (200 - 40) + 40

    # Label tempo
    if tempo < 80:
        tempo_label = "Slow"
    elif tempo < 130:
        tempo_label = "Medium"
    else:
        tempo_label = "Fast"

    # Top 3 mood labels
    top_indices = np.argsort(mood_values)[::-1][:3]
    mood_names = [MOOD_LABELS[i] for i in top_indices]
    mood_part = "_".join(mood_names).title()

    # Final name
    full_name = f"{mood_part}_{tempo_label}"

    # Return name and mood centroid info for display
    top_mood_scores = {MOOD_LABELS[i]: mood_values[i] for i in top_indices}
    extra_info = {"tempo": round(tempo_norm, 2)}

    return full_name, {**top_mood_scores, **extra_info}

def update_playlist_table(playlists):
    print(f"DEBUG: Updating playlist table in DB at {DB_PATH}")
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM playlist") # Clear existing playlists
        for name, cluster in playlists.items():
            for item_id, title, author in cluster:
                cur.execute("INSERT INTO playlist VALUES (?, ?, ?, ?)", (name, item_id, title, author))
        conn.commit()
        conn.close()
        print(f"DEBUG: Playlist table updated successfully.")
    except sqlite3.Error as e:
        print(f"ERROR: Failed to update playlist table in DB: {e}")
        raise


def delete_old_automatic_playlists():
    print(f"DEBUG: Checking for and deleting old automatic playlists on Jellyfin.")
    url = f"{JELLYFIN_URL}/Users/{JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(url, headers=HEADERS, params=params)
        r.raise_for_status()
        for item in r.json().get("Items", []):
            if "_automatic" in item.get("Name", ""):
                del_url = f"{JELLYFIN_URL}/Items/{item['Id']}"
                del_resp = requests.delete(del_url, headers=HEADERS)
                if del_resp.ok:
                    print(f"🗑️ Deleted old playlist: {item['Name']}")
                else:
                    print(f"❌ Failed to delete old playlist {item['Name']}: {del_resp.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to clean old playlists on Jellyfin: {e}")


def create_or_update_playlists_on_jellyfin(playlists, cluster_centers, pca_model=None): # Renamed pca to pca_model
    print(f"DEBUG: Creating/Updating playlists on Jellyfin.")
    delete_old_automatic_playlists()
    for base_name, cluster in playlists.items():
        chunks = [cluster[i:i+MAX_SONGS_PER_CLUSTER] for i in range(0, len(cluster), MAX_SONGS_PER_CLUSTER)]
        for idx, chunk in enumerate(chunks, 1):
            playlist_name = f"{base_name}_automatic_{idx}" if len(chunks) > 1 else f"{base_name}_automatic"
            item_ids = [item_id for item_id, _, _ in chunk]
            if not item_ids:
                print(f"DEBUG: Skipping empty chunk for {playlist_name}")
                continue
            body = {"Name": playlist_name, "Ids": item_ids, "UserId": JELLYFIN_USER_ID}
            try:
                r = requests.post(f"{JELLYFIN_URL}/Playlists", headers=HEADERS, json=body)
                if r.ok:
                    centroid_info = cluster_centers[base_name]
                    # Separate mood scores and extra info
                    top_moods = {k: v for k, v in centroid_info.items() if k in MOOD_LABELS}
                    extra_info = {k: v for k, v in centroid_info.items() if k not in MOOD_LABELS}

                    centroid_str = ", ".join(f"{k}: {v:.2f}" for k, v in top_moods.items())
                    extras_str = ", ".join(f"{k}: {v:.2f}" for k, v in extra_info.items())

                    print(f"✅ Created playlist {playlist_name} with {len(item_ids)} tracks (Centroid: {centroid_str} | {extras_str})")

                else:
                    print(f"❌ Failed to create playlist {playlist_name} -> Status: {r.status_code}, Response: {r.text}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Exception creating {playlist_name}: {e}")

# --- Celery Task Definition (Modified main function) ---
@celery.task(bind=True)
def run_analysis_task(self, jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods):
    # Update global config for this task run
    # These global assignments are important because the task runs in a separate process
    # and needs to ensure it uses the parameters passed to it, not just what was
    # loaded at initial app startup for the main Flask process.
    global JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, NUM_RECENT_ALBUMS, TOP_N_MOODS, HEADERS
    JELLYFIN_URL = jellyfin_url
    JELLYFIN_USER_ID = jellyfin_user_id
    JELLYFIN_TOKEN = jellyfin_token
    HEADERS = {"X-Emby-Token": JELLYFIN_TOKEN}
    NUM_RECENT_ALBUMS = num_recent_albums
    TOP_N_MOODS = top_n_moods

    self.update_state(state='PENDING', meta={'progress': 0, 'status': 'Starting analysis...', 'log_output': [], 'current_album': 'N/A', 'current_album_idx': 0, 'total_albums': 0})
    log_messages = []
    
    def log_and_update(message, progress, current_album=None, current_album_idx=0, total_albums=0):
        log_messages.append(message)
        self.update_state(state='PROGRESS', meta={
            'progress': progress,
            'status': message,
            'log_output': log_messages,
            'current_album': current_album,
            'current_album_idx': current_album_idx,
            'total_albums': total_albums
        })

    try:
        log_and_update("🚀 Starting mood-based analysis and playlist generation...", 0)
        clean_temp() # Uses TEMP_DIR from config.py/env
        init_db()    # Uses DB_PATH from config.py/env

        albums = get_recent_albums() # Uses JELLYFIN_URL, JELLYFIN_USER_ID, HEADERS from updated globals
        if not albums:
            log_and_update("⚠️ No new albums to analyze. Proceeding with existing data.", 10)
        else:
            total_albums = len(albums)
            for idx, album in enumerate(albums, 1):
                log_and_update(f"🎵 Album: {album['Name']} - {idx} out of {total_albums}",
                               10 + int(80 * (idx / total_albums) * 0.5), # Scale analysis progress
                               current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                tracks = get_tracks_from_album(album['Id'])
                for item in tracks:
                    log_and_update(f"   🎶 Analyzing track: {item['Name']} by {item.get('AlbumArtist', 'Unknown')}",
                                   10 + int(80 * (idx / total_albums) * 0.5), # Scale analysis progress
                                   current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                    if track_exists(item['Id']): # Uses DB_PATH from config.py/env
                        log_and_update(f"     ⏭️ Skipping (already analyzed)",
                                       10 + int(80 * (idx / total_albums) * 0.5), # Scale analysis progress
                                       current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                        continue
                    path = download_track(item) # Uses TEMP_DIR from config.py/env
                    if not path:
                        log_and_update(f"     ❌ Skipping track due to download failure: {item['Name']}",
                                       10 + int(80 * (idx / total_albums) * 0.5), # Scale analysis progress
                                       current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                        continue
                    try:
                        tempo, key, scale, moods = analyze_track(path) # Uses EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, TOP_N_MOODS from config.py/env
                        save_track_analysis(item['Id'], item['Name'], item.get('AlbumArtist', 'Unknown'), tempo, key, scale, moods) # Uses DB_PATH from config.py/env
                        log_and_update(f"     ✅ Moods: {', '.join(f'{k}:{v:.2f}' for k,v in moods.items())}",
                                       10 + int(80 * (idx / total_albums) * 0.5), # Scale analysis progress
                                       current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                    except Exception as e:
                        log_and_update(f"     ❌ Error analyzing track {item['Name']}: {e}",
                                       10 + int(80 * (idx / total_albums) * 0.5), # Scale analysis progress
                                       current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                    finally:
                        # Clean up downloaded track immediately after analysis
                        if os.path.exists(path):
                            os.remove(path)
                            print(f"DEBUG: Removed temporary track file: {path}")

            clean_temp() # Final cleanup of temp directory

        log_and_update("Analysis phase complete. Starting clustering...", 90)

        rows = get_all_tracks() # Uses DB_PATH from config.py/env
        if len(rows) < 2:
            log_and_update("⚠️ Not enough data for clustering.", 100)
            return {"status": "SUCCESS", "message": "Analysis complete, but not enough data for clustering."}

        X_original = [score_vector(row) for row in rows]
        X_scaled = X_original # Assuming no scaling needed beyond score_vector's internal normalization

        pca_model = None
        if PCA_ENABLED: # Uses PCA_ENABLED from config.py/env
            # PCA_COMPONENTS is not directly in config.py, but PCA_ENABLED implies it.
            # If you want to configure n_components via env var, add it to config.py and here.
            # For now, assuming fixed 3 components if PCA_ENABLED.
            pca_model = PCA(n_components=3) # Hardcoded n_components=3 if PCA_ENABLED
            X_pca = pca_model.fit_transform(X_scaled)
            data_for_clustering = X_pca
        else:
            data_for_clustering = X_scaled

        if CLUSTER_ALGORITHM == "kmeans": # Uses CLUSTER_ALGORITHM from config.py/env
            k = NUM_CLUSTERS if NUM_CLUSTERS > 0 else max(1, len(rows) // MAX_SONGS_PER_CLUSTER) # Uses NUM_CLUSTERS and MAX_SONGS_PER_CLUSTER from config.py/env
            kmeans = KMeans(n_clusters=min(k, len(rows)), random_state=42, n_init='auto')
            labels = kmeans.fit_predict(data_for_clustering)
            cluster_centers = {i: kmeans.cluster_centers_[i] for i in range(min(k, len(rows)))}
            centers_for_points = kmeans.cluster_centers_[labels]
            raw_distances = np.linalg.norm(data_for_clustering - centers_for_points, axis=1)
        elif CLUSTER_ALGORITHM == "dbscan": # Uses CLUSTER_ALGORITHM from config.py/env
            dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES) # Uses DBSCAN_EPS and DBSCAN_MIN_SAMPLES from config.py/env
            labels = dbscan.fit_predict(data_for_clustering)
            cluster_centers = {}
            raw_distances = np.zeros(len(data_for_clustering))
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue
                indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                cluster_points = np.array([data_for_clustering[i] for i in indices])
                center = cluster_points.mean(axis=0)
                for i in indices:
                    raw_distances[i] = np.linalg.norm(data_for_clustering[i] - center)
                cluster_centers[cluster_id] = center
        else:
            raise ValueError(f"Unsupported clustering algorithm: {CLUSTER_ALGORITHM}")

        max_dist = raw_distances.max()
        normalized_distances = raw_distances / max_dist if max_dist > 0 else raw_distances

        track_info = []
        for row, label, vec, dist in zip(rows, labels, data_for_clustering, normalized_distances):
            if label == -1: # DBSCAN noise points
                continue
            track_info.append({"row": row, "label": label, "vector": vec, "distance": dist})

        filtered_clusters = defaultdict(list)
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            cluster_tracks = [t for t in track_info if t["label"] == cluster_id and t["distance"] <= MAX_DISTANCE] # Uses MAX_DISTANCE from config.py/env
            if not cluster_tracks:
                continue

            cluster_tracks.sort(key=lambda x: x["distance"])
            count_per_artist = defaultdict(int)
            selected = []
            for t in cluster_tracks:
                author = t["row"][2]
                if count_per_artist[author] < MAX_SONGS_PER_ARTIST: # Uses MAX_SONGS_PER_ARTIST from config.py/env
                    selected.append(t)
                    count_per_artist[author] += 1
                if len(selected) >= MAX_SONGS_PER_CLUSTER: # Uses MAX_SONGS_PER_CLUSTER from config.py/env
                    break

            for t in selected:
                item_id, title, author = t["row"][0], t["row"][1], t["row"][2]
                filtered_clusters[cluster_id].append((item_id, title, author))

        named_playlists = defaultdict(list)
        playlist_centroids = {}
        for label, songs in filtered_clusters.items():
            center = cluster_centers[label]
            name, top_scores = name_cluster(center, pca_model) # Pass pca_model
            named_playlists[name].extend(songs)
            playlist_centroids[name] = top_scores
        
        log_and_update("Clustering complete. Updating playlists...", 95)
        update_playlist_table(named_playlists) # Uses DB_PATH from config.py/env
        create_or_update_playlists_on_jellyfin(named_playlists, playlist_centroids, pca_model) # Uses JELLYFIN_URL, JELLYFIN_USER_ID, HEADERS from updated globals
        
        log_and_update("✅ All done!", 100)
        return {"status": "SUCCESS", "message": "Analysis and playlist generation complete!"}

    except Exception as e:
        # This catch-all exception handling is crucial for Celery to report errors properly.
        # The error will be serialized using 'pickle' and sent to the result backend.
        error_message = f"Analysis failed: {type(e).__name__}: {e}"
        print(f"ERROR: {error_message}", flush=True) # Ensure error is printed to logs immediately
        log_and_update(f"❌ {error_message}", 100)
        self.update_state(state='FAILURE', meta={'progress': 100, 'status': error_message, 'log_output': log_messages})
        # Re-raise the exception so Celery marks the task as FAILED with proper exception info
        raise e


# --- API Endpoints ---

@app.route('/')
def index():
    # Flask will automatically look for index.html in the 'static' folder
    # because of app = Flask(__name__, static_folder='static', static_url_path='')
    return render_template('index.html')

@app.route('/api/analysis/start', methods=['POST'])
def start_analysis():
    data = request.json
    # Pass all relevant parameters to the Celery task
    # This ensures the task has the correct Jellyfin credentials and analysis parameters
    task = run_analysis_task.delay(
        data.get('jellyfin_url', JELLYFIN_URL),
        data.get('jellyfin_user_id', JELLYFIN_USER_ID),
        data.get('jellyfin_token', JELLYFIN_TOKEN),
        int(data.get('num_recent_albums', NUM_RECENT_ALBUMS)),
        int(data.get('top_n_moods', TOP_N_MOODS))
    )
    save_analysis_task_id(task.id, "PENDING")
    return jsonify({"task_id": task.id, "status": "PENDING"}), 202

@app.route('/api/analysis/status/<task_id>', methods=['GET'])
def analysis_status(task_id):
    task = AsyncResult(task_id, app=celery)
    response = {
        'task_id': task.id,
        'state': task.state,
        'status': 'Processing...'
    }
    # task.info contains the 'meta' dictionary updated by self.update_state
    if isinstance(task.info, dict):
        response.update(task.info)
    else: # Fallback if task.info is not a dict (e.g., initial PENDING state before first update_state)
        if task.state == 'PENDING':
            response.update({'progress': 0, 'status': 'Initializing...', 'log_output': ['Task pending...'], 'current_album': 'N/A', 'current_album_idx': 0, 'total_albums': 0})
        elif task.state == 'SUCCESS':
            response.update({'progress': 100, 'status': 'Analysis complete!', 'log_output': ['Task completed successfully.']})
        elif task.state == 'FAILURE':
            # task.info for FAILURE contains the exception object when using pickle
            error_message = str(task.info)
            response.update({'progress': 100, 'status': f'Analysis failed: {error_message}', 'log_output': [error_message]})
        elif task.state == 'REVOKED':
            response.update({'progress': 100, 'status': 'Task revoked.', 'log_output': ['Task was cancelled.']})

    # Update status DB for final states
    if task.state in ['SUCCESS', 'FAILURE', 'REVOKED']:
        save_analysis_task_id(task_id, task.state)
    
    return jsonify(response)


@app.route('/api/analysis/cancel/<task_id>', methods=['POST'])
def cancel_analysis(task_id):
    task = AsyncResult(task_id, app=celery)
    if task.state in ['PENDING', 'STARTED', 'PROGRESS']:
        task.revoke(terminate=True)
        save_analysis_task_id(task_id, "REVOKED")
        print(f"DEBUG: Task {task_id} revoked.")
        return jsonify({"message": "Analysis task cancelled.", "task_id": task_id}), 200
    else:
        print(f"DEBUG: Attempted to cancel task {task_id} in state {task.state}, but it cannot be cancelled.")
        return jsonify({"message": "Task cannot be cancelled in its current state.", "state": task.state}), 400

@app.route('/api/analysis/last_task', methods=['GET'])
def get_last_analysis_status():
    last_task = get_last_analysis_task_id()
    if last_task:
        # For the frontend, we need to return the task's actual state from Celery backend
        # if it's still active, otherwise just the stored status.
        task_id = last_task['task_id']
        task = AsyncResult(task_id, app=celery)
        
        # If task is still active, return its current state from Celery
        if task.state in ['PENDING', 'STARTED', 'PROGRESS']:
            response = {
                'task_id': task.id,
                'status': task.state, # Use Celery's actual state
            }
            if isinstance(task.info, dict):
                response.update(task.info) # Include progress, log_output etc.
            return jsonify(response), 200
        else:
            # If task is not active, return the stored status from our SQLite DB
            return jsonify(last_task), 200
    return jsonify({"task_id": None, "status": "NO_PREVIOUS_TASK"}), 200


@app.route('/api/config', methods=['GET'])
def get_config():
    # Expose current config values to the frontend
    # Ensure all values are pulled from config.py which is reading from env vars
    return jsonify({
        "jellyfin_url": JELLYFIN_URL,
        "jellyfin_user_id": JELLYFIN_USER_ID,
        "jellyfin_token": JELLYFIN_TOKEN, # Be cautious exposing tokens in real apps
        "num_recent_albums": NUM_RECENT_ALBUMS,
        "max_distance": MAX_DISTANCE,
        "max_songs_per_cluster": MAX_SONGS_PER_CLUSTER,
        "max_songs_per_artist": MAX_SONGS_PER_ARTIST,
        "cluster_algorithm": CLUSTER_ALGORITHM,
        "pca_enabled": PCA_ENABLED,
        "dbscan_eps": DBSCAN_EPS,
        "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
        "num_clusters": NUM_CLUSTERS,
        "top_n_moods": TOP_N_MOODS,
        "mood_labels": MOOD_LABELS,
    })

@app.route('/api/playlists', methods=['GET'])
def get_playlists():
    print(f"DEBUG: Fetching playlists from DB at {DB_PATH}")
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT playlist, item_id, title, author FROM playlist ORDER BY playlist, title")
        rows = cur.fetchall()
        conn.close()

        playlists = defaultdict(list)
        for playlist, item_id, title, author in rows:
            playlists[playlist].append({"item_id": item_id, "title": title, "author": author})
        
        print(f"DEBUG: Fetched {len(playlists)} distinct playlists.")
        # Convert defaultdict to regular dict for JSON serialization
        return jsonify(dict(playlists)), 200
    except sqlite3.Error as e:
        print(f"ERROR: Failed to fetch playlists from DB: {e}")
        return jsonify({"status": "error", "message": f"Failed to load playlists: {e}"}), 500


@app.route('/api/clustering', methods=['POST'])
def run_clustering():
    # Note: For clustering, you are doing it synchronously in the Flask app.
    # If this becomes a long-running operation, you might want to convert it
    # into a Celery task as well, similar to run_analysis_task.
    
    # Update global config for this clustering run based on request
    # These global assignments are important because the clustering functions
    # use these global variables.
    global CLUSTER_ALGORITHM, NUM_CLUSTERS, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, PCA_ENABLED
    
    data = request.json
    
    CLUSTER_ALGORITHM = data.get('clustering_method', CLUSTER_ALGORITHM)
    NUM_CLUSTERS = int(data.get('num_clusters', NUM_CLUSTERS))
    DBSCAN_EPS = float(data.get('dbscan_eps', DBSCAN_EPS))
    DBSCAN_MIN_SAMPLES = int(data.get('dbscan_min_samples', DBSCAN_MIN_SAMPLES))
    
    pca_components = int(data.get('pca_components', 0))
    PCA_ENABLED = (pca_components > 0)

    print(f"DEBUG: Starting synchronous clustering with: CLUSTER_ALGORITHM={CLUSTER_ALGORITHM}, NUM_CLUSTERS={NUM_CLUSTERS}, PCA_ENABLED={PCA_ENABLED}")

    try:
        rows = get_all_tracks()
        if len(rows) < 2:
            return jsonify({"status": "error", "message": "Not enough analyzed tracks for clustering."}), 400

        X_original = [score_vector(row) for row in rows]
        X_scaled = X_original

        pca_model = None
        if PCA_ENABLED:
            pca_model = PCA(n_components=pca_components)
            X_pca = pca_model.fit_transform(X_scaled)
            data_for_clustering = X_pca
        else:
            data_for_clustering = X_scaled

        if CLUSTER_ALGORITHM == "kmeans":
            k = NUM_CLUSTERS if NUM_CLUSTERS > 0 else max(1, len(rows) // MAX_SONGS_PER_CLUSTER)
            kmeans = KMeans(n_clusters=min(k, len(rows)), random_state=42, n_init='auto')
            labels = kmeans.fit_predict(data_for_clustering)
            cluster_centers = {i: kmeans.cluster_centers_[i] for i in range(min(k, len(rows)))}
            centers_for_points = kmeans.cluster_centers_[labels]
            raw_distances = np.linalg.norm(data_for_clustering - centers_for_points, axis=1)
        elif CLUSTER_ALGORITHM == "dbscan":
            dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
            labels = dbscan.fit_predict(data_for_clustering)
            cluster_centers = {}
            raw_distances = np.zeros(len(data_for_clustering))
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue
                indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                cluster_points = np.array([data_for_clustering[i] for i in indices])
                center = cluster_points.mean(axis=0)
                for i in indices:
                    raw_distances[i] = np.linalg.norm(data_for_clustering[i] - center)
                cluster_centers[cluster_id] = center
        else:
            return jsonify({"status": "error", "message": f"Unsupported clustering algorithm: {CLUSTER_ALGORITHM}"}), 400

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

            cluster_tracks = [t for t in track_info if t["label"] == cluster_id and t["distance"] <= MAX_DISTANCE]
            if not cluster_tracks:
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
            center = cluster_centers[label]
            name, top_scores = name_cluster(center, pca_model)
            named_playlists[name].extend(songs)
            playlist_centroids[name] = top_scores

        update_playlist_table(named_playlists)
        create_or_update_playlists_on_jellyfin(named_playlists, playlist_centroids, pca_model)

        print(f"DEBUG: Clustering and playlist update complete.")
        return jsonify({"status": "success", "message": "Playlists generated and updated on Jellyfin!"}), 200

    except Exception as e:
        print(f"ERROR: Error during clustering: {type(e).__name__}: {e}", flush=True)
        return jsonify({"status": "error", "message": f"Clustering failed: {type(e).__name__}: {e}"}), 500


if __name__ == '__main__':
    # Initialize main DB
    init_db() # This uses DB_PATH from config.py (from env vars)
    # Ensure temp directory exists
    os.makedirs(TEMP_DIR, exist_ok=True) # This uses TEMP_DIR from config.py (from env vars)
    app.run(debug=True, host='0.0.0.0', port=8000)
