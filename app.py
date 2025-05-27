import os
import shutil
import sqlite3
import requests
from collections import defaultdict
import numpy as np
from flask import Flask, jsonify, request, render_template, g
from celery import Celery
from celery.result import AsyncResult
from contextlib import closing
import json
import time
import random # Import random for parameter sampling

# Import your existing analysis functions
from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor, TensorflowPredictMusiCNN, TensorflowPredict2D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Your existing config - assuming this is from config.py and sets global variables
from config import *

# --- Flask App Setup ---
app = Flask(__name__)
# Celery configuration is now read from config.py
app.config['CELERY_BROKER_URL'] = CELERY_BROKER_URL
app.config['CELERY_RESULT_BACKEND'] = CELERY_RESULT_BACKEND

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# --- Status DB Setup ---
def init_status_db():
    """
    Initializes the SQLite database for storing task statuses.
    This table now includes a 'task_type' column to differentiate between
    analysis and clustering tasks.
    """
    with closing(sqlite3.connect(STATUS_DB_PATH)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute('''CREATE TABLE IF NOT EXISTS task_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE,
                task_type TEXT, -- 'analysis' or 'clustering'
                status TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )''')
            conn.commit()

with app.app_context():
    init_status_db()

def get_status_db():
    """
    Provides a connection to the status database.
    Ensures the connection is per-request and closed automatically.
    """
    if 'status_db' not in g:
        g.status_db = sqlite3.connect(STATUS_DB_PATH)
        g.status_db.row_factory = sqlite3.Row
    return g.status_db

@app.teardown_appcontext
def close_status_db(exception):
    """
    Closes the database connection at the end of the request.
    """
    status_db = g.pop('status_db', None)
    if status_db is not None:
        status_db.close()

def save_task_status(task_id, task_type, status="PENDING"):
    """
    Saves or updates the status of a given task in the database.
    Includes the task type.
    """
    conn = get_status_db()
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO task_status (task_id, task_type, status) VALUES (?, ?, ?)",
                (task_id, task_type, status))
    conn.commit()

def get_last_task_status():
    """
    Retrieves the status of the most recent task (analysis or clustering).
    """
    conn = get_status_db()
    cur = conn.cursor()
    cur.execute("SELECT task_id, task_type, status FROM task_status ORDER BY timestamp DESC LIMIT 1")
    row = cur.fetchone()
    return dict(row) if row else None

# --- Existing Script Functions (No Global Mutation) ---

def clean_temp(temp_dir):
    """Cleans up the temporary directory."""
    os.makedirs(temp_dir, exist_ok=True)
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Warning: Could not remove {file_path} from {temp_dir}: {e}")

def init_db(db_path):
    """Initializes the main application database for scores and playlists."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS score (
            item_id TEXT PRIMARY KEY, title TEXT, author TEXT,
            tempo REAL, key TEXT, scale TEXT, mood_vector TEXT
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS playlist (
            playlist TEXT, item_id TEXT, title TEXT, author TEXT
        )''')
        conn.commit()

def get_recent_albums(jellyfin_url, jellyfin_user_id, headers, limit):
    """Fetches recent albums from Jellyfin."""
    url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {
        "IncludeItemTypes": "MusicAlbum",
        "SortBy": "DateCreated",
        "SortOrder": "Descending",
        "Limit": limit,
        "Recursive": True,
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        print(f"ERROR: get_recent_albums: {e}")
        return []

def get_tracks_from_album(jellyfin_url, jellyfin_user_id, headers, album_id):
    """Fetches tracks belonging to a specific album from Jellyfin."""
    url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {"ParentId": album_id, "IncludeItemTypes": "Audio"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("Items", []) if r.ok else []
    except Exception as e:
        print(f"ERROR: get_tracks_from_album {album_id}: {e}")
        return []

def download_track(jellyfin_url, headers, temp_dir, item):
    """Downloads a track from Jellyfin to a temporary directory."""
    filename = f"{item['Name'].replace('/', '_')}-{item.get('AlbumArtist', 'Unknown')}.mp3"
    path = os.path.join(temp_dir, filename)
    try:
        r = requests.get(f"{jellyfin_url}/Items/{item['Id']}/Download", headers=headers, timeout=120)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        return path
    except Exception as e:
        print(f"ERROR: download_track {item['Name']}: {e}")
        return None

def predict_moods(file_path, embedding_model_path, prediction_model_path, mood_labels, top_n_moods):
    """Predicts moods for an audio file using pre-trained models."""
    audio = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictMusiCNN(
        graphFilename=embedding_model_path, output="model/dense/BiasAdd"
    )
    embeddings = embedding_model(audio)
    model = TensorflowPredict2D(
        graphFilename=prediction_model_path,
        input="serving_default_model_Placeholder",
        output="PartitionedCall"
    )
    predictions = model(embeddings)[0]
    results = dict(zip(mood_labels, predictions))
    return {label: float(score) for label, score in sorted(results.items(), key=lambda x: -x[1])[:top_n_moods]}

def analyze_track(file_path, embedding_model_path, prediction_model_path, mood_labels, top_n_moods):
    """Analyzes a single track for tempo, key, scale, and moods."""
    audio = MonoLoader(filename=file_path)()
    tempo, _, _, _, _ = RhythmExtractor2013()(audio)
    key, scale, _ = KeyExtractor()(audio)
    moods = predict_moods(file_path, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, MOOD_LABELS, top_n_moods) # Use global paths and top_n_moods
    return tempo, key, scale, moods

def track_exists(db_path, item_id):
    """Checks if a track's analysis already exists in the database."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM score WHERE item_id=?", (item_id,))
        row = cur.fetchone()
    return row

def save_track_analysis(db_path, item_id, title, author, tempo, key, scale, moods):
    """Saves the analysis results for a track to the database."""
    mood_str = ','.join(f"{k}:{v:.3f}" for k, v in moods.items())
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO score VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (item_id, title, author, tempo, key, scale, mood_str))
        conn.commit()

def score_vector(row, mood_labels):
    """Converts a database row into a numerical feature vector for clustering."""
    tempo = float(row[3]) if row[3] is not None else 0.0
    mood_str = row[6] or ""
    tempo_norm = (tempo - 40) / (200 - 40)
    tempo_norm = np.clip(tempo_norm, 0.0, 1.0)
    mood_scores = np.zeros(len(mood_labels))
    if mood_str:
        for pair in mood_str.split(","):
            if ":" not in pair:
                continue
            label, score = pair.split(":")
            if label in mood_labels:
                try:
                    mood_scores[mood_labels.index(label)] = float(score)
                except ValueError:
                    continue
    full_vector = [tempo_norm] + list(mood_scores)
    return full_vector

def get_all_tracks(db_path):
    """Retrieves all analyzed tracks from the database."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM score")
        rows = cur.fetchall()
    return rows

def name_cluster(centroid_scaled_vector, pca_model, pca_enabled, mood_labels):
    """
    Generates a human-readable name for a cluster based on its centroid's
    tempo and predominant moods. Also returns top mood scores for diversity calculation.
    """
    if pca_enabled and pca_model is not None:
        # Inverse transform to get back to the original feature space
        try:
            scaled_vector = pca_model.inverse_transform(centroid_scaled_vector.reshape(1, -1))[0]
        except ValueError:
            print("Warning: PCA inverse_transform failed. Using original scaled vector.")
            scaled_vector = centroid_scaled_vector
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
    
    if len(mood_values) == 0 or np.sum(mood_values) == 0:
        top_indices = []
    else:
        top_indices = np.argsort(mood_values)[::-1][:3] # Get top 3 moods

    mood_names = [mood_labels[i] for i in top_indices if i < len(mood_labels)]
    mood_part = "_".join(mood_names).title() if mood_names else "Mixed"
    full_name = f"{mood_part}_{tempo_label}"
    
    top_mood_scores = {mood_labels[i]: mood_values[i] for i in top_indices if i < len(mood_labels)}
    extra_info = {"tempo": round(tempo_norm, 2)}
    
    return full_name, {**top_mood_scores, **extra_info}

def update_playlist_table(db_path, playlists):
    """Updates the playlist table in the database with new playlist data."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM playlist") # Clear existing playlists
        for name, cluster in playlists.items():
            for item_id, title, author in cluster:
                cur.execute("INSERT INTO playlist VALUES (?, ?, ?, ?)", (name, item_id, title, author))
        conn.commit()

def delete_old_automatic_playlists(jellyfin_url, jellyfin_user_id, headers):
    """Deletes old automatically generated playlists from Jellyfin."""
    url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        for item in r.json().get("Items", []):
            if "_automatic" in item.get("Name", ""):
                del_url = f"{jellyfin_url}/Items/{item['Id']}"
                del_resp = requests.delete(del_url, headers=headers, timeout=10)
                if del_resp.ok:
                    print(f"🗑️ Deleted old playlist: {item['Name']}")
    except Exception as e:
        print(f"Failed to clean old playlists: {e}")

def create_or_update_playlists_on_jellyfin(jellyfin_url, jellyfin_user_id, headers, playlists, cluster_centers, mood_labels):
    """Creates or updates playlists on Jellyfin based on clustering results."""
    delete_old_automatic_playlists(jellyfin_url, jellyfin_user_id, headers)
    for base_name, cluster in playlists.items():
        # MAX_SONGS_PER_CLUSTER is used here to chunk playlists if they exceed Jellyfin's practical limits
        chunks = [cluster[i:i+MAX_SONGS_PER_CLUSTER] for i in range(0, len(cluster), MAX_SONGS_PER_CLUSTER)]
        for idx, chunk in enumerate(chunks, 1):
            playlist_name = f"{base_name}_automatic_{idx}" if len(chunks) > 1 else f"{base_name}_automatic"
            item_ids = [item_id for item_id, _, _ in chunk]
            if not item_ids:
                continue
            body = {"Name": playlist_name, "Ids": item_ids, "UserId": jellyfin_user_id}
            try:
                r = requests.post(f"{jellyfin_url}/Playlists", headers=headers, json=body, timeout=30)
                if r.ok:
                    centroid_info = cluster_centers.get(base_name, {})
                    top_moods = {k: v for k, v in centroid_info.items() if k in mood_labels}
                    extra_info = {k: v for k, v in centroid_info.items() if k not in mood_labels}
                    centroid_str = ", ".join(f"{k}:{v:.2f}" for k, v in top_moods.items())
                    extras_str = ", ".join(f"{k}:{v:.2f}" for k, v in extra_info.items())
                    print(f"✅ Created playlist {playlist_name} with {len(item_ids)} tracks (Centroid: {centroid_str} | {extras_str})")
            except Exception as e:
                print(f"Exception creating {playlist_name}: {e}")

# --- Celery Task Definitions ---

@celery.task(bind=True)
def run_analysis_task(self, jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods):
    """
    Celery task to run the music analysis process.
    Updates task state with progress and log messages.
    """
    headers = {"X-Emby-Token": jellyfin_token}
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
        clean_temp(TEMP_DIR)
        init_db(DB_PATH) # Ensure DB is initialized before analysis
        albums = get_recent_albums(jellyfin_url, jellyfin_user_id, headers, num_recent_albums)
        if not albums:
            log_and_update("⚠️ No new albums to analyze. Proceeding with existing data.", 10)
        else:
            total_albums = len(albums)
            analysis_start_progress = 5
            analysis_end_progress = 85
            for idx, album in enumerate(albums, 1):
                album_progress_base = analysis_start_progress + int((analysis_end_progress - analysis_start_progress) * ((idx - 1) / total_albums))
                log_and_update(f"🎵 Processing Album: {album['Name']} ({idx}/{total_albums})", album_progress_base, current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                tracks = get_tracks_from_album(jellyfin_url, jellyfin_user_id, headers, album['Id'])
                if not tracks:
                    log_and_update(f"   ⚠️ No tracks found for album: {album['Name']}", album_progress_base, current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                    continue
                total_tracks_in_album = len(tracks)
                for track_idx, item in enumerate(tracks, 1):
                    track_name_full = f"{item['Name']} by {item.get('AlbumArtist', 'Unknown')}"
                    track_progress_within_album = int((analysis_end_progress - analysis_start_progress) * (1 / total_albums) * (track_idx / total_tracks_in_album))
                    current_overall_progress = album_progress_base + track_progress_within_album
                    log_and_update(f"   🎶 Analyzing track: {track_name_full} ({track_idx}/{total_tracks_in_album})", current_overall_progress, current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                    if track_exists(DB_PATH, item['Id']):
                        log_and_update(f"     ⏭️ Skipping '{track_name_full}' (already analyzed)", current_overall_progress, current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                        continue
                    path = download_track(jellyfin_url, headers, TEMP_DIR, item)
                    if not path:
                        log_and_update(f"     ❌ Failed to download '{track_name_full}'. Skipping.", current_overall_progress, current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                        continue
                    try:
                        analysis_start_time = time.time()
                        tempo, key, scale, moods = analyze_track(path, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, MOOD_LABELS, top_n_moods)
                        analysis_duration = time.time() - analysis_start_time
                        save_track_analysis(DB_PATH, item['Id'], item['Name'], item.get('AlbumArtist', 'Unknown'), tempo, key, scale, moods)
                        log_and_update(f"     ✅ Analyzed '{track_name_full}' in {analysis_duration:.2f}s. Moods: {', '.join(f'{k}:{v:.2f}' for k,v in moods.items())}", current_overall_progress, current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                    except Exception as e:
                        log_and_update(f"     ❌ Error analyzing '{track_name_full}': {e}", current_overall_progress, current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                    finally:
                        if path and os.path.exists(path):
                            try:
                                os.remove(path)
                            except Exception as cleanup_e:
                                print(f"WARNING: Failed to clean up temp file {path}: {cleanup_e}")
            clean_temp(TEMP_DIR)
        log_and_update("Analysis phase complete.", 90)
        return {"status": "SUCCESS", "message": "Analysis and playlist generation complete!"}
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"FATAL ERROR: Analysis failed: {e}\n{error_traceback}")
        log_and_update(f"❌ Analysis failed: {e}", 100)
        self.update_state(state='FAILURE', meta={'progress': 100, 'status': f'Analysis failed: {e}', 'log_output': log_messages + [f"Error Traceback: {error_traceback}"]})
        return {"status": "FAILURE", "message": f"Analysis failed: {e}"}

@celery.task(bind=True)
def run_clustering_task(self, clustering_method, num_clusters_min, num_clusters_max, dbscan_eps_min, dbscan_eps_max, dbscan_min_samples_min, dbscan_min_samples_max, pca_components_min, pca_components_max, num_clustering_runs, min_songs_per_playlist, max_songs_per_playlist):
    """
    Celery task to run the clustering and playlist generation process with an
    evolutionary approach for parameter selection.
    Updates task state with progress and log messages.
    Includes weighted diversity score calculation.
    Now includes min/max songs per playlist for evolutionary search.
    """
    log_messages = []
    def log_and_update(message, progress):
        log_messages.append(message)
        self.update_state(state='PROGRESS', meta={
            'progress': progress,
            'status': message,
            'log_output': log_messages,
        })
    try:
        log_and_update("📊 Starting playlist clustering with evolutionary parameter search...", 0)
        rows = get_all_tracks(DB_PATH)
        if len(rows) < 2: # Ensure enough tracks for meaningful clustering
            # Consider a higher threshold, e.g., min_songs_per_playlist * num_clusters_min
            log_and_update("Not enough analyzed tracks for clustering. Please run analysis first.", 100)
            self.update_state(state='FAILURE', meta={'progress': 100, 'status': 'Not enough tracks for clustering', 'log_output': log_messages})
            return {"status": "FAILURE", "message": "Not enough analyzed tracks for clustering."}
        
        log_and_update(f"Fetched {len(rows)} tracks for clustering.", 5)
        
        X_original = [score_vector(row, MOOD_LABELS) for row in rows]
        X_scaled = np.array(X_original)

        best_diversity_score = -1.0 
        best_clustering_results = None
        best_parameters_found = {} 

        for run_idx in range(num_clustering_runs):
            progress_base = 10 + int(80 * (run_idx / num_clustering_runs)) 
            
            current_num_clusters = 0
            current_dbscan_eps = 0.0
            current_dbscan_min_samples = 0
            current_pca_components = 0
            current_pca_enabled = False
            
            min_songs_per_playlist_sampled = random.randint(min_songs_per_playlist, max_songs_per_playlist)
            max_songs_per_playlist_sampled = random.randint(min_songs_per_playlist_sampled, max_songs_per_playlist)

            if clustering_method == "kmeans":
                current_num_clusters = random.randint(num_clusters_min, num_clusters_max)
                current_num_clusters = min(current_num_clusters, len(rows))
                if current_num_clusters == 0: 
                     current_num_clusters = max(1, len(rows) // max_songs_per_playlist_sampled if max_songs_per_playlist_sampled > 0 else 1)
                
            elif clustering_method == "dbscan":
                current_dbscan_eps = round(random.uniform(dbscan_eps_min, dbscan_eps_max), 2)
                current_dbscan_min_samples = random.randint(dbscan_min_samples_min, dbscan_min_samples_max)
            
            current_pca_components = random.randint(pca_components_min, pca_components_max)
            current_pca_enabled = (current_pca_components > 0)

            log_and_update(f"Running clustering iteration {run_idx + 1}/{num_clustering_runs} with parameters: "
                           f"Method={clustering_method}, K={current_num_clusters}, "
                           f"Eps={current_dbscan_eps}, MinS={current_dbscan_min_samples}, "
                           f"PCA_Comp={current_pca_components}, "
                           f"Playlist_Min={min_songs_per_playlist_sampled}, Playlist_Max={max_songs_per_playlist_sampled}...", progress_base)
            
            pca_model = None
            data_for_clustering = X_scaled

            if current_pca_enabled:
                log_and_update(f"  Applying PCA with {current_pca_components} components...", progress_base + 2)
                n_components_actual = min(current_pca_components, X_scaled.shape[1], len(rows) - 1 if len(rows) > 1 else 1)
                if n_components_actual > 0:
                    pca_model = PCA(n_components=n_components_actual)
                    X_pca = pca_model.fit_transform(X_scaled)
                    data_for_clustering = X_pca
                else:
                    log_and_update("  PCA components too low or data insufficient, PCA disabled for this run.", progress_base + 2)
                    current_pca_enabled = False 
                    pca_model = None


            labels = None
            cluster_centers_map = {} # Using a different name to avoid confusion with external 'cluster_centers'
            raw_distances = np.zeros(len(data_for_clustering))

            if clustering_method == "kmeans":
                k = current_num_clusters
                log_and_update(f"  Running KMeans with {k} clusters...", progress_base + 5)
                if k < 1: # K must be at least 1
                    log_and_update(f"  KMeans k must be >= 1. k={k} is invalid. Skipping this run.", progress_base + 5)
                    continue
                # KMeans requires n_samples >= n_clusters.
                if len(data_for_clustering) < k:
                    log_and_update(f"  Not enough samples ({len(data_for_clustering)}) for KMeans with K={k}. Adjusting K to {len(data_for_clustering)}.", progress_base + 5)
                    k = len(data_for_clustering)
                    if k < 1: # Still not enough after adjustment
                        log_and_update("  Not enough samples for KMeans. Skipping this run.", progress_base + 5)
                        continue
                
                kmeans = KMeans(n_clusters=k, random_state=None, n_init='auto')
                labels = kmeans.fit_predict(data_for_clustering)
                cluster_centers_map = {i: kmeans.cluster_centers_[i] for i in range(k)}
                centers_for_points = kmeans.cluster_centers_[labels]
                raw_distances = np.linalg.norm(data_for_clustering - centers_for_points, axis=1)

            elif clustering_method == "dbscan":
                log_and_update(f"  Running DBSCAN (eps={current_dbscan_eps}, min_samples={current_dbscan_min_samples})...", progress_base + 5)
                dbscan = DBSCAN(eps=current_dbscan_eps, min_samples=current_dbscan_min_samples)
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
                        cluster_centers_map[cluster_id] = center
            else:
                log_and_update(f"Unsupported clustering algorithm: {clustering_method}", 100)
                continue 

            if labels is None or len(set(labels) - {-1}) == 0: 
                log_and_update(f"  Run {run_idx + 1}: No valid clusters generated for this parameter set. Skipping.", progress_base + 8)
                continue

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
                    if len(selected) >= max_songs_per_playlist_sampled: # Apply sampled max songs for this original cluster
                        break
                
                if len(selected) >= min_songs_per_playlist_sampled:
                    for t in selected:
                        item_id, title, author = t["row"][0], t["row"][1], t["row"][2]
                        filtered_clusters[cluster_id].append((item_id, title, author))

            current_named_playlists_merged = defaultdict(list)
            current_playlist_centroids_info = {} 
            
            unique_predominant_mood_scores = {} 

            for label_id, songs_in_cluster in filtered_clusters.items():
                if songs_in_cluster: # songs_in_cluster is already limited by max_songs_per_playlist_sampled for this original cluster
                    center = cluster_centers_map.get(label_id) 
                    if center is None:
                        continue 

                    name, top_scores = name_cluster(center, pca_model, current_pca_enabled, MOOD_LABELS)
                    
                    if top_scores and any(mood in MOOD_LABELS for mood in top_scores.keys()):
                        predominant_mood_key = max(top_scores, key=lambda k: top_scores[k] if k in MOOD_LABELS else -1)
                        if predominant_mood_key in MOOD_LABELS:
                            current_mood_score = top_scores.get(predominant_mood_key, 0.0)
                            unique_predominant_mood_scores[predominant_mood_key] = max(
                                unique_predominant_mood_scores.get(predominant_mood_key, 0.0),
                                current_mood_score
                            )
                    # This is where multiple original clusters with the same name merge their songs
                    current_named_playlists_merged[name].extend(songs_in_cluster)
                    # Store centroid info, potentially overwritten if multiple original clusters map to the same name
                    current_playlist_centroids_info[name] = top_scores


            # --- Apply max_songs_per_playlist_sampled to the MERGED playlists ---
            processed_current_playlists = defaultdict(list)
            for name, song_list in current_named_playlists_merged.items():
                if len(song_list) > max_songs_per_playlist_sampled:
                    # Optional: random.shuffle(song_list) before truncating if desired
                    processed_current_playlists[name] = song_list[:max_songs_per_playlist_sampled]
                else:
                    processed_current_playlists[name] = song_list
            # --- End of MERGED playlist truncation ---

            diversity_score = sum(unique_predominant_mood_scores.values())
            log_and_update(f"  Run {run_idx + 1}: Weighted Diversity Score: {diversity_score:.2f}.", progress_base + 8)

            if diversity_score > best_diversity_score:
                best_diversity_score = diversity_score
                best_clustering_results = {
                    "named_playlists": processed_current_playlists, # Store the TRUNCATED merged playlists
                    "playlist_centroids": current_playlist_centroids_info, 
                    "pca_model": pca_model, 
                    "parameters": { 
                        "clustering_method": clustering_method,
                        "num_clusters": current_num_clusters if clustering_method == "kmeans" else len(set(labels) - {-1}), # Store actual number of clusters
                        "dbscan_eps": current_dbscan_eps,
                        "dbscan_min_samples": current_dbscan_min_samples,
                        "pca_enabled": current_pca_enabled,
                        "pca_components": current_pca_components,
                        "min_songs_per_playlist": min_songs_per_playlist_sampled,
                        "max_songs_per_playlist": max_songs_per_playlist_sampled # This is the crucial value
                    }
                }
                best_parameters_found = best_clustering_results["parameters"] 

        if not best_clustering_results:
            log_and_update("No valid clusters found after multiple runs.", 100)
            self.update_state(state='FAILURE', meta={'progress': 100, 'status': 'No valid clusters found', 'log_output': log_messages})
            return {"status": "FAILURE", "message": "No valid clusters found after multiple runs."}

        log_and_update(f"Applying best clustering results (Weighted Diversity Score: {best_diversity_score:.2f})...", 90)
        log_and_update(f"Best parameters: {best_parameters_found}", 90)
        
        final_named_playlists = best_clustering_results["named_playlists"]
        final_playlist_centroids = best_clustering_results["playlist_centroids"]
        # final_pca_model = best_clustering_results["pca_model"] # Not directly used later but good to have

        log_and_update("Updating playlist database...", 95)
        update_playlist_table(DB_PATH, final_named_playlists)
        
        log_and_update("Creating/Updating playlists on Jellyfin...", 98)
        # Note: create_or_update_playlists_on_jellyfin uses global MAX_SONGS_PER_CLUSTER for chunking to Jellyfin API
        # This is separate from the max_songs_per_playlist logic applied above.
        create_or_update_playlists_on_jellyfin(JELLYFIN_URL, JELLYFIN_USER_ID, {"X-Emby-Token": JELLYFIN_TOKEN}, final_named_playlists, final_playlist_centroids, MOOD_LABELS)
        
        log_and_update(f"Playlists generated and updated on Jellyfin! Best run had weighted diversity score of {best_diversity_score:.2f}.", 100)
        return {"status": "SUCCESS", "message": f"Playlists generated and updated on Jellyfin! Best run had weighted diversity score of {best_diversity_score:.2f}."}

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"FATAL ERROR: Clustering failed: {e}\n{error_traceback}")
        log_and_update(f"❌ Clustering failed: {e}", 100)
        self.update_state(state='FAILURE', meta={'progress': 100, 'status': f'Clustering failed: {e}', 'log_output': log_messages + [f"Error Traceback: {error_traceback}"]})
        return {"status": "FAILURE", "message": f"Clustering failed: {e}"}


# --- API Endpoints ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/api/analysis/start', methods=['POST'])
def start_analysis():
    """
    Starts the music analysis as an asynchronous Celery task.
    Records the task ID and type in the database.
    """
    data = request.json or {}
    jellyfin_url = data.get('jellyfin_url', JELLYFIN_URL)
    jellyfin_user_id = data.get('jellyfin_user_id', JELLYFIN_USER_ID)
    jellyfin_token = data.get('jellyfin_token', JELLYFIN_TOKEN)
    num_recent_albums = int(data.get('num_recent_albums', NUM_RECENT_ALBUMS)) # NUM_RECENT_ALBUMS from config
    top_n_moods = int(data.get('top_n_moods', TOP_N_MOODS))
    task = run_analysis_task.delay(jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods)
    save_task_status(task.id, "analysis", "PENDING")
    return jsonify({"task_id": task.id, "task_type": "analysis", "status": "PENDING"}), 202

@app.route('/api/clustering/start', methods=['POST'])
def start_clustering():
    """
    Starts the playlist clustering as an asynchronous Celery task.
    Accepts parameter ranges for evolutionary search, including min/max songs per playlist.
    Records the task ID and type in the database.
    """
    data = request.json

    clustering_method = data.get('clustering_method', CLUSTER_ALGORITHM)
    num_clusters_min = int(data.get('num_clusters_min', NUM_CLUSTERS_MIN))
    num_clusters_max = int(data.get('num_clusters_max', NUM_CLUSTERS_MAX))
    dbscan_eps_min = float(data.get('dbscan_eps_min', DBSCAN_EPS_MIN))
    dbscan_eps_max = float(data.get('dbscan_eps_max', DBSCAN_EPS_MAX))
    dbscan_min_samples_min = int(data.get('dbscan_min_samples_min', DBSCAN_MIN_SAMPLES_MIN))
    dbscan_min_samples_max = int(data.get('dbscan_min_samples_max', DBSCAN_MIN_SAMPLES_MAX))
    pca_components_min = int(data.get('pca_components_min', PCA_COMPONENTS_MIN))
    pca_components_max = int(data.get('pca_components_max', PCA_COMPONENTS_MAX))
    num_clustering_runs = int(data.get('clustering_runs', CLUSTERING_RUNS))
    min_songs_per_playlist = int(data.get('min_songs_per_playlist', MIN_SONGS_PER_PLAYLIST))
    max_songs_per_playlist = int(data.get('max_songs_per_playlist', MAX_SONGS_PER_PLAYLIST))

    task = run_clustering_task.delay(
        clustering_method, 
        num_clusters_min, num_clusters_max,
        dbscan_eps_min, dbscan_eps_max,
        dbscan_min_samples_min, dbscan_min_samples_max,
        pca_components_min, pca_components_max,
        num_clustering_runs,
        min_songs_per_playlist, max_songs_per_playlist 
    )
    save_task_status(task.id, "clustering", "PENDING")
    return jsonify({"task_id": task.id, "task_type": "clustering", "status": "PENDING"}), 202


@app.route('/api/status/<task_id>', methods=['GET'])
def get_task_status_endpoint(task_id):
    """
    Retrieves the status of any task (analysis or clustering).
    """
    task = AsyncResult(task_id, app=celery)
    # Ensure task.info is a dictionary before trying to get 'task_type'
    task_info_dict = task.info if isinstance(task.info, dict) else {}
    task_type = task_info_dict.get('task_type', 'unknown') # Get task_type from Celery meta if available

    response = {
        'task_id': task.id,
        'state': task.state,
        'status': 'Processing...', # Default status message
        'task_type': task_type # Include task_type in the response
    }
    
    # Update response with detailed task information from Celery meta
    response.update(task_info_dict)


    if task.state == 'PENDING':
        response['status'] = 'Task is pending or not yet started.'
        # Ensure default progress/log fields exist if not in task_info_dict
        response.setdefault('progress', 0)
        response.setdefault('log_output', ['Task pending...'])
        if task_type == 'analysis': # Specific defaults for analysis task type
            response.setdefault('current_album', 'N/A')
            response.setdefault('current_album_idx', 0)
            response.setdefault('total_albums', 0)
    elif task.state == 'SUCCESS':
        response['status'] = task_info_dict.get('message', 'Task complete!')
        response.setdefault('progress', 100)
        save_task_status(task_id, task_type, "SUCCESS")
    elif task.state == 'FAILURE':
        response['status'] = task_info_dict.get('message', str(task.info)) # Use detailed message if available
        response.setdefault('progress', 100)
        # Ensure log_output contains the error if not already present
        if not task_info_dict.get('log_output') or str(task.info) not in task_info_dict.get('log_output', []):
            current_logs = task_info_dict.get('log_output', [])
            if isinstance(current_logs, list):
                 current_logs.append(f"Error: {str(task.info)}")
                 response['log_output'] = current_logs
            else: # if log_output was not a list for some reason
                 response['log_output'] = [f"Error: {str(task.info)}"]

        save_task_status(task_id, task_type, "FAILURE")
    elif task.state == 'REVOKED':
        response['status'] = 'Task revoked.'
        response.setdefault('progress', 100) # Typically revocation implies completion of the cancellation
        response.setdefault('log_output', ['Task was cancelled.'])
        save_task_status(task_id, task_type, "REVOKED")
    elif task.state == 'PROGRESS':
        # Status message should already be in task_info_dict from log_and_update
        response['status'] = task_info_dict.get('status', 'Processing...')
    else: # Covers STARTED or any other unknown Celery states
        response['status'] = f'Task state: {task.state}'
        response.setdefault('progress', task_info_dict.get('progress', 0)) # Use progress from meta if available

    return jsonify(response)

@app.route('/api/cancel/<task_id>', methods=['POST'])
def cancel_task_endpoint(task_id):
    """
    Cancels an active task (analysis or clustering).
    """
    task = AsyncResult(task_id, app=celery)
    task_type_from_db = 'unknown' # Default
    
    # Attempt to get task_type from Celery's result backend first
    celery_task_info = task.info if isinstance(task.info, dict) else {}
    task_type_from_celery = celery_task_info.get('task_type')

    if task_type_from_celery:
        task_type_from_db = task_type_from_celery
    else:
        # Fallback to DB if not in Celery meta (e.g., if task never fully started to update meta)
        conn = get_status_db()
        cur = conn.cursor()
        cur.execute("SELECT task_type FROM task_status WHERE task_id = ?", (task_id,))
        row = cur.fetchone()
        if row:
            task_type_from_db = row['task_type']
        conn.close() # Ensure connection is closed if opened here

    if task.state in ['PENDING', 'STARTED', 'PROGRESS']:
        task.revoke(terminate=True, signal='SIGKILL') # Using SIGKILL for more forceful termination
        save_task_status(task_id, task_type_from_db, "REVOKED")
        return jsonify({"message": "Task cancellation requested.", "task_id": task_id, "task_type": task_type_from_db}), 200
    else:
        return jsonify({"message": "Task cannot be cancelled in its current state.", "state": task.state, "task_type": task_type_from_db}), 400

@app.route('/api/last_task', methods=['GET'])
def get_last_overall_task_status():
    """
    Retrieves the status of the last recorded task, regardless of type.
    """
    last_task_db = get_last_task_status() # Fetches from DB: task_id, task_type, status
    if last_task_db and last_task_db.get('task_id'):
        # Get the detailed status from Celery for the last task ID
        task_id = last_task_db['task_id']
        task = AsyncResult(task_id, app=celery)
        celery_info = task.info if isinstance(task.info, dict) else {}

        # Prioritize Celery's state and info, but use DB's task_type if Celery's is missing
        response = {
            'task_id': task_id,
            'state': task.state, # Celery's current state
            'status': celery_info.get('status', last_task_db.get('status', 'Unknown')), # Celery status, fallback to DB status
            'task_type': celery_info.get('task_type', last_task_db.get('task_type')), # Celery type, fallback to DB type
        }
        # Merge other details from Celery if available
        response.update(celery_info)
        return jsonify(response), 200
    return jsonify({"task_id": None, "task_type": None, "status": "NO_PREVIOUS_TASK"}), 200


@app.route('/api/config', methods=['GET'])
def get_config():
    """Returns the current configuration parameters, including new ranges."""
    # NUM_RECENT_ALBUMS is used for analysis, not min/max like in config.py comments
    return jsonify({
        "jellyfin_url": JELLYFIN_URL,
        "jellyfin_user_id": JELLYFIN_USER_ID,
        "jellyfin_token": JELLYFIN_TOKEN,
        "num_recent_albums": NUM_RECENT_ALBUMS, # This is the one used by analysis task
        "max_distance": MAX_DISTANCE,
        "max_songs_per_cluster": MAX_SONGS_PER_CLUSTER, # Used for Jellyfin API chunking
        "max_songs_per_artist": MAX_SONGS_PER_ARTIST,
        "cluster_algorithm": CLUSTER_ALGORITHM,
        "num_clusters_min": NUM_CLUSTERS_MIN,
        "num_clusters_max": NUM_CLUSTERS_MAX,
        "dbscan_eps_min": DBSCAN_EPS_MIN,
        "dbscan_eps_max": DBSCAN_EPS_MAX,
        "dbscan_min_samples_min": DBSCAN_MIN_SAMPLES_MIN,
        "dbscan_min_samples_max": DBSCAN_MIN_SAMPLES_MAX,
        "pca_components_min": PCA_COMPONENTS_MIN,
        "pca_components_max": PCA_COMPONENTS_MAX,
        "top_n_moods": TOP_N_MOODS,
        "mood_labels": MOOD_LABELS,
        "clustering_runs": CLUSTERING_RUNS,
        "min_songs_per_playlist": MIN_SONGS_PER_PLAYLIST, 
        "max_songs_per_playlist": MAX_SONGS_PER_PLAYLIST, 
    })

@app.route('/api/playlists', methods=['GET'])
def get_playlists():
    """Retrieves all saved playlists from the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT playlist, item_id, title, author FROM playlist ORDER BY playlist, title")
        rows = cur.fetchall()
    playlists = defaultdict(list)
    for playlist, item_id, title, author in rows:
        playlists[playlist].append({"item_id": item_id, "title": title, "author": author})
    return jsonify(dict(playlists)), 200

if __name__ == '__main__':
    init_db(DB_PATH)
    os.makedirs(TEMP_DIR, exist_ok=True)
    # Use host='0.0.0.0' to be accessible externally, e.g., in Docker
    app.run(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true', host='0.0.0.0', port=int(os.getenv('FLASK_PORT', 8000)))
