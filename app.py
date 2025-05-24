import os
import shutil
import sqlite3
import requests
from collections import defaultdict
import numpy as np
from flask import Flask, jsonify, request, render_template, g
from celery import Celery, signals # Import signals for process initialization
from celery.result import AsyncResult
from celery.exceptions import Terminated # Import Terminated exception for cancellation
from contextlib import closing
import json
import time
# REMOVED: import tensorflow as tf # This line is absolutely removed now!

# Import your existing analysis functions from essentia
from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor, TensorflowPredictMusiCNN, TensorflowPredict2D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Your existing config - assuming this is from config.py and sets global variables
from config import *

# --- Flask App Setup ---
app = Flask(__name__)
# Flask app config for Celery broker/backend
app.config['CELERY_BROKER_URL'] = CELERY_BROKER_URL
app.config['CELERY_RESULT_BACKEND'] = CELERY_RESULT_BACKEND

# --- Celery App Setup ---
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
celery.conf.update(task_track_started=True) # Ensure task state updates are tracked

# --- Global model instances for each worker process ---
# We will NOT be pre-loading them globally anymore. Instead, they will be
# instantiated on demand within the `predict_moods` function.
# This makes the _embedding_model_instance and _prediction_model_instance
# global variables redundant, as they won't be used for persistent model loading.
# They are being kept as 'None' for now to avoid breaking existing references,
# but their actual usage will be removed.
_embedding_model_instance = None
_prediction_model_instance = None


# --- Celery Signal Handler for TensorFlow Model Loading (NO LONGER NEEDED FOR TF CLEANUP) ---
# Since we are instantiating models per-call, there's no shared TF graph/session
# to clear from the parent process. This signal handler can be removed.
# @signals.worker_process_init.connect
# def init_tf_models_in_worker(**kwargs):
#     # This function is now entirely redundant since TensorFlow models are
#     # instantiated on demand within `predict_moods` and Essentia handles
#     # the underlying TF C++ library.
#     pass # No action needed here


# --- Status DB Setup ---
def init_status_db():
    with closing(sqlite3.connect(STATUS_DB_PATH)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute('''CREATE TABLE IF NOT EXISTS analysis_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE,
                status TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )''')
            conn.commit()

with app.app_context():
    init_status_db()

def get_status_db():
    if 'status_db' not in g:
        g.status_db = sqlite3.connect(STATUS_DB_PATH)
        g.status_db.row_factory = sqlite3.Row
    return g.status_db

@app.teardown_appcontext
def close_status_db(exception):
    status_db = g.pop('status_db', None)
    if status_db is not None:
        status_db.close()

def save_analysis_task_id(task_id, status="PENDING"):
    conn = get_status_db()
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO analysis_status (task_id, status) VALUES (?, ?)", (task_id, status))
    conn.commit()

def get_last_analysis_task_id():
    conn = get_status_db()
    cur = conn.cursor()
    cur.execute("SELECT task_id, status FROM analysis_status ORDER BY timestamp DESC LIMIT 1")
    row = cur.fetchone()
    return dict(row) if row else None

# --- Existing Script Functions (No Global Mutation) ---

def clean_temp(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"WARNING: Could not remove {file_path} from {temp_dir}: {e}")

def init_db(db_path):
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

def get_recent_albums(jellyfin_url, headers, jellyfin_user_id, limit): # Corrected order
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

def get_tracks_from_album(jellyfin_url, headers, jellyfin_user_id, album_id): # Corrected order
    url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {"ParentId": album_id, "IncludeItemTypes": "Audio"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("Items", []) if r.ok else []
    except Exception as e:
        print(f"ERROR: get_tracks_from_album {album_id}: {e}")
        return []

def get_jellyfin_audio_url(jellyfin_url, item_id, jellyfin_user_id):
    # This is a helper that was missing, assuming it constructs the correct URL
    return f"{jellyfin_url}/Audio/{item_id}/stream?api_key={JELLYFIN_TOKEN}&UserId={jellyfin_user_id}"

def download_track(jellyfin_url, headers, temp_dir, item_id, item_name, jellyfin_user_id):
    # Modified to take item_id and item_name directly, and jellyfin_user_id
    filename_safe = f"{item_name.replace('/', '_')}-{item_id}.mp3" # Use ID for uniqueness
    path = os.path.join(temp_dir, filename_safe)
    try:
        download_url = get_jellyfin_audio_url(jellyfin_url, item_id, jellyfin_user_id) # Use the helper
        r = requests.get(download_url, headers=headers, timeout=120)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        return path
    except Exception as e:
        print(f"ERROR: download_track {item_name} (ID: {item_id}): {e}")
        return None

# --- Reverted predict_moods to instantiate models per-call ---
def predict_moods(file_path, embedding_model_path, prediction_model_path, mood_labels, top_n_moods):
    print(f"DEBUG: predict_moods: Starting prediction for {os.path.basename(file_path)} by loading models.")
    audio = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()

    # Instantiate models here, on demand for each call
    embedding_model = TensorflowPredictMusiCNN(
        graphFilename=embedding_model_path, output="model/dense/BiasAdd"
    )
    embeddings = embedding_model(audio)

    prediction_model = TensorflowPredict2D(
        graphFilename=prediction_model_path,
        input="serving_default_model_Placeholder",
        output="PartitionedCall"
    )
    predictions = prediction_model(embeddings)[0]

    results = dict(zip(mood_labels, predictions))
    return {label: float(score) for label, score in sorted(results.items(), key=lambda x: -x[1])[:top_n_moods]}

# --- Modified analyze_track to pass model paths ---
def analyze_track(file_path, mood_labels, top_n_moods):
    audio = MonoLoader(filename=file_path)()
    tempo, _, _, _, _ = RhythmExtractor2013()(audio)
    key, scale, _ = KeyExtractor()(audio)
    # Pass the model paths to predict_moods
    moods = predict_moods(file_path, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, mood_labels, top_n_moods)
    return tempo, key, scale, moods

def track_exists(db_path, item_id):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM score WHERE item_id=?", (item_id,))
        row = cur.fetchone()
    return row

def save_track_analysis(db_path, item_id, title, author, tempo, key, scale, moods):
    mood_str = ','.join(f"{k}:{v:.3f}" for k, v in moods.items())
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO score VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (item_id, title, author, tempo, key, scale, mood_str))
        conn.commit()

def score_vector(row, mood_labels):
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
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM score")
        rows = cur.fetchall()
    return rows

def name_cluster(centroid_scaled_vector, pca_model, pca_enabled, mood_labels):
    if pca_enabled and pca_model is not None:
        scaled_vector = pca_model.inverse_transform(centroid_scaled_vector)
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
    top_indices = np.argsort(mood_values)[::-1][:3]
    mood_names = [mood_labels[i] for i in top_indices]
    mood_part = "_".join(mood_names).title()
    full_name = f"{mood_part}_{tempo_label}"
    top_mood_scores = {mood_labels[i]: mood_values[i] for i in top_indices}
    extra_info = {"tempo": round(tempo_norm, 2)}
    return full_name, {**top_mood_scores, **extra_info}

def update_playlist_table(db_path, playlists):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM playlist")
        for name, cluster in playlists.items():
            for item_id, title, author in cluster:
                cur.execute("INSERT INTO playlist VALUES (?, ?, ?, ?)", (name, item_id, title, author))
        conn.commit()

def delete_old_automatic_playlists(jellyfin_url, jellyfin_user_id, headers):
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

def create_or_update_playlists_on_jellyfin(jellyfin_url, jellyfin_user_id, headers, playlists, cluster_centers, pca_model, mood_labels):
    delete_old_automatic_playlists(jellyfin_url, jellyfin_user_id, headers)
    for base_name, cluster in playlists.items():
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
                    centroid_info = cluster_centers[base_name]
                    top_moods = {k: v for k, v in centroid_info.items() if k in mood_labels}
                    extra_info = {k: v for k, v in centroid_info.items() if k not in mood_labels}
                    centroid_str = ", ".join(f"{k}:{v:.2f}" for k, v in top_moods.items())
                    extras_str = ", ".join(f"{k}:{v:.2f}" for k, v in extra_info.items())
                    print(f"✅ Created playlist {playlist_name} with {len(item_ids)} tracks (Centroid: {centroid_str} | {extras_str})")
            except Exception as e:
                print(f"Exception creating {playlist_name}: {e}")

# --- Celery Task Definition (with self.is_revoked() checks) ---
@celery.task(bind=True)
def run_analysis_task(self, jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods):
    headers = {"X-Emby-Token": jellyfin_token}
    log_messages = []

    def log_and_update(message, progress, current_album=None, current_album_idx=0, total_albums=0, current_track=None, current_track_idx=0, total_tracks_in_album=0):
        log_messages.append(message)
        self.update_state(state='PROGRESS', meta={
            'progress': progress,
            'status': message,
            'log_output': log_messages,
            'current_album': current_album,
            'current_album_idx': current_album_idx,
            'total_albums': total_albums,
            'current_track': current_track,
            'current_track_idx': current_track_idx,
            'total_tracks_in_album': total_tracks_in_album
        })
        print(f"DEBUG: Task {self.request.id} - {message} (Progress: {progress:.2f}%)")
        
        # Crucial check for task revocation
        if self.request.id and self.is_revoked():
            print(f"WARNING: Task {self.request.id} has been revoked during '{message}'. Aborting.")
            # Raise Terminated exception to stop execution immediately
            raise Terminated(f"Task was revoked during step: {message}")

    try:
        log_and_update("🚀 Starting mood-based analysis and playlist generation...", 0)
        clean_temp(TEMP_DIR)
        init_db(DB_PATH)

        # 1. Fetch albums
        log_and_update("Fetching recent albums from Jellyfin...", 5)
        albums = get_recent_albums(jellyfin_url, headers, jellyfin_user_id, num_recent_albums)
        if not albums:
            log_and_update("⚠️ No new albums to analyze. Proceeding with existing data or finishing.", 10)
            # If no albums, we might skip track analysis and go straight to clustering existing data
            # Or exit if no tracks exist for clustering
            all_tracks_to_process = []
        else:
            total_albums = len(albums)
            all_tracks_to_process = []
            for idx, album in enumerate(albums, 1):
                log_and_update(f"🎵 Fetching tracks for Album: {album['Name']} ({idx}/{total_albums})", 5 + (idx / total_albums) * 5, current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                tracks = get_tracks_from_album(jellyfin_url, headers, jellyfin_user_id, album['Id'])
                if tracks:
                    all_tracks_to_process.extend([ (track, album['Name'], idx, total_albums) for track in tracks])
                else:
                    log_and_update(f"   ⚠️ No tracks found for album: {album['Name']}", 5 + (idx / total_albums) * 5, current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
        
        total_tracks_overall = len(all_tracks_to_process)
        if total_tracks_overall == 0 and not get_all_tracks(DB_PATH): # Check if DB is empty and no new tracks
             log_and_update("No tracks found or analyzed. Nothing to do.", 100)
             return {"status": "SUCCESS", "message": "No tracks found or analyzed. Nothing to do."}
        
        processed_count = 0
        analysis_start_progress = 10
        analysis_end_progress = 90 # Leave some space for clustering

        # 2. Process each track (analysis)
        for i, (item, album_name, album_idx, total_albums) in enumerate(all_tracks_to_process, 1):
            track_name_full = f"{item['Name']} by {item.get('AlbumArtist', 'Unknown')}"
            item_id = item['Id']

            # Calculate progress more precisely
            current_overall_progress = analysis_start_progress + int((analysis_end_progress - analysis_start_progress) * (processed_count / max(1, total_tracks_overall)))
            
            log_and_update(f"   🎶 Analyzing track: {track_name_full}", current_overall_progress, 
                           current_album=album_name, current_album_idx=album_idx, total_albums=total_albums,
                           current_track=item['Name'], current_track_idx=i, total_tracks_in_album=total_tracks_overall) # total_tracks_overall here
            
            # Check for revocation before processing each track
            if self.is_revoked():
                raise Terminated(f"Task revoked during processing of track {track_name_full}.")

            try:
                if track_exists(DB_PATH, item_id):
                    log_and_update(f"     ⏭️ Skipping '{track_name_full}' (already analyzed)", current_overall_progress, current_album=album_name, current_album_idx=album_idx, total_albums=total_albums)
                    processed_count += 1 # Still count it for progress
                    continue

                # Download track
                download_start_time = time.time()
                log_and_update(f"     ⬇️ Downloading '{track_name_full}'...", current_overall_progress, current_album=album_name, current_album_idx=album_idx, total_albums=total_albums)
                path = download_track(jellyfin_url, headers, TEMP_DIR, item_id, item['Name'], jellyfin_user_id) # Pass correct args
                if not path:
                    log_and_update(f"     ❌ Failed to download '{track_name_full}'. Skipping.", current_overall_progress, current_album=album_name, current_album_idx=album_idx, total_albums=total_albums)
                    processed_count += 1
                    continue
                log_and_update(f"     Downloaded '{track_name_full}' in {time.time() - download_start_time:.2f}s.", current_overall_progress, current_album=album_name, current_album_idx=album_idx, total_albums=total_albums)
                
                # Check for revocation after download
                if self.is_revoked():
                    if path and os.path.exists(path): os.remove(path) # Cleanup
                    raise Terminated(f"Task revoked after downloading {track_name_full}.")

                # Analyze track
                analysis_start_time = time.time()
                log_and_update(f"     🔬 Analyzing '{track_name_full}' with Essentia/TF...", current_overall_progress, current_album=album_name, current_album_idx=album_idx, total_albums=total_albums)
                # Call analyze_track with correct args (passing model paths implicitly from config)
                tempo, key, scale, moods = analyze_track(path, MOOD_LABELS, top_n_moods) 
                analysis_duration = time.time() - analysis_start_time
                save_track_analysis(DB_PATH, item_id, item['Name'], item.get('AlbumArtist', 'Unknown'), tempo, key, scale, moods)
                log_and_update(f"     ✅ Analyzed '{track_name_full}' in {analysis_duration:.2f}s. Moods: {', '.join(f'{k}:{v:.2f}' for k,v in moods.items())}", current_overall_progress, current_album=album_name, current_album_idx=album_idx, total_albums=total_albums)
                
                # Check for revocation after analysis
                if self.is_revoked():
                    if path and os.path.exists(path): os.remove(path) # Cleanup
                    raise Terminated(f"Task revoked after analyzing {track_name_full}.")

                processed_count += 1

            except Terminated:
                # This catches Terminated raised from within the loop or `log_and_update`
                raise # Re-raise to let the outer try-except handle it

            except Exception as e:
                print(f"ERROR: Error processing track {track_name_full} (ID: {item_id}): {e}")
                import traceback
                print(traceback.format_exc()) # Print full traceback for track-specific errors
                log_and_update(f"     ❌ Error processing '{track_name_full}': {e}", current_overall_progress, current_album=album_name, current_album_idx=album_idx, total_albums=total_albums)
                processed_count += 1 # Still count for progress, even if failed
            finally:
                # Ensure temporary file is cleaned up regardless of success or failure
                if 'path' in locals() and path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as cleanup_e:
                        print(f"WARNING: Failed to clean up temp file {path}: {cleanup_e}")
        
        clean_temp(TEMP_DIR) # Final cleanup of temp directory

        # 3. Clustering
        log_and_update("Starting clustering and playlist creation...", 90)
        
        # Check for revocation before clustering
        if self.is_revoked():
            raise Terminated("Task revoked before clustering.")

        rows = get_all_tracks(DB_PATH)
        if len(rows) < 2:
            log_and_update("Not enough analyzed tracks for clustering. Skipping playlist generation.", 100)
            return {"status": "SUCCESS", "message": "Analysis complete, but not enough tracks for clustering."}

        X_original = [score_vector(row, MOOD_LABELS) for row in rows]
        X_scaled = X_original
        pca_model = None

        if PCA_ENABLED:
            # You might need to make PCA_COMPONENTS configurable in config.py
            pca_components_val = min(len(MOOD_LABELS) + 1, len(rows) - 1) # Ensure components are valid
            pca_model = PCA(n_components=pca_components_val)
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
            raw_distances = np.zeros(len(data_for_clustering)) # Initialize
            for cluster_id in set(labels):
                if cluster_id == -1: # Noise points
                    continue
                indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                cluster_points = np.array([data_for_clustering[i] for i in indices])
                if len(cluster_points) > 0:
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
            if label == -1: # Exclude noise points from clusters
                continue
            track_info.append({"row": row, "label": label, "vector": vec, "distance": dist})

        filtered_clusters = defaultdict(list)
        for cluster_id in set(labels):
            if cluster_id == -1: # Exclude noise points
                continue
            cluster_tracks = [t for t in track_info if t["label"] == cluster_id and t["distance"] <= MAX_DISTANCE]
            if not cluster_tracks:
                continue
            cluster_tracks.sort(key=lambda x: x["distance"]) # Sort by distance to centroid
            
            # Apply MAX_SONGS_PER_ARTIST and MAX_SONGS_PER_CLUSTER
            count_per_artist = defaultdict(int)
            selected_tracks_for_playlist = []
            for t in cluster_tracks:
                author = t["row"][2] # Author is at index 2 in the `score` table row
                if count_per_artist[author] < MAX_SONGS_PER_ARTIST:
                    selected_tracks_for_playlist.append(t)
                    count_per_artist[author] += 1
                if len(selected_tracks_for_playlist) >= MAX_SONGS_PER_CLUSTER:
                    break
            
            for t in selected_tracks_for_playlist:
                item_id, title, author = t["row"][0], t["row"][1], t["row"][2]
                filtered_clusters[cluster_id].append((item_id, title, author))

        named_playlists = defaultdict(list)
        playlist_centroids = {}
        for label, songs in filtered_clusters.items():
            if songs:
                center = cluster_centers[label]
                name, top_scores = name_cluster(center, pca_model, PCA_ENABLED, MOOD_LABELS)
                named_playlists[name].extend(songs)
                playlist_centroids[name] = top_scores
        
        log_and_update("Updating internal playlist database...", 95)
        update_playlist_table(DB_PATH, named_playlists)

        log_and_update("Creating/Updating playlists on Jellyfin...", 98)
        create_or_update_playlists_on_jellyfin(jellyfin_url, jellyfin_user_id, headers, named_playlists, playlist_centroids, pca_model, MOOD_LABELS)

        log_and_update("Analysis and playlist generation complete!", 100)
        return {"status": "SUCCESS", "message": "Analysis and playlist generation complete!"}

    except Terminated as e:
        # This block catches the Terminated exception raised by self.is_revoked() checks
        print(f"Task {self.request.id} was terminated by revocation: {e}")
        self.update_state(state='REVOKED', meta={'progress': 0, 'status': 'Task cancelled by user.'})
        return {"status": "REVOKED", "message": f"Task was cancelled: {e}"}

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"FATAL ERROR in run_analysis_task: {e}\n{error_traceback}")
        log_and_update(f"❌ Analysis failed: {e}", 100)
        self.update_state(state='FAILURE', meta={'progress': 100, 'status': f'Analysis failed: {e}', 'log_output': log_messages + [f"Error Traceback: {error_traceback}"]})
        return {"status": "FAILURE", "message": f"Analysis failed: {e}"}

# --- API Endpoints ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analysis/start', methods=['POST'])
def start_analysis():
    data = request.json or {}
    # Use default values from config.py if not provided in request
    jellyfin_url = data.get('jellyfin_url', JELLYFIN_URL)
    jellyfin_user_id = data.get('jellyfin_user_id', JELLYFIN_USER_ID)
    jellyfin_token = data.get('jellyfin_token', JELLYFIN_TOKEN)
    num_recent_albums = int(data.get('num_recent_albums', NUM_RECENT_ALBUMS))
    top_n_moods = int(data.get('top_n_moods', TOP_N_MOODS))

    task = run_analysis_task.delay(jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods)
    save_analysis_task_id(task.id, "PENDING")
    return jsonify({"task_id": task.id, "status": "PENDING"}), 202

@app.route('/api/analysis/status/<task_id>', methods=['GET'])
def analysis_status(task_id):
    task = AsyncResult(task_id, app=celery)
    response = {
        'task_id': task.id,
        'state': task.state,
        'status': 'Processing...' # Default status if not PROGRESS or final
    }
    task_info = task.info if isinstance(task.info, dict) else {}

    if task.state == 'PENDING':
        response.update({'progress': 0, 'status': 'Task is pending or not yet started.', 'log_output': ['Task pending...']})
    elif task.state == 'PROGRESS':
        response.update(task_info) # Already contains progress, status, log_output etc.
    elif task.state == 'SUCCESS':
        response.update({'progress': 100, 'status': 'Analysis complete!', 'log_output': task_info.get('log_output', ['Analysis complete!'])})
        save_analysis_task_id(task_id, "SUCCESS")
    elif task.state == 'FAILURE':
        response.update({'progress': 100, 'status': task_info.get('status', 'Analysis failed!'), 'log_output': task_info.get('log_output', ['Analysis failed.'])})
        save_analysis_task_id(task_id, "FAILURE")
    elif task.state == 'REVOKED':
        response.update({'progress': 100, 'status': 'Task cancelled by user.', 'log_output': task_info.get('log_output', ['Task was cancelled.'])})
        save_analysis_task_id(task_id, "REVOKED")
    else:
        response.update({'progress': 0, 'status': f'Unknown state: {task.state}', 'log_output': [f'Unknown state: {task.state}']})
        
    return jsonify(response)

@app.route('/api/analysis/cancel/<task_id>', methods=['POST'])
def cancel_analysis(task_id):
    task = AsyncResult(task_id, app=celery)
    if task.state in ['PENDING', 'STARTED', 'PROGRESS']:
        # This sends a SIGTERM signal. The task's `self.is_revoked()` checks
        # will respond to this and raise the Terminated exception.
        task.revoke(terminate=False) # signal='SIGTERM' is default
        save_analysis_task_id(task_id, "REVOKED") # Manually update DB status
        print(f"INFO: Requested soft cancellation for task {task_id}.")
        return jsonify({"message": "Analysis task cancellation requested.", "task_id": task_id}), 200
    else:
        return jsonify({"message": "Task cannot be cancelled in its current state.", "state": task.state}), 400

@app.route('/api/analysis/last_task', methods=['GET'])
def get_last_analysis_status():
    last_task = get_last_analysis_task_id()
    if last_task:
        return jsonify(last_task), 200
    return jsonify({"task_id": None, "status": "NO_PREVIOUS_TASK"}), 200

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        "jellyfin_url": JELLYFIN_URL,
        "jellyfin_user_id": JELLYFIN_USER_ID,
        "jellyfin_token": '***hidden***' if JELLYFIN_TOKEN else 'None', # Don't expose token in API
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
        "temp_dir": TEMP_DIR, # Expose temp_dir for debugging
        "db_path": DB_PATH, # Expose db_path for debugging
        "status_db_path": STATUS_DB_PATH, # Expose status_db_path for debugging
    })

@app.route('/api/clustering', methods=['POST'])
def run_clustering():
    data = request.json
    clustering_method = data.get('clustering_method', CLUSTER_ALGORITHM)
    num_clusters = int(data.get('num_clusters', NUM_CLUSTERS))
    dbscan_eps = float(data.get('dbscan_eps', DBSCAN_EPS))
    dbscan_min_samples = int(data.get('dbscan_min_samples', DBSCAN_MIN_SAMPLES))
    # Note: If PCA_ENABLED is read from config, `pca_components` from request might be redundant
    # or you'd need a specific `PCA_COMPONENTS` config variable.
    pca_enabled_request = data.get('pca_enabled', PCA_ENABLED) # Use PCA_ENABLED from config as default
    pca_components = int(data.get('pca_components', 0)) # If 0, it means auto or off

    try:
        rows = get_all_tracks(DB_PATH)
        if len(rows) < 2:
            return jsonify({"status": "error", "message": "Not enough analyzed tracks for clustering (need at least 2)."}, 400)
        
        # Original vectors (tempo + mood scores)
        X_original = [score_vector(row, MOOD_LABELS) for row in rows]
        
        data_for_clustering = np.array(X_original)
        pca_model = None

        # Apply PCA if enabled
        if pca_enabled_request and pca_components > 0:
            # Ensure n_components is valid: <= number of features and < number of samples
            n_features_for_pca = data_for_clustering.shape[1]
            pca_components_actual = min(pca_components, n_features_for_pca, data_for_clustering.shape[0] -1)
            if pca_components_actual <= 0:
                print("WARNING: PCA enabled but not enough components/samples for PCA. Skipping PCA.")
                pca_enabled_request = False # Effectively disable PCA if invalid
            else:
                pca_model = PCA(n_components=pca_components_actual)
                data_for_clustering = pca_model.fit_transform(data_for_clustering)
                print(f"DEBUG: PCA applied with {pca_components_actual} components.")

        # Clustering
        if clustering_method == "kmeans":
            # Ensure k is valid: >= 1 and <= number of samples
            k = num_clusters if num_clusters > 0 else max(1, len(rows) // MAX_SONGS_PER_CLUSTER)
            k_actual = min(k, data_for_clustering.shape[0])
            if k_actual < 1:
                return jsonify({"status": "error", "message": "Not enough data for KMeans clustering with 1 or more clusters."}), 400
            
            kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(data_for_clustering)
            cluster_centers = {i: kmeans.cluster_centers_[i] for i in range(k_actual)}
            centers_for_points = kmeans.cluster_centers_[labels]
            raw_distances = np.linalg.norm(data_for_clustering - centers_for_points, axis=1)
            print(f"DEBUG: KMeans clustering applied with {k_actual} clusters.")

        elif clustering_method == "dbscan":
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            labels = dbscan.fit_predict(data_for_clustering)
            cluster_centers = {}
            raw_distances = np.zeros(len(data_for_clustering)) # Initialize for all points
            
            # Calculate centroids for DBSCAN clusters (excluding noise -1)
            for cluster_id in set(labels):
                if cluster_id == -1: # Noise points
                    continue
                indices = np.where(labels == cluster_id)[0]
                cluster_points = data_for_clustering[indices]
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    for i in indices:
                        raw_distances[i] = np.linalg.norm(data_for_clustering[i] - center)
                    cluster_centers[cluster_id] = center
            print(f"DEBUG: DBSCAN clustering applied (eps={dbscan_eps}, min_samples={dbscan_min_samples}).")

        else:
            return jsonify({"status": "error", "message": f"Unsupported clustering algorithm: {clustering_method}"}), 400
        
        max_dist = raw_distances.max()
        normalized_distances = raw_distances / max_dist if max_dist > 0 else raw_distances

        track_info = []
        for i, row in enumerate(rows):
            label = labels[i]
            vec = data_for_clustering[i]
            dist = normalized_distances[i]
            if label == -1: # Exclude noise points from clusters
                continue
            track_info.append({"row": row, "label": label, "vector": vec, "distance": dist})

        filtered_clusters = defaultdict(list)
        for cluster_id in set(labels):
            if cluster_id == -1: # Exclude noise points
                continue
            # Filter tracks by MAX_DISTANCE and then sort by distance
            cluster_tracks_filtered_by_dist = [t for t in track_info if t["label"] == cluster_id and t["distance"] <= MAX_DISTANCE]
            if not cluster_tracks_filtered_by_dist:
                continue
            cluster_tracks_filtered_by_dist.sort(key=lambda x: x["distance"]) # Sort by distance to centroid

            # Apply MAX_SONGS_PER_ARTIST and MAX_SONGS_PER_CLUSTER
            count_per_artist = defaultdict(int)
            selected_tracks_for_playlist = []
            for t in cluster_tracks_filtered_by_dist:
                author = t["row"][2] # Author is at index 2 in the `score` table row
                if count_per_artist[author] < MAX_SONGS_PER_ARTIST:
                    selected_tracks_for_playlist.append(t)
                    count_per_artist[author] += 1
                if len(selected_tracks_for_playlist) >= MAX_SONGS_PER_CLUSTER:
                    break
            
            for t in selected_tracks_for_playlist:
                item_id, title, author = t["row"][0], t["row"][1], t["row"][2]
                filtered_clusters[cluster_id].append((item_id, title, author))

        named_playlists = defaultdict(list)
        playlist_centroids = {}
        for label, songs in filtered_clusters.items():
            if songs:
                center = cluster_centers[label]
                name, top_scores = name_cluster(center, pca_model, PCA_ENABLED, MOOD_LABELS)
                named_playlists[name].extend(songs)
                playlist_centroids[name] = top_scores
        
        print("Updating internal playlist database...")
        update_playlist_table(DB_PATH, named_playlists)

        print("Creating/Updating playlists on Jellyfin...")
        create_or_update_playlists_on_jellyfin(JELLYFIN_URL, JELLYFIN_USER_ID, {"X-Emby-Token": JELLYFIN_TOKEN}, named_playlists, playlist_centroids, pca_model, MOOD_LABELS)

        print("Clustering and playlist generation complete!")
        return jsonify({"status": "SUCCESS", "message": "Clustering and playlist generation complete!"})

    except Exception as e:
        import traceback
        print(f"Error during clustering: {e}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"Clustering failed: {e}"}), 500


@app.route('/api/playlists', methods=['GET'])
def get_playlists():
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
    app.run(debug=True, host='0.0.0.0', port=8000)
