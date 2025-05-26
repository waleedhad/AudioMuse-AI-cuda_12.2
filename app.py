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
import traceback # Import traceback for better error logging

# Import your existing analysis functions
# Reverting to original Essentia imports and usage for analysis functions
from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor, TensorflowPredictMusiCNN, TensorflowPredict2D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Your existing config - assuming this is from config.py and sets global variables
from config import *

# --- Flask App Setup ---
app = Flask(__app.name__)
# Update to use environment variables for Celery broker and backend
app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(
    app.config,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='Europe/Rome', # Set your appropriate timezone
    enable_utc=True,
)


# --- Status DB Setup ---
def init_status_db():
    with closing(sqlite3.connect(STATUS_DB_PATH)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute('''CREATE TABLE IF NOT EXISTS analysis_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )''')
            conn.commit()

def get_status_db():
    db = getattr(g, '_status_db', None)
    if db is None:
        db = g._status_db = sqlite3.connect(STATUS_DB_PATH)
        db.row_factory = sqlite3.Row # Allows accessing columns by name
    return db

@app.teardown_appcontext
def close_status_db(exception):
    db = getattr(g, '_status_db', None)
    if db is not None:
        db.close()

def save_task_status(task_id, status, task_type, message=""):
    conn = get_status_db()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO analysis_status (task_id, task_type, status, message) VALUES (?, ?, ?, ?)",
                   (task_id, task_type, status, message))
    conn.commit()

def get_task_type_from_db(task_id):
    conn = get_status_db()
    cursor = conn.cursor()
    cursor.execute("SELECT task_type FROM analysis_status WHERE task_id = ?", (task_id,))
    result = cursor.fetchone()
    return result['task_type'] if result else None


# --- Jellyfin API Functions ---
def get_recent_albums(jellyfin_url, jellyfin_user_id, headers, limit):
    url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {
        "IncludeItemTypes": "MusicAlbum",
        "SortBy": "DateCreated",
        "SortOrder": "Descending",
        "Limit": limit,
        "Recursive": "true",
        "Fields": "PrimaryImage"
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json().get('Items', [])

def get_tracks_from_album(jellyfin_url, jellyfin_user_id, headers, album_id):
    url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {
        "ParentId": album_id,
        "IncludeItemTypes": "Audio",
        "Recursive": "true"
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json().get('Items', [])

def download_track(jellyfin_url, headers, temp_dir, track_item):
    if not track_item.get('Path'):
        print(f"Skipping track {track_item['Name']} due to missing path.")
        return None

    # Construct the download URL using the ItemId
    download_url = f"{jellyfin_url}/Audio/{track_item['Id']}/stream"

    # Sanitize the track name for filename, handle potential duplicates
    filename = os.path.join(temp_dir, f"{track_item['Id']}_{track_item['Name'].replace('/', '_').replace('\\', '_')}.flac")

    try:
        with requests.get(download_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading track {track_item['Name']} ({track_item['Id']}): {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return None

def create_or_update_playlists_on_jellyfin(jellyfin_url, jellyfin_user_id, headers, named_playlists, playlist_centroids, mood_labels):
    print("Attempting to update playlists on Jellyfin...")
    # Get existing playlists
    existing_playlists_url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": "true"}
    response = requests.get(existing_playlists_url, headers=headers, params=params)
    response.raise_for_status()
    existing_playlists = {p['Name']: p['Id'] for p in response.json().get('Items', [])}

    for name, tracks in named_playlists.items():
        track_ids = [track_id for track_id, _, _ in tracks]
        playlist_id = existing_playlists.get(name)

        # Generate description from centroid moods
        centroid_scores = playlist_centroids.get(name, {})
        description_parts = []
        for mood in MOOD_LABELS: # Iterate through all possible moods to ensure order
            if mood in centroid_scores:
                description_parts.append(f"{mood}: {centroid_scores[mood]:.2f}")
        description = "Moods: " + ", ".join(description_parts) if description_parts else "A collection of tracks."

        if playlist_id:
            # Update existing playlist (add/remove tracks, update description)
            print(f"Updating playlist: {name}")
            # Get current tracks in playlist to avoid adding duplicates
            current_playlist_tracks_url = f"{jellyfin_url}/Playlists/{playlist_id}/Items"
            current_tracks_response = requests.get(current_playlist_tracks_url, headers=headers)
            current_tracks_response.raise_for_status()
            current_track_ids = {item['Id'] for item in current_tracks_response.json().get('Items', [])}

            tracks_to_add = [tid for tid in track_ids if tid not in current_track_ids]
            if tracks_to_add:
                add_tracks_url = f"{jellyfin_url}/Playlists/{playlist_id}/Items"
                add_response = requests.post(add_tracks_url, headers=headers, json=tracks_to_add)
                add_response.raise_for_status()
                print(f"Added {len(tracks_to_add)} tracks to playlist '{name}'.")

            # Update description
            update_playlist_url = f"{jellyfin_url}/Items/{playlist_id}"
            update_data = {"Name": name, "Overview": description} # Jellyfin uses Overview for description
            update_response = requests.post(update_playlist_url, headers=headers, json=update_data)
            update_response.raise_for_status()
            print(f"Updated description for playlist '{name}'.")

        else:
            # Create new playlist
            print(f"Creating new playlist: {name}")
            create_playlist_url = f"{jellyfin_url}/Playlists"
            create_data = {
                "Name": name,
                "UserId": jellyfin_user_id,
                "Ids": track_ids,
                "Overview": description # Set description during creation
            }
            create_response = requests.post(create_playlist_url, headers=headers, json=create_data)
            create_response.raise_for_status()
            print(f"Created new playlist '{name}' with {len(track_ids)} tracks.")

# --- Essentia and Analysis Functions (Restored from previous working version) ---

# Load models outside functions to avoid reloading for every track
try:
    EMBEDDING_MODEL = TensorflowPredictMusiCNN(graph=EMBEDDING_MODEL_PATH, output="pool_embeddings")
    PREDICTION_MODEL = TensorflowPredict2D(graph=PREDICTION_MODEL_PATH, input="serving_default_dense_input", output="StatefulPartitionedCall")
except Exception as e:
    print(f"Error loading Essentia models: {e}")
    EMBEDDING_MODEL = None
    PREDICTION_MODEL = None

def analyze_track(audio_path, embedding_model_path, prediction_model_path, mood_labels, top_n):
    if EMBEDDING_MODEL is None or PREDICTION_MODEL is None:
        raise RuntimeError("Essentia models failed to load. Cannot analyze track.")

    # 1. Load audio
    audio = MonoLoader(filename=audio_path, sampleRate=44100)()

    # 2. Extract rhythm features (for tempo and key detection context)
    rhythm = RhythmExtractor2013()
    tempo, beats, beats_confidence, bpm_intervals = rhythm(audio)

    # 3. Key and Scale detection
    key_extractor = KeyExtractor()
    key, scale, strength, key_p, scale_p = key_extractor(audio)

    # 4. Extract embeddings
    embeddings = EMBEDDING_MODEL(audio)

    # 5. Predict moods
    # The model expects a batch dimension, so we wrap embeddings in a list
    mood_predictions_raw = PREDICTION_MODEL(np.array([embeddings]))[0] # Take the first (and only) item from the batch

    # Normalize predictions to sum to 1, if they are not already probabilities
    mood_probabilities = np.exp(mood_predictions_raw) / np.sum(np.exp(mood_predictions_raw))

    # Create a dictionary of moods and their probabilities
    mood_scores = {label: prob for label, prob in zip(mood_labels, mood_probabilities)}

    # Get top N moods
    sorted_moods = sorted(mood_scores.items(), key=lambda item: item[1], reverse=True)
    top_moods = {k: v for k, v in sorted_moods[:top_n]}

    return tempo, key, scale, top_moods

# --- Database Functions ---
def init_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
                item_id TEXT PRIMARY KEY,
                title TEXT,
                author TEXT,
                tempo REAL,
                key TEXT,
                scale TEXT,
                moods TEXT, -- Stored as JSON string
                analysis_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS playlist (
                playlist TEXT,
                item_id TEXT,
                title TEXT,
                author TEXT,
                PRIMARY KEY (playlist, item_id)
            )
        ''')
        conn.commit()

def track_exists(db_path, item_id):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM tracks WHERE item_id = ?", (item_id,))
        return cursor.fetchone() is not None

def save_track_analysis(db_path, item_id, title, author, tempo, key, scale, moods):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        moods_json = json.dumps(moods) # Convert dict to JSON string
        cursor.execute('''
            INSERT OR REPLACE INTO tracks (item_id, title, author, tempo, key, scale, moods)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (item_id, title, author, tempo, key, scale, moods_json))
        conn.commit()

def get_all_tracks(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row # Return rows as dict-like objects
        cursor = conn.cursor()
        cursor.execute("SELECT item_id, title, author, tempo, key, scale, moods FROM tracks")
        return cursor.fetchall()

def update_playlist_table(db_path, named_playlists):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM playlist") # Clear existing playlists
        for name, tracks in named_playlists.items():
            for item_id, title, author in tracks:
                cursor.execute("INSERT INTO playlist (playlist, item_id, title, author) VALUES (?, ?, ?, ?)",
                               (name, item_id, title, author))
        conn.commit()

def score_vector(row, mood_labels):
    moods_dict = json.loads(row['moods'])
    # Create a vector where each element corresponds to a mood label's score
    # Ensure all mood_labels are present, defaulting to 0 if not in moods_dict
    return [moods_dict.get(label, 0.0) for label in mood_labels]

def name_cluster(centroid, pca_model, pca_enabled, mood_labels):
    # If PCA was applied, inverse transform the centroid to original mood space
    if pca_enabled and pca_model is not None:
        original_space_centroid = pca_model.inverse_transform(centroid.reshape(1, -1))[0]
    else:
        original_space_centroid = centroid

    # Create a dictionary of mood scores from the (possibly inverse-transformed) centroid
    mood_scores = {label: score for label, score in zip(mood_labels, original_space_centroid)}

    # Get the top N moods for naming
    sorted_moods = sorted(mood_scores.items(), key=lambda item: item[1], reverse=True)

    # Filter out moods with very low scores to make names more meaningful
    significant_moods = [(m, s) for m, s in sorted_moods if s >= 0.1] # Threshold of 0.1

    if significant_moods:
        # Take the top 2 or 3 most significant moods for the name
        top_mood_names = [mood for mood, score in significant_moods[:3]]
        name = " & ".join(top_mood_names).title() + " Mix"
    else:
        name = "Mixed Moods" # Fallback if no significant moods

    # Return the full scores dictionary for the playlist description
    return name, mood_scores

def clean_temp(temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

# --- Celery Tasks ---

@celery.task(bind=True)
def run_analysis_task(self, jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods):
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
            'total_albums': total_albums,
            'task_type': 'analysis'
        })
    try:
        log_and_update("🚀 Starting mood-based analysis...", 0)
        clean_temp(TEMP_DIR)
        init_db(DB_PATH)
        albums = get_recent_albums(jellyfin_url, jellyfin_user_id, headers, num_recent_albums)
        if not albums:
            log_and_update("⚠️ No new albums to analyze. Proceeding with existing data.", 10)
        else:
            total_albums = len(albums)
            analysis_start_progress = 5
            analysis_end_progress = 85
            for idx, album in enumerate(albums, 1):
                # Check for revocation at each album
                if self.request.is_revoked():
                    log_and_update(f"Task {self.request.id} revoked during album processing. Exiting.", 100)
                    return {"status": "REVOKED", "message": "Analysis task was revoked.", 'task_type': 'analysis'}

                album_progress_base = analysis_start_progress + int((analysis_end_progress - analysis_start_progress) * ((idx - 1) / total_albums))
                log_and_update(f"🎵 Processing Album: {album['Name']} ({idx}/{total_albums})", album_progress_base, current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                tracks = get_tracks_from_album(jellyfin_url, jellyfin_user_id, headers, album['Id'])
                if not tracks:
                    log_and_update(f"   ⚠️ No tracks found for album: {album['Name']}", album_progress_base, current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                    continue
                total_tracks_in_album = len(tracks)
                for track_idx, item in enumerate(tracks, 1):
                    # Check for revocation at each track
                    if self.request.is_revoked():
                        log_and_update(f"Task {self.request.id} revoked during track processing. Exiting.", 100)
                        return {"status": "REVOKED", "message": "Analysis task was revoked.", 'task_type': 'analysis'}

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
        return {"status": "SUCCESS", "message": "Analysis complete!", 'task_type': 'analysis'}
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"FATAL ERROR: Analysis failed: {e}\n{error_traceback}")
        log_and_update(f"❌ Analysis failed: {e}", 100)
        self.update_state(state='FAILURE', meta={'progress': 100, 'status': f'Analysis failed: {e}', 'log_output': log_messages + [f"Error Traceback: {error_traceback}"], 'task_type': 'analysis'})
        return {"status": "FAILURE", "message": f"Analysis failed: {e}", 'task_type': 'analysis'}


@celery.task(bind=True)
def run_clustering_task(self, jellyfin_url, jellyfin_user_id, jellyfin_token,
                        clustering_method, num_clusters, dbscan_eps,
                        dbscan_min_samples, pca_components, pca_enabled, num_clustering_runs):

    headers = {"X-Emby-Token": jellyfin_token}
    log_messages = []
    unique_predominant_moods = [] # To store the moods for the frontend

    def log_and_update(message, progress):
        log_messages.append(message)
        self.update_state(state='PROGRESS', meta={
            'progress': progress,
            'status': message,
            'log_output': log_messages,
            'task_type': 'clustering'
        })

    try:
        log_and_update("▶️ Starting clustering and playlist generation...", 0)

        rows = get_all_tracks(DB_PATH)
        if len(rows) < 2:
            log_and_update("⛔ Not enough analyzed tracks for clustering. Requires at least 2.", 100)
            return {"status": "FAILURE", "message": "Not enough analyzed tracks for clustering.", 'task_type': 'clustering'}

        X_original = [score_vector(row, MOOD_LABELS) for row in rows]
        X_scaled = np.array(X_original)

        best_diversity_score = -1
        best_clustering_results = None
        best_unique_predominant_moods = [] # Store for the best run

        log_and_update(f"📊 Running {num_clustering_runs} clustering iterations...", 10)

        for run_idx in range(num_clustering_runs):
            # Check for revocation at each run iteration
            if self.request.is_revoked():
                log_and_update(f"Task {self.request.id} revoked during clustering iterations. Exiting.", 100)
                return {"status": "REVOKED", "message": "Clustering task was revoked.", 'task_type': 'clustering'}

            current_run_progress = 10 + int(80 * ((run_idx + 1) / num_clustering_runs)) # 10% for initial, 80% for runs
            log_and_update(f"🔄 Clustering Run {run_idx + 1}/{num_clustering_runs}", current_run_progress)

            pca_model = None
            data_for_clustering = X_scaled

            if pca_enabled:
                pca_model = PCA(n_components=pca_components)
                X_pca = pca_model.fit_transform(X_scaled)
                data_for_clustering = X_pca

            labels = None
            cluster_centers_raw = {}
            raw_distances = np.zeros(len(data_for_clustering))

            if clustering_method == "kmeans":
                k = num_clusters if num_clusters > 0 else max(1, len(rows) // MAX_SONGS_PER_CLUSTER)
                kmeans = KMeans(n_clusters=min(k, len(rows)), random_state=None, n_init='auto')
                labels = kmeans.fit_predict(data_for_clustering)
                cluster_centers_raw = {i: kmeans.cluster_centers_[i] for i in range(min(k, len(rows)))}
                centers_for_points = kmeans.cluster_centers_[labels]
                raw_distances = np.linalg.norm(data_for_clustering - centers_for_points, axis=1)
            elif clustering_method == "dbscan":
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
                        cluster_centers_raw[cluster_id] = center
            else:
                log_and_update(f"❌ Unsupported clustering algorithm: {clustering_method}", current_run_progress)
                return {"status": "FAILURE", "message": f"Unsupported clustering algorithm: {clustering_method}", 'task_type': 'clustering'}

            max_dist = raw_distances.max()
            normalized_distances = raw_distances / max_dist if max_dist > 0 else raw_distances

            track_info = []
            for row, label, vec, dist in zip(rows, labels, data_for_clustering, normalized_distances):
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

            current_named_playlists = defaultdict(list)
            current_playlist_centroids = {}
            predominant_moods_found_this_run = set()

            for label, songs in filtered_clusters.items():
                if songs:
                    center_raw = cluster_centers_raw[label]
                    name, top_scores = name_cluster(center_raw, pca_model, pca_enabled, MOOD_LABELS)

                    if top_scores:
                        for mood_label in MOOD_LABELS:
                            if mood_label in top_scores and top_scores[mood_label] > 0.1:
                                predominant_moods_found_this_run.add(mood_label)

                    current_named_playlists[name].extend(songs)
                    current_playlist_centroids[name] = top_scores

            diversity_score = len(predominant_moods_found_this_run)
            log_and_update(f"Run {run_idx + 1}: Found {diversity_score} unique predominant moods.", current_run_progress)

            if diversity_score > best_diversity_score:
                best_diversity_score = diversity_score
                best_clustering_results = {
                    "named_playlists": current_named_playlists,
                    "playlist_centroids": current_playlist_centroids,
                }
                best_unique_predominant_moods = list(predominant_moods_found_this_run)

        if not best_clustering_results:
            log_and_update("❌ No valid clusters found after multiple runs.", 100)
            return {"status": "FAILURE", "message": "No valid clusters found after multiple runs.", 'task_type': 'clustering'}

        log_and_update("✅ Clustering complete. Updating Jellyfin playlists...", 95)

        final_named_playlists = best_clustering_results["named_playlists"]
        final_playlist_centroids = best_clustering_results["playlist_centroids"]

        update_playlist_table(DB_PATH, final_named_playlists)
        create_or_update_playlists_on_jellyfin(jellyfin_url, jellyfin_user_id, headers, final_named_playlists, final_playlist_centroids, MOOD_LABELS)

        log_and_update(f"🎉 Playlists generated and updated on Jellyfin! Best run had {best_diversity_score} unique predominant moods.", 100)
        return {"status": "SUCCESS", "message": f"Playlists generated and updated on Jellyfin! Best run had {best_diversity_score} unique predominant moods.", 'task_type': 'clustering', 'unique_moods': best_unique_predominant_moods}

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"FATAL ERROR: Clustering failed: {e}\n{error_traceback}")
        log_and_update(f"❌ Clustering failed: {e}", 100)
        self.update_state(state='FAILURE', meta={'progress': 100, 'status': f'Clustering failed: {e}', 'log_output': log_messages + [f"Error Traceback: {error_traceback}"], 'task_type': 'clustering'})
        return {"status": "FAILURE", "message": f"Clustering failed: {e}", 'task_type': 'clustering'}


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.before_request
def before_request():
    init_status_db()

@app.route('/api/analysis/start', methods=['POST'])
def start_analysis():
    data = request.get_json()
    jellyfin_url = data.get('jellyfin_url', JELLYFIN_URL)
    jellyfin_user_id = data.get('jellyfin_user_id', JELLYFIN_USER_ID)
    jellyfin_token = data.get('jellyfin_token', JELLYFIN_TOKEN)
    num_recent_albums = int(data.get('num_recent_albums', NUM_RECENT_ALBUMS))
    top_n_moods = int(data.get('top_n_moods', TOP_N_MOODS))

    task = run_analysis_task.delay(jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods)
    save_task_status(task.id, "PENDING", "analysis")
    return jsonify({"task_id": task.id, "status": "PENDING", "task_type": "analysis"}), 202

@app.route('/api/clustering/start', methods=['POST'])
def start_clustering():
    data = request.get_json()
    clustering_method = data.get('clustering_method', CLUSTERING_METHOD)
    num_clusters = int(data.get('num_clusters', NUM_CLUSTERS))
    dbscan_eps = float(data.get('dbscan_eps', DBSCAN_EPS))
    dbscan_min_samples = int(data.get('dbscan_min_samples', DBSCAN_MIN_SAMPLES))
    pca_components = int(data.get('pca_components', 0))
    pca_enabled = (pca_components > 0)
    num_clustering_runs = int(data.get('clustering_runs', CLUSTERING_RUNS))

    # Pass Jellyfin credentials for playlist creation within the task
    jellyfin_url = data.get('jellyfin_url', JELLYFIN_URL)
    jellyfin_user_id = data.get('jellyfin_user_id', JELLYFIN_USER_ID)
    jellyfin_token = data.get('jellyfin_token', JELLYFIN_TOKEN)

    task = run_clustering_task.delay(
        jellyfin_url, jellyfin_user_id, jellyfin_token,
        clustering_method, num_clusters, dbscan_eps,
        dbscan_min_samples, pca_components, pca_enabled, num_clustering_runs
    )
    # Ensure task_type is correctly passed here
    save_task_status(task.id, "PENDING", "clustering")
    return jsonify({"task_id": task.id, "status": "PENDING", "task_type": "clustering"}), 202


@app.route('/api/task/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = AsyncResult(task_id, app=celery)
    response = {
        'task_id': task.id,
        'state': task.state,
        'status': 'Processing...'
    }

    if task.state == 'PENDING':
        # For PENDING, get type from DB if possible
        db_task_type = get_task_type_from_db(task_id)
        if db_task_type:
            response['task_type'] = db_task_type
        response['status'] = 'Task is pending.'
    elif task.state == 'STARTED':
        # For STARTED, get info from live task result if available
        if task.info and isinstance(task.info, dict):
            response.update(task.info)
        else:
            response['status'] = 'Task has started.'
    elif task.state == 'PROGRESS':
        if task.info and isinstance(task.info, dict):
            response.update(task.info)
    elif task.state == 'SUCCESS':
        db_task_type = get_task_type_from_db(task_id)
        response['state'] = 'SUCCESS' # Ensure state is explicitly set
        if task.info and isinstance(task.info, dict):
            response.update(task.info)
        else:
            response['status'] = 'Task completed successfully.'

        if db_task_type:
            response['task_type'] = db_task_type
    elif task.state == 'FAILURE':
        db_task_type = get_task_type_from_db(task_id)
        response['state'] = 'FAILURE' # Ensure state is explicitly set
        if task.info and isinstance(task.info, dict):
            response.update(task.info)
        else:
            response['status'] = f'Task failed: {task.info}'

        if db_task_type:
            response['task_type'] = db_task_type
    elif task.state == 'REVOKED':
        db_task_type = get_task_type_from_db(task_id)
        response['state'] = 'REVOKED'
        if task.info and isinstance(task.info, dict):
            response.update(task.info)
        else:
            response['status'] = 'Task was revoked.'
        if db_task_type:
            response['task_type'] = db_task_type
    else:
        # Fallback for unexpected states, try to get from DB
        db_status = get_status_db().execute("SELECT status, task_type FROM analysis_status WHERE task_id = ?", (task_id,)).fetchone()
        if db_status:
            response['status'] = db_status['status']
            response['task_type'] = db_status['task_type']
        else:
            response['status'] = 'Unknown task state.'

    return jsonify(response)


@app.route('/api/task/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    task = AsyncResult(task_id, app=celery)

    # Get the task type from the database BEFORE revoking,
    # as the live task might disappear quickly after revocation.
    original_task_type = get_task_type_from_db(task_id)

    if task.state in ['PENDING', 'STARTED', 'PROGRESS']:
        task.revoke(terminate=True)
        # Update the database status to REVOKED immediately
        save_task_status(task_id, "REVOKED", original_task_type, "Task was manually cancelled.")
        message = f"Task {task_id} ({original_task_type}) has been cancelled."
        print(message)
        return jsonify({"task_id": task_id, "status": "REVOKED", "message": message, "task_type": original_task_type}), 200
    else:
        message = f"Task {task_id} ({original_task_type}) cannot be cancelled in state: {task.state}."
        print(message)
        return jsonify({"task_id": task_id, "status": task.state, "message": message, "task_type": original_task_type}), 400

@app.route('/api/analysis/last_task', methods=['GET'])
def get_last_analysis_task():
    conn = get_status_db()
    cursor = conn.cursor()
    cursor.execute("SELECT task_id, status, message FROM analysis_status WHERE task_type = 'analysis' ORDER BY timestamp DESC LIMIT 1")
    last_task = cursor.fetchone()
    if last_task:
        # Fetch full Celery status to get progress etc., if available
        task = AsyncResult(last_task['task_id'], app=celery)
        response = {
            'task_id': task.id,
            'state': task.state,
            'status': last_task['status'],
            'message': last_task['message'],
            'task_type': 'analysis'
        }
        if task.info and isinstance(task.info, dict):
            response.update(task.info)
        return jsonify(response)
    return jsonify({"message": "No previous analysis task found."}), 404

@app.route('/api/clustering/last_task', methods=['GET'])
def get_last_clustering_task():
    conn = get_status_db()
    cursor = conn.cursor()
    cursor.execute("SELECT task_id, status, message FROM analysis_status WHERE task_type = 'clustering' ORDER BY timestamp DESC LIMIT 1")
    last_task = cursor.fetchone()
    if last_task:
        # Fetch full Celery status to get progress etc., if available
        task = AsyncResult(last_task['task_id'], app=celery)
        response = {
            'task_id': task.id,
            'state': task.state,
            'status': last_task['status'],
            'message': last_task['message'],
            'task_type': 'clustering'
        }
        if task.info and isinstance(task.info, dict):
            response.update(task.info)
        return jsonify(response)
    return jsonify({"message": "No previous clustering task found."}), 404


@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        "jellyfin_url": JELLYFIN_URL,
        "jellyfin_user_id": JELLYFIN_USER_ID,
        "jellyfin_token": JELLYFIN_TOKEN,
        "num_recent_albums": NUM_RECENT_ALBUMS,
        "top_n_moods": TOP_N_MOODS,
        "clustering_method": CLUSTERING_METHOD,
        "num_clusters": NUM_CLUSTERS,
        "dbscan_eps": DBSCAN_EPS,
        "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
        "max_songs_per_cluster": MAX_SONGS_PER_CLUSTER,
        "max_songs_per_artist": MAX_SONGS_PER_ARTIST,
        "max_distance": MAX_DISTANCE,
        "pca_components": PCA_COMPONENTS,
        "clustering_runs": CLUSTERING_RUNS,
        "mood_labels": MOOD_LABELS # Expose mood labels to frontend
    })

@app.route('/api/playlists', methods=['GET'])
def get_playlists():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT playlist, item_id, title, author FROM playlist ORDER BY playlist, title")
        rows = cur.fetchall()
    playlists = defaultdict(list)
    for playlist, item_id, title, author in rows:
        playlists[playlist].append({"item_id": item_id, "title": title, "author": author})
    return jsonify(playlists)


if __name__ == '__main__':
    # Initialize the status database when the app starts
    with app.app_context():
        init_status_db()
    # Ensure temp directory exists for downloads
    os.makedirs(TEMP_DIR, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8000)
