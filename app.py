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

# Import your existing analysis functions
from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor, TensorflowPredictMusiCNN, TensorflowPredict2D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Your existing config - assuming this is from config.py and sets global variables
from config import *

# --- Flask App Setup ---
app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

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
    # Let SQLite autoincrement: don't fix id=1
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
            print(f"Warning: Could not remove {file_path} from {temp_dir}: {e}")

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

def get_recent_albums(jellyfin_url, jellyfin_user_id, headers, limit):
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
    audio = MonoLoader(filename=file_path)()
    tempo, _, _, _, _ = RhythmExtractor2013()(audio)
    key, scale, _ = KeyExtractor()(audio)
    moods = predict_moods(file_path, embedding_model_path, prediction_model_path, mood_labels, top_n_moods)
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
        # Inverse transform to get back to the original feature space
        # Make sure the input centroid_scaled_vector has the correct shape for inverse_transform
        # It should be 2D: (1, n_components) if pca_model was fitted on 2D data
        try:
            scaled_vector = pca_model.inverse_transform(centroid_scaled_vector.reshape(1, -1))[0]
        except ValueError:
            # Handle cases where inverse_transform might fail due to shape mismatch
            # or if pca_model is None or not fitted correctly
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
    
    # Ensure mood_values is not empty or all zeros
    if len(mood_values) == 0 or np.sum(mood_values) == 0:
        top_indices = []
    else:
        top_indices = np.argsort(mood_values)[::-1][:3] # Get top 3 moods

    mood_names = [mood_labels[i] for i in top_indices if i < len(mood_labels)] # Ensure index is valid
    mood_part = "_".join(mood_names).title() if mood_names else "Mixed" # Handle case with no strong moods
    full_name = f"{mood_part}_{tempo_label}"
    
    top_mood_scores = {mood_labels[i]: mood_values[i] for i in top_indices if i < len(mood_labels)}
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
                    centroid_info = cluster_centers.get(base_name, {}) # Use .get to prevent KeyError
                    top_moods = {k: v for k, v in centroid_info.items() if k in mood_labels}
                    extra_info = {k: v for k, v in centroid_info.items() if k not in mood_labels}
                    centroid_str = ", ".join(f"{k}:{v:.2f}" for k, v in top_moods.items())
                    extras_str = ", ".join(f"{k}:{v:.2f}" for k, v in extra_info.items())
                    print(f"✅ Created playlist {playlist_name} with {len(item_ids)} tracks (Centroid: {centroid_str} | {extras_str})")
            except Exception as e:
                print(f"Exception creating {playlist_name}: {e}")

# --- Celery Task Definition ---
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
            'total_albums': total_albums
        })
    try:
        log_and_update("🚀 Starting mood-based analysis and playlist generation...", 0)
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

# --- API Endpoints ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analysis/start', methods=['POST'])
def start_analysis():
    data = request.json or {}
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
        'status': 'Processing...'
    }
    task_info = task.info if isinstance(task.info, dict) else {}
    if task.state == 'PENDING':
        response['status'] = 'Task is pending or not yet started.'
        response.update({'progress': 0, 'status': 'Initializing...', 'log_output': ['Task pending...'], 'current_album': 'N/A', 'current_album_idx': 0, 'total_albums': 0})
        response.update(task_info)
    elif task.state == 'PROGRESS':
        response.update(task_info)
    elif task.state == 'SUCCESS':
        response['status'] = 'Analysis complete!'
        response.update({'progress': 100, 'status': 'Analysis complete!', 'log_output': []})
        response.update(task_info)
        save_analysis_task_id(task_id, "SUCCESS")
    elif task.state == 'FAILURE':
        response['status'] = str(task.info)
        response.update({'progress': 100, 'status': 'Analysis failed!', 'log_output': [str(task.info)]})
        response.update(task_info)
        save_analysis_task_id(task_id, "FAILURE")
    elif task.state == 'REVOKED':
        response['status'] = 'Task revoked.'
        response.update({'progress': 100, 'status': 'Task revoked.', 'log_output': ['Task was cancelled.']})
        response.update(task_info)
        save_analysis_task_id(task_id, "REVOKED")
    else:
        response['status'] = f'Unknown state: {task.state}'
        response.update(task_info)
    return jsonify(response)

@app.route('/api/analysis/cancel/<task_id>', methods=['POST'])
def cancel_analysis(task_id):
    task = AsyncResult(task_id, app=celery)
    if task.state in ['PENDING', 'STARTED', 'PROGRESS']:
        task.revoke(terminate=True, signal='SIGKILL')
        save_analysis_task_id(task_id, "REVOKED")
        return jsonify({"message": "Analysis task cancelled.", "task_id": task_id}), 200
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
        "jellyfin_token": JELLYFIN_TOKEN,
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
        "clustering_runs": CLUSTERING_RUNS, # New config parameter
    })

@app.route('/api/clustering', methods=['POST'])
def run_clustering():
    data = request.json
    clustering_method = data.get('clustering_method', CLUSTER_ALGORITHM)
    num_clusters = int(data.get('num_clusters', NUM_CLUSTERS))
    dbscan_eps = float(data.get('dbscan_eps', DBSCAN_EPS))
    dbscan_min_samples = int(data.get('dbscan_min_samples', DBSCAN_MIN_SAMPLES))
    pca_components = int(data.get('pca_components', 0))
    pca_enabled = (pca_components > 0)
    num_clustering_runs = int(data.get('clustering_runs', CLUSTERING_RUNS)) # New parameter

    try:
        rows = get_all_tracks(DB_PATH)
        if len(rows) < 2:
            return jsonify({"status": "error", "message": "Not enough analyzed tracks for clustering."}), 400
        
        X_original = [score_vector(row, MOOD_LABELS) for row in rows]
        X_scaled = np.array(X_original) # Convert to numpy array for PCA/clustering

        best_diversity_score = -1
        best_clustering_results = None

        for run_idx in range(num_clustering_runs):
            print(f"Clustering Run {run_idx + 1}/{num_clustering_runs}")
            pca_model = None
            data_for_clustering = X_scaled

            if pca_enabled:
                pca_model = PCA(n_components=pca_components)
                X_pca = pca_model.fit_transform(X_scaled)
                data_for_clustering = X_pca

            labels = None
            cluster_centers = {}
            raw_distances = np.zeros(len(data_for_clustering))

            if clustering_method == "kmeans":
                k = num_clusters if num_clusters > 0 else max(1, len(rows) // MAX_SONGS_PER_CLUSTER)
                kmeans = KMeans(n_clusters=min(k, len(rows)), random_state=None, n_init='auto') # Set random_state=None for true randomness across runs
                labels = kmeans.fit_predict(data_for_clustering)
                cluster_centers = {i: kmeans.cluster_centers_[i] for i in range(min(k, len(rows)))}
                centers_for_points = kmeans.cluster_centers_[labels]
                raw_distances = np.linalg.norm(data_for_clustering - centers_for_points, axis=1)
            elif clustering_method == "dbscan":
                dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
                labels = dbscan.fit_predict(data_for_clustering)
                
                # DBSCAN does not have explicit centroids, so calculate them for existing clusters
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
                return jsonify({"status": "error", "message": f"Unsupported clustering algorithm: {clustering_method}"}), 400

            max_dist = raw_distances.max()
            normalized_distances = raw_distances / max_dist if max_dist > 0 else raw_distances
            
            # Prepare track_info for current run
            track_info = []
            for row, label, vec, dist in zip(rows, labels, data_for_clustering, normalized_distances):
                if label == -1: # Skip noise points for now
                    continue
                track_info.append({"row": row, "label": label, "vector": vec, "distance": dist})

            # Filter clusters based on MAX_DISTANCE and MAX_SONGS_PER_ARTIST/CLUSTER
            filtered_clusters = defaultdict(list)
            for cluster_id in set(labels):
                if cluster_id == -1: # Skip noise points
                    continue
                cluster_tracks = [t for t in track_info if t["label"] == cluster_id and t["distance"] <= MAX_DISTANCE]
                if not cluster_tracks:
                    continue
                cluster_tracks.sort(key=lambda x: x["distance"]) # Sort by distance for selection
                
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

            # Name playlists and calculate diversity score
            current_named_playlists = defaultdict(list)
            current_playlist_centroids = {}
            predominant_moods_found = set()

            for label, songs in filtered_clusters.items():
                if songs:
                    center = cluster_centers[label]
                    name, top_scores = name_cluster(center, pca_model, pca_enabled, MOOD_LABELS)
                    
                    # Extract predominant mood for diversity calculation
                    # The name_cluster function returns 'top_mood_scores' as part of 'top_scores'
                    # which contains the top 3 moods. We can take the very first one as the predominant.
                    if top_scores and any(mood in MOOD_LABELS for mood in top_scores.keys()):
                        # Find the actual mood label with the highest score from top_scores
                        predominant_mood_key = max(top_scores, key=lambda k: top_scores[k] if k in MOOD_LABELS else -1)
                        if predominant_mood_key in MOOD_LABELS:
                             predominant_moods_found.add(predominant_mood_key)


                    current_named_playlists[name].extend(songs)
                    current_playlist_centroids[name] = top_scores # Store the centroid info including tempo

            diversity_score = len(predominant_moods_found)
            print(f"Run {run_idx + 1}: Found {diversity_score} unique predominant moods.")

            if diversity_score > best_diversity_score:
                best_diversity_score = diversity_score
                best_clustering_results = {
                    "named_playlists": current_named_playlists,
                    "playlist_centroids": current_playlist_centroids,
                    "pca_model": pca_model
                }

        if not best_clustering_results:
            return jsonify({"status": "error", "message": "No valid clusters found after multiple runs."}), 500

        # Apply the best clustering results
        final_named_playlists = best_clustering_results["named_playlists"]
        final_playlist_centroids = best_clustering_results["playlist_centroids"]
        final_pca_model = best_clustering_results["pca_model"]

        update_playlist_table(DB_PATH, final_named_playlists)
        create_or_update_playlists_on_jellyfin(JELLYFIN_URL, JELLYFIN_USER_ID, {"X-Emby-Token": JELLYFIN_TOKEN}, final_named_playlists, final_playlist_centroids, final_pca_model, MOOD_LABELS)
        
        return jsonify({"status": "success", "message": f"Playlists generated and updated on Jellyfin! Best run had {best_diversity_score} unique predominant moods."}), 200

    except Exception as e:
        print(f"Error during clustering: {e}")
        import traceback
        traceback.print_exc()
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
