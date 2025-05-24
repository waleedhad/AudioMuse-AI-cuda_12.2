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
import time # Added for timing operations

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
# STATUS_DB_PATH is assumed to be defined in config.py
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

# Ensure status DB is initialized on app startup
with app.app_context():
    init_status_db()

def get_status_db():
    if 'status_db' not in g:
        g.status_db = sqlite3.connect(STATUS_DB_PATH)
        g.status_db.row_factory = sqlite3.Row # Return rows as dictionaries
    return g.status_db

@app.teardown_appcontext
def close_status_db(exception):
    status_db = g.pop('status_db', None)
    if status_db is not None:
        status_db.close()

def save_analysis_task_id(task_id, status="PENDING"):
    conn = get_status_db()
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO analysis_status (id, task_id, status) VALUES (1, ?, ?)", (task_id, status))
    conn.commit()

def get_last_analysis_task_id():
    conn = get_status_db()
    cur = conn.cursor()
    cur.execute("SELECT task_id, status FROM analysis_status ORDER BY timestamp DESC LIMIT 1")
    row = cur.fetchone()
    return dict(row) if row else None

# --- Existing Script Functions (Minimum Changes) ---

def clean_temp():
    os.makedirs(TEMP_DIR, exist_ok=True)
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Warning: Could not remove {file_path} from {TEMP_DIR}: {e}")
            
def init_db():
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

def get_recent_albums(limit=NUM_RECENT_ALBUMS):
    url = f"{JELLYFIN_URL}/Users/{JELLYFIN_USER_ID}/Items"
    params = {
        "IncludeItemTypes": "MusicAlbum",
        "SortBy": "DateCreated",
        "SortOrder": "Descending",
        "Limit": limit,
        "Recursive": True,
    }
    
    print(f"DEBUG: get_recent_albums - JELLYFIN_URL: {JELLYFIN_URL}")
    print(f"DEBUG: get_recent_albums - JELLYFIN_USER_ID: {JELLYFIN_USER_ID}")
    print(f"DEBUG: get_recent_albums - JELLYFIN_TOKEN (from HEADERS): {'***hidden***' if HEADERS.get('X-Emby-Token') else 'None'}")
    print(f"DEBUG: get_recent_albums - Request URL: {url}")
    print(f"DEBUG: get_recent_albums - Request Params: {json.dumps(params, indent=2)}")
    print(f"DEBUG: get_recent_albums - Request Headers: {json.dumps(HEADERS, indent=2)}")

    try:
        # Added timeout to prevent indefinite hangs
        print(f"DEBUG: About to make request to {url} with params {json.dumps(params)} and headers {json.dumps(HEADERS)}")
        r = requests.get(url, headers=HEADERS, params=params, timeout=30) # <<< Added timeout
        r.raise_for_status()
        print(f"DEBUG: Request completed with status code {r.status_code}")
        
        print(f"DEBUG: get_recent_albums - Successful response status: {r.status_code}")
        return r.json().get("Items", [])
    
    except requests.exceptions.Timeout as timeout_err:
        print(f"ERROR: Jellyfin request timed out after 30 seconds: {timeout_err} for URL: {url}")
        return []
    except requests.exceptions.HTTPError as http_err:
        print(f"ERROR: Failed to get albums from Jellyfin: {http_err} for URL: {url}")
        print(f"DEBUG: get_recent_albums - HTTP Response Status Code: {http_err.response.status_code}")
        print(f"DEBUG: get_recent_albums - HTTP Response Body: {http_err.response.text}")
        return []
    except requests.exceptions.ConnectionError as conn_err:
        print(f"ERROR: Failed to connect to Jellyfin: {conn_err} for URL: {url}. Is Jellyfin service running and accessible?")
        return []
    except requests.exceptions.RequestException as req_err:
        print(f"ERROR: A general network or request error occurred while getting albums: {req_err}")
        return []
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in get_recent_albums: {e}")
        return []


def get_tracks_from_album(album_id):
    url = f"{JELLYFIN_URL}/Users/{JELLYFIN_USER_ID}/Items"
    params = {"ParentId": album_id, "IncludeItemTypes": "Audio"}
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30) # <<< Added timeout
        r.raise_for_status()
        return r.json().get("Items", []) if r.ok else []
    except Exception as e:
        print(f"ERROR: Failed to get tracks from album {album_id}: {e}")
        return []


def download_track(item):
    filename = f"{item['Name'].replace('/', '_')}-{item.get('AlbumArtist', 'Unknown')}.mp3"
    path = os.path.join(TEMP_DIR, filename)
    try:
        # Added timeout for download as well
        print(f"DEBUG: Attempting to download {item['Name']} to {path}")
        r = requests.get(f"{JELLYFIN_URL}/Items/{item['Id']}/Download", headers=HEADERS, timeout=120) # Increased timeout for downloads
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        print(f"DEBUG: Successfully downloaded {item['Name']}")
        return path
    except requests.exceptions.Timeout as timeout_err:
        print(f"ERROR: Download of {item['Name']} timed out: {timeout_err}")
        return None
    except requests.exceptions.HTTPError as http_err:
        print(f"ERROR: HTTP error downloading {item['Name']}: {http_err}. Response: {http_err.response.text}")
        return None
    except Exception as e:
        print(f"ERROR: Download failed for {item['Name']}: {e}")
        return None

def predict_moods(file_path):
    print(f"DEBUG: predict_moods: Prediction started")
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
    print(f"DEBUG: predict_moods: Prediction finished")
    return {label: float(score) for label, score in sorted(results.items(), key=lambda x: -x[1])[:TOP_N_MOODS]}



def analyze_track(file_path):
    start_time = time.time()
    print(f"DEBUG: analyze_track: Starting for {os.path.basename(file_path)}")

    loader_start = time.time()
    loader = MonoLoader(filename=file_path)
    audio = loader()
    print(f"DEBUG: analyze_track: MonoLoader complete ({time.time() - loader_start:.2f}s).")

    rhythm_start = time.time()
    tempo, _, _, _, _ = RhythmExtractor2013()(audio)
    print(f"DEBUG: analyze_track: RhythmExtractor complete ({time.time() - rhythm_start:.2f}s). Tempo: {tempo:.2f}")

    key_start = time.time()
    key, scale, _ = KeyExtractor()(audio)
    print(f"DEBUG: analyze_track: KeyExtractor complete ({time.time() - key_start:.2f}s). Key: {key}, Scale: {scale}")

    moods_start = time.time()
    moods = predict_moods(file_path)
    print(f"DEBUG: analyze_track: predict_moods complete ({time.time() - moods_start:.2f}s).")
    
    print(f"DEBUG: analyze_track: Total time for {os.path.basename(file_path)}: {time.time() - start_time:.2f}s")
    return tempo, key, scale, moods


def track_exists(item_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM score WHERE item_id=?", (item_id,))
    row = cur.fetchone()
    conn.close()
    return row


def save_track_analysis(item_id, title, author, tempo, key, scale, moods):
    mood_str = ','.join(f"{k}:{v:.3f}" for k, v in moods.items())
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO score VALUES (?, ?, ?, ?, ?, ?, ?)",
                (item_id, title, author, tempo, key, scale, mood_str))
    conn.commit()
    conn.close()


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
                    continue

    full_vector = [tempo_norm] + list(mood_scores)
    return full_vector


def get_all_tracks():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM score")
    rows = cur.fetchall()
    conn.close()
    return rows


def limit_songs_per_artist(cluster): # This function is defined but not currently used in the main flow
    count = defaultdict(int)
    result = []
    for item_id, title, author in cluster:
        if count[author] < MAX_SONGS_PER_ARTIST:
            result.append((item_id, title, author))
            count[author] += 1
    return result


def name_cluster(centroid_scaled_vector, pca_model=None):
    if PCA_ENABLED and pca_model is not None:
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
    mood_names = [MOOD_LABELS[i] for i in top_indices]
    mood_part = "_".join(mood_names).title()

    full_name = f"{mood_part}_{tempo_label}"

    top_mood_scores = {MOOD_LABELS[i]: mood_values[i] for i in top_indices}
    extra_info = {"tempo": round(tempo_norm, 2)}

    return full_name, {**top_mood_scores, **extra_info}

def update_playlist_table(playlists):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM playlist")
    for name, cluster in playlists.items():
        for item_id, title, author in cluster:
            cur.execute("INSERT INTO playlist VALUES (?, ?, ?, ?)", (name, item_id, title, author))
    conn.commit()
    conn.close()


def delete_old_automatic_playlists():
    url = f"{JELLYFIN_URL}/Users/{JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30) # <<< Added timeout
        r.raise_for_status()
        for item in r.json().get("Items", []):
            if "_automatic" in item.get("Name", ""):
                del_url = f"{JELLYFIN_URL}/Items/{item['Id']}"
                # Added timeout for delete request
                del_resp = requests.delete(del_url, headers=HEADERS, timeout=10)
                if del_resp.ok:
                    print(f"🗑️ Deleted old playlist: {item['Name']}")
                else:
                    print(f"WARNING: Failed to delete old playlist {item['Name']}: {del_resp.status_code} - {del_resp.text}")
    except Exception as e:
        print(f"Failed to clean old playlists: {e}")


def create_or_update_playlists_on_jellyfin(playlists, cluster_centers, pca_model=None):
    print("DEBUG: Starting delete_old_automatic_playlists...")
    delete_old_automatic_playlists()
    print("DEBUG: Finished delete_old_automatic_playlists. Starting playlist creation...")
    for base_name, cluster in playlists.items():
        chunks = [cluster[i:i+MAX_SONGS_PER_CLUSTER] for i in range(0, len(cluster), MAX_SONGS_PER_CLUSTER)]
        for idx, chunk in enumerate(chunks, 1):
            playlist_name = f"{base_name}_automatic_{idx}" if len(chunks) > 1 else f"{base_name}_automatic"
            item_ids = [item_id for item_id, _, _ in chunk]
            if not item_ids:
                print(f"WARNING: Skipping empty chunk for playlist {playlist_name}")
                continue
            body = {"Name": playlist_name, "Ids": item_ids, "UserId": JELLYFIN_USER_ID}
            try:
                # Added timeout for playlist creation request
                r = requests.post(f"{JELLYFIN_URL}/Playlists", headers=HEADERS, json=body, timeout=30)
                if r.ok:
                    centroid_info = cluster_centers[base_name]
                    top_moods = {k: v for k, v in centroid_info.items() if k in MOOD_LABELS}
                    extra_info = {k: v for k, v in centroid_info.items() if k not in MOOD_LABELS}

                    centroid_str = ", ".join(f"{k}:{v:.2f}" for k, v in top_moods.items())
                    extras_str = ", ".join(f"{k}:{v:.2f}" for k, v in extra_info.items())

                    print(f"✅ Created playlist {playlist_name} with {len(item_ids)} tracks (Centroid: {centroid_str} | {extras_str})")

                else:
                    print(f"❌ Failed to create playlist {playlist_name}: {r.status_code} - {r.text}")
            except requests.exceptions.Timeout as timeout_err:
                print(f"ERROR: Playlist creation for {playlist_name} timed out: {timeout_err}")
            except Exception as e:
                print(f"❌ Exception creating {playlist_name}: {e}")

# --- Celery Task Definition ---
@celery.task(bind=True)
def run_analysis_task(self, jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods):
    # Update global config for this task run
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
        clean_temp()
        init_db()

        albums = get_recent_albums()
        if not albums:
            log_and_update("⚠️ No new albums to analyze. Proceeding with existing data.", 10)
        else:
            total_albums = len(albums)
            # Adjust progress calculation to reflect full analysis scope more accurately
            analysis_start_progress = 5
            analysis_end_progress = 85 # Leaving 15% for clustering and playlist creation
            
            for idx, album in enumerate(albums, 1):
                album_progress_base = analysis_start_progress + int((analysis_end_progress - analysis_start_progress) * ((idx - 1) / total_albums))
                log_and_update(f"🎵 Processing Album: {album['Name']} ({idx}/{total_albums})",
                               album_progress_base, # Base progress for the album
                               current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                
                tracks = get_tracks_from_album(album['Id'])
                if not tracks:
                    log_and_update(f"   ⚠️ No tracks found for album: {album['Name']}", album_progress_base, current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                    continue

                total_tracks_in_album = len(tracks)
                for track_idx, item in enumerate(tracks, 1):
                    track_name_full = f"{item['Name']} by {item.get('AlbumArtist', 'Unknown')}"
                    track_progress_within_album = int((analysis_end_progress - analysis_start_progress) * (1 / total_albums) * (track_idx / total_tracks_in_album))
                    current_overall_progress = album_progress_base + track_progress_within_album

                    log_and_update(f"   🎶 Analyzing track: {track_name_full} ({track_idx}/{total_tracks_in_album})",
                                   current_overall_progress,
                                   current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                    
                    if track_exists(item['Id']):
                        log_and_update(f"     ⏭️ Skipping '{track_name_full}' (already analyzed)",
                                       current_overall_progress,
                                       current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                        continue
                    
                    path = download_track(item)
                    if not path:
                        log_and_update(f"     ❌ Failed to download '{track_name_full}'. Skipping.",
                                       current_overall_progress,
                                       current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                        continue
                    
                    try:
                        analysis_start_time = time.time()
                        tempo, key, scale, moods = analyze_track(path)
                        analysis_duration = time.time() - analysis_start_time
                        save_track_analysis(item['Id'], item['Name'], item.get('AlbumArtist', 'Unknown'), tempo, key, scale, moods)
                        log_and_update(f"     ✅ Analyzed '{track_name_full}' in {analysis_duration:.2f}s. Moods: {', '.join(f'{k}:{v:.2f}' for k,v in moods.items())}",
                                       current_overall_progress,
                                       current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                    except Exception as e:
                        log_and_update(f"     ❌ Error analyzing '{track_name_full}': {e}",
                                       current_overall_progress,
                                       current_album=album['Name'], current_album_idx=idx, total_albums=total_albums)
                    finally:
                        # Ensure temporary file is cleaned up after analysis, even if it failed
                        if os.path.exists(path):
                            try:
                                os.remove(path)
                                print(f"DEBUG: Cleaned up temporary file: {path}")
                            except Exception as cleanup_e:
                                print(f"WARNING: Failed to clean up temp file {path}: {cleanup_e}")

            clean_temp() # Final cleanup of temp directory

        log_and_update("Analysis phase complete. Starting clustering...", 90)

        rows = get_all_tracks()
        if len(rows) < 2:
            log_and_update("⚠️ Not enough data for clustering. Skipping playlist generation.", 100)
            return {"status": "SUCCESS", "message": "Analysis complete, but not enough data for clustering."}

        X_original = [score_vector(row) for row in rows]
        X_scaled = X_original

        pca_model = None
        if PCA_ENABLED:
            print("DEBUG: PCA enabled. Performing PCA...")
            pca_model = PCA(n_components=3)
            X_pca = pca_model.fit_transform(X_scaled)
            data_for_clustering = X_pca
            print(f"DEBUG: PCA transformation complete. Data shape: {data_for_clustering.shape}")
        else:
            data_for_clustering = X_scaled
            print(f"DEBUG: PCA disabled. Using original data for clustering. Data shape: {data_for_clustering.shape}")

        if CLUSTER_ALGORITHM == "kmeans":
            print(f"DEBUG: Clustering with KMeans. NUM_CLUSTERS: {NUM_CLUSTERS}")
            k = NUM_CLUSTERS if NUM_CLUSTERS > 0 else max(1, len(rows) // MAX_SONGS_PER_CLUSTER)
            kmeans = KMeans(n_clusters=min(k, len(rows)), random_state=42, n_init='auto')
            labels = kmeans.fit_predict(data_for_clustering)
            cluster_centers = {i: kmeans.cluster_centers_[i] for i in range(min(k, len(rows)))}
            centers_for_points = kmeans.cluster_centers_[labels]
            raw_distances = np.linalg.norm(data_for_clustering - centers_for_points, axis=1)
        elif CLUSTER_ALGORITHM == "dbscan":
            print(f"DEBUG: Clustering with DBSCAN. EPS: {DBSCAN_EPS}, MIN_SAMPLES: {DBSCAN_MIN_SAMPLES}")
            dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
            labels = dbscan.fit_predict(data_for_clustering)
            cluster_centers = {}
            raw_distances = np.zeros(len(data_for_clustering))
            for cluster_id in set(labels):
                if cluster_id == -1: # Noise points in DBSCAN
                    continue
                indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                cluster_points = np.array([data_for_clustering[i] for i in indices])
                if len(cluster_points) > 0: # Ensure cluster is not empty
                    center = cluster_points.mean(axis=0)
                    for i in indices:
                        raw_distances[i] = np.linalg.norm(data_for_clustering[i] - center)
                    cluster_centers[cluster_id] = center
            print(f"DEBUG: DBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")
        else:
            raise ValueError(f"Unsupported clustering algorithm: {CLUSTER_ALGORITHM}")

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
                    count_per_artist[author] += 1 # Corrected: Use count_per_artist here
                if len(selected) >= MAX_SONGS_PER_CLUSTER:
                    break

            for t in selected:
                item_id, title, author = t["row"][0], t["row"][1], t["row"][2]
                filtered_clusters[cluster_id].append((item_id, title, author))
        
        named_playlists = defaultdict(list)
        playlist_centroids = {}
        for label, songs in filtered_clusters.items():
            if songs: # Only create playlist if there are songs in the filtered cluster
                center = cluster_centers[label]
                name, top_scores = name_cluster(center, pca_model)
                named_playlists[name].extend(songs)
                playlist_centroids[name] = top_scores
            else:
                print(f"DEBUG: Skipping empty filtered cluster {label}")
        
        log_and_update("Clustering complete. Updating playlists...", 95)
        update_playlist_table(named_playlists)
        create_or_update_playlists_on_jellyfin(named_playlists, playlist_centroids, pca_model)
        
        log_and_update("✅ All done!", 100)
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
    data = request.json
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
    # Celery's info can be None or not a dict initially
    task_info = task.info if isinstance(task.info, dict) else {}

    if task.state == 'PENDING':
        response['status'] = 'Task is pending or not yet started.'
        response.update({'progress': 0, 'status': 'Initializing...', 'log_output': ['Task pending...'], 'current_album': 'N/A', 'current_album_idx': 0, 'total_albums': 0})
        response.update(task_info) # Overwrite with any actual info if present
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
        response.update(task_info) # Include any info for unknown states

    return jsonify(response)


@app.route('/api/analysis/cancel/<task_id>', methods=['POST'])
def cancel_analysis(task_id):
    task = AsyncResult(task_id, app=celery)
    if task.state in ['PENDING', 'STARTED', 'PROGRESS']:
        task.revoke(terminate=True)
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
    })

@app.route('/api/clustering', methods=['POST'])
def run_clustering():
    global CLUSTER_ALGORITHM, NUM_CLUSTERS, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, PCA_ENABLED
    
    data = request.json
    
    CLUSTER_ALGORITHM = data.get('clustering_method', CLUSTER_ALGORITHM)
    NUM_CLUSTERS = int(data.get('num_clusters', NUM_CLUSTERS))
    DBSCAN_EPS = float(data.get('dbscan_eps', DBSCAN_EPS))
    DBSCAN_MIN_SAMPLES = int(data.get('dbscan_min_samples', DBSCAN_MIN_SAMPLES))
    
    pca_components = int(data.get('pca_components', 0))
    PCA_ENABLED = (pca_components > 0)

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
                if len(cluster_points) > 0:
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
            if songs:
                center = cluster_centers[label]
                name, top_scores = name_cluster(center, pca_model)
                named_playlists[name].extend(songs)
                playlist_centroids[name] = top_scores
            else:
                print(f"DEBUG: Skipping empty filtered cluster {label}")


        update_playlist_table(named_playlists)
        create_or_update_playlists_on_jellyfin(named_playlists, playlist_centroids, pca_model)

        return jsonify({"status": "success", "message": "Playlists generated and updated on Jellyfin!"}), 200

    except Exception as e:
        print(f"Error during clustering: {e}")
        return jsonify({"status": "error", "message": f"Clustering failed: {e}"}), 500

@app.route('/api/playlists', methods=['GET'])
def get_playlists():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT playlist, item_id, title, author FROM playlist ORDER BY playlist, title")
    rows = cur.fetchall()
    conn.close()

    playlists = defaultdict(list)
    for playlist, item_id, title, author in rows:
        playlists[playlist].append({"item_id": item_id, "title": title, "author": author})
    
    return jsonify(dict(playlists)), 200

if __name__ == '__main__':
    init_db()
    os.makedirs(TEMP_DIR, exist_ok=True) 
    app.run(debug=True, host='0.0.0.0', port=8000)
