import os
import shutil
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor # For dictionary-like row access
import requests
from collections import defaultdict
import numpy as np
from flask import Flask, jsonify, request, render_template, g
from celery import Celery, group, chord
from celery.result import AsyncResult
from contextlib import closing
import json
import time
import random
from urllib.parse import urljoin # Added for robust URL joining

# Import your existing analysis functions
from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor, TensorflowPredictMusiCNN, TensorflowPredict2D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Your existing config - assuming this is from config.py and sets global variables
from config import *

# --- Flask App Setup ---
app = Flask(__name__)

# --- Celery Configuration ---
celery = Celery(app.name)
celery.conf.broker_url = CELERY_BROKER_URL
celery.conf.result_backend = CELERY_RESULT_BACKEND
celery.conf.task_track_started = True

# --- Database Connection Management ---
def get_db_connection():
    """
    Establishes a new PostgreSQL database connection.
    Uses Flask's `g` object to store the connection per request.
    This function requires an active Flask application context.
    """
    if 'db_conn' not in g:
        try:
            g.db_conn = psycopg2.connect(DATABASE_URL)
            g.db_conn.autocommit = False # Manage transactions manually
        except psycopg2.Error as e:
            print(f"Database connection failed: {e}")
            raise
    return g.db_conn

def get_db_cursor(conn):
    """
    Provides a cursor for the database connection, using RealDictCursor for dict-like rows.
    """
    return conn.cursor(cursor_factory=RealDictCursor)

@app.teardown_appcontext
def close_db_connection(exception):
    """
    Closes the database connection at the end of the request.
    """
    db_conn = g.pop('db_conn', None)
    if db_conn is not None:
        db_conn.close()

# --- DB Initialization Functions ---
def init_status_db():
    """
    Initializes the PostgreSQL database table for storing task statuses.
    This function should be called within an application context or a standalone script.
    """
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    id SERIAL PRIMARY KEY,
                    task_id TEXT UNIQUE NOT NULL,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """).format(sql.Identifier(STATUS_DB_TABLE_NAME)))
        conn.commit()
        print(f"Table '{STATUS_DB_TABLE_NAME}' ensured to exist.")
    except psycopg2.Error as e:
        print(f"Error initializing status database table: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def init_main_db():
    """
    Initializes the main application database tables for scores and playlists.
    This function should be called within an application context or a standalone script.
    """
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    item_id TEXT PRIMARY KEY,
                    title TEXT,
                    author TEXT,
                    tempo REAL,
                    key TEXT,
                    scale TEXT,
                    mood_vector TEXT
                )
            """).format(sql.Identifier(SCORE_DB_TABLE_NAME)))
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    playlist TEXT,
                    item_id TEXT,
                    title TEXT,
                    author TEXT
                )
            """).format(sql.Identifier(PLAYLIST_DB_TABLE_NAME)))
        conn.commit()
        print(f"Tables '{SCORE_DB_TABLE_NAME}' and '{PLAYLIST_DB_TABLE_NAME}' ensured to exist.")
    except psycopg2.Error as e:
        print(f"Error initializing main database tables: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

with app.app_context():
    init_status_db()
    init_main_db()

def save_task_status(task_id, task_type, status="PENDING"):
    """
    Saves or updates the status of a given task in the database.
    Includes the task type.
    This function requires an active Flask application context.
    """
    conn = get_db_connection()
    with get_db_cursor(conn) as cur:
        cur.execute(sql.SQL("""
            INSERT INTO {} (task_id, task_type, status)
            VALUES (%s, %s, %s)
            ON CONFLICT (task_id) DO UPDATE SET
                status = EXCLUDED.status,
                timestamp = CURRENT_TIMESTAMP
        """).format(sql.Identifier(STATUS_DB_TABLE_NAME)),
        (task_id, task_type, status))
    conn.commit()

def get_last_task_status():
    """
    Retrieves the status of the most recent task (analysis or clustering).
    This function requires an active Flask application context.
    """
    conn = get_db_connection()
    with get_db_cursor(conn) as cur:
        cur.execute(sql.SQL("SELECT task_id, task_type, status FROM {} ORDER BY timestamp DESC LIMIT 1").format(sql.Identifier(STATUS_DB_TABLE_NAME)))
        row = cur.fetchone()
    return dict(row) if row else None

# --- Helper Functions (adapted for PostgreSQL) ---

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

def get_recent_albums(jellyfin_url, jellyfin_user_id, headers, limit):
    """Fetches recent albums from Jellyfin."""
    # Ensure jellyfin_url has a trailing slash for robust joining
    base_url_with_slash = jellyfin_url if jellyfin_url.endswith('/') else jellyfin_url + '/'
    # Construct path segments carefully
    path = f"Users/{jellyfin_user_id.strip('/')}/Items"
    full_url = urljoin(base_url_with_slash, path)

    params = {
        "IncludeItemTypes": "MusicAlbum",
        "SortBy": "DateCreated",
        "SortOrder": "Descending",
        "Limit": limit if limit > 0 else None,
        "Recursive": True,
    }
    try:
        print(f"DEBUG: Fetching recent albums from URL: {full_url} with params: {params}")
        r = requests.get(full_url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        print(f"ERROR: get_recent_albums: {e} for url: {r.url if 'r' in locals() and hasattr(r, 'url') else full_url}")
        return []

def get_tracks_from_album(jellyfin_url, jellyfin_user_id, headers, album_id):
    """Fetches tracks belonging to a specific album from Jellyfin."""
    base_url_with_slash = jellyfin_url if jellyfin_url.endswith('/') else jellyfin_url + '/'
    path = f"Users/{jellyfin_user_id.strip('/')}/Items"
    full_url = urljoin(base_url_with_slash, path)

    params = {"ParentId": album_id, "IncludeItemTypes": "Audio"}
    try:
        print(f"DEBUG: Fetching tracks for album {album_id} from URL: {full_url} with params: {params}")
        r = requests.get(full_url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("Items", []) if r.ok else []
    except Exception as e:
        print(f"ERROR: get_tracks_from_album {album_id}: {e} for url: {r.url if 'r' in locals() and hasattr(r, 'url') else full_url}")
        return []

def download_track(jellyfin_url, headers, temp_dir, item):
    """Downloads a track from Jellyfin to a temporary directory."""
    filename = f"{item['Id']}_{item['Name'].replace('/', '_')}-{item.get('AlbumArtist', 'Unknown')}.mp3"
    path = os.path.join(temp_dir, filename)
    
    base_url_with_slash = jellyfin_url if jellyfin_url.endswith('/') else jellyfin_url + '/'
    download_path = f"Items/{item['Id']}/Download"
    full_download_url = urljoin(base_url_with_slash, download_path)

    try:
        r = requests.get(full_download_url, headers=headers, timeout=120)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        return path
    except Exception as e:
            print(f"ERROR: download_track {item['Name']}: {e} for url: {full_download_url}")
            return None

def predict_moods(file_path, embedding_model_path, prediction_model_path, mood_labels_list, top_n_moods_count):
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
    results = dict(zip(mood_labels_list, predictions))
    return {label: float(score) for label, score in sorted(results.items(), key=lambda x: -x[1])[:top_n_moods_count]}

def analyze_track_audio(file_path, embedding_model_path, prediction_model_path, mood_labels_list, top_n_moods_count):
    """Analyzes a single track for tempo, key, scale, and moods."""
    audio = MonoLoader(filename=file_path)()
    tempo, _, _, _, _ = RhythmExtractor2013()(audio)
    key, scale, _ = KeyExtractor()(audio)
    moods = predict_moods(file_path, embedding_model_path, prediction_model_path, mood_labels_list, top_n_moods_count)
    return tempo, key, scale, moods

def track_exists_in_db(item_id):
    """Checks if a track's analysis already exists in the database."""
    conn = get_db_connection()
    with get_db_cursor(conn) as cur:
        cur.execute(sql.SQL("SELECT * FROM {} WHERE item_id = %s").format(sql.Identifier(SCORE_DB_TABLE_NAME)), (item_id,))
        row = cur.fetchone()
    return row

def save_track_analysis_to_db(item_id, title, author, tempo, key, scale, moods):
    """Saves the analysis results for a track to the database."""
    mood_str = json.dumps(moods) # Store moods as JSON string
    conn = get_db_connection()
    with get_db_cursor(conn) as cur:
        cur.execute(sql.SQL("""
            INSERT INTO {} (item_id, title, author, tempo, key, scale, mood_vector)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (item_id) DO NOTHING; -- Only insert if not exists
        """).format(sql.Identifier(SCORE_DB_TABLE_NAME)),
        (item_id, title, author, tempo, key, scale, mood_str))
    conn.commit()

def score_vector(row, mood_labels_list):
    """Converts a database row into a numerical feature vector for clustering."""
    tempo = float(row['tempo']) if row['tempo'] is not None else 0.0
    mood_str = row['mood_vector'] or "{}" 
    
    try:
        moods_dict = json.loads(mood_str)
    except json.JSONDecodeError:
        moods_dict = {} 

    tempo_norm = (tempo - 40) / (200 - 40)
    tempo_norm = np.clip(tempo_norm, 0.0, 1.0)
    mood_scores = np.zeros(len(mood_labels_list))
    if moods_dict:
        for label, score in moods_dict.items():
            if label in mood_labels_list:
                mood_scores[mood_labels_list.index(label)] = float(score)
    full_vector = [tempo_norm] + list(mood_scores)
    return full_vector

def get_all_tracks_from_db():
    """Retrieves all analyzed tracks from the database."""
    conn = get_db_connection()
    with get_db_cursor(conn) as cur:
        cur.execute(sql.SQL("SELECT item_id, title, author, tempo, key, scale, mood_vector FROM {}").format(sql.Identifier(SCORE_DB_TABLE_NAME)))
        rows = cur.fetchall()
    return rows

def name_cluster(centroid_scaled_vector, pca_model, pca_enabled, mood_labels_list):
    if pca_enabled and pca_model is not None:
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
        top_indices = np.argsort(mood_values)[::-1][:3] 

    mood_names = [mood_labels_list[i] for i in top_indices if i < len(mood_labels_list)]
    mood_part = "_".join(mood_names).title() if mood_names else "Mixed"
    full_name = f"{mood_part}_{tempo_label}"
    
    top_mood_scores = {mood_labels_list[i]: mood_values[i] for i in top_indices if i < len(mood_labels_list)}
    extra_info = {"tempo": round(tempo_norm, 2)}
    
    return full_name, {**top_mood_scores, **extra_info}

def update_playlist_table_in_db(playlists):
    conn = get_db_connection()
    with get_db_cursor(conn) as cur:
        cur.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(PLAYLIST_DB_TABLE_NAME))) 
        for name, cluster in playlists.items():
            for item_id, title, author in cluster:
                cur.execute(sql.SQL("INSERT INTO {} (playlist, item_id, title, author) VALUES (%s, %s, %s, %s)").format(sql.Identifier(PLAYLIST_DB_TABLE_NAME)),
                            (name, item_id, title, author))
    conn.commit()

def delete_old_automatic_playlists_from_jellyfin(jellyfin_url, jellyfin_user_id, headers):
    url_base = jellyfin_url if jellyfin_url.endswith('/') else jellyfin_url + '/'
    path = f"Users/{jellyfin_user_id.strip('/')}/Items"
    full_url = urljoin(url_base, path)
    params = {"IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(full_url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        for item in r.json().get("Items", []):
            if "_automatic" in item.get("Name", ""):
                del_path = f"Items/{item['Id']}"
                del_url = urljoin(url_base, del_path)
                del_resp = requests.delete(del_url, headers=headers, timeout=10)
                if del_resp.ok:
                    print(f"🗑️ Deleted old playlist: {item['Name']}")
    except Exception as e:
        print(f"Failed to clean old playlists: {e}")

def create_or_update_playlists_on_jellyfin(jellyfin_url_val, jellyfin_user_id_val, headers_val, playlists_val, cluster_centers_val, mood_labels_list, max_songs_val):
    delete_old_automatic_playlists_from_jellyfin(jellyfin_url_val, jellyfin_user_id_val, headers_val)
    url_base = jellyfin_url_val if jellyfin_url_val.endswith('/') else jellyfin_url_val + '/'
    playlist_creation_url = urljoin(url_base, "Playlists")

    for base_name, cluster in playlists_val.items():
        chunks = [cluster[i:i+max_songs_val] for i in range(0, len(cluster), max_songs_val)]
        for idx, chunk in enumerate(chunks, 1):
            playlist_name = f"{base_name}_automatic_{idx}" if len(chunks) > 1 else f"{base_name}_automatic"
            item_ids = [item_id for item_id, _, _ in chunk]
            if not item_ids:
                continue
            body = {"Name": playlist_name, "Ids": item_ids, "UserId": jellyfin_user_id_val}
            try:
                r = requests.post(playlist_creation_url, headers=headers_val, json=body, timeout=30)
                if r.ok:
                    centroid_info = cluster_centers_val.get(base_name, {})
                    top_moods = {k: v for k, v in centroid_info.items() if k in mood_labels_list}
                    extra_info = {k: v for k, v in centroid_info.items() if k not in mood_labels_list}
                    centroid_str = ", ".join(f"{k}:{v:.2f}" for k, v in top_moods.items())
                    extras_str = ", ".join(f"{k}:{v:.2f}" for k, v in extra_info.items())
                    print(f"✅ Created playlist {playlist_name} with {len(item_ids)} tracks (Centroid: {centroid_str} | {extras_str})")
            except Exception as e:
                print(f"Exception creating {playlist_name}: {e}")

# --- Individual Celery Tasks ---

@celery.task
def analyze_album_task(album_details, jellyfin_url_val, jellyfin_user_id_val, jellyfin_token_val, top_n_moods_val, temp_dir_base, embedding_path, prediction_path, mood_labels_list_val):
    with app.app_context():
        album_id = album_details['Id']
        album_name = album_details['Name']
        headers = {"X-Emby-Token": jellyfin_token_val}
        
        task_temp_dir = os.path.join(temp_dir_base, f"album_{album_id}_{random.randint(1000,9999)}")
        os.makedirs(task_temp_dir, exist_ok=True)

        tracks_analyzed_count = 0 
        tracks_failed_count = 0
        error_messages = []
        
        print(f"🎵 Processing Album: {album_name}")
        tracks = get_tracks_from_album(jellyfin_url_val, jellyfin_user_id_val, headers, album_id)
        if not tracks:
            print(f"   ⚠️ No tracks found for album: {album_name}")
            shutil.rmtree(task_temp_dir) 
            return (album_name, 0, 0, ["No tracks found"])

        for item in tracks:
            track_name_full = f"{item['Name']} by {item.get('AlbumArtist', 'Unknown')}"
            print(f"   🎶 Analyzing track: {track_name_full}")
            if track_exists_in_db(item['Id']):
                print(f"     ⏭️ Skipping '{track_name_full}' (already analyzed)")
                tracks_analyzed_count +=1 
                continue
            
            path = download_track(jellyfin_url_val, headers, task_temp_dir, item)
            if not path:
                print(f"     ❌ Failed to download '{track_name_full}'. Skipping.")
                tracks_failed_count += 1
                error_messages.append(f"Download failed for {track_name_full}")
                continue
            try:
                analysis_start_time = time.time()
                tempo, key, scale, moods = analyze_track_audio(path, embedding_path, prediction_path, mood_labels_list_val, top_n_moods_val)
                analysis_duration = time.time() - analysis_start_time
                save_track_analysis_to_db(item['Id'], item['Name'], item.get('AlbumArtist', 'Unknown'), tempo, key, scale, moods)
                print(f"     ✅ Analyzed '{track_name_full}' in {analysis_duration:.2f}s. Moods: {', '.join(f'{k}:{v:.2f}' for k,v in moods.items())}")
                tracks_analyzed_count += 1
            except Exception as e:
                print(f"     ❌ Error analyzing '{track_name_full}': {e}")
                tracks_failed_count += 1
                error_messages.append(f"Analysis error for {track_name_full}: {str(e)}")
            finally:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as cleanup_e:
                        print(f"WARNING: Failed to clean up temp file {path}: {cleanup_e}")
        
        shutil.rmtree(task_temp_dir) 
        return (album_name, tracks_analyzed_count, tracks_failed_count, error_messages)

@celery.task
def run_single_clustering_iteration_task(run_idx, X_scaled_data, rows_data, mood_labels_list,
                                         clustering_method_val, num_clusters_min_val, num_clusters_max_val,
                                         dbscan_eps_min_val, dbscan_eps_max_val,
                                         dbscan_min_samples_min_val, dbscan_min_samples_max_val,
                                         gmm_n_components_min_val, gmm_n_components_max_val,
                                         pca_components_min_val, pca_components_max_val,
                                         max_songs_per_cluster_val, current_max_songs_per_artist):
    with app.app_context():
        current_num_clusters = 0
        current_dbscan_eps = 0.0
        current_dbscan_min_samples = 0
        current_gmm_n_components = 0
        
        current_pca_components = random.randint(pca_components_min_val, pca_components_max_val)
        current_pca_enabled = (current_pca_components > 0)
        
        params_used = {
            "clustering_method": clustering_method_val,
            "pca_components": current_pca_components,
            "pca_enabled": current_pca_enabled,
            "max_songs_per_cluster": max_songs_per_cluster_val,
            "run_index": run_idx
        }

        if clustering_method_val == "kmeans":
            current_num_clusters = random.randint(num_clusters_min_val, num_clusters_max_val)
            current_num_clusters = min(current_num_clusters, len(rows_data))
            if current_num_clusters == 0: 
                current_num_clusters = max(1, len(rows_data) // max_songs_per_cluster_val if max_songs_per_cluster_val > 0 else 1)
            params_used["num_clusters"] = current_num_clusters
        elif clustering_method_val == "dbscan":
            current_dbscan_eps = round(random.uniform(dbscan_eps_min_val, dbscan_eps_max_val), 2)
            current_dbscan_min_samples = random.randint(dbscan_min_samples_min_val, dbscan_min_samples_max_val)
            params_used["dbscan_eps"] = current_dbscan_eps
            params_used["dbscan_min_samples"] = current_dbscan_min_samples
        elif clustering_method_val == "gmm":
            current_gmm_n_components = random.randint(gmm_n_components_min_val, gmm_n_components_max_val)
            current_gmm_n_components = min(current_gmm_n_components, len(rows_data))
            if current_gmm_n_components == 0:
                current_gmm_n_components = max(1, len(rows_data) // max_songs_per_cluster_val if max_songs_per_cluster_val > 0 else 1)
            params_used["gmm_n_components"] = current_gmm_n_components
        
        print(f"Cluster Run {run_idx}: Params: {params_used}")

        pca_model_instance = None
        data_for_clustering_run = np.array(X_scaled_data)

        if current_pca_enabled:
            n_components_actual = min(current_pca_components, data_for_clustering_run.shape[1], len(rows_data) - 1)
            if n_components_actual > 0:
                pca_model_instance = PCA(n_components=n_components_actual)
                X_pca = pca_model_instance.fit_transform(data_for_clustering_run)
                data_for_clustering_run = X_pca
            else:
                current_pca_enabled = False
                params_used["pca_enabled"] = False

        labels = None
        cluster_centers_map = {}
        raw_distances_arr = np.zeros(len(data_for_clustering_run))

        try:
            if clustering_method_val == "kmeans":
                kmeans = KMeans(n_clusters=current_num_clusters, random_state=None, n_init='auto')
                labels = kmeans.fit_predict(data_for_clustering_run)
                cluster_centers_map = {i: kmeans.cluster_centers_[i] for i in range(current_num_clusters)}
                centers_for_points = kmeans.cluster_centers_[labels]
                raw_distances_arr = np.linalg.norm(data_for_clustering_run - centers_for_points, axis=1)
            elif clustering_method_val == "dbscan":
                dbscan = DBSCAN(eps=current_dbscan_eps, min_samples=current_dbscan_min_samples)
                labels = dbscan.fit_predict(data_for_clustering_run)
                
                for cluster_id_val in set(labels):
                    if cluster_id_val == -1: 
                        continue
                    indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id_val]
                    cluster_points = np.array([data_for_clustering_run[i] for i in indices])
                    if len(cluster_points) > 0:
                        center = cluster_points.mean(axis=0)
                        for i in indices:
                            raw_distances_arr[i] = np.linalg.norm(data_for_clustering_run[i] - center)
                        cluster_centers_map[cluster_id_val] = center
            elif clustering_method_val == "gmm":
                gmm = GaussianMixture(n_components=current_gmm_n_components, covariance_type=GMM_COVARIANCE_TYPE, random_state=None, max_iter=200)
                gmm.fit(data_for_clustering_run)
                labels = gmm.predict(data_for_clustering_run)
                
                for i in range(current_gmm_n_components):
                    cluster_centers_map[i] = gmm.means_[i]
                
                centers_for_points = gmm.means_[labels]
                raw_distances_arr = np.linalg.norm(data_for_clustering_run - centers_for_points, axis=1)

            else:
                return {'score': -1.0, 'params': params_used, 'playlists': {}, 'centroids': {}, 'pca_model_components': None, 'pca_model_mean': None, 'error': 'Unsupported clustering method'}

            if labels is None or len(set(labels) - {-1}) == 0: 
                 return {'score': -1.0, 'params': params_used, 'playlists': {}, 'centroids': {}, 'pca_model_components': None, 'pca_model_mean': None, 'error': 'No valid clusters generated for this parameter set.'}
        except Exception as e:
            print(f"Error during clustering iteration {run_idx} with params {params_used}: {e}")
            return {'score': -1.0, 'params': params_used, 'playlists': {}, 'centroids': {}, 'pca_model_components': None, 'pca_model_mean': None, 'error': str(e)}


        max_dist_val = raw_distances_arr.max()
        normalized_distances = raw_distances_arr / max_dist_val if max_dist_val > 0 else raw_distances_arr
        
        track_info_list = []
        for row_item, label_item, vec_item, dist_item in zip(rows_data, labels, data_for_clustering_run, normalized_distances):
            if label_item == -1: 
                continue
            track_info_list.append({"row": row_item, "label": label_item, "vector": vec_item, "distance": dist_item})

        filtered_clusters_map = defaultdict(list)
        for cluster_id_val in set(labels):
            if cluster_id_val == -1:
                continue
            cluster_tracks_list = [t for t in track_info_list if t["label"] == cluster_id_val and t["distance"] <= MAX_DISTANCE]
            if not cluster_tracks_list:
                continue
            cluster_tracks_list.sort(key=lambda x: x["distance"])
            
            count_per_artist = defaultdict(int)
            selected_tracks = []
            for t in cluster_tracks_list:
                author = t["row"]["author"] 
                if count_per_artist[author] < current_max_songs_per_artist:
                    selected_tracks.append(t)
                    count_per_artist[author] += 1
                if len(selected_tracks) >= max_songs_per_cluster_val:
                    break
            for t in selected_tracks:
                item_id = t["row"]["item_id"] 
                title = t["row"]["title"]
                author = t["row"]["author"]
                filtered_clusters_map[cluster_id_val].append((item_id, title, author))

        current_named_playlists_map = defaultdict(list)
        current_playlist_centroids_map = {}
        
        unique_predominant_mood_scores_map = {} 

        for label_item, songs_list in filtered_clusters_map.items():
            if songs_list:
                center = cluster_centers_map.get(label_item) 
                if center is None:
                    continue 

                temp_pca_model = None
                if current_pca_enabled and pca_model_instance:
                    temp_pca_model = PCA(n_components=pca_model_instance.n_components_)
                    temp_pca_model.components_ = pca_model_instance.components_
                    temp_pca_model.mean_ = pca_model_instance.mean_

                name, top_scores = name_cluster(center, temp_pca_model, current_pca_enabled, mood_labels_list)
                
                if top_scores and any(mood in mood_labels_list for mood in top_scores.keys()):
                    predominant_mood_key = max(top_scores, key=lambda k: top_scores[k] if k in mood_labels_list else -1)
                    if predominant_mood_key in mood_labels_list:
                        current_mood_score = top_scores.get(predominant_mood_key, 0.0)
                        unique_predominant_mood_scores_map[predominant_mood_key] = max(
                            unique_predominant_mood_scores_map.get(predominant_mood_key, 0.0),
                            current_mood_score
                        )

                current_named_playlists_map[name].extend(songs_list)
                current_playlist_centroids_map[name] = top_scores
        
        diversity_score_val = sum(unique_predominant_mood_scores_map.values())
        
        pca_components_data = pca_model_instance.components_.tolist() if pca_model_instance and hasattr(pca_model_instance, 'components_') else None
        pca_mean_data = pca_model_instance.mean_.tolist() if pca_model_instance and hasattr(pca_model_instance, 'mean_') else None

        return {
            'score': float(diversity_score_val), 
            'params': params_used,
            'playlists': dict(current_named_playlists_map), 
            'centroids': current_playlist_centroids_map,
            'pca_model_components': pca_components_data,
            'pca_model_mean': pca_mean_data
        }

# --- Main Celery Tasks (Coordinators) ---

@celery.task(bind=True)
def run_analysis_task(self, jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums_val, top_n_moods_val):
    with app.app_context():
        task_id = self.request.id
        save_task_status(task_id, "analysis", "STARTED")
        log_messages = ["🚀 Starting mood-based analysis orchestration..."]
        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Fetching album list...', 'log_output': log_messages, 'processed_albums': 0, 'total_albums': 0, 'task_type': 'analysis'})
        
        headers = {"X-Emby-Token": jellyfin_token}
        
        try:
            clean_temp(TEMP_DIR) 
            albums = get_recent_albums(jellyfin_url, jellyfin_user_id, headers, num_recent_albums_val)
            
            if not albums:
                log_messages.append("⚠️ No new albums to analyze. Analysis complete.")
                self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'No new albums to analyze.', 'log_output': log_messages, 'processed_albums': 0, 'total_albums': 0, 'task_type': 'analysis'})
                save_task_status(task_id, "analysis", "SUCCESS")
                return {"status": "SUCCESS", "message": "No new albums to analyze."}

            total_albums_count = len(albums)
            log_messages.append(f"Found {total_albums_count} albums to process.")
            self.update_state(state='PROGRESS', meta={'progress': 5, 'status': f'Dispatching {total_albums_count} album analysis tasks...', 'log_output': log_messages, 'processed_albums': 0, 'total_albums': total_albums_count, 'task_type': 'analysis'})

            analysis_subtasks = [
                analyze_album_task.s(
                    album, jellyfin_url, jellyfin_user_id, jellyfin_token, top_n_moods_val,
                    TEMP_DIR, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, MOOD_LABELS
                ) for album in albums
            ]
            group_task = group(analysis_subtasks)
            group_result = group_task.apply_async()
            
            processed_count = 0
            while not group_result.ready():
                completed_count = group_result.completed_count()
                if completed_count > processed_count : 
                    processed_count = completed_count
                    progress = 5 + int(90 * (processed_count / total_albums_count))
                    log_messages.append(f"Processed {processed_count}/{total_albums_count} albums...")
                    short_log = log_messages[-5:] if len(log_messages) > 5 else log_messages
                    self.update_state(state='PROGRESS', meta={
                        'progress': progress, 
                        'status': f'Analyzing albums: {processed_count}/{total_albums_count} complete.',
                        'log_output': short_log + [f"...See Celery worker logs for full details of task {task_id}..." ],
                        'processed_albums': processed_count, 
                        'total_albums': total_albums_count,
                        'task_type': 'analysis'
                    })
                time.sleep(5)

            all_results = group_result.get(timeout=300) 
            
            final_analyzed_count = 0
            final_failed_count = 0
            aggregated_errors = []
            for res_tuple in all_results:
                if isinstance(res_tuple, tuple) and len(res_tuple) == 4:
                    album_n, analyzed_c, failed_c, errors_list = res_tuple
                    final_analyzed_count += analyzed_c
                    final_failed_count += failed_c
                    if errors_list:
                        aggregated_errors.extend(f"Album '{album_n}': {e}" for e in errors_list)
                else: 
                    final_failed_count +=1 
                    aggregated_errors.append(f"Unknown album processing issue for one task: {str(res_tuple)}")


            log_messages.append(f"Analysis complete. Total tracks analyzed (incl. skips): {final_analyzed_count}. Total tracks failed: {final_failed_count}.")
            if aggregated_errors:
                log_messages.append("Summary of errors during analysis:")
                log_messages.extend(aggregated_errors[:10]) 
                if len(aggregated_errors) > 10:
                    log_messages.append(f"...and {len(aggregated_errors) - 10} more errors. Check worker logs.")
            
            final_status_message = f"Analysis complete. Processed {total_albums_count} albums."
            if final_failed_count > 0 :
                final_status_message += f" Encountered {final_failed_count} track processing failures."

            self.update_state(state='SUCCESS', meta={
                'progress': 100, 
                'status': final_status_message,
                'log_output': log_messages[-10:], 
                'processed_albums': total_albums_count, 
                'total_albums': total_albums_count,
                'task_type': 'analysis'
            })
            save_task_status(task_id, "analysis", "SUCCESS")
            return {"status": "SUCCESS", "message": final_status_message, "details": {"analyzed": final_analyzed_count, "failed": final_failed_count, "errors": aggregated_errors}}

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            log_messages.append(f"❌ Analysis orchestration failed: {e}")
            self.update_state(state='FAILURE', meta={
                'progress': 100, 
                'status': f'Analysis orchestration failed: {e}', 
                'log_output': log_messages + [f"Error Traceback: {error_traceback}"],
                'task_type': 'analysis'
                })
            save_task_status(task_id, "analysis", "FAILURE")
            return {"status": "FAILURE", "message": f"Analysis orchestration failed: {e}"}


@celery.task(bind=True)
def run_clustering_task(self, clustering_method, num_clusters_min, num_clusters_max,
                        dbscan_eps_min, dbscan_eps_max, dbscan_min_samples_min, dbscan_min_samples_max,
                        gmm_n_components_min, gmm_n_components_max,
                        pca_components_min, pca_components_max,
                        num_clustering_runs, max_songs_per_cluster, current_max_songs_per_artist=MAX_SONGS_PER_ARTIST):
    with app.app_context():
        task_id = self.request.id
        save_task_status(task_id, "clustering", "STARTED")
        log_messages = ["📊 Starting playlist clustering orchestration with parallel iterations..."]
        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Preparing for clustering...', 'log_output': log_messages, 'iterations_done': 0, 'total_iterations': num_clustering_runs, 'task_type': 'clustering'})

        try:
            rows = get_all_tracks_from_db() 
            if len(rows) < 2: 
                log_messages.append("Not enough analyzed tracks for clustering. Please run analysis first.")
                self.update_state(state='FAILURE', meta={'progress': 100, 'status': 'Not enough tracks', 'log_output': log_messages, 'task_type': 'clustering'})
                save_task_status(task_id, "clustering", "FAILURE")
                return {"status": "FAILURE", "message": "Not enough analyzed tracks for clustering."}

            log_messages.append(f"Fetched {len(rows)} tracks for clustering.")
            X_original = [score_vector(row, MOOD_LABELS) for row in rows]
            X_scaled_np = np.array(X_original)
            
            X_scaled_list = X_scaled_np.tolist() 

            self.update_state(state='PROGRESS', meta={'progress': 5, 'status': f'Dispatching {num_clustering_runs} clustering iterations...', 'log_output': log_messages, 'iterations_done': 0, 'total_iterations': num_clustering_runs, 'task_type': 'clustering'})

            iteration_subtasks = [
                run_single_clustering_iteration_task.s(
                    i, X_scaled_list, rows, MOOD_LABELS, 
                    clustering_method, num_clusters_min, num_clusters_max,
                    dbscan_eps_min, dbscan_eps_max, dbscan_min_samples_min, dbscan_min_samples_max,
                    gmm_n_components_min, gmm_n_components_max,
                    pca_components_min, pca_components_max,
                    max_songs_per_cluster, current_max_songs_per_artist
                ) for i in range(num_clustering_runs)
            ]
            
            group_task = group(iteration_subtasks)
            group_result = group_task.apply_async()

            iterations_completed_count = 0
            while not group_result.ready():
                current_completed = group_result.completed_count()
                if current_completed > iterations_completed_count:
                    iterations_completed_count = current_completed
                    progress = 5 + int(85 * (iterations_completed_count / num_clustering_runs))
                    log_messages.append(f"Clustering iterations completed: {iterations_completed_count}/{num_clustering_runs}")
                    self.update_state(state='PROGRESS', meta={
                        'progress': progress, 
                        'status': f'Running clustering: {iterations_completed_count}/{num_clustering_runs} iterations done.',
                        'log_output': log_messages[-5:] + [f"...See Celery worker logs for full details of task {task_id}..." ], 
                        'iterations_done': iterations_completed_count, 
                        'total_iterations': num_clustering_runs,
                        'task_type': 'clustering'
                    })
                time.sleep(3) 

            all_iteration_results = group_result.get(timeout=600) 

            best_diversity_score = -1.0
            best_result_package = None
            
            for res in all_iteration_results:
                if isinstance(res, dict) and 'score' in res and 'params' in res:
                    if res.get('error'):
                        print(f"Clustering iteration {res['params'].get('run_index', '')} failed or had an issue: {res['error']}")
                        continue 
                    if res['score'] > best_diversity_score:
                        best_diversity_score = res['score']
                        best_result_package = res
                else:
                    print(f"Warning: Received malformed result from a clustering subtask: {res}")


            if not best_result_package or best_diversity_score < 0: 
                log_messages.append("No valid clustering results found after all iterations.")
                self.update_state(state='FAILURE', meta={'progress': 100, 'status': 'No valid clusters from iterations', 'log_output': log_messages, 'task_type': 'clustering'})
                save_task_status(task_id, "clustering", "FAILURE")
                return {"status": "FAILURE", "message": "No valid clusters found after multiple runs."}

            log_messages.append(f"Best clustering iteration found with score: {best_diversity_score:.2f}. Parameters: {best_result_package['params']}")
            self.update_state(state='PROGRESS', meta={'progress': 95, 'status': 'Finalizing best playlists...', 'log_output': log_messages, 'iterations_done': num_clustering_runs, 'total_iterations': num_clustering_runs, 'task_type': 'clustering'})

            final_named_playlists = best_result_package['playlists']
            final_playlist_centroids = best_result_package['centroids']
            
            final_pca_model = None
            if best_result_package['params'].get('pca_enabled') and best_result_package.get('pca_model_components') is not None:
                final_pca_model = PCA(n_components=len(best_result_package['pca_model_components']))
                final_pca_model.components_ = np.array(best_result_package['pca_model_components'])
                if best_result_package.get('pca_model_mean') is not None:
                    final_pca_model.mean_ = np.array(best_result_package['pca_model_mean'])
                else:
                    final_pca_model.mean_ = np.zeros(final_pca_model.components_.shape[1])

            final_max_songs_per_cluster = best_result_package['params']['max_songs_per_cluster']

            log_messages.append("Updating playlist database...")
            update_playlist_table_in_db(final_named_playlists) 
            
            log_messages.append("Creating/Updating playlists on Jellyfin...")
            create_or_update_playlists_on_jellyfin(JELLYFIN_URL, JELLYFIN_USER_ID, {"X-Emby-Token": JELLYFIN_TOKEN}, final_named_playlists, final_playlist_centroids, MOOD_LABELS, final_max_songs_per_cluster)
            
            final_message = f"Playlists generated and updated on Jellyfin! Best diversity score: {best_diversity_score:.2f}."
            log_messages.append(final_message)
            self.update_state(state='SUCCESS', meta={'progress': 100, 'status': final_message, 'log_output': log_messages[-10:], 'task_type': 'clustering'})
            save_task_status(task_id, "clustering", "SUCCESS")
            return {"status": "SUCCESS", "message": final_message, "best_score": best_diversity_score, "best_params": best_result_package['params']}

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            log_messages.append(f"❌ Clustering orchestration failed: {e}")
            self.update_state(state='FAILURE', meta={'progress': 100, 'status': f'Clustering failed: {e}', 'log_output': log_messages + [f"Error Traceback: {error_traceback}"], 'task_type': 'clustering'})
            save_task_status(task_id, "clustering", "FAILURE")
            return {"status": "FAILURE", "message": f"Clustering orchestration failed: {e}"}


# --- API Endpoints ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analysis/start', methods=['POST'])
def start_analysis():
    data = request.json or {}
    jellyfin_url_req = data.get('jellyfin_url', JELLYFIN_URL)
    jellyfin_user_id_req = data.get('jellyfin_user_id', JELLYFIN_USER_ID)
    jellyfin_token_req = data.get('jellyfin_token', JELLYFIN_TOKEN)
    num_recent_albums_req = int(data.get('num_recent_albums', NUM_RECENT_ALBUMS))
    top_n_moods_req = int(data.get('top_n_moods', TOP_N_MOODS))
    
    task = run_analysis_task.delay(jellyfin_url_req, jellyfin_user_id_req, jellyfin_token_req, num_recent_albums_req, top_n_moods_req)
    # save_task_status is called inside the task now, no need to call here if we want STARTED status from task
    # For PENDING status from API immediately:
    with app.app_context(): # Need app context for DB ops here if not in task
        save_task_status(task.id, "analysis", "PENDING") 
    return jsonify({"task_id": task.id, "task_type": "analysis", "status": "PENDING"}), 202

@app.route('/api/clustering/start', methods=['POST'])
def start_clustering():
    data = request.json
    clustering_method_req = data.get('clustering_method', CLUSTER_ALGORITHM)
    num_clusters_min_req = int(data.get('num_clusters_min', NUM_CLUSTERS_MIN))
    num_clusters_max_req = int(data.get('num_clusters_max', NUM_CLUSTERS_MAX))
    dbscan_eps_min_req = float(data.get('dbscan_eps_min', DBSCAN_EPS_MIN))
    dbscan_eps_max_req = float(data.get('dbscan_eps_max', DBSCAN_EPS_MAX))
    dbscan_min_samples_min_req = int(data.get('dbscan_min_samples_min', DBSCAN_MIN_SAMPLES_MIN))
    dbscan_min_samples_max_req = int(data.get('dbscan_min_samples_max', DBSCAN_MIN_SAMPLES_MAX))
    gmm_n_components_min_req = int(data.get('gmm_n_components_min', GMM_N_COMPONENTS_MIN))
    gmm_n_components_max_req = int(data.get('gmm_n_components_max', GMM_N_COMPONENTS_MAX))
    pca_components_min_req = int(data.get('pca_components_min', PCA_COMPONENTS_MIN))
    pca_components_max_req = int(data.get('pca_components_max', PCA_COMPONENTS_MAX))
    num_clustering_runs_req = int(data.get('clustering_runs', CLUSTERING_RUNS))
    max_songs_per_cluster_req = int(data.get('max_songs_per_cluster', MAX_SONGS_PER_CLUSTER))
    
    task = run_clustering_task.delay(
        clustering_method_req, 
        num_clusters_min_req, num_clusters_max_req,
        dbscan_eps_min_req, dbscan_eps_max_req,
        dbscan_min_samples_min_req, dbscan_min_samples_max_req,
        gmm_n_components_min_req, gmm_n_components_max_req,
        pca_components_min_req, pca_components_max_req,
        num_clustering_runs_req,
        max_songs_per_cluster_req
    )
    with app.app_context(): # Need app context for DB ops here if not in task
      save_task_status(task.id, "clustering", "PENDING")
    return jsonify({"task_id": task.id, "task_type": "clustering", "status": "PENDING"}), 202


@app.route('/api/status/<task_id>', methods=['GET'])
def get_task_status_endpoint(task_id):
    task = AsyncResult(task_id, app=celery)
    response = {
        'task_id': task.id,
        'state': task.state, 
        'status': 'Processing...', 
        'log_output': [],
        'progress': 0,
    }
    task_info = task.info if isinstance(task.info, dict) else {}

    if 'task_type' in task_info:
        response['task_type'] = task_info['task_type']
        if task_info['task_type'] == 'analysis':
            response.update({'processed_albums': 0, 'total_albums': 0, 'current_album_progress': '0/0', 'currentAlbumName': 'N/A'})
        elif task_info['task_type'] == 'clustering':
            response.update({'iterations_done': 0, 'total_iterations': 0})
    
    response.update(task_info)

    if task.state == 'PENDING':
        response['status'] = 'Task is pending or not yet started.'
        if not response.get('log_output'): response['log_output'] = ['Task pending...']
    elif task.state == 'STARTED': 
        response['status'] = task_info.get('status', 'Task has started...')
    elif task.state == 'PROGRESS':
        response['status'] = task_info.get('status', 'Task in progress...')
    elif task.state == 'SUCCESS':
        response['status'] = task_info.get('status', 'Task complete!')
        response['progress'] = 100
        with app.app_context(): save_task_status(task_id, task_info.get('task_type', 'unknown'), "SUCCESS")
    elif task.state == 'FAILURE':
        response['status'] = task_info.get('status', str(task.info)) 
        response['progress'] = 100
        with app.app_context(): save_task_status(task_id, task_info.get('task_type', 'unknown'), "FAILURE")
    elif task.state == 'REVOKED':
        response['status'] = 'Task revoked/cancelled.'
        response['progress'] = 100
        with app.app_context(): save_task_status(task_id, task_info.get('task_type', 'unknown'), "REVOKED")
    else: 
        response['status'] = f'Task state: {task.state}'
        if not response.get('log_output'): response['log_output'] = [f'Current state: {task.state}']

    if response.get('task_type') == 'analysis':
        response['currentAlbumName'] = task_info.get('current_album', 'N/A') 
        response['currentAlbumProgress'] = f"{task_info.get('processed_albums',0)}/{task_info.get('total_albums',0)}"
        response['current_album_idx'] = task_info.get('processed_albums', 0) 
        response['total_albums'] = task_info.get('total_albums', 0) 

    return jsonify(response)

@app.route('/api/cancel/<task_id>', methods=['POST'])
def cancel_task_endpoint(task_id):
    task = AsyncResult(task_id, app=celery)
    task_type_from_db = 'unknown' # Default
    with app.app_context():
        conn = get_db_connection()
        with get_db_cursor(conn) as cur:
            cur.execute(sql.SQL("SELECT task_type FROM {} WHERE task_id = %s").format(sql.Identifier(STATUS_DB_TABLE_NAME)), (task_id,))
            row = cur.fetchone()
        if row: task_type_from_db = row['task_type']


    if task.state in ['PENDING', 'STARTED', 'PROGRESS']:
        task.revoke(terminate=True, signal='SIGKILL') 

        try:
            if task.result and hasattr(task.result, 'revoke'): 
                 task.result.revoke(terminate=True, signal='SIGKILL')
            elif task.children: 
                for child_task_res in task.children:
                    child_task_res.revoke(terminate=True, signal='SIGKILL')
        except Exception as e:
            print(f"Error trying to revoke children of task {task_id}: {e}")
        with app.app_context():
          save_task_status(task_id, task_type_from_db, "REVOKED")
        return jsonify({"message": "Task cancellation requested.", "task_id": task_id}), 200
    else:
        return jsonify({"message": "Task cannot be cancelled in its current state.", "state": task.state}), 400

@app.route('/api/last_task', methods=['GET'])
def get_last_overall_task_status():
    with app.app_context(): # Ensure app context for get_last_task_status
        last_task = get_last_task_status()
    if last_task:
        return jsonify(last_task), 200
    return jsonify({"task_id": None, "task_type": None, "status": "NO_PREVIOUS_TASK"}), 200

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
        "num_clusters_min": NUM_CLUSTERS_MIN,
        "num_clusters_max": NUM_CLUSTERS_MAX,
        "dbscan_eps_min": DBSCAN_EPS_MIN,
        "dbscan_eps_max": DBSCAN_EPS_MAX,
        "dbscan_min_samples_min": DBSCAN_MIN_SAMPLES_MIN,
        "dbscan_min_samples_max": DBSCAN_MIN_SAMPLES_MAX,
        "gmm_n_components_min": GMM_N_COMPONENTS_MIN, 
        "gmm_n_components_max": GMM_N_COMPONENTS_MAX, 
        "pca_components_min": PCA_COMPONENTS_MIN,
        "pca_components_max": PCA_COMPONENTS_MAX,
        "top_n_moods": TOP_N_MOODS,
        "mood_labels": MOOD_LABELS,
        "clustering_runs": CLUSTERING_RUNS,
    })

@app.route('/api/playlists', methods=['GET'])
def get_playlists():
    with app.app_context(): # Ensure app context for DB operations
        conn = get_db_connection()
        with get_db_cursor(conn) as cur:
            cur.execute(sql.SQL("SELECT playlist, item_id, title, author FROM {} ORDER BY playlist, title").format(sql.Identifier(PLAYLIST_DB_TABLE_NAME)))
            rows = cur.fetchall()
    playlists_data = defaultdict(list)
    for row in rows:
        playlists_data[row['playlist']].append({"item_id": row['item_id'], "title": row['title'], "author": row['author']})
    return jsonify(dict(playlists_data)), 200

if __name__ == '__main__':
    os.makedirs(TEMP_DIR, exist_ok=True) 
    app.run(debug=True, host='0.0.0.0', port=8000)
