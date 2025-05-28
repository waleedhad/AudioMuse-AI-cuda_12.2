import os
import shutil
import requests
from collections import defaultdict
import psycopg2
from psycopg2.extras import DictCursor
import numpy as np
from flask import Flask, jsonify, request, render_template, g, current_app
from celery import Celery
from celery.result import AsyncResult, GroupResult
from celery.signals import task_prerun, task_postrun
from contextlib import closing
import json
import time
import random # Import random for parameter sampling

# Import your existing analysis functions
from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor, TensorflowPredictMusiCNN, TensorflowPredict2D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture # Import GaussianMixture
from celery import current_task # Import current_task
# Your existing config - assuming this is from config.py and sets global variables
from config import JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, HEADERS, TEMP_DIR, MAX_DISTANCE, \
    MAX_SONGS_PER_CLUSTER, MAX_SONGS_PER_ARTIST, NUM_RECENT_ALBUMS, CLUSTER_ALGORITHM, NUM_CLUSTERS_MIN, \
    NUM_CLUSTERS_MAX, DBSCAN_EPS_MIN, DBSCAN_EPS_MAX, DBSCAN_MIN_SAMPLES_MIN, DBSCAN_MIN_SAMPLES_MAX, \
    GMM_N_COMPONENTS_MIN, GMM_N_COMPONENTS_MAX, GMM_COVARIANCE_TYPE, PCA_COMPONENTS_MIN, PCA_COMPONENTS_MAX, \
    CLUSTERING_RUNS, CELERY_BROKER_URL, CELERY_RESULT_BACKEND, MOOD_LABELS, TOP_N_MOODS, \
    EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH

# --- Flask App Setup ---
app = Flask(__name__)
# Celery configuration is now read from config.py
app.config['CELERY_BROKER_URL'] = CELERY_BROKER_URL
app.config['CELERY_RESULT_BACKEND'] = CELERY_RESULT_BACKEND

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# --- Database Setup (PostgreSQL) ---
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://audiomuse:audiomusepassword@postgres-service.playlist:5432/audiomusedb")

def get_db():
    if 'db' not in g:
        g.db = psycopg2.connect(DATABASE_URL)
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS score (
            item_id TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            tempo REAL,
            key TEXT,
            scale TEXT,
            mood_vector TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS playlist (
            id SERIAL PRIMARY KEY,
            playlist_name TEXT,
            item_id TEXT,
            title TEXT,
            author TEXT,
            UNIQUE (playlist_name, item_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS task_status (
            id SERIAL PRIMARY KEY,
            task_id TEXT UNIQUE NOT NULL,
            parent_task_id TEXT, -- For sub-tasks like album analysis or single clustering run
            task_type TEXT NOT NULL, -- 'main_analysis', 'album_analysis', 'main_clustering', 'single_clustering_run'
            sub_type_identifier TEXT, -- e.g., album_id for 'album_analysis', run_id for 'single_clustering_run'
            status TEXT,
            progress INTEGER DEFAULT 0,
            details TEXT, -- JSON string for additional details
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
    cur.close()

with app.app_context():
    init_db()

def save_task_status(task_id, task_type, status="PENDING", parent_task_id=None, sub_type_identifier=None, progress=0, details=None):
    db = get_db()
    cur = db.cursor()
    details_json = json.dumps(details) if details else None
    cur.execute("""
        INSERT INTO task_status (task_id, parent_task_id, task_type, sub_type_identifier, status, progress, details, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (task_id) DO UPDATE SET
            status = EXCLUDED.status,
            parent_task_id = EXCLUDED.parent_task_id,
            sub_type_identifier = EXCLUDED.sub_type_identifier,
            progress = EXCLUDED.progress,
            details = EXCLUDED.details,
            timestamp = NOW()
    """, (task_id, parent_task_id, task_type, sub_type_identifier, status, progress, details_json))
    db.commit()
    cur.close()

def get_task_info_from_db(task_id):
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT task_id, parent_task_id, task_type, sub_type_identifier, status, progress, details, timestamp FROM task_status WHERE task_id = %s", (task_id,))
    row = cur.fetchone()
    cur.close()
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
    moods = predict_moods(file_path, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, MOOD_LABELS, TOP_N_MOODS) # Use global paths and top_n_moods
    return tempo, key, scale, moods

def track_exists(db_path, item_id):
    """Checks if a track's analysis already exists in the PostgreSQL database."""
    # db_path is no longer used, connection comes from g.db
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT item_id FROM score WHERE item_id = %s", (item_id,))
    row = cur.fetchone()
    cur.close()
    return row

def save_track_analysis(db_path, item_id, title, author, tempo, key, scale, moods):
    """Saves the analysis results for a track to the PostgreSQL database."""
    # db_path is no longer used
    mood_str = ','.join(f"{k}:{v:.3f}" for k, v in moods.items())
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO score (item_id, title, author, tempo, key, scale, mood_vector) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (item_id) DO NOTHING",
                    (item_id, title, author, tempo, key, scale, mood_str))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error saving track analysis for {item_id}: {e}")
    finally:
        cur.close()

def score_vector(row, mood_labels):
    """Converts a database row into a numerical feature vector for clustering."""
    tempo = float(row['tempo']) if row['tempo'] is not None else 0.0 # Assuming row is a dict or DictRow
    mood_str = row['mood_vector'] or ""
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
    """Retrieves all analyzed tracks from the PostgreSQL database."""
    # db_path is no longer used
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor) # Use DictCursor for easy column access by name
    cur.execute("SELECT item_id, title, author, tempo, key, scale, mood_vector FROM score")
    rows = cur.fetchall()
    cur.close()
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
    """Updates the playlist table in the PostgreSQL database with new playlist data."""
    # db_path is no longer used
    conn = get_db()
    cur = conn.cursor()
    try:
        # Clear existing automatically generated playlists
        cur.execute("DELETE FROM playlist WHERE playlist_name LIKE '%_automatic%'")
        for name, cluster in playlists.items():
            for item_id, title, author in cluster:
                cur.execute("INSERT INTO playlist (playlist_name, item_id, title, author) VALUES (%s, %s, %s, %s) ON CONFLICT (playlist_name, item_id) DO NOTHING", (name, item_id, title, author))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error updating playlist table: {e}")
    finally:
        cur.close()

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

def create_or_update_playlists_on_jellyfin(jellyfin_url, jellyfin_user_id, headers, playlists, cluster_centers, mood_labels, max_songs_per_cluster):
    """Creates or updates playlists on Jellyfin based on clustering results."""
    delete_old_automatic_playlists(jellyfin_url, jellyfin_user_id, headers)
    for base_name, cluster in playlists.items():
        # Use the passed max_songs_per_cluster here
        chunks = [cluster[i:i+max_songs_per_cluster] for i in range(0, len(cluster), max_songs_per_cluster)]
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

# --- Celery Task Signal Handlers for DB status ---
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extras):
    # Determine task_type and other details from sender.name or kwargs
    # This is a generic handler; specific tasks might update their status more granularly.
    # For now, we rely on tasks calling save_task_status themselves upon starting.
    print(f"Task {task_id} ({sender.name}) is about to run.")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **extras):
    # This is where we can reliably update the final status from Celery's perspective
    # Tasks should ideally update their own status in the DB for more control, but this ensures a final update.
    with app.app_context(): # Ensure Flask application context is available
        task_info = get_task_info_from_db(task_id)
        if task_info:
            # If the task cooperatively cancelled itself, its status might already be REVOKED in DB
            # Only update if Celery's state is more definitive or different
            if task_info.get('status') != state or state in ['SUCCESS', 'FAILURE']: # Ensure final states from Celery are recorded
                save_task_status(task_id, task_info['task_type'], state,
                                 parent_task_id=task_info.get('parent_task_id'),
                                 sub_type_identifier=task_info.get('sub_type_identifier'),
                                 progress=100 if state in ['SUCCESS', 'FAILURE', 'REVOKED'] else task_info.get('progress',0),
                                 details={"message": str(retval)} if state != 'REVOKED' else {"message": "Task revoked."}) # Use a generic revoked message
    # close_db() is handled by app.teardown_appcontext when context pops
    print(f"Task {task_id} ({sender.name}) finished with state {state}.")


# --- Celery Task Definitions ---

@celery.task(bind=True)
def analyze_album_task(self, album_id, album_name, jellyfin_url, jellyfin_user_id, jellyfin_token, top_n_moods, parent_task_id):
    """Celery task to analyze a single album."""
    with app.app_context(): # Ensure Flask application context
        current_task_id = self.request.id
        # Create AsyncResult once for efficiency if task ID is available
        task_result_checker = AsyncResult(current_task_id, app=celery) if current_task_id else None
        save_task_status(current_task_id, "album_analysis", "STARTED", parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=0, details={"album_name": album_name})

        headers = {"X-Emby-Token": jellyfin_token}
        log_messages = []
        tracks_analyzed_count = 0

        def log_and_update_album_task(message, progress, current_track_name=None, task_state='PROGRESS'):
            log_messages.append(message)
            print(f"[AlbumTask-{current_task_id}-{album_name}] {message}") # Celery container log
            details = {"album_name": album_name, "log": log_messages, "current_track": current_track_name}
            if task_state != 'PROGRESS': # For final states like REVOKED
                 self.update_state(state=task_state, meta={'progress': progress, 'status': message, 'details': details})
            else: # For ongoing progress
                 self.update_state(state='PROGRESS', meta={'progress': progress, 'status': message, 'details': details})
            save_task_status(current_task_id, "album_analysis", task_state, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=progress, details=details)

        try:
            log_and_update_album_task(f"Fetching tracks for album: {album_name}", 5)
            tracks = get_tracks_from_album(jellyfin_url, jellyfin_user_id, headers, album_id)
            if not tracks:
                log_and_update_album_task(f"No tracks found for album: {album_name}", 100, task_state='SUCCESS')
                return {"status": "SUCCESS", "message": f"No tracks in album {album_name}", "tracks_analyzed": 0}

            total_tracks_in_album = len(tracks)
            for idx, item in enumerate(tracks, 1):
                track_name_full = f"{item['Name']} by {item.get('AlbumArtist', 'Unknown')}"
                current_progress_val = 10 + int(85 * (idx / float(total_tracks_in_album))) if total_tracks_in_album > 0 else 10
                
                if task_result_checker and task_result_checker.state == 'REVOKED':
                    log_and_update_album_task(f"Task revoked, stopping analysis of album {album_name} before track {track_name_full}.", current_progress_val, current_track_name=track_name_full, task_state='REVOKED')
                    return {"status": "REVOKED", "message": f"Task revoked during album analysis: {album_name}"}
                
                log_and_update_album_task(f"Analyzing track: {track_name_full} ({idx}/{total_tracks_in_album})", current_progress_val, current_track_name=track_name_full)

                if track_exists(None, item['Id']): # db_path not needed
                    log_and_update_album_task(f"Skipping '{track_name_full}' (already analyzed)", current_progress_val, current_track_name=track_name_full)
                    tracks_analyzed_count +=1 # Count as "processed" for progress
                    continue

                path = download_track(jellyfin_url, headers, TEMP_DIR, item)
                
                if task_result_checker and task_result_checker.state == 'REVOKED': # Check after potentially long I/O
                    log_and_update_album_task(f"Task revoked, stopping analysis of album {album_name} after download attempt for {track_name_full}.", current_progress_val, current_track_name=track_name_full, task_state='REVOKED')
                    return {"status": "REVOKED", "message": f"Task revoked during album analysis: {album_name}"}
                
                log_and_update_album_task(f"Download attempt for '{track_name_full}': {'Success' if path else 'Failed'}", current_progress_val, current_track_name=track_name_full)
                if not path:
                    log_and_update_album_task(f"Failed to download '{track_name_full}'. Skipping.", current_progress_val, current_track_name=track_name_full)
                    continue

                try:
                    if task_result_checker and task_result_checker.state == 'REVOKED': # Check before heavy computation
                        log_and_update_album_task(f"Task revoked, stopping before analyzing track {track_name_full}.", current_progress_val, current_track_name=track_name_full, task_state='REVOKED')
                        return {"status": "REVOKED", "message": f"Task revoked during album analysis: {album_name}"}
                    
                    tempo, key, scale, moods = analyze_track(path, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, MOOD_LABELS, top_n_moods)
                    save_track_analysis(None, item['Id'], item['Name'], item.get('AlbumArtist', 'Unknown'), tempo, key, scale, moods) # db_path not needed
                    tracks_analyzed_count += 1
                    mood_details_str = ', '.join(f'{k}:{v:.2f}' for k,v in moods.items())
                    log_and_update_album_task(f"Analyzed '{track_name_full}'. Tempo: {tempo:.2f}, Key: {key} {scale}. Moods: {mood_details_str}", current_progress_val, current_track_name=track_name_full)
                except Exception as e:
                    log_and_update_album_task(f"Error analyzing '{track_name_full}': {e}", current_progress_val, current_track_name=track_name_full)
                finally:
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception as cleanup_e:
                            print(f"WARNING: Failed to clean up temp file {path}: {cleanup_e}")

            log_and_update_album_task(f"Album '{album_name}' analysis complete.", 100, task_state='SUCCESS')
            return {"status": "SUCCESS", "message": f"Album '{album_name}' analysis complete.", "tracks_analyzed": tracks_analyzed_count, "total_tracks": total_tracks_in_album}
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            # Determine current progress before potential failure for logging
            current_progress_on_failure = 0 
            if 'current_progress_val' in locals():
                current_progress_on_failure = current_progress_val
            elif log_messages: # Try to get progress from last log message if possible
                try:
                    last_log_details = json.loads(log_messages[-1].get('details', '{}'))
                    current_progress_on_failure = last_log_details.get('progress', 0)
                except: pass # Ignore if parsing fails

            log_and_update_album_task(f"Failed to analyze album '{album_name}': {e}", current_progress_on_failure, task_state='FAILURE')
            print(f"ERROR: Album analysis {album_id} failed: {e}\n{error_traceback}")
            raise # Re-raise to mark task as FAILED in Celery

@celery.task(bind=True)
def run_analysis_task(self, jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods):
    """Main Celery task to orchestrate the analysis of multiple albums."""
    with app.app_context(): # Ensure Flask application context
        current_task_id = self.request.id
        task_result_checker = AsyncResult(current_task_id, app=celery) if current_task_id else None
        save_task_status(current_task_id, "main_analysis", "STARTED", progress=0, details={"message": "Fetching albums..."})

        headers = {"X-Emby-Token": jellyfin_token}
        log_messages = []
        current_progress = 0 # Initialize current_progress for the scope

        def log_and_update_main_analysis(message, progress, details_extra=None, task_state='PROGRESS'):
            nonlocal current_progress # Allow modification of outer scope variable
            current_progress = progress # Keep track of the latest progress
            log_messages.append(message)
            print(f"[MainAnalysisTask-{current_task_id}] {message}") # Celery container log
            current_details = {"log": log_messages, "overall_status": message}
            if details_extra:
                current_details.update(details_extra)
            
            if task_state != 'PROGRESS':
                self.update_state(state=task_state, meta={'progress': progress, 'status': message, 'details': current_details})
            else:
                self.update_state(state='PROGRESS', meta={'progress': progress, 'status': message, 'details': current_details})
            save_task_status(current_task_id, "main_analysis", task_state, progress=progress, details=current_details)

        try:
            log_and_update_main_analysis("🚀 Starting main analysis process...", 0)
            clean_temp(TEMP_DIR)

            if task_result_checker and task_result_checker.state == 'REVOKED':
                log_and_update_main_analysis("Main analysis task revoked at start.", 0, task_state='REVOKED')
                return {"status": "REVOKED", "message": "Main analysis task revoked."}
            
            albums = get_recent_albums(jellyfin_url, jellyfin_user_id, headers, num_recent_albums)
            if not albums:
                log_and_update_main_analysis("⚠️ No new albums to analyze.", 100, {"albums_found": 0}, task_state='SUCCESS')
                return {"status": "SUCCESS", "message": "No new albums to analyze.", "albums_processed": 0}

            total_albums = len(albums)
            log_and_update_main_analysis(f"Found {total_albums} albums to process.", 5, {"albums_found": total_albums, "album_tasks_ids": []})

            album_tasks_group = []
            album_task_ids = []
            for album_idx, album in enumerate(albums): # Use enumerate for progress calculation
                # Update progress before checking for revocation
                # This progress is for launching sub-tasks, not their completion
                launch_progress = 5 + int(5 * (album_idx / float(total_albums))) if total_albums > 0 else 5 
                if task_result_checker and task_result_checker.state == 'REVOKED':
                    log_and_update_main_analysis("Main analysis task revoked before processing all albums.", launch_progress, task_state='REVOKED')
                    return {"status": "REVOKED", "message": "Main analysis task revoked."}
                
                album_task = analyze_album_task.s(
                    album['Id'], album['Name'], jellyfin_url, jellyfin_user_id, jellyfin_token, top_n_moods, current_task_id
                ).apply_async()
                album_tasks_group.append(album_task)
                album_task_ids.append(album_task.id)

            log_and_update_main_analysis(f"Launched {total_albums} album analysis tasks.", 10, 
                             details={"message": f"Launched {total_albums} album analysis tasks.", "album_task_ids": album_task_ids})

            while not all(t.ready() for t in album_tasks_group):
                completed_count = sum(1 for t in album_tasks_group if t.ready())
                # Use the 'current_progress' variable that is updated by log_and_update_main_analysis
                # This ensures we use the latest known progress if revoked.
                progress_while_waiting = 10 + int(80 * (completed_count / float(total_albums))) if total_albums > 0 else 10
                if task_result_checker and task_result_checker.state == 'REVOKED':
                    log_and_update_main_analysis("Main analysis task revoked while waiting for album sub-tasks.", progress_while_waiting, task_state='REVOKED')
                    return {"status": "REVOKED", "message": "Main analysis task revoked."}
                
                log_and_update_main_analysis(
                    f"Processing albums: {completed_count}/{total_albums} completed.",
                    progress_while_waiting,
                    {"albums_completed": completed_count, "total_albums": total_albums, "album_task_ids": album_task_ids}
                )
                time.sleep(5) 

            successful_albums = 0
            failed_albums = 0
            total_tracks_analyzed_all_albums = 0
            for t_res in album_tasks_group:
                if t_res.successful():
                    successful_albums += 1
                    if isinstance(t_res.result, dict):
                        total_tracks_analyzed_all_albums += t_res.result.get("tracks_analyzed", 0)
                else:
                    failed_albums += 1
            
            final_message = f"Main analysis complete. Successful albums: {successful_albums}, Failed albums: {failed_albums}. Total tracks analyzed: {total_tracks_analyzed_all_albums}."
            log_and_update_main_analysis(final_message, 100, {"albums_completed": successful_albums, "albums_failed": failed_albums, "total_tracks_analyzed": total_tracks_analyzed_all_albums}, task_state='SUCCESS')
            clean_temp(TEMP_DIR)
            return {"status": "SUCCESS", "message": final_message, "successful_albums": successful_albums, "failed_albums": failed_albums, "total_tracks_analyzed": total_tracks_analyzed_all_albums}

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"FATAL ERROR: Analysis failed: {e}\n{error_traceback}")
            # Use the last known 'current_progress' for logging failure
            log_and_update_main_analysis(f"❌ Main analysis failed: {e}", current_progress, {"error": str(e)}, task_state='FAILURE')
            raise 

@celery.task(bind=True)
def run_single_clustering_iteration_task(self, run_id, all_tracks_data_json, clustering_method_config, pca_config, max_songs_per_cluster, parent_task_id):
    """Celery task for a single clustering iteration with specific parameters."""
    with app.app_context(): # Ensure Flask application context
        current_task_id = self.request.id
        task_result_checker = AsyncResult(current_task_id, app=celery) if current_task_id else None
        log_messages_iter = []
        initial_details = {"run_id": run_id, "params": clustering_method_config, "log": log_messages_iter}
        
        def log_and_update_single_run(message, progress, details_extra=None, task_state='PROGRESS'):
            log_messages_iter.append(message)
            print(f"[SingleClusteringRun-{current_task_id}] Run {run_id}: {message}")
            current_run_details = {"run_id": run_id, "log": log_messages_iter, "params": clustering_method_config}
            if details_extra:
                current_run_details.update(details_extra)
            
            # Update Celery's state for the sub-task
            if task_state != 'PROGRESS':
                 self.update_state(state=task_state, meta={'progress': progress, 'status': message, 'details': current_run_details})
            else:
                 self.update_state(state='PROGRESS', meta={'progress': progress, 'status': message, 'details': current_run_details})
            save_task_status(current_task_id, "single_clustering_run", task_state, parent_task_id=parent_task_id, sub_type_identifier=str(run_id), progress=progress, details=current_run_details)

        log_and_update_single_run(f"Starting with method {clustering_method_config['method']}, params: {clustering_method_config['params']}, PCA: {pca_config}", 0)

        all_tracks_data = json.loads(all_tracks_data_json) 
        if task_result_checker and task_result_checker.state == 'REVOKED':
            log_and_update_single_run("Task revoked at start.", 0, task_state='REVOKED')
            return {"status": "REVOKED", "message": f"Single clustering run {run_id} revoked."}
        
        rows = [type('DictRow', (), item)() for item in all_tracks_data] 
        X_original = [score_vector(row, MOOD_LABELS) for row in rows]
        X_scaled = np.array(X_original)
        data_for_clustering = X_scaled
        pca_model = None
        n_components_actual = 0 # Initialize

        if pca_config["enabled"]:
            if task_result_checker and task_result_checker.state == 'REVOKED':
                log_and_update_single_run("Task revoked before PCA.", 20, task_state='REVOKED')
                return {"status": "REVOKED", "message": f"Single clustering run {run_id} revoked."}
            
            n_components_actual = min(pca_config["components"], X_scaled.shape[1], len(rows) -1 if len(rows) >1 else 1)
            if n_components_actual > 0:
                pca_model = PCA(n_components=n_components_actual)
                data_for_clustering = pca_model.fit_transform(X_scaled)
            else:
                pca_config["enabled"] = False 
        log_and_update_single_run(f"PCA {'enabled with ' + str(n_components_actual) + ' components' if pca_config['enabled'] else 'disabled'}.", 25)

        labels = None
        cluster_centers_map = {} 
        raw_distances = np.zeros(len(data_for_clustering))

        method = clustering_method_config["method"]
        params = clustering_method_config["params"]

        if task_result_checker and task_result_checker.state == 'REVOKED':
            log_and_update_single_run("Task revoked before clustering method execution.", 30, task_state='REVOKED')
            return {"status": "REVOKED", "message": f"Single clustering run {run_id} revoked."}
        
        if method == "kmeans":
            kmeans = KMeans(n_clusters=params["n_clusters"], random_state=None, n_init='auto')
            labels = kmeans.fit_predict(data_for_clustering)
            cluster_centers_map = {i: kmeans.cluster_centers_[i] for i in range(params["n_clusters"])}
            centers_for_points = kmeans.cluster_centers_[labels]
            raw_distances = np.linalg.norm(data_for_clustering - centers_for_points, axis=1)
        elif method == "dbscan":
            dbscan = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
            labels = dbscan.fit_predict(data_for_clustering)
            for cluster_id in set(labels):
                if cluster_id == -1: continue
                indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                cluster_points = np.array([data_for_clustering[i] for i in indices])
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    for i in indices: raw_distances[i] = np.linalg.norm(data_for_clustering[i] - center)
                    cluster_centers_map[cluster_id] = center
        elif method == "gmm":
            gmm = GaussianMixture(n_components=params["n_components"], covariance_type=GMM_COVARIANCE_TYPE, random_state=None, max_iter=1000)
            gmm.fit(data_for_clustering)
            labels = gmm.predict(data_for_clustering)
            log_and_update_single_run(f"GMM fitting complete. Found {len(set(labels))} clusters (incl. noise if applicable).", 45)
            cluster_centers_map = {i: gmm.means_[i] for i in range(params["n_components"])}
            centers_for_points = gmm.means_[labels]
            raw_distances = np.linalg.norm(data_for_clustering - centers_for_points, axis=1)
        log_and_update_single_run("Clustering algorithm applied.", 50)

        if labels is None or len(set(labels) - {-1}) == 0:
            log_and_update_single_run("No valid clusters found.", 100, {"diversity_score": -1}, task_state='SUCCESS')
            return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_components": None, "parameters": clustering_method_config}

        max_dist = raw_distances.max()
        normalized_distances = raw_distances / max_dist if max_dist > 0 else raw_distances

        track_info_list = [{"row": rows[i], "label": labels[i], "distance": normalized_distances[i]} for i in range(len(rows))]

        filtered_clusters = defaultdict(list)
        for cluster_id_val in set(labels):
            if cluster_id_val == -1: continue
            cluster_tracks_list = [t for t in track_info_list if t["label"] == cluster_id_val and t["distance"] <= MAX_DISTANCE]
            if not cluster_tracks_list: continue
            cluster_tracks_list.sort(key=lambda x: x["distance"])
            
            count_per_artist = defaultdict(int)
            selected_tracks = []
            for t_item in cluster_tracks_list:
                author = t_item["row"].author 
                if count_per_artist[author] < MAX_SONGS_PER_ARTIST:
                    selected_tracks.append(t_item)
                    count_per_artist[author] += 1
                if len(selected_tracks) >= max_songs_per_cluster: break
            for t_item in selected_tracks:
                item_id, title, author = t_item["row"].item_id, t_item["row"].title, t_item["row"].author
                filtered_clusters[cluster_id_val].append((item_id, title, author))
        log_and_update_single_run("Tracks filtered into clusters.", 65)

        current_named_playlists = defaultdict(list)
        current_playlist_centroids = {}
        unique_predominant_mood_scores = {}

        for label_val, songs_list in filtered_clusters.items():
            if songs_list:
                center_val = cluster_centers_map.get(label_val)
                if center_val is None: continue
                name, top_scores = name_cluster(center_val, pca_model, pca_config["enabled"], MOOD_LABELS)
                if top_scores and any(mood in MOOD_LABELS for mood in top_scores.keys()):
                    predominant_mood_key = max(top_scores, key=lambda k: top_scores[k] if k in MOOD_LABELS else -1)
                    if predominant_mood_key in MOOD_LABELS:
                        current_mood_score = top_scores.get(predominant_mood_key, 0.0)
                        unique_predominant_mood_scores[predominant_mood_key] = max(unique_predominant_mood_scores.get(predominant_mood_key, 0.0), current_mood_score)
                current_named_playlists[name].extend(songs_list)
                current_playlist_centroids[name] = top_scores

        diversity_score = sum(unique_predominant_mood_scores.values())
        log_and_update_single_run(f"Named {len(current_named_playlists)} playlists. Diversity score: {diversity_score:.2f}", 75)
        
        pca_model_details = {"n_components": pca_model.n_components_, "mean": pca_model.mean_.tolist()} if pca_model else None

        result = {
            "diversity_score": float(diversity_score),
            "named_playlists": dict(current_named_playlists), 
            "playlist_centroids": current_playlist_centroids,
            "pca_model_details": pca_model_details, 
            "parameters": {"clustering_method_config": clustering_method_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster}
        }
        log_and_update_single_run("Iteration complete.", 100, {"diversity_score": diversity_score, "final_result": result}, task_state='SUCCESS')
        return result

@celery.task(bind=True)
def run_clustering_task(self, clustering_method, num_clusters_min, num_clusters_max, dbscan_eps_min, dbscan_eps_max, dbscan_min_samples_min, dbscan_min_samples_max, pca_components_min, pca_components_max, num_clustering_runs, max_songs_per_cluster, gmm_n_components_min, gmm_n_components_max):
    """
    Celery task to run the clustering and playlist generation process with an
    evolutionary approach for parameter selection.
    Updates task state with progress and log messages.
    Includes weighted diversity score calculation.
    """
    with app.app_context(): # Ensure Flask application context
        current_task_id = self.request.id
        task_result_checker = AsyncResult(current_task_id, app=celery) if current_task_id else None
        save_task_status(current_task_id, "main_clustering", "STARTED", progress=0, details={"message": "Initializing clustering..."})

        log_messages = []
        current_progress = 0 # Initialize current_progress for the scope

        def log_and_update_main_clustering(message, progress, details_extra=None, task_state='PROGRESS'):
            nonlocal current_progress
            current_progress = progress
            log_messages.append(message)
            print(f"[MainClusteringTask-{current_task_id}] {message}") # Celery container log
            current_details = {"log": log_messages, "overall_status": message}
            if details_extra:
                current_details.update(details_extra)
            
            if task_state != 'PROGRESS':
                 self.update_state(state=task_state, meta={'progress': progress, 'status': message, 'details': current_details})
            else:
                 self.update_state(state='PROGRESS', meta={'progress': progress, 'status': message, 'details': current_details})
            save_task_status(current_task_id, "main_clustering", task_state, progress=progress, details=current_details)

        try:
            log_and_update_main_clustering("📊 Starting main clustering process with evolutionary parameter search...", 0)
            rows = get_all_tracks(None) # db_path not needed
            if len(rows) < 2:
                log_and_update_main_clustering("Not enough analyzed tracks for clustering.", 100, {"error": "Insufficient data"}, task_state='FAILURE')
                return {"status": "FAILURE", "message": "Not enough analyzed tracks for clustering."}
            
            if task_result_checker and task_result_checker.state == 'REVOKED':
                log_and_update_main_clustering("Main clustering task revoked at start.", 0, task_state='REVOKED')
                return {"status": "REVOKED", "message": "Main clustering task revoked."}
            log_and_update_main_clustering(f"Fetched {len(rows)} tracks for clustering.", 5)

            serializable_rows = [dict(row) for row in rows]
            all_tracks_data_json = json.dumps(serializable_rows)

            best_diversity_score = -1.0 
            best_clustering_results = None
            clustering_run_tasks = []
            clustering_run_task_ids = []

            for run_idx in range(num_clustering_runs):
                # Determine progress for logging before the check
                progress_before_launch = 5 + int(5 * (run_idx / float(num_clustering_runs))) if num_clustering_runs > 0 else 5
                if task_result_checker and task_result_checker.state == 'REVOKED':
                    log_and_update_main_clustering("Main clustering task revoked before launching all sub-runs.", progress_before_launch, task_state='REVOKED')
                    return {"status": "REVOKED", "message": "Main clustering task revoked."}
                
                current_num_clusters = 0
                current_dbscan_eps = 0.0
                current_dbscan_min_samples = 0
                current_gmm_n_components = 0 
                
                method_params = {}
                if clustering_method == "kmeans":
                    current_num_clusters = random.randint(num_clusters_min, num_clusters_max)
                    current_num_clusters = min(current_num_clusters, len(rows))
                    current_num_clusters = max(1, current_num_clusters) 
                    method_params = {"method": "kmeans", "params": {"n_clusters": current_num_clusters}}
                elif clustering_method == "dbscan":
                    current_dbscan_eps = round(random.uniform(dbscan_eps_min, dbscan_eps_max), 2)
                    current_dbscan_min_samples = random.randint(dbscan_min_samples_min, dbscan_min_samples_max)
                    method_params = {"method": "dbscan", "params": {"eps": current_dbscan_eps, "min_samples": current_dbscan_min_samples}}
                elif clustering_method == "gmm": 
                    current_gmm_n_components = random.randint(gmm_n_components_min, gmm_n_components_max)
                    current_gmm_n_components = min(current_gmm_n_components, len(rows))
                    current_gmm_n_components = max(1, current_gmm_n_components) 
                    method_params = {"method": "gmm", "params": {"n_components": current_gmm_n_components}}
                else:
                    log_and_update_main_clustering(f"Unsupported clustering algorithm: {clustering_method}", 100, {"error": "Unsupported algorithm"}, task_state='FAILURE')
                    return {"status": "FAILURE", "message": f"Unsupported clustering algorithm: {clustering_method}"}
                
                sampled_pca_components = random.randint(pca_components_min, pca_components_max)
                pca_config = {"enabled": sampled_pca_components > 0, "components": sampled_pca_components}

                run_task = run_single_clustering_iteration_task.s(
                    run_idx, all_tracks_data_json, method_params, pca_config, max_songs_per_cluster, current_task_id
                ).apply_async()
                clustering_run_tasks.append(run_task)
                clustering_run_task_ids.append(run_task.id)

            log_and_update_main_clustering(f"Launched {num_clustering_runs} clustering iteration tasks.", 10, {"clustering_run_task_ids": clustering_run_task_ids})

            while not all(t.ready() for t in clustering_run_tasks):
                completed_count = sum(1 for t in clustering_run_tasks if t.ready())
                progress_while_waiting = 10 + int(80 * (completed_count / float(num_clustering_runs))) if num_clustering_runs > 0 else 10
                if task_result_checker and task_result_checker.state == 'REVOKED':
                    log_and_update_main_clustering("Main clustering task revoked while waiting for sub-runs.", progress_while_waiting, task_state='REVOKED')
                    return {"status": "REVOKED", "message": "Main clustering task revoked."}
                
                log_and_update_main_clustering(
                    f"Processing clustering runs: {completed_count}/{num_clustering_runs} completed.",
                    progress_while_waiting,
                    {"runs_completed": completed_count, "total_runs": num_clustering_runs, "clustering_run_task_ids": clustering_run_task_ids}
                )
                time.sleep(2) 

            for t_res in clustering_run_tasks:
                if t_res.successful() and isinstance(t_res.result, dict):
                    run_result = t_res.result
                    current_diversity_score = run_result.get("diversity_score", -1.0)
                    if current_diversity_score > best_diversity_score:
                        best_diversity_score = current_diversity_score
                        best_clustering_results = run_result
                        log_and_update_main_clustering(f"New best clustering iteration found (Run ID from subtask: {run_result.get('parameters', {}).get('run_id', 'N/A')}, Diversity: {current_diversity_score:.2f})", 85)
                else:
                    log_and_update_main_clustering(f"A clustering run task ({t_res.id}) failed or returned unexpected result.", 85) 

            if not best_clustering_results:
                log_and_update_main_clustering("No valid clustering solution found after all runs.", 100, {"error": "No suitable clustering found"}, task_state='FAILURE')
                return {"status": "FAILURE", "message": "No valid clusters found after multiple runs."}

            log_and_update_main_clustering(f"Best clustering found with diversity score: {best_diversity_score:.2f}.", 90, {"best_score": best_diversity_score, "best_params": best_clustering_results.get("parameters")})
            
            final_named_playlists = best_clustering_results["named_playlists"]
            final_playlist_centroids = best_clustering_results["playlist_centroids"]
            final_max_songs_per_cluster = best_clustering_results["parameters"]["max_songs_per_cluster"]

            log_and_update_main_clustering("Updating playlist database...", 95)
            update_playlist_table(None, final_named_playlists) 
            
            log_and_update_main_clustering("Creating/Updating playlists on Jellyfin...", 98)
            create_or_update_playlists_on_jellyfin(JELLYFIN_URL, JELLYFIN_USER_ID, {"X-Emby-Token": JELLYFIN_TOKEN}, final_named_playlists, final_playlist_centroids, MOOD_LABELS, final_max_songs_per_cluster)
            
            log_and_update_main_clustering(f"Playlists generated and updated on Jellyfin! Best diversity score: {best_diversity_score:.2f}.", 100, task_state='SUCCESS')
            return {"status": "SUCCESS", "message": f"Playlists generated and updated on Jellyfin! Best run had weighted diversity score of {best_diversity_score:.2f}."}

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"FATAL ERROR: Clustering failed: {e}\n{error_traceback}")
            log_and_update_main_clustering(f"❌ Main clustering failed: {e}", current_progress, {"error": str(e)}, task_state='FAILURE')
            raise 


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
    num_recent_albums = int(data.get('num_recent_albums', NUM_RECENT_ALBUMS))
    top_n_moods = int(data.get('top_n_moods', TOP_N_MOODS))
    task = run_analysis_task.delay(jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods)
    save_task_status(task.id, "main_analysis", "PENDING") # Changed task_type
    return jsonify({"task_id": task.id, "task_type": "main_analysis", "status": "PENDING"}), 202

@app.route('/api/clustering/start', methods=['POST'])
def start_clustering():
    """
    Starts the playlist clustering as an asynchronous Celery task.
    Accepts parameter ranges for evolutionary search.
    Records the task ID and type in the database.
    """
    data = request.json

    # Get clustering method
    clustering_method = data.get('clustering_method', CLUSTER_ALGORITHM)

    # Get ranges for K-Means parameters
    num_clusters_min = int(data.get('num_clusters_min', NUM_CLUSTERS_MIN))
    num_clusters_max = int(data.get('num_clusters_max', NUM_CLUSTERS_MAX))

    # Get ranges for DBSCAN parameters
    dbscan_eps_min = float(data.get('dbscan_eps_min', DBSCAN_EPS_MIN))
    dbscan_eps_max = float(data.get('dbscan_eps_max', DBSCAN_EPS_MAX))
    dbscan_min_samples_min = int(data.get('dbscan_min_samples_min', DBSCAN_MIN_SAMPLES_MIN))
    dbscan_min_samples_max = int(data.get('dbscan_min_samples_max', DBSCAN_MIN_SAMPLES_MAX))

    # Get ranges for GMM parameters
    gmm_n_components_min = int(data.get('gmm_n_components_min', GMM_N_COMPONENTS_MIN))
    gmm_n_components_max = int(data.get('gmm_n_components_max', GMM_N_COMPONENTS_MAX))

    # Get ranges for PCA components
    pca_components_min = int(data.get('pca_components_min', PCA_COMPONENTS_MIN))
    pca_components_max = int(data.get('pca_components_max', PCA_COMPONENTS_MAX))
    
    # Get total clustering runs
    num_clustering_runs = int(data.get('clustering_runs', CLUSTERING_RUNS))

    # Get max_songs_per_cluster
    max_songs_per_cluster = int(data.get('max_songs_per_cluster', MAX_SONGS_PER_CLUSTER))


    task = run_clustering_task.delay(
        clustering_method, 
        num_clusters_min, num_clusters_max,
        dbscan_eps_min, dbscan_eps_max,
        dbscan_min_samples_min, dbscan_min_samples_max,
        pca_components_min, pca_components_max,
        num_clustering_runs,
        max_songs_per_cluster,
        gmm_n_components_min, gmm_n_components_max # Pass new GMM parameters
    )
    save_task_status(task.id, "main_clustering", "PENDING") # Changed task_type
    return jsonify({"task_id": task.id, "task_type": "main_clustering", "status": "PENDING"}), 202


@app.route('/api/status/<task_id>', methods=['GET'])
def get_task_status_endpoint(task_id):
    """
    Retrieves the status of any task (analysis or clustering).
    """
    task = AsyncResult(task_id, app=celery)
    db_task_info = get_task_info_from_db(task_id) # Get info from our DB

    response = {
        'task_id': task.id,
        'state': task.state,
        'status_message': 'Processing...', # Renamed from 'status' to avoid conflict
        'progress': 0,
        'details': {},
        'task_type_from_db': None
    }

    if db_task_info:
        response['progress'] = db_task_info.get('progress', 0)
        response['details'] = json.loads(db_task_info.get('details')) if db_task_info.get('details') else {}
        response['task_type_from_db'] = db_task_info.get('task_type')
        response['status_message'] = db_task_info.get('status') # Use DB status as primary message

    # Celery state overrides some parts if it's more definitive (e.g. SUCCESS/FAILURE)
    celery_meta = task.info if isinstance(task.info, dict) else {}

    if task.state == 'PENDING':
        response['status_message'] = response.get('status_message', 'Task is pending or not yet started.')
        if not response['details']: # if DB had no details yet
             response['details'] = {'log_output': ['Task pending...']}
    elif task.state == 'PROGRESS':
        # Celery's meta during PROGRESS is what the task itself updates
        if celery_meta:
            response['progress'] = celery_meta.get('progress', response['progress'])
            response['status_message'] = celery_meta.get('status', response['status_message'])
            # Merge details, giving preference to Celery's live update
            response['details'] = {**response['details'], **celery_meta.get('details', {})}
    elif task.state == 'SUCCESS':
        response['status_message'] = celery_meta.get('message', 'Task complete!') if celery_meta else 'Task complete!'
        response['progress'] = 100
        if celery_meta: response['details'] = {**response['details'], **celery_meta}
        # DB status already updated by postrun signal
    elif task.state == 'FAILURE':
        response['status_message'] = str(task.info) # Celery's .info is the exception for FAILURE
        response['progress'] = 100
        if celery_meta: response['details'] = {**response['details'], **celery_meta}
        if 'log_output' not in response['details']: response['details']['log_output'] = []
        response['details']['log_output'].append(f"Error: {str(task.info)}")
        # DB status already updated by postrun signal
    elif task.state == 'REVOKED':
        response['status_message'] = 'Task revoked.'
        response['progress'] = 100
        response['details']['log_output'] = ['Task was cancelled.']
        # DB status already updated by postrun signal
    else:
        response['status_message'] = response.get('status_message', f'Unknown state: {task.state}')
        if celery_meta: response['details'] = {**response['details'], **celery_meta}

    return jsonify(response)

def revoke_task_and_children(task_id, task_type_hint=None):
    """Helper to revoke a task and its potential children based on DB records."""
    revoked_count = 0
    task = AsyncResult(task_id, app=celery)
    if task.state in ['PENDING', 'STARTED', 'PROGRESS']:
        # Changed to not use terminate=True as thread pool doesn't support it
        task.revoke() 
        revoked_count += 1
        db_task_info = get_task_info_from_db(task_id)
        task_type = db_task_info.get('task_type') if db_task_info else task_type_hint
        if task_type:
            save_task_status(task_id, task_type, "REVOKED", progress=100, details={"message": "Task cancelled by API."})

    # Attempt to revoke children if it's a main task
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT task_id FROM task_status WHERE parent_task_id = %s AND status NOT IN ('SUCCESS', 'FAILURE', 'REVOKED')", (task_id,))
    children_tasks = cur.fetchall()
    cur.close()

    for child_task_row in children_tasks:
        child_task_id = child_task_row['task_id']
        child_celery_task = AsyncResult(child_task_id, app=celery)
        if child_celery_task.state in ['PENDING', 'STARTED', 'PROGRESS']:
            # Changed to not use terminate=True
            child_celery_task.revoke()
            revoked_count +=1
            child_db_info = get_task_info_from_db(child_task_id)
            child_task_type = child_db_info.get('task_type') if child_db_info else 'unknown_child'
            save_task_status(child_task_id, child_task_type, "REVOKED", parent_task_id=task_id, progress=100, details={"message": "Child task cancelled as part of parent cancellation."})
    return revoked_count

@app.route('/api/cancel/<task_id>', methods=['POST'])
def cancel_task_endpoint(task_id):
    """Cancels a specific task and its children if applicable."""
    revoked_count = revoke_task_and_children(task_id)
    if revoked_count > 0:
        return jsonify({"message": f"Task {task_id} and {revoked_count-1} children (if any) cancellation initiated.", "task_id": task_id, "revoked_count": revoked_count}), 200
    return jsonify({"message": "Task cannot be cancelled or was not found in an active state.", "task_id": task_id}), 400

@app.route('/api/cancel_all/<task_type_prefix>', methods=['POST'])
def cancel_all_tasks_by_type(task_type_prefix):
    """Cancels all active tasks of a given type (e.g., 'main_analysis', 'main_clustering')."""
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    # Find main tasks of the specified type that are still active
    cur.execute("SELECT task_id, task_type FROM task_status WHERE task_type = %s AND status NOT IN ('SUCCESS', 'FAILURE', 'REVOKED')", (task_type_prefix,))
    tasks_to_cancel = cur.fetchall()
    cur.close()

    total_revoked_count = 0
    cancelled_ids = []
    for task_row in tasks_to_cancel:
        total_revoked_count += revoke_task_and_children(task_row['task_id'], task_row['task_type'])
        cancelled_ids.append(task_row['task_id'])

    if total_revoked_count > 0:
        return jsonify({"message": f"Cancellation initiated for {len(cancelled_ids)} main tasks of type '{task_type_prefix}' and their children. Total tasks/subtasks affected: {total_revoked_count}.", "cancelled_main_tasks": cancelled_ids}), 200
    return jsonify({"message": f"No active tasks of type '{task_type_prefix}' found to cancel."}), 404

@app.route('/api/last_task', methods=['GET'])
def get_last_overall_task_status():
    """
    Retrieves the status of the last recorded task, regardless of type.
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    # Get the most recent main task (not sub-tasks)
    cur.execute("SELECT task_id, task_type, status, progress, details FROM task_status WHERE parent_task_id IS NULL ORDER BY timestamp DESC LIMIT 1")
    last_task_row = cur.fetchone()
    cur.close()

    if last_task_row:
        last_task = dict(last_task_row)
        if last_task.get('details'):
            last_task['details'] = json.loads(last_task['details'])
        return jsonify(last_task), 200
    return jsonify({"task_id": None, "task_type": None, "status": "NO_PREVIOUS_MAIN_TASK"}), 200

@app.route('/api/active_tasks', methods=['GET'])
def get_active_tasks():
    """Retrieves all tasks not in a final state (SUCCESS, FAILURE, REVOKED)."""
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    cur.execute("""
        SELECT task_id, parent_task_id, task_type, sub_type_identifier, status, progress, details, timestamp
        FROM task_status
        WHERE status NOT IN ('SUCCESS', 'FAILURE', 'REVOKED')
        ORDER BY timestamp DESC
    """)
    active_tasks_rows = cur.fetchall()
    cur.close()
    active_tasks_list = [] # Renamed to avoid conflict with last_task variable name
    for row in active_tasks_rows:
        task_item = dict(row)
        if task_item.get('details'):
            try:
                task_item['details'] = json.loads(task_item['details'])
            except json.JSONDecodeError:
                task_item['details'] = {"raw_details": task_item['details']} # Keep raw if not JSON
        active_tasks_list.append(task_item)
    if active_tasks_list: # Check if the list is not empty
        return jsonify(active_tasks_list), 200 # Return the list of active tasks
    return jsonify([]), 200 # Return an empty list if no active tasks

@app.route('/api/config', methods=['GET'])
def get_config():
    """Returns the current configuration parameters, including new ranges."""
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
        "gmm_n_components_min": GMM_N_COMPONENTS_MIN, # Added GMM min components
        "gmm_n_components_max": GMM_N_COMPONENTS_MAX, # Added GMM max components
        "pca_components_min": PCA_COMPONENTS_MIN,
        "pca_components_max": PCA_COMPONENTS_MAX,
        "top_n_moods": TOP_N_MOODS,
        "mood_labels": MOOD_LABELS,
        "clustering_runs": CLUSTERING_RUNS,
    })

@app.route('/api/playlists', methods=['GET'])
def get_playlists():
    """Retrieves all saved playlists from the database."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT playlist_name, item_id, title, author FROM playlist ORDER BY playlist_name, title")
    rows = cur.fetchall()
    cur.close()

    playlists = defaultdict(list)
    for row in rows:
        playlists[row['playlist_name']].append({"item_id": row['item_id'], "title": row['title'], "author": row['author']})
    return jsonify(dict(playlists)), 200

if __name__ == '__main__':
    # init_db() is called within app_context now, so not explicitly here.
    # However, for standalone script execution, you might want to ensure the DB is ready.
    # with app.app_context(): init_db() # If running this script directly and need DB init
    os.makedirs(TEMP_DIR, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8000)
