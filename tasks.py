import os
import shutil
import requests
from collections import defaultdict
import numpy as np
import json
import time
import random
import uuid
import traceback

# Essentia and ML imports
from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor, TensorflowPredictMusiCNN, TensorflowPredict2D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# RQ import
from rq import get_current_job

# Import necessary components from the main app.py file
# This assumes app.py will define 'app', 'redis_conn', and the DB utility functions
from app import (app, redis_conn, get_db, save_task_status, get_task_info_from_db,
                track_exists, save_track_analysis, get_all_tracks, update_playlist_table, JobStatus,
                TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
                TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)

# Import configuration (ensure config.py is in PYTHONPATH or same directory)
from config import TEMP_DIR, MAX_DISTANCE, MAX_SONGS_PER_CLUSTER, MAX_SONGS_PER_ARTIST, \
    GMM_COVARIANCE_TYPE, MOOD_LABELS, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, \
    JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, USE_AI_PLAYLIST_NAMING, \
    OLLAMA_SERVER_URL, OLLAMA_MODEL_NAME, \
    SCORE_WEIGHT_DIVERSITY, \
    SCORE_WEIGHT_PURITY, \
    MUTATION_KMEANS_COORD_FRACTION # For create_or_update_playlists_on_jellyfin
from rq.job import Job # Import Job class
from rq.exceptions import NoSuchJobError, InvalidJobOperation

# --- Task-specific Helper Functions ---

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

def predict_moods(file_path, embedding_model_path, prediction_model_path, mood_labels_list, top_n_moods):
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
    return {label: float(score) for label, score in sorted(results.items(), key=lambda x: -x[1])[:top_n_moods]}

def analyze_track(file_path, embedding_model_path, prediction_model_path, mood_labels_list, top_n_moods):
    """Analyzes a single track for tempo, key, scale, and moods."""
    audio = MonoLoader(filename=file_path)()
    tempo, _, _, _, _ = RhythmExtractor2013()(audio)
    key, scale, _ = KeyExtractor()(audio)
    moods = predict_moods(file_path, embedding_model_path, prediction_model_path, mood_labels_list, top_n_moods)
    return tempo, key, scale, moods

def score_vector(row, mood_labels_list):
    """Converts a database row into a numerical feature vector for clustering."""
    tempo = float(row['tempo']) if row['tempo'] is not None else 0.0
    mood_str = row['mood_vector'] or ""
    tempo_norm = (tempo - 40) / (200 - 40)
    tempo_norm = np.clip(tempo_norm, 0.0, 1.0)
    mood_scores = np.zeros(len(mood_labels_list))
    if mood_str:
        for pair in mood_str.split(","):
            if ":" not in pair:
                continue
            label, score_str = pair.split(":")
            if label in mood_labels_list:
                try:
                    mood_scores[mood_labels_list.index(label)] = float(score_str)
                except ValueError:
                    continue
    full_vector = [tempo_norm] + list(mood_scores)
    return full_vector

def name_cluster(centroid_scaled_vector, pca_model, pca_enabled, mood_labels_list):
    """Generates a human-readable name for a cluster."""
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
    if tempo < 80: tempo_label = "Slow"
    elif tempo < 130: tempo_label = "Medium"
    else: tempo_label = "Fast"
    
    if len(mood_values) == 0 or np.sum(mood_values) == 0: top_indices = []
    else: top_indices = np.argsort(mood_values)[::-1][:3]

    mood_names = [mood_labels_list[i] for i in top_indices if i < len(mood_labels_list)]
    mood_part = "_".join(mood_names).title() if mood_names else "Mixed"
    full_name = f"{mood_part}_{tempo_label}"
    
    top_mood_scores = {mood_labels_list[i]: mood_values[i] for i in top_indices if i < len(mood_labels_list)}
    extra_info = {"tempo": round(tempo_norm, 2)}
    
    return full_name, {**top_mood_scores, **extra_info}

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
                if del_resp.ok: print(f"🗑️ Deleted old playlist: {item['Name']}")
    except Exception as e:
        print(f"Failed to clean old playlists: {e}")

def create_or_update_playlists_on_jellyfin(jellyfin_url_param, jellyfin_user_id_param, headers_param, playlists, cluster_centers, mood_labels_list, max_songs_per_cluster_param):
    """Creates or updates playlists on Jellyfin based on clustering results."""
    delete_old_automatic_playlists(jellyfin_url_param, jellyfin_user_id_param, headers_param)
    for base_name, cluster in playlists.items():
        chunks = [cluster[i:i+max_songs_per_cluster_param] for i in range(0, len(cluster), max_songs_per_cluster_param)]
        for idx, chunk in enumerate(chunks, 1):
            playlist_name = f"{base_name}_automatic_{idx}" if len(chunks) > 1 else f"{base_name}_automatic"
            item_ids = [item_id for item_id, _, _ in chunk]
            if not item_ids: continue
            body = {"Name": playlist_name, "Ids": item_ids, "UserId": jellyfin_user_id_param}
            try:
                r = requests.post(f"{jellyfin_url_param}/Playlists", headers=headers_param, json=body, timeout=30)
                if r.ok:
                    centroid_info = cluster_centers.get(base_name, {})
                    top_moods = {k: v for k, v in centroid_info.items() if k in mood_labels_list}
                    extra_info = {k: v for k, v in centroid_info.items() if k not in mood_labels_list}
                    centroid_str = ", ".join(f"{k}:{v:.2f}" for k, v in top_moods.items())
                    extras_str = ", ".join(f"{k}:{v:.2f}" for k, v in extra_info.items())
                    print(f"✅ Created playlist {playlist_name} with {len(item_ids)} tracks (Centroid: {centroid_str} | {extras_str})")
            except Exception as e:
                print(f"Exception creating {playlist_name}: {e}")

# --- RQ Task Definitions ---

def analyze_album_task(album_id, album_name, jellyfin_url, jellyfin_user_id, jellyfin_token, top_n_moods, parent_task_id):
    """RQ task to analyze a single album."""
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        initial_details = {"album_name": album_name, "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Album analysis task started."]}
        save_task_status(current_task_id, "album_analysis", TASK_STATUS_STARTED, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=0, details=initial_details)
        headers = {"X-Emby-Token": jellyfin_token}
        tracks_analyzed_count = 0
        current_progress_val = 0
        # Accumulate logs here for the current task
        current_task_logs = initial_details["log"]

        def log_and_update_album_task(message, progress, current_track_name=None, task_state=TASK_STATUS_PROGRESS, current_track_analysis_details=None, final_summary_details=None):
            nonlocal current_progress_val
            nonlocal current_task_logs
            current_progress_val = progress
            print(f"[AlbumTask-{current_task_id}-{album_name}] {message}")

            # Base details for DB
            db_details = {"album_name": album_name}
            if current_track_name: db_details["current_track"] = current_track_name
            if current_track_analysis_details: db_details["current_track_analysis"] = current_track_analysis_details
            if final_summary_details: # Merge final summary if provided
                db_details.update(final_summary_details)

            # Manage logs
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            if task_state == TASK_STATUS_SUCCESS:
                db_details["log"] = [f"Task completed successfully. Final status: {message}"]
                # Ensure other summary fields from final_summary_details are in db_details
            elif task_state == TASK_STATUS_FAILURE or task_state == TASK_STATUS_REVOKED:
                current_task_logs.append(log_entry) # Add final error/revoked message
                db_details["log"] = current_task_logs # Keep all logs for failure/revocation
                if "error" not in db_details and task_state == TASK_STATUS_FAILURE : db_details["error"] = message # Ensure error message is captured
            else: # PROGRESS or STARTED
                current_task_logs.append(log_entry)
                db_details["log"] = current_task_logs

            # Details for RQ job.meta (leaner)
            meta_details = {
                "album_name": album_name,
                "current_track": current_track_name,
                "status_message": message # Main status message for meta
            }
            if current_track_analysis_details: meta_details["current_track_analysis_short"] = {k:v for k,v in current_track_analysis_details.items() if k in ['tempo', 'key']}

            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.meta['details'] = meta_details
                current_job.save_meta()
            
            save_task_status(current_task_id, "album_analysis", task_state, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=progress, details=db_details)

        try:
            log_and_update_album_task(f"Fetching tracks for album: {album_name}", 5)
            tracks = get_tracks_from_album(jellyfin_url, jellyfin_user_id, headers, album_id)
            if not tracks:
                log_and_update_album_task(f"No tracks found for album: {album_name}", 100, task_state='SUCCESS')
                return {"status": "SUCCESS", "message": f"No tracks in album {album_name}", "tracks_analyzed": 0}
            
            total_tracks_in_album = len(tracks)
            for idx, item in enumerate(tracks, 1):
                # === Cooperative Cancellation Check for Album Task ===
                if current_job: # Check if running as an RQ job
                    with app.app_context(): # Ensure DB access for status check
                        album_task_db_info = get_task_info_from_db(current_task_id)
                        parent_task_db_info = get_task_info_from_db(parent_task_id) if parent_task_id else None
                        
                        is_self_revoked = album_task_db_info and album_task_db_info.get('status') == 'REVOKED'
                        # Check if parent is in a terminal failure or revoked state
                        is_parent_failed_or_revoked = parent_task_db_info and parent_task_db_info.get('status') in [TASK_STATUS_REVOKED, TASK_STATUS_FAILURE]

                        if is_self_revoked or is_parent_failed_or_revoked:
                            parent_status_for_reason = parent_task_db_info.get('status') if parent_task_db_info else "N/A"
                            revocation_reason = "self was REVOKED" if is_self_revoked else f"parent task {parent_task_id} status is {parent_status_for_reason}"
                            # Ensure path is cleaned up if it exists at this point
                            temp_file_to_clean = locals().get('path') # Get 'path' if defined in this scope
                            if temp_file_to_clean and os.path.exists(temp_file_to_clean):
                                try: os.remove(temp_file_to_clean)
                                except Exception as e_cleanup: print(f"Warning: Failed to clean up {temp_file_to_clean} during revocation: {e_cleanup}")
                            log_and_update_album_task(f"🛑 Album analysis task {current_task_id} for '{album_name}' stopping because {revocation_reason}.", current_progress_val, task_state=TASK_STATUS_REVOKED)
                            return {"status": "REVOKED", "message": f"Album analysis for '{album_name}' stopped because {revocation_reason}."}
                # === End Cooperative Cancellation Check ===

                track_name_full = f"{item['Name']} by {item.get('AlbumArtist', 'Unknown')}"
                current_progress_val = 10 + int(85 * (idx / float(total_tracks_in_album))) if total_tracks_in_album > 0 else 10
                # Clear previous track's analysis details when starting a new track
                log_and_update_album_task(
                    f"Analyzing track: {track_name_full} ({idx}/{total_tracks_in_album})",
                    current_progress_val,
                    current_track_name=track_name_full,
                    current_track_analysis_details=None)

                if track_exists(item['Id']):
                    log_and_update_album_task(f"Skipping '{track_name_full}' (already analyzed)", current_progress_val, current_track_name=track_name_full)
                    tracks_analyzed_count +=1
                    continue

                path = download_track(jellyfin_url, headers, TEMP_DIR, item)
                log_and_update_album_task(f"Download attempt for '{track_name_full}': {'Success' if path else 'Failed'}", current_progress_val, current_track_name=track_name_full)
                if not path:
                    log_and_update_album_task(f"Failed to download '{track_name_full}'. Skipping.", current_progress_val, current_track_name=track_name_full)
                    continue

                try:
                    tempo, key, scale, moods = analyze_track(path, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, MOOD_LABELS, top_n_moods)
                    current_track_details_for_api = {
                        "name": track_name_full,
                        "tempo": round(tempo, 2),
                        "key": key,
                        "scale": scale,
                        "moods": {k: round(v, 2) for k, v in moods.items()}
                    }
                    save_track_analysis(item['Id'], item['Name'], item.get('AlbumArtist', 'Unknown'), tempo, key, scale, moods)
                    tracks_analyzed_count += 1
                    mood_details_str = ', '.join(f'{k}:{v:.2f}' for k,v in moods.items())
                    analysis_summary_msg = f"Tempo: {tempo:.2f}, Key: {key} {scale}."
                    if mood_details_str:
                        analysis_summary_msg += f" Moods: {mood_details_str}"
                    else:
                        analysis_summary_msg += " Moods: (No moods reported or TOP_N_MOODS is low/zero)"
                    log_and_update_album_task(
                        f"Analyzed '{track_name_full}'. {analysis_summary_msg}",
                        current_progress_val,
                        current_track_name=track_name_full,
                        current_track_analysis_details=current_track_details_for_api)
                except Exception as e_analyze:
                    log_and_update_album_task(f"Error analyzing '{track_name_full}': {e_analyze}", current_progress_val, current_track_name=track_name_full)
                finally:
                    if path and os.path.exists(path):
                        try: os.remove(path)
                        except Exception as cleanup_e: print(f"WARNING: Failed to clean up temp file {path}: {cleanup_e}")
            
            success_summary = {
                "tracks_analyzed": tracks_analyzed_count,
                "total_tracks_in_album": total_tracks_in_album,
                "message": f"Album '{album_name}' analysis complete."
            }
            log_and_update_album_task(f"Album '{album_name}' analysis complete. Analyzed {tracks_analyzed_count}/{total_tracks_in_album} tracks.", 
                                      100, task_state=TASK_STATUS_SUCCESS, final_summary_details=success_summary)
            return {"status": "SUCCESS", "message": f"Album '{album_name}' analysis complete.", "tracks_analyzed": tracks_analyzed_count, "total_tracks": total_tracks_in_album}
        except Exception as e:
            error_tb = traceback.format_exc()
            failure_details = {"error": str(e), "traceback": error_tb, "album_name": album_name}
            log_and_update_album_task(f"Failed to analyze album '{album_name}': {e}", current_progress_val, task_state=TASK_STATUS_FAILURE, final_summary_details=failure_details)
            print(f"ERROR: Album analysis {album_id} failed: {e}\n{error_tb}")
            raise

def run_analysis_task(jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods):
    """Main RQ task to orchestrate the analysis of multiple albums."""
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        initial_details = {"message": "Fetching albums...", "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Main analysis task started."]}
        save_task_status(current_task_id, "main_analysis", TASK_STATUS_STARTED, progress=0, details=initial_details)
        headers = {"X-Emby-Token": jellyfin_token}
        current_progress = 0
        current_task_logs = initial_details["log"] # Initialize with the first log

        def log_and_update_main_analysis(message, progress, details_extra=None, task_state=TASK_STATUS_PROGRESS):
            nonlocal current_progress
            nonlocal current_task_logs
            current_progress = progress
            print(f"[MainAnalysisTask-{current_task_id}] {message}")

            current_details_for_db = dict(details_extra) if details_extra else {}
            current_details_for_db["status_message"] = message

            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            if task_state == TASK_STATUS_SUCCESS:
                current_details_for_db["log"] = [f"Task completed successfully. Final status: {message}"]
                current_details_for_db.pop('log_storage_info', None)
            elif task_state == TASK_STATUS_FAILURE or task_state == TASK_STATUS_REVOKED:
                current_task_logs.append(log_entry)
                current_details_for_db["log"] = current_task_logs
                if "error" not in current_details_for_db and task_state == TASK_STATUS_FAILURE: current_details_for_db["error"] = message
            else: # PROGRESS or STARTED
                current_task_logs.append(log_entry)
                current_details_for_db["log"] = current_task_logs

            meta_details = {"status_message": message}
            if details_extra:
                for key in ["albums_completed", "total_albums", "successful_albums", "failed_albums", "total_tracks_analyzed", "albums_found", "album_task_ids"]:
                    if key in details_extra:
                        meta_details[key] = details_extra[key]
            
            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.meta['details'] = meta_details
                current_job.save_meta()
            save_task_status(current_task_id, "main_analysis", task_state, progress=progress, details=current_details_for_db)

        try:
            log_and_update_main_analysis("🚀 Starting main analysis process...", 0)
            clean_temp(TEMP_DIR)
            albums = get_recent_albums(jellyfin_url, jellyfin_user_id, headers, num_recent_albums)
            if not albums:
                log_and_update_main_analysis("⚠️ No new albums to analyze.", 100, {"albums_found": 0}, task_state=TASK_STATUS_SUCCESS)
                return {"status": "SUCCESS", "message": "No new albums to analyze.", "albums_processed": 0}

            total_albums = len(albums)
            log_and_update_main_analysis(f"Found {total_albums} albums to process.", 5, {"albums_found": total_albums, "album_tasks_ids": []})
            album_jobs = []
            album_job_ids = []
            from app import rq_queue as main_rq_queue

            for album_idx, album in enumerate(albums):
                sub_task_id = str(uuid.uuid4())
                save_task_status(sub_task_id, "album_analysis", "PENDING", parent_task_id=current_task_id, sub_type_identifier=album['Id'], details={"album_name": album['Name']})
                job = main_rq_queue.enqueue(
                    analyze_album_task,
                    args=(album['Id'], album['Name'], jellyfin_url, jellyfin_user_id, jellyfin_token, top_n_moods, current_task_id),
                    job_id=sub_task_id,
                    description=f"Analyzing album: {album['Name']}",
                    job_timeout=-1, # Set timeout for album analysis
                    meta={'parent_task_id': current_task_id}
                )
                album_jobs.append(job)
                album_job_ids.append(job.id)
            
            log_and_update_main_analysis(f"Launched {total_albums} album analysis tasks.", 10, {"album_task_ids": album_job_ids})

            while True:
                # === Cooperative Cancellation Check for Main Analysis Task ===
                if current_job: # Check if running as an RQ job
                    with app.app_context(): # Ensure DB access
                    # Check self status first
                        main_task_db_info = get_task_info_from_db(current_task_id)
                        if main_task_db_info and main_task_db_info.get('status') == TASK_STATUS_REVOKED:
                            log_and_update_main_analysis(f"🛑 Main analysis task {current_task_id} has been REVOKED. Stopping and attempting to update children.", current_progress, task_state=TASK_STATUS_REVOKED)
                            # Update DB status of child jobs to REVOKED
                            for child_job_instance in album_jobs:
                                if not child_job_instance: continue 
                                try:
                                    # No need to call child_job_instance.cancel() here,
                                    # as the child task's cooperative cancellation will pick up the DB status change.
                                    with app.app_context(): # New context for each save
                                        child_db_info = get_task_info_from_db(child_job_instance.id)
                                        if child_db_info and child_db_info.get('status') not in [TASK_STATUS_REVOKED, TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, JobStatus.CANCELED]: # Avoid re-revoking
                                            print(f"[MainAnalysisTask-{current_task_id}] Marking child job {child_job_instance.id} as REVOKED in DB.")
                                            save_task_status(child_job_instance.id, "album_analysis", TASK_STATUS_REVOKED, parent_task_id=current_task_id, progress=100, details={"message": "Parent task was revoked.", "log": ["Parent task revoked."]})
                                except Exception as e_cancel_child:
                                    print(f"[MainAnalysisTask-{current_task_id}] Error trying to mark child job {child_job_instance.id} as REVOKED: {e_cancel_child}")
                            clean_temp(TEMP_DIR)
                            return {"status": "REVOKED", "message": "Main analysis task was revoked."}
                # === End Cooperative Cancellation Check ===
                completed_count = 0

                all_done = True

                for job_instance in album_jobs:
                    try:
                        is_child_completed = False

                        try:
                            job_instance.refresh() # Get latest status from Redis
                            child_status_rq = job_instance.get_status()

                            if job_instance.is_finished or job_instance.is_failed or child_status_rq == JobStatus.CANCELED:
                                is_child_completed = True
                            else: # Still active in RQ
                                pass # Child is active, all_done will be set to False later

                        except NoSuchJobError:
                            print(f"[MainAnalysisTask-{current_task_id}] Warning: Child job {job_instance.id} not found in Redis. Checking DB.")
                            with app.app_context(): # Ensure DB context
                                db_task_info = get_task_info_from_db(job_instance.id)
                            
                            if db_task_info:
                                db_status = db_task_info.get('status')
                                if db_status in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED, JobStatus.CANCELED, JobStatus.FINISHED, JobStatus.FAILED]:
                                    is_child_completed = True
                                    if db_status in [TASK_STATUS_FAILURE, TASK_STATUS_REVOKED, JobStatus.CANCELED, JobStatus.FAILED]:
                                        print(f"[MainAnalysisTask-{current_task_id}] Info: Child job {job_instance.id} (from DB) has status: {db_status}.")
                                    # If SUCCESS/FINISHED, it's completed, no special print needed here.
                                else:
                                    print(f"[MainAnalysisTask-{current_task_id}] Warning: Child job {job_instance.id} (from DB) has non-terminal status '{db_status}' but missing from RQ. Treating as completed.")
                                    is_child_completed = True 
                            else:
                                print(f"[MainAnalysisTask-{current_task_id}] CRITICAL: Child job {job_instance.id} not found in Redis or DB. Treating as completed.")
                                is_child_completed = True
                        
                        if is_child_completed:
                            completed_count += 1
                        else:
                            all_done = False
                    except Exception as e_monitor_child:
                        job_id_for_error = job_instance.id if job_instance else "Unknown_ID"
                        print(f"[MainAnalysisTask-{current_task_id}] ERROR monitoring child job {job_id_for_error}: {e_monitor_child}. Treating as completed to avoid hanging.")
                        traceback.print_exc()
                        completed_count += 1 # Ensure progress
                        # This job will likely be counted as 'failed' in the final tally or be missing its results.
                
                status_message_for_log = f"Processing albums: {completed_count}/{total_albums} completed."
                progress_while_waiting = 10 + int(80 * (completed_count / float(total_albums))) if total_albums > 0 else 10
                log_and_update_main_analysis(
                    status_message_for_log,
                    progress_while_waiting,
                    {"albums_completed": completed_count, "total_albums": total_albums})
                if all_done: break
                time.sleep(5)
            successful_albums = 0
            failed_albums = 0
            total_tracks_analyzed_all_albums = 0
            for job_instance in album_jobs:
                job_is_successful = False
                job_is_failed_or_canceled = False
                album_tracks_analyzed = 0

                # Wrap final counting for each job in a try-except
                try: 
                    try:
                        job_instance.refresh() # Try to get fresh status from RQ
                        if job_instance.is_finished:
                            job_is_successful = True
                            if isinstance(job_instance.result, dict):
                                album_tracks_analyzed = job_instance.result.get("tracks_analyzed", 0)
                        elif job_instance.is_failed or job_instance.get_status() == JobStatus.CANCELED:
                            # print(f"[MainAnalysisTask-{current_task_id}] Info: Child job {job_instance.id} (from RQ) ended with status: {job_instance.get_status()}.")
                            job_is_failed_or_canceled = True
                    except NoSuchJobError:
                        print(f"[MainAnalysisTask-{current_task_id}] Warning: Final check for job {job_instance.id} - not in Redis. Checking DB.")
                        with app.app_context():
                            db_task_info = get_task_info_from_db(job_instance.id)
                        if db_task_info:
                            db_status = db_task_info.get('status')
                            if db_status in [TASK_STATUS_SUCCESS, JobStatus.FINISHED]:
                                job_is_successful = True
                                if db_task_info.get('details'):
                                    try:
                                        details_dict = json.loads(db_task_info.get('details'))
                                        if 'tracks_analyzed' in details_dict:
                                            album_tracks_analyzed = details_dict.get("tracks_analyzed", 0)
                                    except (json.JSONDecodeError, TypeError): pass # Log already cleaned or minimal
                            elif db_status in [TASK_STATUS_FAILURE, TASK_STATUS_REVOKED, JobStatus.CANCELED, JobStatus.FAILED]:
                                print(f"[MainAnalysisTask-{current_task_id}] Info: Child job {job_instance.id} (from DB) has final status: {db_status}.")
                                job_is_failed_or_canceled = True
                            else: 
                                print(f"[MainAnalysisTask-{current_task_id}] Warning: Final check for job {job_instance.id} (from DB) - non-terminal status '{db_status}' but not in RQ. Counted as failed/canceled.")
                                job_is_failed_or_canceled = True
                        else: 
                            print(f"[MainAnalysisTask-{current_task_id}] CRITICAL: Final check for job {job_instance.id} - not found in Redis or DB. Counted as failed/canceled.")
                            job_is_failed_or_canceled = True
                except Exception as e_final_count:
                    job_id_for_error = job_instance.id if job_instance else "Unknown_ID"
                    print(f"[MainAnalysisTask-{current_task_id}] ERROR during final result processing for job {job_id_for_error}: {e_final_count}. Counting as failed.")
                    traceback.print_exc()
                    job_is_successful = False
                    job_is_failed_or_canceled = True # Ensure it's counted as failed
                    album_tracks_analyzed = 0 # Reset tracks analyzed for this problematic job

                if job_is_successful:
                    successful_albums += 1
                    total_tracks_analyzed_all_albums += album_tracks_analyzed
                elif job_is_failed_or_canceled:
                    failed_albums += 1
            
            final_summary_for_db = {
                "successful_albums": successful_albums, 
                "failed_albums": failed_albums, 
                "total_tracks_analyzed": total_tracks_analyzed_all_albums
            }
            final_message = f"Main analysis complete. Successful albums: {successful_albums}, Failed albums: {failed_albums}. Total tracks analyzed: {total_tracks_analyzed_all_albums}."
            log_and_update_main_analysis(final_message, 100, details_extra=final_summary_for_db, task_state=TASK_STATUS_SUCCESS)
            clean_temp(TEMP_DIR)
            return {"status": "SUCCESS", "message": final_message, "successful_albums": successful_albums, "failed_albums": failed_albums, "total_tracks_analyzed": total_tracks_analyzed_all_albums}
        except Exception as e:
            error_tb = traceback.format_exc()
            print(f"FATAL ERROR: Analysis failed: {e}\n{error_tb}")
            with app.app_context():
                failure_details = {"error_message": str(e), "traceback": error_tb}
                log_and_update_main_analysis(f"❌ Main analysis failed: {e}", current_progress, details_extra=failure_details, task_state=TASK_STATUS_FAILURE)
                # Attempt to mark children as REVOKED if the parent fails
                if 'album_jobs' in locals() and album_jobs: # Check if album_jobs was initialized
                    print(f"[MainAnalysisTask-{current_task_id}] Parent task failed. Attempting to mark {len(album_jobs)} children as REVOKED.")
                    for child_job_instance in album_jobs:
                        try: # Guard against issues with individual child job objects during cleanup
                            if not child_job_instance: continue # Skip if somehow a None entry exists
                            with app.app_context(): # Ensure context for DB ops
                                child_db_info = get_task_info_from_db(child_job_instance.id)
                                if child_db_info and child_db_info.get('status') not in [TASK_STATUS_REVOKED, TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, JobStatus.CANCELED]:
                                    save_task_status(child_job_instance.id, "album_analysis", TASK_STATUS_REVOKED, parent_task_id=current_task_id, progress=100, details={"message": "Parent task failed.", "log": ["Parent task failed."]})
                        except Exception as e_cancel_child_on_fail:
                            print(f"[MainAnalysisTask-{current_task_id}] Error marking child job {child_job_instance.id} as REVOKED during parent failure: {e_cancel_child_on_fail}")
            raise

# --- Helper for mutation ---
def _mutate_param(value, min_val, max_val, delta, is_float=False, round_digits=None):
    """Mutates a parameter value within its bounds."""
    if is_float:
        mutation = random.uniform(-delta, delta)
        new_value = value + mutation
        if round_digits is not None:
            new_value = round(new_value, round_digits)
    else: # Integer
        # Ensure delta is at least 1 for integer mutation if it's derived from a float
        int_delta = max(1, int(delta)) if isinstance(delta, float) else int(delta)
        mutation = random.randint(-int_delta, int_delta)
        new_value = value + mutation

    # Clip to min/max bounds; ensure correct type after clipping for integers
    new_value = np.clip(new_value, min_val, max_val)
    return int(new_value) if not is_float else new_value

def _generate_or_mutate_kmeans_initial_centroids(
    n_clusters_current, data_for_clustering_current,
    elite_kmeans_params_original, elite_pca_config_original, new_pca_config_current,
    mutation_coord_fraction=0.05, log_prefix=""):
    """
    Generates or mutates initial centroids for KMeans.
    Centroids are returned as a list of lists.
    """
    use_mutated_elite_centroids = False
    if elite_kmeans_params_original and elite_pca_config_original:
        elite_initial_centroids_list = elite_kmeans_params_original.get("initial_centroids")
        elite_n_clusters = elite_kmeans_params_original.get("n_clusters")

        pca_compatible = (elite_pca_config_original.get("enabled") == new_pca_config_current.get("enabled"))
        if pca_compatible and elite_pca_config_original.get("enabled"): # Both enabled
            pca_compatible = (elite_pca_config_original.get("components") == new_pca_config_current.get("components"))

        if elite_initial_centroids_list and isinstance(elite_initial_centroids_list, list) and \
           pca_compatible and elite_n_clusters == n_clusters_current:
            use_mutated_elite_centroids = True

    if use_mutated_elite_centroids:
        # print(f"{log_prefix} Attempting to mutate KMeans centroids.")
        mutated_centroids = []
        elite_centroids_arr = np.array(elite_initial_centroids_list)
        
        if data_for_clustering_current.shape[0] > 0 and data_for_clustering_current.shape[1] > 0 and \
           elite_centroids_arr.ndim == 2 and elite_centroids_arr.shape[1] == data_for_clustering_current.shape[1]:
            
            data_min = np.min(data_for_clustering_current, axis=0)
            data_max = np.max(data_for_clustering_current, axis=0)
            data_range = data_max - data_min
            coord_mutation_deltas = data_range * mutation_coord_fraction
            
            for centroid_coords in elite_centroids_arr: # Iterate over rows (centroids)
                mutation_vector = np.random.uniform(-coord_mutation_deltas, coord_mutation_deltas, size=centroid_coords.shape)
                mutated_coord = centroid_coords + mutation_vector
                mutated_coord = np.clip(mutated_coord, data_min, data_max) # Clip to current data bounds
                mutated_centroids.append(mutated_coord.tolist())
            # print(f"{log_prefix} Successfully mutated {len(mutated_centroids)} KMeans centroids.")
            return mutated_centroids
        else:
            # print(f"{log_prefix} KMeans centroid mutation condition not met (data shape, elite centroid format, or dim mismatch). Falling back to random.")
            pass # Fall through to random generation

    # Fallback: Randomly pick from current data
    # print(f"{log_prefix} Generating random KMeans centroids.")
    if data_for_clustering_current.shape[0] == 0 or n_clusters_current == 0:
        # print(f"{log_prefix} No data points or zero clusters requested for KMeans centroid generation.")
        return [] 
    
    num_available_points = data_for_clustering_current.shape[0]
    actual_n_clusters = min(n_clusters_current, num_available_points) # Cannot pick more unique centroids than available points
    if actual_n_clusters == 0 and num_available_points > 0: actual_n_clusters = 1 # Ensure at least one if possible
    if actual_n_clusters == 0: return []

    indices = np.random.choice(num_available_points, actual_n_clusters, replace=(actual_n_clusters > num_available_points))
    initial_centroids = data_for_clustering_current[indices]
    return initial_centroids.tolist()

def _perform_single_clustering_iteration(
    run_idx, all_tracks_data_parsed, 
    clustering_method, num_clusters_min_max, dbscan_params_ranges, gmm_params_ranges, pca_params_ranges, 
    max_songs_per_cluster, log_prefix="",
    elite_solutions_params_list=None, exploitation_probability=0.0, mutation_config=None):
    """
    Internal helper to perform a single clustering iteration. Not an RQ task.
    Returns a result dictionary or None on failure.
    `num_clusters_min_max` is a tuple (min, max)
    `dbscan_params_ranges` is a dict like {"eps_min": ..., "eps_max": ..., "samples_min": ..., "samples_max": ...}
    `gmm_params_ranges` is a dict like {"n_components_min": ..., "n_components_max": ...}
    `pca_params_ranges` is a dict like {"components_min": ..., "components_max": ...}
    `elite_solutions_params_list`: A list of 'parameters' dicts from previous best runs.
    `exploitation_probability`: Chance to use an elite solution for parameter generation.
    `mutation_config`: Dict with mutation strengths, e.g., {"int_abs_delta": 2, "float_abs_delta": 0.05}.
    """
    try:
        elite_solutions_params_list = elite_solutions_params_list or []
        mutation_config = mutation_config or {"int_abs_delta": 2, "float_abs_delta": 0.05, "coord_mutation_fraction": MUTATION_KMEANS_COORD_FRACTION}
        if "coord_mutation_fraction" not in mutation_config: # Ensure default if not passed
            mutation_config["coord_mutation_fraction"] = MUTATION_KMEANS_COORD_FRACTION

        # --- Data Preparation ---
        X_original = [score_vector(row, MOOD_LABELS) for row in all_tracks_data_parsed]
        X_scaled = np.array(X_original)
        if X_scaled.shape[0] == 0:
            print(f"{log_prefix} Iteration {run_idx}: No data to cluster.")
            return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "parameters": {}}
        
        data_after_pca_for_this_iteration = X_scaled # Default if PCA is off or fails
        pca_model_for_this_iteration = None
        # --- End Data Preparation ---

        # Parameter generation for this specific iteration
        # PCA parameters are determined first, then PCA is applied.
        # Then, clustering method parameters (like n_clusters) are determined,
        # potentially using the shape of the data *after* PCA.
        # Finally, for KMeans, initial_centroids are generated/mutated.

        method_params_config = {}
        pca_config = {} # Renamed from current_pca_config
        params_generated_by_mutation = False

        if elite_solutions_params_list and random.random() < exploitation_probability:
            chosen_elite_params_set = random.choice(elite_solutions_params_list)
            elite_method_config_original = chosen_elite_params_set.get("clustering_method_config")
            elite_pca_config_original = chosen_elite_params_set.get("pca_config", {"enabled": False, "components": 0})

            if elite_method_config_original and elite_pca_config_original and \
               elite_method_config_original.get("method") == clustering_method:
                try:
                    # 1. Mutate PCA components first
                    elite_pca_comps = elite_pca_config_original.get("components", pca_params_ranges["components_min"])
                    mutated_pca_comps = _mutate_param(
                        elite_pca_comps,
                        pca_params_ranges["components_min"], pca_params_ranges["components_max"],
                        mutation_config.get("int_abs_delta", 2)
                    )
                    # Max PCA components also limited by number of features in X_scaled and number of samples
                    max_pca_by_features = X_scaled.shape[1]
                    max_pca_by_samples = (X_scaled.shape[0] - 1) if X_scaled.shape[0] > 1 else 1
                    
                    max_allowable_pca_mutated = min(mutated_pca_comps, len(MOOD_LABELS) + 1, max_pca_by_features, max_pca_by_samples)
                    max_allowable_pca_mutated = max(0, max_allowable_pca_mutated) # Ensure not negative
                    temp_pca_config = {"enabled": max_allowable_pca_mutated > 0, "components": max_allowable_pca_mutated}

                    # 2. Apply this new PCA config to get data for this iteration
                    if temp_pca_config["enabled"]:
                        if temp_pca_config["components"] > 0:
                            pca_model_for_this_iteration = PCA(n_components=temp_pca_config["components"])
                            data_after_pca_for_this_iteration = pca_model_for_this_iteration.fit_transform(X_scaled)
                            temp_pca_config["components"] = pca_model_for_this_iteration.n_components_ # Update with actual components used
                        else: # Components somehow became 0
                            temp_pca_config["enabled"] = False
                            data_after_pca_for_this_iteration = X_scaled # Fallback
                    else:
                        data_after_pca_for_this_iteration = X_scaled # PCA not enabled

                    # 3. Mutate clustering method parameters (e.g., n_clusters)
                    #    This must happen AFTER PCA, as n_clusters can depend on data_after_pca_for_this_iteration.shape[0]
                    temp_method_params_config = None
                    max_clusters_or_components = data_after_pca_for_this_iteration.shape[0]
                    if max_clusters_or_components == 0: # No data points after PCA (should be rare)
                        raise ValueError("No data points available after PCA to determine cluster parameters.")

                    if clustering_method == "kmeans":
                        elite_n_clusters = elite_method_config_original.get("params", {}).get("n_clusters", num_clusters_min_max[0])
                        mutated_n_clusters = _mutate_param(
                            elite_n_clusters, 
                            num_clusters_min_max[0], 
                            min(num_clusters_min_max[1], max_clusters_or_components), # Max clusters capped by available points
                            mutation_config.get("int_abs_delta", 2)
                        )
                        temp_method_params_config = {"method": "kmeans", "params": {"n_clusters": max(1, mutated_n_clusters)}}
                    elif clustering_method == "dbscan":
                        elite_eps = elite_method_config_original.get("params", {}).get("eps", dbscan_params_ranges["eps_min"])
                        elite_min_samples = elite_method_config_original.get("params", {}).get("min_samples", dbscan_params_ranges["samples_min"])
                        mutated_eps = _mutate_param(
                            elite_eps, dbscan_params_ranges["eps_min"], dbscan_params_ranges["eps_max"],
                            mutation_config.get("float_abs_delta", 0.05), is_float=True, round_digits=2
                        )
                        mutated_min_samples = _mutate_param(
                            elite_min_samples, dbscan_params_ranges["samples_min"], dbscan_params_ranges["samples_max"],
                            mutation_config.get("int_abs_delta", 2)
                        )
                        temp_method_params_config = {"method": "dbscan", "params": {"eps": mutated_eps, "min_samples": mutated_min_samples}}
                    elif clustering_method == "gmm":
                        elite_n_components = elite_method_config_original.get("params", {}).get("n_components", gmm_params_ranges["n_components_min"])
                        mutated_n_components = _mutate_param(
                            elite_n_components, 
                            gmm_params_ranges["n_components_min"], 
                            min(gmm_params_ranges["n_components_max"], max_clusters_or_components), # Max components capped
                            mutation_config.get("int_abs_delta", 2)
                        )
                        temp_method_params_config = {"method": "gmm", "params": {"n_components": max(1, mutated_n_components)}}
                    
                    if temp_method_params_config and temp_pca_config is not None:
                        # 4. For KMeans, generate/mutate initial_centroids
                        if clustering_method == "kmeans":
                            kmeans_initial_centroids = _generate_or_mutate_kmeans_initial_centroids(
                                temp_method_params_config["params"]["n_clusters"],
                                data_after_pca_for_this_iteration,
                                elite_method_config_original.get("params"), # Pass full elite Kmeans params
                                elite_pca_config_original, # Elite's PCA config
                                temp_pca_config,           # Current iteration's PCA config
                                mutation_config.get("coord_mutation_fraction"),
                                log_prefix=f"{log_prefix} Iteration {run_idx} (mutation)"
                            )
                            temp_method_params_config["params"]["initial_centroids"] = kmeans_initial_centroids

                        method_params_config = temp_method_params_config
                        pca_config = temp_pca_config
                        params_generated_by_mutation = True
                except Exception as e_mutate:
                    print(f"{log_prefix} Iteration {run_idx}: Error mutating elite params: {e_mutate}. Falling back to random.")
                    params_generated_by_mutation = False
        if not params_generated_by_mutation:
            # print(f"{log_prefix} Iteration {run_idx}: Using random parameters.")
            # Original random parameter generation
            sampled_pca_components_rand = random.randint(pca_params_ranges["components_min"], pca_params_ranges["components_max"])
            max_allowable_pca_rand = min(sampled_pca_components_rand, len(MOOD_LABELS) + 1, len(all_tracks_data_parsed) -1 if len(all_tracks_data_parsed) > 1 else 1)
            max_allowable_pca_rand = max(0, max_allowable_pca_rand)
            pca_config = {"enabled": max_allowable_pca_rand > 0, "components": max_allowable_pca_rand}
            
            # Apply PCA for random generation path to get data_after_pca_for_this_iteration
            if pca_config["enabled"]:
                n_comps_rand = min(pca_config["components"], X_scaled.shape[1], (X_scaled.shape[0] - 1) if X_scaled.shape[0] > 1 else 1)
                if n_comps_rand > 0:
                    pca_model_for_this_iteration = PCA(n_components=n_comps_rand)
                    data_after_pca_for_this_iteration = pca_model_for_this_iteration.fit_transform(X_scaled)
                    pca_config["components"] = pca_model_for_this_iteration.n_components_ # Update with actual
                else:
                    pca_config["enabled"] = False
                    data_after_pca_for_this_iteration = X_scaled
            else:
                data_after_pca_for_this_iteration = X_scaled

            max_clusters_or_components_rand = data_after_pca_for_this_iteration.shape[0]
            if max_clusters_or_components_rand == 0: # No data points after PCA
                 print(f"{log_prefix} Iteration {run_idx}: No data points available after PCA for random parameter generation.")
                 return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "parameters": {"pca_config": pca_config}}

            if clustering_method == "kmeans":
                k_rand = random.randint(num_clusters_min_max[0], min(num_clusters_min_max[1], max_clusters_or_components_rand))
                k_rand = max(1, k_rand)
                kmeans_initial_centroids_rand = _generate_or_mutate_kmeans_initial_centroids(
                    k_rand, data_after_pca_for_this_iteration, None, None, pca_config, # No elite for random
                    log_prefix=f"{log_prefix} Iteration {run_idx} (random)"
                )
                method_params_config = {"method": "kmeans", "params": {"n_clusters": k_rand, "initial_centroids": kmeans_initial_centroids_rand}}
            elif clustering_method == "dbscan":
                current_dbscan_eps_rand = round(random.uniform(dbscan_params_ranges["eps_min"], dbscan_params_ranges["eps_max"]), 2)
                current_dbscan_min_samples_rand = random.randint(dbscan_params_ranges["samples_min"], dbscan_params_ranges["samples_max"])
                method_params_config = {"method": "dbscan", "params": {"eps": current_dbscan_eps_rand, "min_samples": current_dbscan_min_samples_rand}}
            elif clustering_method == "gmm":
                gmm_n_rand = random.randint(gmm_params_ranges["n_components_min"], min(gmm_params_ranges["n_components_max"], max_clusters_or_components_rand))
                method_params_config = {"method": "gmm", "params": {"n_components": max(1, gmm_n_rand)}}
            else:
                print(f"{log_prefix} Iteration {run_idx}: Unsupported clustering method {clustering_method}")
                return None
        
        # Ensure pca_model_for_this_iteration is set if pca_config is enabled but model wasn't created during mutation path
        if pca_config["enabled"] and pca_model_for_this_iteration is None and pca_config["components"] > 0:
            pca_model_for_this_iteration = PCA(n_components=pca_config["components"])
            data_after_pca_for_this_iteration = pca_model_for_this_iteration.fit_transform(X_scaled) # Refit if needed

        if not method_params_config or pca_config is None:
            print(f"{log_prefix} Iteration {run_idx}: Critical error: parameters not configured.")
            return None

        # --- Start of core logic from original run_single_clustering_iteration_task ---
        X_original = [score_vector(row, MOOD_LABELS) for row in all_tracks_data_parsed]
        # X_scaled, data_after_pca_for_this_iteration, and pca_model_for_this_iteration are already prepared above.
        # Use data_after_pca_for_this_iteration for clustering.

        labels = None
        cluster_centers_map = {}
        raw_distances = np.zeros(data_after_pca_for_this_iteration.shape[0])
        method_from_config = method_params_config["method"]
        params_from_config = method_params_config["params"]

        if method_from_config == "kmeans":
            if not params_from_config.get("initial_centroids") or not isinstance(params_from_config["initial_centroids"], list) or len(params_from_config["initial_centroids"]) == 0:
                print(f"{log_prefix} Iteration {run_idx}: KMeans initial_centroids missing or empty. Cannot cluster with KMeans.")
                return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}}
            
            initial_centroids_np = np.array(params_from_config["initial_centroids"])
            
            # Ensure n_clusters matches the number of initial centroids provided
            if initial_centroids_np.ndim == 1 and initial_centroids_np.shape[0] == 0: # Empty array from empty list
                 print(f"{log_prefix} Iteration {run_idx}: KMeans initial_centroids resulted in empty numpy array. Cannot cluster.")
                 return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}}

            if initial_centroids_np.shape[0] != params_from_config["n_clusters"]:
                # print(f"{log_prefix} Iteration {run_idx}: Mismatch n_clusters ({params_from_config['n_clusters']}) and num initial_centroids ({initial_centroids_np.shape[0]}). Adjusting n_clusters.")
                params_from_config["n_clusters"] = initial_centroids_np.shape[0]
            
            if params_from_config["n_clusters"] == 0:
                print(f"{log_prefix} Iteration {run_idx}: n_clusters is 0 for KMeans after adjustment. Cannot cluster.")
                return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}}

            kmeans = KMeans(n_clusters=params_from_config["n_clusters"], init=initial_centroids_np, n_init=1)
            labels = kmeans.fit_predict(data_after_pca_for_this_iteration)
            cluster_centers_map = {i: kmeans.cluster_centers_[i] for i in range(params_from_config["n_clusters"])}
            centers_for_points = kmeans.cluster_centers_[labels]
            raw_distances = np.linalg.norm(data_after_pca_for_this_iteration - centers_for_points, axis=1)
        elif method_from_config == "dbscan":
            dbscan = DBSCAN(eps=params_from_config["eps"], min_samples=params_from_config["min_samples"])
            labels = dbscan.fit_predict(data_after_pca_for_this_iteration)
            for cluster_id_val in set(labels):
                if cluster_id_val == -1: continue
                indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id_val]
                cluster_points = data_after_pca_for_this_iteration[indices]
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    for i_idx in indices: raw_distances[i_idx] = np.linalg.norm(data_after_pca_for_this_iteration[i_idx] - center)
                    cluster_centers_map[cluster_id_val] = center
        elif method_from_config == "gmm":
            gmm = GaussianMixture(n_components=params_from_config["n_components"], covariance_type=GMM_COVARIANCE_TYPE, random_state=None, max_iter=1000)
            gmm.fit(data_after_pca_for_this_iteration)
            labels = gmm.predict(data_after_pca_for_this_iteration)
            cluster_centers_map = {i: gmm.means_[i] for i in range(params_from_config["n_components"])}
            centers_for_points = gmm.means_[labels] # type: ignore
            raw_distances = np.linalg.norm(data_after_pca_for_this_iteration - centers_for_points, axis=1)
        
        del data_after_pca_for_this_iteration # Free memory
        
        if labels is None or len(set(labels) - {-1}) == 0:
            # print(f"{log_prefix} Iteration {run_idx}: No valid clusters found.")
            return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}}

        max_dist = raw_distances.max()
        normalized_distances = raw_distances / max_dist if max_dist > 0 else raw_distances
        track_info_list = [{"row": all_tracks_data_parsed[i], "label": labels[i], "distance": normalized_distances[i]} for i in range(len(all_tracks_data_parsed))]

        filtered_clusters = defaultdict(list)
        for cid in set(labels):
            if cid == -1: continue
            cluster_tracks = [t for t in track_info_list if t["label"] == cid and t["distance"] <= MAX_DISTANCE]
            if not cluster_tracks: continue
            cluster_tracks.sort(key=lambda x: x["distance"])
            
            count_per_artist = defaultdict(int)
            selected_tracks = []
            for t_item in cluster_tracks:
                author = t_item["row"]["author"]
                if count_per_artist[author] < MAX_SONGS_PER_ARTIST:
                    selected_tracks.append(t_item)
                    count_per_artist[author] += 1
                if len(selected_tracks) >= max_songs_per_cluster: break
            for t_item in selected_tracks:
                item_id, title, author_val = t_item["row"]["item_id"], t_item["row"]["title"], t_item["row"]["author"]
                filtered_clusters[cid].append((item_id, title, author_val))

        current_named_playlists = defaultdict(list)
        current_playlist_centroids = {}
        unique_predominant_mood_scores = {}

        for label_val, songs_list in filtered_clusters.items():
            if songs_list:
                center_val = cluster_centers_map.get(label_val)
                if center_val is None: continue
                name, top_scores = name_cluster(center_val, pca_model_for_this_iteration, pca_config["enabled"], MOOD_LABELS)
                if top_scores and any(mood in MOOD_LABELS for mood in top_scores.keys()):
                    predominant_mood_key = max((k for k in top_scores if k in MOOD_LABELS), key=top_scores.get, default=None)
                    if predominant_mood_key:
                        current_mood_score = top_scores.get(predominant_mood_key, 0.0)
                        unique_predominant_mood_scores[predominant_mood_key] = max(unique_predominant_mood_scores.get(predominant_mood_key, 0.0), current_mood_score)
                current_named_playlists[name].extend(songs_list)
                current_playlist_centroids[name] = top_scores
        
        # --- Enhanced Score Calculation ---
        base_diversity_score = sum(unique_predominant_mood_scores.values())

        # Calculate playlist_purity_component
        all_individual_playlist_purities = []
        item_id_to_song_index_map = {row['item_id']: i for i, row in enumerate(all_tracks_data_parsed)}

        if current_named_playlists:
            for playlist_name_key, songs_in_playlist_info_list in current_named_playlists.items():
                # playlist_name_key is the generated name like "Rock_Fast"
                # songs_in_playlist_info_list is list of (item_id, title, author)

                # Get the centroid data for this named playlist
                playlist_centroid_mood_data = current_playlist_centroids.get(playlist_name_key)
                if not playlist_centroid_mood_data or not songs_in_playlist_info_list:
                    continue

                # Determine the predominant mood for THIS specific playlist based on its centroid's mood data
                predominant_mood_for_this_playlist = None
                max_score_for_predominant = -1.0
                for mood_label, mood_score in playlist_centroid_mood_data.items():
                    if mood_label in MOOD_LABELS: # Ensure it's a mood label
                        if mood_score > max_score_for_predominant:
                            max_score_for_predominant = mood_score
                            predominant_mood_for_this_playlist = mood_label
                
                if not predominant_mood_for_this_playlist:
                    continue

                try:
                    predominant_mood_index_in_labels = MOOD_LABELS.index(predominant_mood_for_this_playlist)
                except ValueError:
                    print(f"{log_prefix} Iteration {run_idx}: Warning: Predominant mood '{predominant_mood_for_this_playlist}' for playlist '{playlist_name_key}' not in MOOD_LABELS list.")
                    continue
                    
                scores_of_predominant_mood_for_songs_in_playlist = []
                for item_id, _, _ in songs_in_playlist_info_list:
                    song_original_index = item_id_to_song_index_map.get(item_id)
                    if song_original_index is not None:
                        song_mood_scores_vector = X_original[song_original_index][1:] # Get only the mood scores part
                        if predominant_mood_index_in_labels < len(song_mood_scores_vector):
                            song_specific_score_for_predominant_mood = song_mood_scores_vector[predominant_mood_index_in_labels]
                            scores_of_predominant_mood_for_songs_in_playlist.append(song_specific_score_for_predominant_mood)

                if scores_of_predominant_mood_for_songs_in_playlist:
                    avg_purity_for_this_playlist = sum(scores_of_predominant_mood_for_songs_in_playlist) / len(scores_of_predominant_mood_for_songs_in_playlist)
                    all_individual_playlist_purities.append(avg_purity_for_this_playlist)

        playlist_purity_component = 0.0
        if all_individual_playlist_purities:
            playlist_purity_component = sum(all_individual_playlist_purities) / len(all_individual_playlist_purities)

        final_enhanced_score = (SCORE_WEIGHT_DIVERSITY * base_diversity_score) + (SCORE_WEIGHT_PURITY * playlist_purity_component)
        # print(f"{log_prefix} Iteration {run_idx}: BaseDiv: {base_diversity_score:.2f}, PurityComp: {playlist_purity_component:.2f}, FinalScore: {final_enhanced_score:.2f}")

        pca_model_details = {"n_components": pca_model_for_this_iteration.n_components_, "explained_variance_ratio": pca_model_for_this_iteration.explained_variance_ratio_.tolist(), "mean": pca_model_for_this_iteration.mean_.tolist()} if pca_model_for_this_iteration and pca_config["enabled"] else None
        result = {
            "diversity_score": float(final_enhanced_score), # Use the new enhanced score
            "named_playlists": dict(current_named_playlists), 
            "playlist_centroids": current_playlist_centroids,
            "pca_model_details": pca_model_details, 
            "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}
        }
        return result
    except Exception as e_iter:
        print(f"{log_prefix} Iteration {run_idx} failed: {e_iter}")
        traceback.print_exc()
        return None # Indicate failure for this iteration

def run_clustering_batch_task(
    batch_id_str, start_run_idx, num_iterations_in_batch, all_tracks_data_json,
    clustering_method, num_clusters_min_max_tuple, dbscan_params_ranges_dict, gmm_params_ranges_dict, pca_params_ranges_dict,
    max_songs_per_cluster, parent_task_id,
    elite_solutions_params_list_json=None, exploitation_probability=0.0, mutation_config_json=None):
    """RQ task to run a batch of clustering iterations."""
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4()) # This is the ID of the batch task itself

    with app.app_context():
        initial_log_message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Clustering Batch Task {batch_id_str} (Job ID: {current_task_id}) started. Iterations: {start_run_idx} to {start_run_idx + num_iterations_in_batch - 1}."
        initial_details = {
            "batch_id": batch_id_str, 
            "start_run_idx": start_run_idx, 
            "num_iterations_in_batch": num_iterations_in_batch,
            "log": [initial_log_message]
        }
        save_task_status(current_task_id, "clustering_batch", TASK_STATUS_STARTED, parent_task_id=parent_task_id, sub_type_identifier=batch_id_str, progress=0, details=initial_details)
        
        current_progress_batch = 0
        current_task_logs_batch = initial_details["log"]
        log_prefix_for_iter = f"[Batch-{current_task_id}]"

        def log_and_update_batch_task(message, progress, details_extra=None, task_state=TASK_STATUS_PROGRESS):
            nonlocal current_progress_batch, current_task_logs_batch
            current_progress_batch = progress
            print(f"[ClusteringBatchTask-{current_task_id}] {message}")

            db_details_batch = {"batch_id": batch_id_str, "start_run_idx": start_run_idx, "num_iterations_in_batch": num_iterations_in_batch}
            if details_extra: db_details_batch.update(details_extra)
            db_details_batch["status_message"] = message

            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            if task_state == TASK_STATUS_SUCCESS:
                db_details_batch["log"] = [f"Task completed successfully. Final status: {message}"]
            elif task_state in [TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]:
                current_task_logs_batch.append(log_entry)
                db_details_batch["log"] = current_task_logs_batch
                if "error" not in db_details_batch and task_state == TASK_STATUS_FAILURE: db_details_batch["error"] = message
            else:
                current_task_logs_batch.append(log_entry)
                db_details_batch["log"] = current_task_logs_batch

            meta_details_batch = {"batch_id": batch_id_str, "status_message": message}
            if details_extra and "best_score_in_batch" in details_extra: meta_details_batch["best_score_in_batch"] = details_extra["best_score_in_batch"]
            
            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.meta['details'] = meta_details_batch
                current_job.save_meta()
            save_task_status(current_task_id, "clustering_batch", task_state, parent_task_id=parent_task_id, sub_type_identifier=batch_id_str, progress=progress, details=db_details_batch)

        try:
            log_and_update_batch_task(f"Processing {num_iterations_in_batch} iterations.", 5)
            
            # === Cooperative Cancellation Check for Batch Task ===
            if current_job:
                with app.app_context():
                    task_db_info = get_task_info_from_db(current_task_id)
                    parent_task_db_info = get_task_info_from_db(parent_task_id) if parent_task_id else None
                    
                    is_self_revoked = task_db_info and task_db_info.get('status') == 'REVOKED'
                    is_parent_failed_or_revoked = parent_task_db_info and parent_task_db_info.get('status') in [TASK_STATUS_REVOKED, TASK_STATUS_FAILURE]

                    if is_self_revoked or is_parent_failed_or_revoked:
                        parent_status_for_reason = parent_task_db_info.get('status') if parent_task_db_info else "N/A"
                        revocation_reason = "self was REVOKED" if is_self_revoked else f"parent task {parent_task_id} status is {parent_status_for_reason}"
                        log_and_update_batch_task(f"🛑 Batch task {batch_id_str} stopping because {revocation_reason}.", current_progress_batch, task_state=TASK_STATUS_REVOKED)
                        return {"status": "REVOKED", "message": f"Batch task {batch_id_str} stopped.", "iterations_completed_in_batch": 0, "best_result_from_batch": None}
            # === End Cooperative Cancellation Check ===

            all_tracks_data = json.loads(all_tracks_data_json)

            elite_solutions_params_list_for_iter = []
            if elite_solutions_params_list_json:
                try:
                    elite_solutions_params_list_for_iter = json.loads(elite_solutions_params_list_json)
                except json.JSONDecodeError:
                    print(f"{log_prefix_for_iter} Warning: Could not decode elite solutions JSON. Proceeding without elites for this batch.")
            
            mutation_config_for_iter = {}
            if mutation_config_json:
                try:
                    mutation_config_for_iter = json.loads(mutation_config_json)
                except json.JSONDecodeError:
                    print(f"{log_prefix_for_iter} Warning: Could not decode mutation config JSON. Using default mutation behavior.")
            
            best_result_in_this_batch = None
            best_score_in_this_batch = -1.0
            iterations_actually_completed = 0

            for i in range(num_iterations_in_batch):
                current_run_global_idx = start_run_idx + i
                
                # Cooperative cancellation check before starting each iteration within the batch
                if current_job:
                    with app.app_context():
                        task_db_info_iter_check = get_task_info_from_db(current_task_id) # Check self
                        parent_task_db_info_iter_check = get_task_info_from_db(parent_task_id) if parent_task_id else None
                        if (task_db_info_iter_check and task_db_info_iter_check.get('status') == TASK_STATUS_REVOKED) or \
                           (parent_task_db_info_iter_check and parent_task_db_info_iter_check.get('status') in [TASK_STATUS_REVOKED, TASK_STATUS_FAILURE]):
                            log_and_update_batch_task(f"Batch task {batch_id_str} stopping mid-batch due to cancellation/parent failure before iteration {current_run_global_idx}.", current_progress_batch, task_state=TASK_STATUS_REVOKED)
                            return {"status": "REVOKED", "message": "Batch task revoked mid-process.", "iterations_completed_in_batch": iterations_actually_completed, "best_result_from_batch": best_result_in_this_batch}

                iteration_result = _perform_single_clustering_iteration(
                    current_run_global_idx, all_tracks_data, clustering_method,
                    num_clusters_min_max_tuple, dbscan_params_ranges_dict, gmm_params_ranges_dict, pca_params_ranges_dict,
                    max_songs_per_cluster, log_prefix=log_prefix_for_iter,
                    elite_solutions_params_list=(elite_solutions_params_list_for_iter if elite_solutions_params_list_for_iter else []),
                    exploitation_probability=exploitation_probability,
                    mutation_config=(mutation_config_for_iter if mutation_config_for_iter else {})
                )
                iterations_actually_completed += 1 # Count even if result is None, as an attempt was made

                if iteration_result and iteration_result.get("diversity_score", -1.0) > best_score_in_this_batch:
                    best_score_in_this_batch = iteration_result["diversity_score"]
                    best_result_in_this_batch = iteration_result
                
                iter_progress = 5 + int(90 * (iterations_actually_completed / float(num_iterations_in_batch)))
                log_and_update_batch_task(f"Iteration {current_run_global_idx} (in batch: {i+1}/{num_iterations_in_batch}) complete. Batch best score: {best_score_in_this_batch:.2f}", iter_progress)

            final_batch_summary = {
                "best_score_in_batch": best_score_in_this_batch,
                "iterations_completed_in_batch": iterations_actually_completed,
                "full_best_result_from_batch": best_result_in_this_batch # This is what the parent will use
            }
            log_and_update_batch_task(f"Batch {batch_id_str} complete. Best score in batch: {best_score_in_this_batch:.2f}", 100, details_extra=final_batch_summary, task_state=TASK_STATUS_SUCCESS)
            return {"status": "SUCCESS", "iterations_completed_in_batch": iterations_actually_completed, "best_result_from_batch": best_result_in_this_batch}

        except Exception as e:
            error_tb = traceback.format_exc()
            failure_details = {"error": str(e), "traceback": error_tb, "batch_id": batch_id_str}
            log_and_update_batch_task(f"Failed clustering batch {batch_id_str}: {e}", current_progress_batch, details_extra=failure_details, task_state=TASK_STATUS_FAILURE)
            print(f"ERROR: Clustering batch {batch_id_str} failed: {e}\n{error_tb}")
            raise

# Constants for guided search (can be moved to config.py later or read from env)
TOP_N_ELITES = int(os.environ.get("CLUSTERING_TOP_N_ELITES", "10"))
EXPLOITATION_START_FRACTION = float(os.environ.get("CLUSTERING_EXPLOITATION_START_FRACTION", "0.2"))
EXPLOITATION_PROBABILITY_CONFIG = float(os.environ.get("CLUSTERING_EXPLOITATION_PROBABILITY", "0.7"))
MUTATION_INT_ABS_DELTA = int(os.environ.get("CLUSTERING_MUTATION_INT_ABS_DELTA", "3"))
MUTATION_FLOAT_ABS_DELTA = float(os.environ.get("CLUSTERING_MUTATION_FLOAT_ABS_DELTA", "0.05"))
def run_clustering_task(
    clustering_method, num_clusters_min, num_clusters_max, 
    dbscan_eps_min, dbscan_eps_max, dbscan_min_samples_min, dbscan_min_samples_max, 
    pca_components_min, pca_components_max, num_clustering_runs, max_songs_per_cluster, 
    gmm_n_components_min, gmm_n_components_max,
    use_ai_playlist_naming_param, ollama_server_url_param, ollama_model_name_param): # Added AI params
    """Main RQ task for clustering and playlist generation."""
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        initial_log_message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Main clustering task started."
        _main_task_accumulated_details = {
            "message": "Initializing clustering...",
            "log": [initial_log_message],
            "total_runs": num_clustering_runs, 
            "runs_completed": 0,            
            "batch_jobs_launched": 0,       
            "total_batch_jobs": 0,          
            "active_runs_count": 0,         
            "best_score": -1.0,             
            "clustering_run_job_ids": []    
        }
        save_task_status(current_task_id, "main_clustering", TASK_STATUS_STARTED, progress=0, details=_main_task_accumulated_details)
        
        current_progress = 0

        def log_and_update_main_clustering(message, progress, details_to_add_or_update=None, task_state=TASK_STATUS_PROGRESS, print_console=True):
            nonlocal current_progress, _main_task_accumulated_details
            current_progress = progress
            if print_console: print(f"[MainClusteringTask-{current_task_id}] {message}")
            if details_to_add_or_update: _main_task_accumulated_details.update(details_to_add_or_update)
            _main_task_accumulated_details["status_message"] = message
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            current_log_list = _main_task_accumulated_details.get("log", [])
            
            if task_state == TASK_STATUS_SUCCESS:
                _main_task_accumulated_details["log"] = [f"Task completed successfully. Final status: {message}"]
                _main_task_accumulated_details.pop('log_storage_info', None)
            elif task_state in [TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]:
                current_log_list.append(log_entry)
                _main_task_accumulated_details["log"] = current_log_list
                if "error" not in _main_task_accumulated_details and task_state == TASK_STATUS_FAILURE:
                    _main_task_accumulated_details["error"] = message
            else:
                current_log_list.append(log_entry)
                _main_task_accumulated_details["log"] = current_log_list

            meta_details = {"status_message": message}
            for key in ["runs_completed", "total_runs", "best_score", "best_params", 
                        "clustering_run_job_ids", "batch_jobs_launched", "total_batch_jobs", "active_runs_count"]:
                if key in _main_task_accumulated_details:
                    meta_details[key] = _main_task_accumulated_details[key]
            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.meta['details'] = meta_details
                current_job.save_meta()
            save_task_status(current_task_id, "main_clustering", task_state, progress=progress, details=_main_task_accumulated_details)

        try:
            log_and_update_main_clustering("📊 Starting main clustering process...", 0)
            rows = get_all_tracks()
            
            min_tracks_for_kmeans = num_clusters_min if clustering_method == "kmeans" else 2
            min_tracks_for_gmm = gmm_n_components_min if clustering_method == "gmm" else 2
            min_req_overall = max(2, min_tracks_for_kmeans, min_tracks_for_gmm, (pca_components_min + 1) if pca_components_min > 0 else 2)

            if len(rows) < min_req_overall:
                err_msg = f"Not enough analyzed tracks ({len(rows)}) for clustering. Minimum required: {min_req_overall}."
                log_and_update_main_clustering(err_msg, 100, details_to_add_or_update={"error": "Insufficient data"}, task_state=TASK_STATUS_FAILURE)
                return {"status": "FAILURE", "message": err_msg}
            
            if num_clustering_runs == 0:
                log_and_update_main_clustering("Number of clustering runs is 0. Nothing to do.", 100, task_state=TASK_STATUS_SUCCESS)
                return {"status": "SUCCESS", "message": "Number of clustering runs was 0."}

            serializable_rows = [dict(row) for row in rows]
            all_tracks_data_json = json.dumps(serializable_rows)
            best_diversity_score = _main_task_accumulated_details.get("best_score", -1.0)
            best_clustering_results = None # Stores the full result dict of the best iteration
            elite_solutions_list = []      # List of {"score": float, "params": dict}

            mutation_config = {
                "int_abs_delta": MUTATION_INT_ABS_DELTA,
                "float_abs_delta": MUTATION_FLOAT_ABS_DELTA,
                "coord_mutation_fraction": MUTATION_KMEANS_COORD_FRACTION
            }
            mutation_config_json = json.dumps(mutation_config)
            exploitation_start_run_idx = int(num_clustering_runs * EXPLOITATION_START_FRACTION)
            all_launched_child_jobs_instances = [] 
            from app import rq_queue as main_rq_queue

            MAX_CONCURRENT_BATCH_JOBS = 10 
            ITERATIONS_PER_BATCH_JOB = 10 
            active_jobs_map = {}
            total_iterations_completed_count = 0
            next_batch_job_idx_to_launch = 0 
            num_total_batch_jobs = (num_clustering_runs + ITERATIONS_PER_BATCH_JOB - 1) // ITERATIONS_PER_BATCH_JOB
            _main_task_accumulated_details["total_batch_jobs"] = num_total_batch_jobs # Store total batches
            batches_completed_count = 0
            
            log_and_update_main_clustering(f"Fetched {len(rows)} tracks. Preparing {num_clustering_runs} runs in {num_total_batch_jobs} batches.", 5)


            while batches_completed_count < num_total_batch_jobs:
                if current_job:
                    with app.app_context():
                        main_task_db_info = get_task_info_from_db(current_task_id)
                        if main_task_db_info and main_task_db_info.get('status') == TASK_STATUS_REVOKED:
                            log_and_update_main_clustering(f"🛑 Main clustering task {current_task_id} REVOKED.", current_progress, task_state=TASK_STATUS_REVOKED)
                            return {"status": "REVOKED", "message": "Main clustering task revoked."}

                processed_in_this_cycle_ids = []
                for job_id, job_instance_active in list(active_jobs_map.items()):
                    is_child_truly_completed_this_cycle = False
                    try:
                        job_instance_active.refresh()
                        child_status_rq = job_instance_active.get_status()
                        if job_instance_active.is_finished or job_instance_active.is_failed or child_status_rq == JobStatus.CANCELED:
                            is_child_truly_completed_this_cycle = True
                    except NoSuchJobError:
                        with app.app_context(): db_task_info_child = get_task_info_from_db(job_id)
                        if db_task_info_child and db_task_info_child.get('status') in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED, JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED]:
                            is_child_truly_completed_this_cycle = True
                        else:
                            print(f"[MainClusteringTask-{current_task_id}] Warning: Active batch job {job_id} missing from RQ, not terminal in DB.")
                    except Exception as e_monitor_child_active:
                        print(f"[MainClusteringTask-{current_task_id}] ERROR monitoring active batch job {job_id}: {e_monitor_child_active}. Treating as completed.")
                        traceback.print_exc()
                        is_child_truly_completed_this_cycle = True
                    if is_child_truly_completed_this_cycle:
                        processed_in_this_cycle_ids.append(job_id)
                
                for job_id_processed in processed_in_this_cycle_ids:
                    if job_id_processed in active_jobs_map: del active_jobs_map[job_id_processed]
                newly_completed_elite_candidates = []

                if processed_in_this_cycle_ids:
                    for job_id_just_completed in processed_in_this_cycle_ids:
                        batch_job_result = get_job_result_safely(job_id_just_completed, current_task_id, "clustering_batch")
                        batches_completed_count += 1
                        if batch_job_result:
                            iterations_from_batch = batch_job_result.get("iterations_completed_in_batch", 0)
                            total_iterations_completed_count += iterations_from_batch
                            best_from_batch = batch_job_result.get("best_result_from_batch")
                            if best_from_batch and isinstance(best_from_batch, dict):
                                batch_best_score = best_from_batch.get("diversity_score", -1.0)
                                batch_best_params = best_from_batch.get("parameters")
                                if batch_best_score > -1.0 and batch_best_params:
                                    newly_completed_elite_candidates.append({"score": batch_best_score, "params": batch_best_params})
                                
                                if batch_best_score > best_diversity_score:
                                    best_diversity_score = batch_best_score
                                    _main_task_accumulated_details["best_score"] = best_diversity_score
                                    best_clustering_results = best_from_batch # Update overall best
                                    print(f"[MainClusteringTask-{current_task_id}] Intermediate new best score: {best_diversity_score:.2f} from batch job {job_id_just_completed}")
                        else:
                            print(f"[MainClusteringTask-{current_task_id}] Warning: Batch job {job_id_just_completed} completed but no result.")
                
                if newly_completed_elite_candidates:
                    all_potential_elites = elite_solutions_list + newly_completed_elite_candidates
                    all_potential_elites.sort(key=lambda x: x["score"], reverse=True)
                    # Simple top N, relying on run_id in params for distinctness of entries
                    # If multiple runs (different run_ids) yield the same params otherwise, they are treated as distinct elites.
                    # This is acceptable as the goal is to feed good parameter sets to mutation.
                    elite_solutions_list = all_potential_elites[:TOP_N_ELITES]


                # Launch new jobs if slots are available and more batches are pending
                can_launch_more_batch_jobs = next_batch_job_idx_to_launch < num_total_batch_jobs
                if can_launch_more_batch_jobs:
                    num_slots_to_fill = MAX_CONCURRENT_BATCH_JOBS - len(active_jobs_map)
                    for _ in range(num_slots_to_fill):
                        if next_batch_job_idx_to_launch >= num_total_batch_jobs: break 
                        
                        batch_job_task_id = str(uuid.uuid4())
                        current_batch_start_run_idx = next_batch_job_idx_to_launch * ITERATIONS_PER_BATCH_JOB
                        num_iterations_for_this_batch = min(ITERATIONS_PER_BATCH_JOB, num_clustering_runs - current_batch_start_run_idx)
                        if num_iterations_for_this_batch <= 0: # Should not happen if num_total_batch_jobs is correct
                            print(f"[MainClusteringTask-{current_task_id}] Warning: Calculated 0 iterations for batch {next_batch_job_idx_to_launch}. Skipping.")
                            next_batch_job_idx_to_launch +=1 # Ensure progress
                            continue


                        batch_id_for_logging = f"Batch_{next_batch_job_idx_to_launch}"
                        save_task_status(batch_job_task_id, "clustering_batch", "PENDING", parent_task_id=current_task_id, sub_type_identifier=batch_id_for_logging, 
                                         details={"start_run_idx": current_batch_start_run_idx, "num_iterations": num_iterations_for_this_batch})
                        
                        num_clusters_min_max_tuple_for_batch = (num_clusters_min, num_clusters_max)
                        dbscan_params_ranges_dict_for_batch = {"eps_min": dbscan_eps_min, "eps_max": dbscan_eps_max, "samples_min": dbscan_min_samples_min, "samples_max": dbscan_min_samples_max}
                        gmm_params_ranges_dict_for_batch = {"n_components_min": gmm_n_components_min, "n_components_max": gmm_n_components_max}
                        pca_params_ranges_dict_for_batch = {"components_min": pca_components_min, "components_max": pca_components_max}

                        current_elite_params_for_batch_json = json.dumps([item["params"] for item in elite_solutions_list]) if elite_solutions_list else "[]"
                        should_exploit_for_this_batch = (current_batch_start_run_idx >= exploitation_start_run_idx) and elite_solutions_list
                        exploitation_prob_for_this_batch = EXPLOITATION_PROBABILITY_CONFIG if should_exploit_for_this_batch else 0.0

                        
                        new_job = main_rq_queue.enqueue(
                            run_clustering_batch_task,
                            args=(
                                batch_id_for_logging, current_batch_start_run_idx, num_iterations_for_this_batch, all_tracks_data_json,
                                clustering_method, num_clusters_min_max_tuple_for_batch, dbscan_params_ranges_dict_for_batch, 
                                gmm_params_ranges_dict_for_batch, pca_params_ranges_dict_for_batch,
                                max_songs_per_cluster, current_task_id,
                                current_elite_params_for_batch_json,
                                exploitation_prob_for_this_batch,
                                mutation_config_json
                            ),
                            job_id=batch_job_task_id,
                            description=f"Clustering Batch {next_batch_job_idx_to_launch} (Runs {current_batch_start_run_idx}-{current_batch_start_run_idx + num_iterations_for_this_batch -1})",
                            job_timeout=3600 * (ITERATIONS_PER_BATCH_JOB / 2), 
                            meta={'parent_task_id': current_task_id}
                        )
                        active_jobs_map[new_job.id] = new_job
                        all_launched_child_jobs_instances.append(new_job)
                        _main_task_accumulated_details.setdefault("clustering_run_job_ids", []).append(new_job.id)
                        next_batch_job_idx_to_launch += 1

                launch_phase_max_progress = 5 
                execution_phase_max_progress = 80
                current_launch_progress = 0
                if num_total_batch_jobs > 0:
                    current_launch_progress = int(launch_phase_max_progress * (min(next_batch_job_idx_to_launch, num_total_batch_jobs) / float(num_total_batch_jobs)))
                current_execution_progress = 0
                if num_clustering_runs > 0:
                    current_execution_progress = int(execution_phase_max_progress * (total_iterations_completed_count / float(num_clustering_runs)))
                current_progress_val = 5 + current_launch_progress + current_execution_progress
                
                log_and_update_main_clustering(
                    f"Batch Jobs Launched: {next_batch_job_idx_to_launch}/{num_total_batch_jobs}. Active Batch Jobs: {len(active_jobs_map)}. Iterations Completed: {total_iterations_completed_count}/{num_clustering_runs}. Best Score: {best_diversity_score:.2f}",
                    current_progress_val,
                    details_to_add_or_update={
                        "runs_completed": total_iterations_completed_count,
                        "batch_jobs_launched": next_batch_job_idx_to_launch,
                        "active_runs_count": len(active_jobs_map),
                        "best_score": best_diversity_score
                    }
                )
                if batches_completed_count >= num_total_batch_jobs and not active_jobs_map: break 
                time.sleep(3)

            # Best result (best_clustering_results and best_diversity_score) is already tracked incrementally.
            # No need for a separate final aggregation loop over all_launched_child_jobs_instances.
            log_and_update_main_clustering("All clustering batch jobs completed. Finalizing best result...", 90,
                                           details_to_add_or_update={"best_score": best_diversity_score})

            if not best_clustering_results or best_diversity_score < 0:
                log_and_update_main_clustering("No valid clustering solution found after all runs.", 100, details_to_add_or_update={"error": "No suitable clustering found", "best_score": best_diversity_score}, task_state=TASK_STATUS_FAILURE)
                return {"status": "FAILURE", "message": "No valid clusters found after multiple runs."}

            current_progress = 92 
            log_and_update_main_clustering(f"Best clustering found with diversity score: {best_diversity_score:.2f}. Preparing to create playlists.", current_progress, details_to_add_or_update={"best_score": best_diversity_score, "best_params": best_clustering_results.get("parameters")})
            
            final_named_playlists = best_clustering_results["named_playlists"]
            final_playlist_centroids = best_clustering_results["playlist_centroids"]
            final_max_songs_per_cluster = best_clustering_results["parameters"]["max_songs_per_cluster"]
            
            # --- AI Playlist Naming for the BEST result ---
            log_prefix_main_task_ai = f"[MainClusteringTask-{current_task_id} AI Naming]"
            print(f"{log_prefix_main_task_ai} Checking AI Naming. use_ai_param={use_ai_playlist_naming_param}, ollama_url_param_is_set={bool(ollama_server_url_param)}")
            if use_ai_playlist_naming_param and ollama_server_url_param: # Use passed-in parameters
                print(f"{log_prefix_main_task_ai} AI Naming block entered. Attempting to import 'ai' module.")

                # Define the progress range for AI Naming
                ai_naming_start_progress = 70
                ai_naming_end_progress = 95
                ai_naming_progress_range = ai_naming_end_progress - ai_naming_start_progress # 3%

                # Set initial progress for AI naming phase *only if* AI is enabled
                current_progress = ai_naming_start_progress
                log_and_update_main_clustering(f"Preparing for AI Naming...", current_progress, print_console=True) # Initial log for AI phase
                try:
                    from ai import get_ollama_playlist_name, creative_prompt_template # Import the Ollama function
                    print(f"{log_prefix_main_task_ai} 'ai' module with Ollama function imported successfully.")
                    
                    ai_renamed_playlists_final = defaultdict(list)
                    ai_renamed_centroids_final = {}
                    total_playlists_to_name = len(final_named_playlists)
                    playlists_named_count = 0

                    for original_name, songs_in_playlist in final_named_playlists.items():
                        if not songs_in_playlist:
                            ai_renamed_playlists_final[original_name].extend(songs_in_playlist)
                            if original_name in final_playlist_centroids:
                                ai_renamed_centroids_final[original_name] = final_playlist_centroids[original_name]
                            continue

                        song_list_for_ai = [{'title': s_title, 'author': s_author} for _, s_title, s_author in songs_in_playlist]
                        name_parts = original_name.split('_')
                        
                        # Updated logic for feature1 and feature2 extraction
                        feature1 = "Unknown"
                        feature2 = "General" 
                        feature3 = "Music" # Default if not enough parts

                        if len(name_parts) >= 1:
                            feature1 = name_parts[0]
                        if len(name_parts) >= 2:
                            feature2 = name_parts[1]
                        if len(name_parts) >= 3 and name_parts[2].lower() not in ["slow", "medium", "fast"]: # Ensure 3rd part isn't tempo
                            feature3 = name_parts[2]
                        elif len(name_parts) >= 2 and name_parts[1].lower() not in ["slow", "medium", "fast"]: # If only 2 parts and 2nd isn't tempo, use it as f3
                            feature3 = name_parts[1] # Or keep as "Music" if you prefer a distinct 3rd feature always

                        # The prompt variable here is the fully formatted template with feature1, feature2, category_name.
                        # The get_openai_playlist_name function will take this and append the song list.
                        # So, we pass the template itself, and the individual components to the function for Ollama, using passed-in params.
                        print(f"{log_prefix_main_task_ai} Generating AI name for '{original_name}' ({len(song_list_for_ai)} songs) using model '{ollama_model_name_param}'. F1: '{feature1}', F2: '{feature2}', F3: '{feature3}'.")
                        ai_generated_name_str = get_ollama_playlist_name(
                            ollama_server_url_param, ollama_model_name_param, # Use passed-in parameters
                            creative_prompt_template, # Pass the base template
                            feature1, feature2, feature3, # Pass all three features
                            song_list_for_ai)

                        current_playlist_final_name = original_name
                        if ai_generated_name_str and not ai_generated_name_str.startswith("Error") and not ai_generated_name_str.startswith("An unexpected error"):
                            clean_ai_name = ai_generated_name_str.strip().replace("\n", " ")
                            if clean_ai_name:
                                # Check if the generated name is significantly different or just the original name
                                # This helps avoid unnecessary logging if AI just returns the input name
                                if clean_ai_name.lower() != original_name.lower().strip().replace("_", " "): # Simple check for difference
                                     print(f"{log_prefix_main_task_ai} AI: '{original_name}' -> '{clean_ai_name}'")
                                else:
                                     # print(f"{log_prefix_main_task_ai} AI returned name similar to original for '{original_name}'. Using original.")
                                     current_playlist_final_name = original_name # Explicitly keep original if AI name is same after cleaning
                            else:
                                print(f"{log_prefix_main_task_ai} AI for '{original_name}' returned empty after cleaning. Raw: '{ai_generated_name_str}'. Using original.")
                        else:
                            print(f"{log_prefix_main_task_ai} AI naming for '{original_name}' failed or returned error: '{ai_generated_name_str}'. Using original.")
                        
                        ai_renamed_playlists_final[current_playlist_final_name].extend(songs_in_playlist)
                        if original_name in final_playlist_centroids: # Keep original centroid data, just change key
                            ai_renamed_centroids_final[current_playlist_final_name] = final_playlist_centroids[original_name]
                            
                        # Update progress after processing each playlist
                        playlists_named_count += 1
                        if total_playlists_to_name > 0:
                            current_ai_progress = ai_naming_start_progress + (playlists_named_count / total_playlists_to_name) * ai_naming_progress_range
                            log_and_update_main_clustering(
                                f"AI Naming: Processed {playlists_named_count}/{total_playlists_to_name} playlists.",
                                int(current_ai_progress), # Progress should be an integer
                                print_console=False # Avoid spamming console for each playlist
                            )
                    
                    final_named_playlists = ai_renamed_playlists_final
                    final_playlist_centroids = ai_renamed_centroids_final
                    log_and_update_main_clustering(f"{log_prefix_main_task_ai} AI Naming for best playlist set completed.", ai_naming_end_progress, print_console=True)
                    # After successful AI naming, progress is now at 95.
                    # The next step is database update, which starts at 95. This is correct.
                except ImportError:
                    print(f"{log_prefix_main_task_ai} Could not import 'ai' module. Skipping AI naming for final playlists.")
                    traceback.print_exc()
                    # If AI fails, progress should jump from 70 (or wherever it was) to 95 (start of DB update)
                    current_progress = ai_naming_end_progress # Set progress to 95
                    log_and_update_main_clustering(f"{log_prefix_main_task_ai} AI Naming skipped due to import error.", current_progress, print_console=True)
                except Exception as e_ai_final:
                    print(f"{log_prefix_main_task_ai} Error during final AI playlist naming: {e_ai_final}. Using original names.")
                    traceback.print_exc()
                    # If AI fails, progress should jump from 70 (or wherever it was) to 95 (start of DB update)
                    current_progress = ai_naming_end_progress # Set progress to 95
                    log_and_update_main_clustering(f"{log_prefix_main_task_ai} AI Naming skipped due to error.", current_progress, print_console=True)
            else:
                if not ollama_server_url_param: print(f"{log_prefix_main_task_ai} AI Naming skipped: Ollama Server URL (param) not set.")
                elif not use_ai_playlist_naming_param: print(f"{log_prefix_main_task_ai} AI Naming skipped: Use AI Naming (param) is False.")
            # --- End AI Playlist Naming for BEST result ---
            
            current_progress = 95
            log_and_update_main_clustering("Updating playlist database...", current_progress, print_console=False)
            update_playlist_table(final_named_playlists) 
            
            current_progress = 98
            log_and_update_main_clustering("Creating/Updating playlists on Jellyfin...", current_progress, print_console=False)
            create_or_update_playlists_on_jellyfin(JELLYFIN_URL, JELLYFIN_USER_ID, {"X-Emby-Token": JELLYFIN_TOKEN}, final_named_playlists, final_playlist_centroids, MOOD_LABELS, final_max_songs_per_cluster)
            
            final_db_summary = {
                "best_score": best_diversity_score, 
                "best_params": best_clustering_results.get("parameters"),
                "num_playlists_created": len(final_named_playlists)
            }
            current_progress = 100 
            log_and_update_main_clustering(f"Playlists generated and updated on Jellyfin! Best diversity score: {best_diversity_score:.2f}.", current_progress, details_to_add_or_update=final_db_summary, task_state=TASK_STATUS_SUCCESS)
            
            return {"status": "SUCCESS", "message": f"Playlists generated and updated on Jellyfin! Best run had diversity score of {best_diversity_score:.2f}."}

        except Exception as e:
            error_tb = traceback.format_exc()
            print(f"FATAL ERROR: Clustering failed: {e}\n{error_tb}")
            with app.app_context():
                log_and_update_main_clustering(f"❌ Main clustering failed: {e}", current_progress, details_to_add_or_update={"error_message": str(e), "traceback": error_tb}, task_state=TASK_STATUS_FAILURE)
                if 'all_launched_child_jobs_instances' in locals():
                    print(f"[MainClusteringTask-{current_task_id}] Parent task failed. Attempting to mark {len(all_launched_child_jobs_instances)} children (batch jobs) as REVOKED.")
                    for child_job_instance in all_launched_child_jobs_instances:
                        try:
                            if not child_job_instance: continue 
                            with app.app_context():
                                child_db_info = get_task_info_from_db(child_job_instance.id)
                                if child_db_info and child_db_info.get('status') not in [TASK_STATUS_REVOKED, TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, JobStatus.CANCELED]:
                                    save_task_status(child_job_instance.id, "clustering_batch", TASK_STATUS_REVOKED, parent_task_id=current_task_id, progress=100, details={"message": "Parent task failed.", "log": ["Parent task failed."]})
                        except Exception as e_cancel_child_on_fail:
                            print(f"[MainClusteringTask-{current_task_id}] Error marking child batch job {child_job_instance.id} as REVOKED: {e_cancel_child_on_fail}")
            raise
        finally:
            if 'all_tracks_data_json' in locals(): 
                try: del all_tracks_data_json
                except NameError: pass

# --- Helper to get job result safely from RQ or DB ---
def get_job_result_safely(job_id_for_result, parent_task_id_for_logging, task_type_for_logging="child task"):
    """
    Safely retrieves the result of a job, checking RQ and then the database.
    Assumes child tasks store their full result in details['full_result'] if successful (for single runs)
    or details['full_best_result_from_batch'] for batch tasks.
    """
    run_result_data = None
    job_instance = None
    try:
        job_instance = Job.fetch(job_id_for_result, connection=redis_conn)
        job_instance.refresh() 
        if job_instance.is_finished and isinstance(job_instance.result, dict):
            run_result_data = job_instance.result
    except NoSuchJobError:
        print(f"[ParentTask-{parent_task_id_for_logging}] Warning: {task_type_for_logging} {job_id_for_result} not found in RQ. Checking DB.")
    except Exception as e_rq_fetch:
        print(f"[ParentTask-{parent_task_id_for_logging}] Error fetching {task_type_for_logging} {job_id_for_result} from RQ: {e_rq_fetch}. Will check DB.")

    if run_result_data is None: 
        with app.app_context(): 
            db_task_info = get_task_info_from_db(job_id_for_result)
        if db_task_info:
            db_status = db_task_info.get('status')
            if db_status in [TASK_STATUS_SUCCESS, JobStatus.FINISHED]: 
                if db_task_info.get('details'):
                    try:
                        details_dict = json.loads(db_task_info.get('details'))
                        # Check for batch result structure first, then single run structure
                        if 'full_best_result_from_batch' in details_dict:
                             # This is the structure returned by run_clustering_batch_task in its DB details
                            run_result_data = {"status": "SUCCESS", 
                                               "iterations_completed_in_batch": details_dict.get("iterations_completed_in_batch", 0),
                                               "best_result_from_batch": details_dict.get("full_best_result_from_batch")}
                        elif 'full_result' in details_dict: # Fallback for old single run tasks if any
                            run_result_data = details_dict['full_result']
                        else:
                            print(f"[ParentTask-{parent_task_id_for_logging}] Warning: {task_type_for_logging} {job_id_for_result} (DB status: {db_status}) has no 'full_best_result_from_batch' or 'full_result' in details.")
                    except (json.JSONDecodeError, TypeError) as e_json:
                        print(f"[ParentTask-{parent_task_id_for_logging}] Warning: Could not parse details for {task_type_for_logging} {job_id_for_result} from DB: {e_json}")
            elif db_status in [TASK_STATUS_FAILURE, TASK_STATUS_REVOKED, JobStatus.CANCELED, JobStatus.FAILED]:
                print(f"[ParentTask-{parent_task_id_for_logging}] Info: {task_type_for_logging} {job_id_for_result} (DB status: {db_status}) did not succeed. No result to process.")
    return run_result_data
