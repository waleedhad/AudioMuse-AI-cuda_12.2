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
    JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN # For create_or_update_playlists_on_jellyfin
from rq.exceptions import NoSuchJobError

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

def run_single_clustering_iteration_task(run_id, all_tracks_data_json, clustering_method_config, pca_config, max_songs_per_cluster, parent_task_id):
    """RQ task for a single clustering iteration."""
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())
    
    with app.app_context():
        initial_log_message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Single clustering run {run_id} started."
        initial_details = {"run_id": run_id, "params": clustering_method_config, "pca_config": pca_config, "log": [initial_log_message]}
        save_task_status(current_task_id, "single_clustering_run", TASK_STATUS_STARTED, parent_task_id=parent_task_id, sub_type_identifier=str(run_id), progress=0, details=initial_details)
        current_progress_iter = 0
        current_task_logs = initial_details["log"]

        def log_and_update_single_run(message, progress, details_extra=None, task_state=TASK_STATUS_PROGRESS, print_console=True):
            nonlocal current_progress_iter
            nonlocal current_task_logs
            current_progress_iter = progress
            if print_console:
                print(f"[SingleClusteringRun-{current_task_id}] Run {run_id}: {message}")

            current_run_details_for_db = {"run_id": run_id, "params": clustering_method_config, "pca_config": pca_config}
            if details_extra:
                current_run_details_for_db.update(details_extra)
            current_run_details_for_db["status_message"] = message

            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            if task_state == TASK_STATUS_SUCCESS:
                current_run_details_for_db["log"] = [f"Task completed successfully. Final status: {message}"]
                current_run_details_for_db.pop('log_storage_info', None)
            elif task_state == TASK_STATUS_FAILURE or task_state == TASK_STATUS_REVOKED:
                current_task_logs.append(log_entry)
                current_run_details_for_db["log"] = current_task_logs
                if "error" not in current_run_details_for_db and task_state == TASK_STATUS_FAILURE : current_run_details_for_db["error"] = message
            else: # PROGRESS or STARTED
                current_task_logs.append(log_entry)
                current_run_details_for_db["log"] = current_task_logs

            meta_details = {"run_id": run_id, "status_message": message} # Leaner for RQ
            if details_extra and "diversity_score" in details_extra: meta_details["diversity_score"] = details_extra["diversity_score"]
            
            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.meta['details'] = meta_details
                current_job.save_meta()
            save_task_status(current_task_id, "single_clustering_run", task_state, parent_task_id=parent_task_id, sub_type_identifier=str(run_id), progress=progress, details=current_run_details_for_db)

        try:
            log_and_update_single_run(f"Starting with method {clustering_method_config['method']}, params: {clustering_method_config['params']}, PCA: {pca_config}", 0)
            # === Cooperative Cancellation Check for Single Clustering Run Task ===
            if current_job:
                with app.app_context():
                    task_db_info = get_task_info_from_db(current_task_id)
                    parent_task_db_info = get_task_info_from_db(parent_task_id) if parent_task_id else None
                    
                    is_self_revoked = task_db_info and task_db_info.get('status') == 'REVOKED'
                    is_parent_failed_or_revoked = parent_task_db_info and parent_task_db_info.get('status') in [TASK_STATUS_REVOKED, TASK_STATUS_FAILURE]

                    if is_self_revoked or is_parent_failed_or_revoked:
                        parent_status_for_reason = parent_task_db_info.get('status') if parent_task_db_info else "N/A"
                        revocation_reason = "self was REVOKED" if is_self_revoked else f"parent task {parent_task_id} status is {parent_status_for_reason}"
                        log_and_update_single_run(f"🛑 Clustering run {run_id} (Task ID: {current_task_id}) stopping because {revocation_reason}.", current_progress_iter, task_state=TASK_STATUS_REVOKED)
                        return {"status": "REVOKED", "message": f"Clustering run {run_id} stopped because {revocation_reason}."}
            # === End Cooperative Cancellation Check ===

            all_tracks_data = json.loads(all_tracks_data_json)
            
            # Convert dicts back to pseudo-DictRow objects if score_vector expects attribute access
            # Or modify score_vector to accept dicts. Assuming score_vector is adapted for dicts.
            rows = all_tracks_data
            X_original = [score_vector(row, MOOD_LABELS) for row in rows]
            del rows # Potentially free memory from the list of dicts
            X_scaled = np.array(X_original)
            del X_original # Potentially free memory from the list of lists
            data_for_clustering = X_scaled
            pca_model = None
            n_components_actual = 0

            if pca_config["enabled"]:
                # Use len(all_tracks_data) as rows is deleted
                n_components_actual = min(pca_config["components"], X_scaled.shape[1], (len(all_tracks_data) - 1) if len(all_tracks_data) > 1 else 1)
                if n_components_actual > 0:
                    pca_model = PCA(n_components=n_components_actual)
                    data_for_clustering = pca_model.fit_transform(X_scaled)
                    # X_scaled might still be needed if pca_model.inverse_transform uses it or if not all code paths redefine data_for_clustering
                else:
                    pca_config["enabled"] = False # Not enough features/samples
            log_and_update_single_run(f"PCA {'enabled with ' + str(n_components_actual) + ' components' if pca_config['enabled'] else 'disabled'}.", 25, print_console=False)

            labels = None
            cluster_centers_map = {}
            raw_distances = np.zeros(len(data_for_clustering))
            method = clustering_method_config["method"]
            params = clustering_method_config["params"]

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
                    cluster_points = data_for_clustering[indices]
                    if len(cluster_points) > 0:
                        center = cluster_points.mean(axis=0)
                        for i in indices: raw_distances[i] = np.linalg.norm(data_for_clustering[i] - center)
                        cluster_centers_map[cluster_id] = center
            elif method == "gmm":
                gmm = GaussianMixture(n_components=params["n_components"], covariance_type=GMM_COVARIANCE_TYPE, random_state=None, max_iter=1000)
                gmm.fit(data_for_clustering)
                labels = gmm.predict(data_for_clustering)
                cluster_centers_map = {i: gmm.means_[i] for i in range(params["n_components"])}
                centers_for_points = gmm.means_[labels] # type: ignore
                raw_distances = np.linalg.norm(data_for_clustering - centers_for_points, axis=1)
            log_and_update_single_run("Clustering algorithm applied.", 50, print_console=False)
            # ... after clustering algorithm applied and labels are generated ...
            del data_for_clustering # This was the main input to the clustering algorithm
            
            if labels is None or len(set(labels) - {-1}) == 0:
                log_and_update_single_run("No valid clusters found.", 100, details_extra={"diversity_score": -1.0, "final_result_summary": "No clusters"}, task_state=TASK_STATUS_SUCCESS)
                return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "parameters": {"clustering_method_config": clustering_method_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster}}

            max_dist = raw_distances.max()
            normalized_distances = raw_distances / max_dist if max_dist > 0 else raw_distances
            track_info_list = [{"row": all_tracks_data[i], "label": labels[i], "distance": normalized_distances[i]} for i in range(len(all_tracks_data))]

            filtered_clusters = defaultdict(list)
            for cid in set(labels):
                if cid == -1: continue
                cluster_tracks = [t for t in track_info_list if t["label"] == cid and t["distance"] <= MAX_DISTANCE]
                if not cluster_tracks: continue
                cluster_tracks.sort(key=lambda x: x["distance"])
                
                count_per_artist = defaultdict(int)
                selected_tracks = []
                for t_item in cluster_tracks:
                    author = t_item["row"]["author"] # Access as dict
                    if count_per_artist[author] < MAX_SONGS_PER_ARTIST:
                        selected_tracks.append(t_item)
                        count_per_artist[author] += 1
                    if len(selected_tracks) >= max_songs_per_cluster: break
                for t_item in selected_tracks:
                    item_id, title, author = t_item["row"]["item_id"], t_item["row"]["title"], t_item["row"]["author"]
                    filtered_clusters[cid].append((item_id, title, author))
            log_and_update_single_run("Tracks filtered into clusters.", 65, print_console=False)

            current_named_playlists = defaultdict(list)
            current_playlist_centroids = {}
            unique_predominant_mood_scores = {}

            for label_val, songs_list in filtered_clusters.items():
                if songs_list:
                    center_val = cluster_centers_map.get(label_val)
                    if center_val is None: continue
                    name, top_scores = name_cluster(center_val, pca_model, pca_config["enabled"], MOOD_LABELS)
                    if top_scores and any(mood in MOOD_LABELS for mood in top_scores.keys()):
                        predominant_mood_key = max((k for k in top_scores if k in MOOD_LABELS), key=top_scores.get, default=None)
                        if predominant_mood_key:
                            current_mood_score = top_scores.get(predominant_mood_key, 0.0)
                            unique_predominant_mood_scores[predominant_mood_key] = max(unique_predominant_mood_scores.get(predominant_mood_key, 0.0), current_mood_score)
                    current_named_playlists[name].extend(songs_list)
                    current_playlist_centroids[name] = top_scores
            
            diversity_score = sum(unique_predominant_mood_scores.values())
            log_and_update_single_run(f"Named {len(current_named_playlists)} playlists. Diversity score: {diversity_score:.2f}", 75)
            
            pca_model_details = {"n_components": pca_model.n_components_, "explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(), "mean": pca_model.mean_.tolist()} if pca_model and pca_config["enabled"] else None
            result = {
                "diversity_score": float(diversity_score),
                "named_playlists": dict(current_named_playlists), 
                "playlist_centroids": current_playlist_centroids,
                "pca_model_details": pca_model_details, 
                "parameters": {"clustering_method_config": clustering_method_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_id}
            }
            final_summary_db = {
                "diversity_score": diversity_score, 
                "final_result_summary": f"Playlists: {len(result['named_playlists'])}, Diversity: {result['diversity_score']:.2f}",
                "full_result": result # Save the full result in DB details for the main clustering task to pick up
            }
            log_and_update_single_run(
                "Iteration complete.", 100, 
                details_extra=final_summary_db, task_state=TASK_STATUS_SUCCESS)
            return result
        except Exception as e:
            error_tb = traceback.format_exc()
            failure_details = {"error": str(e), "traceback": error_tb, "run_id": run_id}
            log_and_update_single_run(f"Failed clustering iteration {run_id}: {e}", current_progress_iter, details_extra=failure_details, task_state='FAILURE')
            print(f"ERROR: Clustering iteration {run_id} failed: {e}\n{error_tb}")
            raise

def run_clustering_task(clustering_method, num_clusters_min, num_clusters_max, dbscan_eps_min, dbscan_eps_max, dbscan_min_samples_min, dbscan_min_samples_max, pca_components_min, pca_components_max, num_clustering_runs, max_songs_per_cluster, gmm_n_components_min, gmm_n_components_max):
    """Main RQ task for clustering and playlist generation."""
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        initial_log_message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Main clustering task started."
        # _main_task_accumulated_details will hold all details that need to be persisted.
        # Initialize it with values that are known at the start or will be updated.
        _main_task_accumulated_details = {
            "message": "Initializing clustering...",
            "log": [initial_log_message],
            "total_runs": num_clustering_runs, # Known at start
            "runs_completed": 0,             # Will be incremented
            "runs_launched": 0,              # Will be incremented
            "active_runs_count": 0,          # Will be updated
            "best_score": -1.0,              # Default best score
            "clustering_run_job_ids": []     # List of child job IDs
        }
        save_task_status(current_task_id, "main_clustering", TASK_STATUS_STARTED, progress=0, details=_main_task_accumulated_details)
        
        current_progress = 0
        # current_task_logs is now part of _main_task_accumulated_details['log']

        def log_and_update_main_clustering(message, progress, details_to_add_or_update=None, task_state=TASK_STATUS_PROGRESS, print_console=True):
            nonlocal current_progress
            nonlocal _main_task_accumulated_details # Directly modify the shared details dictionary
            
            current_progress = progress
            if print_console:
                print(f"[MainClusteringTask-{current_task_id}] {message}")

            # Update the accumulated details with any new information
            if details_to_add_or_update:
                _main_task_accumulated_details.update(details_to_add_or_update)
            
            # Always ensure the current status message is set in the accumulated details
            _main_task_accumulated_details["status_message"] = message

            # Manage the log list within accumulated_details
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            current_log_list = _main_task_accumulated_details.get("log", [])
            
            if task_state == TASK_STATUS_SUCCESS:
                # For success, replace log with a final message
                _main_task_accumulated_details["log"] = [f"Task completed successfully. Final status: {message}"]
                _main_task_accumulated_details.pop('log_storage_info', None) # Clean up truncation info
            elif task_state == TASK_STATUS_FAILURE or task_state == TASK_STATUS_REVOKED:
                current_log_list.append(log_entry) # Add the final error/revoked message
                _main_task_accumulated_details["log"] = current_log_list # Keep all logs
                if "error" not in _main_task_accumulated_details and task_state == TASK_STATUS_FAILURE:
                    _main_task_accumulated_details["error"] = message # Ensure error message is captured
            else: # PROGRESS or STARTED
                current_log_list.append(log_entry)
                _main_task_accumulated_details["log"] = current_log_list

            # Prepare details for RQ job.meta (can be a leaner version if needed)
            meta_details = {"status_message": message}
            # Populate meta_details from _main_task_accumulated_details for specific keys
            for key in ["runs_completed", "total_runs", "best_score", "best_params", "clustering_run_job_ids", "runs_launched", "active_runs_count"]:
                if key in _main_task_accumulated_details:
                    meta_details[key] = _main_task_accumulated_details[key]

            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.meta['details'] = meta_details # Persist to RQ meta
                current_job.save_meta()
            
            # Save the full _main_task_accumulated_details to the database
            save_task_status(current_task_id, "main_clustering", task_state, progress=progress, details=_main_task_accumulated_details)

        try:
            log_and_update_main_clustering("📊 Starting main clustering process...", 0)
            rows = get_all_tracks()
            
            min_tracks_for_kmeans = num_clusters_min if clustering_method == "kmeans" else 2
            min_tracks_for_gmm = gmm_n_components_min if clustering_method == "gmm" else 2
            min_req_overall = max(2, min_tracks_for_kmeans, min_tracks_for_gmm, (pca_components_min + 1) if pca_components_min > 0 else 2)

            if len(rows) < min_req_overall:
                err_msg = f"Not enough analyzed tracks ({len(rows)}) for clustering. Minimum required for selected parameters: {min_req_overall}."
                log_and_update_main_clustering(err_msg, 100, details_to_add_or_update={"error": "Insufficient data"}, task_state=TASK_STATUS_FAILURE)
                return {"status": "FAILURE", "message": err_msg}
            
            if num_clustering_runs == 0:
                log_and_update_main_clustering("Number of clustering runs is 0. Nothing to do.", 100, task_state=TASK_STATUS_SUCCESS)
                return {"status": "SUCCESS", "message": "Number of clustering runs was 0."}

            # Initial status update after fetching tracks and basic checks
            log_and_update_main_clustering(f"Fetched {len(rows)} tracks for clustering. Preparing {num_clustering_runs} runs.", 5)

            serializable_rows = [dict(row) for row in rows]
            all_tracks_data_json = json.dumps(serializable_rows)

            best_diversity_score = _main_task_accumulated_details.get("best_score", -1.0)
            best_clustering_results = None
            
            # This list will store all Job objects for later result fetching
            all_launched_child_jobs_instances = [] 
            
            from app import rq_queue as main_rq_queue

            MAX_CONCURRENT_CLUSTERING_RUNS = 10 
            active_jobs_map = {} # {job_id: job_instance}
            
            runs_completed_count = 0
            next_run_idx_to_launch = 0

            while runs_completed_count < num_clustering_runs:
                if current_job:
                    with app.app_context():
                        main_task_db_info = get_task_info_from_db(current_task_id)
                        if main_task_db_info and main_task_db_info.get('status') == TASK_STATUS_REVOKED:
                            log_and_update_main_clustering(f"🛑 Main clustering task {current_task_id} REVOKED. Stopping.", current_progress, task_state=TASK_STATUS_REVOKED)
                            # Child cancellation will be handled by children checking parent status or by a dedicated cancel call
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
                            print(f"[MainClusteringTask-{current_task_id}] Warning: Active job {job_id} missing from RQ and not terminal in DB. Will re-check.")
                    except Exception as e_monitor_child_active:
                        print(f"[MainClusteringTask-{current_task_id}] ERROR monitoring active child job {job_id}: {e_monitor_child_active}. Treating as completed.")
                        traceback.print_exc()
                        is_child_truly_completed_this_cycle = True
                    
                    if is_child_truly_completed_this_cycle:
                        processed_in_this_cycle_ids.append(job_id)
                        runs_completed_count +=1 
                
                for job_id_processed in processed_in_this_cycle_ids:
                    if job_id_processed in active_jobs_map:
                        del active_jobs_map[job_id_processed]

                can_launch_more_runs = next_run_idx_to_launch < num_clustering_runs
                if can_launch_more_runs:
                    num_slots_to_fill = MAX_CONCURRENT_CLUSTERING_RUNS - len(active_jobs_map)
                    for _ in range(num_slots_to_fill):
                        if next_run_idx_to_launch >= num_clustering_runs:
                            break 
                        
                        run_idx = next_run_idx_to_launch
                        
                        method_params = {}
                        if clustering_method == "kmeans":
                            current_num_clusters = random.randint(num_clusters_min, min(num_clusters_max, len(rows)))
                            method_params = {"method": "kmeans", "params": {"n_clusters": max(1, current_num_clusters)}}
                        elif clustering_method == "dbscan":
                            current_dbscan_eps = round(random.uniform(dbscan_eps_min, dbscan_eps_max), 2)
                            current_dbscan_min_samples = random.randint(dbscan_min_samples_min, dbscan_min_samples_max)
                            method_params = {"method": "dbscan", "params": {"eps": current_dbscan_eps, "min_samples": current_dbscan_min_samples}}
                        elif clustering_method == "gmm":
                            current_gmm_n_components = random.randint(gmm_n_components_min, min(gmm_n_components_max, len(rows)))
                            method_params = {"method": "gmm", "params": {"n_components": max(1, current_gmm_n_components)}}
                        else: # Should be validated by API
                            log_and_update_main_clustering(f"Unsupported clustering algorithm: {clustering_method}", 100, details_to_add_or_update={"error": "Unsupported algorithm"}, task_state=TASK_STATUS_FAILURE)
                            return {"status": "FAILURE", "message": f"Unsupported clustering algorithm: {clustering_method}"}

                        sampled_pca_components = random.randint(pca_components_min, pca_components_max)
                        max_allowable_pca = min(sampled_pca_components, len(MOOD_LABELS) + 1, len(rows) -1 if len(rows) > 1 else 1)
                        pca_config = {"enabled": max_allowable_pca > 0, "components": max_allowable_pca}
                        
                        new_sub_task_id = str(uuid.uuid4())
                        save_task_status(new_sub_task_id, "single_clustering_run", "PENDING", parent_task_id=current_task_id, sub_type_identifier=str(run_idx), details={"params": method_params, "pca_config": pca_config})
                        
                        new_job = main_rq_queue.enqueue(
                            run_single_clustering_iteration_task,
                            args=(run_idx, all_tracks_data_json, method_params, pca_config, max_songs_per_cluster, current_task_id),
                            job_id=new_sub_task_id,
                            description=f"Clustering Iteration {run_idx}",
                            job_timeout=3600,
                            meta={'parent_task_id': current_task_id}
                        )
                        active_jobs_map[new_job.id] = new_job
                        all_launched_child_jobs_instances.append(new_job)
                        _main_task_accumulated_details.setdefault("clustering_run_job_ids", []).append(new_job.id)
                        next_run_idx_to_launch += 1

                launch_phase_max_progress = 5 
                execution_phase_max_progress = 80
                
                current_launch_progress = 0
                if num_clustering_runs > 0:
                    current_launch_progress = int(launch_phase_max_progress * (min(next_run_idx_to_launch, num_clustering_runs) / float(num_clustering_runs)))
                
                current_execution_progress = 0
                if num_clustering_runs > 0:
                    current_execution_progress = int(execution_phase_max_progress * (runs_completed_count / float(num_clustering_runs)))
                    
                current_progress_val = 5 + current_launch_progress + current_execution_progress
                
                log_and_update_main_clustering(
                    f"Launched: {next_run_idx_to_launch}/{num_clustering_runs}. Active: {len(active_jobs_map)}. Completed: {runs_completed_count}/{num_clustering_runs}.",
                    current_progress_val,
                    details_to_add_or_update={
                        "runs_completed": runs_completed_count,
                        "runs_launched": next_run_idx_to_launch,
                        "active_runs_count": len(active_jobs_map)
                        # "clustering_run_job_ids" is already updated in _main_task_accumulated_details
                    }
                )

                if runs_completed_count >= num_clustering_runs and not active_jobs_map:
                    break 
                
                time.sleep(3)

            log_and_update_main_clustering("All clustering runs completed. Aggregating results...", 90)
            
            for job_instance_result_phase in all_launched_child_jobs_instances:
                if current_job: # Cooperative cancellation check during result aggregation
                    with app.app_context():
                        main_task_db_info = get_task_info_from_db(current_task_id)
                        if main_task_db_info and main_task_db_info.get('status') == TASK_STATUS_REVOKED:
                            log_and_update_main_clustering(f"🛑 Main clustering task {current_task_id} REVOKED during result aggregation.", current_progress, task_state=TASK_STATUS_REVOKED)
                            return {"status": "REVOKED", "message": "Main clustering task revoked during final result aggregation."}

                run_result_data = get_job_result_safely(job_instance_result_phase.id, current_task_id, "single_clustering_run")
                
                try:
                    if run_result_data:
                        current_diversity_score = run_result_data.get("diversity_score", -1.0)
                        if current_diversity_score > best_diversity_score:
                            best_diversity_score = current_diversity_score
                            best_clustering_results = run_result_data
                            log_and_update_main_clustering(f"New best clustering iteration found (Run ID: {run_result_data.get('parameters', {}).get('run_id', 'N/A')}, Diversity: {current_diversity_score:.2f})", current_progress, details_to_add_or_update={"best_score": best_diversity_score})
                    else: 
                        failed_job_id = job_instance_result_phase.id if job_instance_result_phase else "Unknown_ID"
                        job_status_for_failed_log = "UNKNOWN (Result not found)"
                        error_info_str = "No specific error info retrieved."
                        try: 
                            job_instance_result_phase.refresh() 
                            job_status_for_failed_log = job_instance_result_phase.get_status()
                            if job_instance_result_phase.is_failed and job_instance_result_phase.exc_info:
                                error_info_str = str(job_instance_result_phase.exc_info).strip().split('\n')[-1]
                        except NoSuchJobError:
                            with app.app_context():
                                db_info_for_log = get_task_info_from_db(failed_job_id) # type: ignore
                                job_status_for_failed_log = f"DB:{db_info_for_log.get('status')}" if db_info_for_log else "MISSING_FROM_RQ_AND_DB"
                                if db_info_for_log and db_info_for_log.get('details'):
                                    try:
                                        child_details = json.loads(db_info_for_log['details'])
                                        if 'error' in child_details: error_info_str = f"DB error: {child_details['error']}"
                                        elif 'log' in child_details and child_details['log']:
                                            for log_line in reversed(child_details['log']):
                                                if "error" in log_line.lower() or "fail" in log_line.lower():
                                                    error_info_str = f"DB log hint: {log_line}"; break
                                    except Exception: pass
                        log_and_update_main_clustering(
                            f"Clustering run task ({failed_job_id}) did not yield a usable result (status: {job_status_for_failed_log}). Error hint: {error_info_str}",
                            current_progress
                        )
                except Exception as e_final_result_cluster:
                    job_id_for_error = job_instance_result_phase.id if job_instance_result_phase else "Unknown_ID"
                    print(f"[MainClusteringTask-{current_task_id}] ERROR processing final result for child clustering job {job_id_for_error}: {e_final_result_cluster}. This run will not be considered.")
                    traceback.print_exc()

            if not best_clustering_results or best_diversity_score < 0:
                log_and_update_main_clustering("No valid clustering solution found after all runs.", 100, details_to_add_or_update={"error": "No suitable clustering found", "best_score": best_diversity_score}, task_state=TASK_STATUS_FAILURE)
                return {"status": "FAILURE", "message": "No valid clusters found after multiple runs."}

            log_and_update_main_clustering(f"Best clustering found with diversity score: {best_diversity_score:.2f}.", 90, details_to_add_or_update={"best_score": best_diversity_score, "best_params": best_clustering_results.get("parameters")})
            
            final_named_playlists = best_clustering_results["named_playlists"]
            final_playlist_centroids = best_clustering_results["playlist_centroids"]
            final_max_songs_per_cluster = best_clustering_results["parameters"]["max_songs_per_cluster"]

            log_and_update_main_clustering("Updating playlist database...", 95, print_console=False)
            update_playlist_table(final_named_playlists) 
            
            log_and_update_main_clustering("Creating/Updating playlists on Jellyfin...", 98, print_console=False)
            create_or_update_playlists_on_jellyfin(JELLYFIN_URL, JELLYFIN_USER_ID, {"X-Emby-Token": JELLYFIN_TOKEN}, final_named_playlists, final_playlist_centroids, MOOD_LABELS, final_max_songs_per_cluster)
            
            final_db_summary = {
                "best_score": best_diversity_score, 
                "best_params": best_clustering_results.get("parameters"),
                "num_playlists_created": len(final_named_playlists)
            }
            log_and_update_main_clustering(f"Playlists generated and updated on Jellyfin! Best diversity score: {best_diversity_score:.2f}.", 100, details_to_add_or_update=final_db_summary, task_state=TASK_STATUS_SUCCESS)
            
            return {"status": "SUCCESS", "message": f"Playlists generated and updated on Jellyfin! Best run had diversity score of {best_diversity_score:.2f}."}

        except Exception as e:
            error_tb = traceback.format_exc()
            print(f"FATAL ERROR: Clustering failed: {e}\n{error_tb}")
            with app.app_context():
                log_and_update_main_clustering(f"❌ Main clustering failed: {e}", current_progress, details_to_add_or_update={"error_message": str(e), "traceback": error_tb}, task_state=TASK_STATUS_FAILURE)
                if 'all_launched_child_jobs_instances' in locals():
                    print(f"[MainClusteringTask-{current_task_id}] Parent task failed. Attempting to mark {len(all_launched_child_jobs_instances)} children as REVOKED.")
                    for child_job_instance in all_launched_child_jobs_instances:
                        try:
                            if not child_job_instance: continue 
                            with app.app_context():
                                child_db_info = get_task_info_from_db(child_job_instance.id)
                                if child_db_info and child_db_info.get('status') not in [TASK_STATUS_REVOKED, TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, JobStatus.CANCELED]:
                                    save_task_status(child_job_instance.id, "single_clustering_run", TASK_STATUS_REVOKED, parent_task_id=current_task_id, progress=100, details={"message": "Parent task failed.", "log": ["Parent task failed."]})
                        except Exception as e_cancel_child_on_fail:
                            print(f"[MainClusteringTask-{current_task_id}] Error marking child job {child_job_instance.id} as REVOKED during parent failure: {e_cancel_child_on_fail}")
            raise
        finally:
            if 'all_tracks_data_json' in locals(): # Check if defined in local scope
                try:
                    del all_tracks_data_json
                except NameError: # Should not happen if 'in locals()' check passes but good for safety
                    pass
