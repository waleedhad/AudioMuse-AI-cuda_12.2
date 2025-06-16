# tasks/analysis.py

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


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score # type: ignore
from sklearn.preprocessing import StandardScaler

# Essentia imports
from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor, TensorflowPredictMusiCNN, TensorflowPredict2D, Energy

# RQ import
from rq import get_current_job
from rq.job import Job # Import Job class
from rq.exceptions import NoSuchJobError, InvalidJobOperation

# Import necessary components from the main app.py file (ensure these are available)
from app import (app, redis_conn, get_db, save_task_status, get_task_info_from_db,
                track_exists, save_track_analysis, get_all_tracks, update_playlist_table, JobStatus, # Removed save_track_embedding from top
                TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
                TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)

# Import configuration (ensure config.py is in PYTHONPATH or same directory)
from config import (TEMP_DIR, MAX_DISTANCE, MAX_SONGS_PER_CLUSTER, MAX_SONGS_PER_ARTIST,
    GMM_COVARIANCE_TYPE, MOOD_LABELS, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, ENERGY_MIN, ENERGY_MAX,
    TEMPO_MIN_BPM, TEMPO_MAX_BPM, JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, OTHER_FEATURE_LABELS, REDIS_URL, DATABASE_URL,
    OLLAMA_SERVER_URL, OLLAMA_MODEL_NAME, AI_MODEL_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL_NAME,
    DANCEABILITY_MODEL_PATH, AGGRESSIVE_MODEL_PATH, HAPPY_MODEL_PATH, PARTY_MODEL_PATH, RELAXED_MODEL_PATH, SAD_MODEL_PATH,
    SCORE_WEIGHT_SILHOUETTE, SCORE_WEIGHT_DAVIES_BOULDIN, SCORE_WEIGHT_CALINSKI_HARABASZ,
    SCORE_WEIGHT_DIVERSITY, SCORE_WEIGHT_PURITY, SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY, SCORE_WEIGHT_OTHER_FEATURE_PURITY,
    MUTATION_KMEANS_COORD_FRACTION, MUTATION_INT_ABS_DELTA, MUTATION_FLOAT_ABS_DELTA,
    TOP_N_ELITES, EXPLOITATION_START_FRACTION, EXPLOITATION_PROBABILITY_CONFIG, TOP_N_MOODS, TOP_N_OTHER_FEATURES,
    STRATIFIED_GENRES, MIN_SONGS_PER_GENRE_FOR_STRATIFICATION, SAMPLING_PERCENTAGE_CHANGE_PER_RUN, ITERATIONS_PER_BATCH_JOB, MAX_CONCURRENT_BATCH_JOBS, # type: ignore
    MAX_QUEUED_ANALYSIS_JOBS, # New config for limiting queued analysis jobs
    TOP_K_MOODS_FOR_PURITY_CALCULATION, LN_MOOD_DIVERSITY_STATS, LN_MOOD_PURITY_STATS,
    LN_OTHER_FEATURES_DIVERSITY_STATS, LN_OTHER_FEATURES_PURITY_STATS, # Import new stats for other features
    STRATIFIED_SAMPLING_TARGET_PERCENTILE, # Import new config
    OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY as CONFIG_OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY) # Import the new config

# Import AI naming function and prompt template
from ai import get_ai_playlist_name, creative_prompt_template
from .commons import score_vector # Import from commons

# Task specific function
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
    # Base parameters
    params = {
        "IncludeItemTypes": "MusicAlbum",
        "SortBy": "DateCreated",
        "SortOrder": "Descending",
        "Recursive": True,
    }
    # If limit is greater than 0, add it to the parameters.
    # If limit is 0 (or less, though typically 0 means all), omit the Limit parameter to fetch all.
    if limit > 0:
        params["Limit"] = limit
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

def predict_moods(embeddings_input, prediction_model_path, mood_labels_list):
    """Predicts moods using pre-computed embeddings and a mood classification model."""
    # embeddings_input is the 1D track-level embedding from TensorflowPredictMusiCNN
    model = TensorflowPredict2D(
        graphFilename=prediction_model_path,
        input="serving_default_model_Placeholder", # Ensure this matches your mood model's input tensor name
        output="PartitionedCall" # Ensure this matches your mood model's output tensor name
    )
    # model() will take the 1D embeddings_input, promote to (1, D), and output (1, num_moods)
    # [0] extracts the 1D array of mood scores
    mood_predictions = model(embeddings_input)[0]
    mood_results = dict(zip(mood_labels_list, mood_predictions))
    return {label: float(score) for label, score in mood_results.items()}

def predict_other_models(embeddings): # Now accepts embeddings
    """Predicts danceability, aggression, happiness, party, relaxed, and sadness using MusiCNN embeddings."""
    # MusiCNN embeddings are passed directly to this function.
    model_paths = {
        # Ensure these paths in config.py point to your new MusiCNN-based models
        "danceable": DANCEABILITY_MODEL_PATH,
        "aggressive": AGGRESSIVE_MODEL_PATH,
        "happy": HAPPY_MODEL_PATH,
        "party": PARTY_MODEL_PATH,
        "relaxed": RELAXED_MODEL_PATH,
        "sad": SAD_MODEL_PATH,
    }
    predictions = {}
    for mood, path in model_paths.items():
        try:
            model = TensorflowPredict2D(graphFilename=path, output="model/Softmax") # Using "model/Softmax" as per your example
            prediction = model(embeddings)
            if prediction is not None and prediction.size > 0 and len(prediction) > 0 and len(prediction[0]) == 2:  # Assuming binary classification
                predictions[mood] = float(prediction[0][0])  # Probability of the positive class
            else:
                predictions[mood] = 0.0  # Default value if prediction is invalid
        except Exception as e:
            print(f"Error predicting {mood}: {e}")
            predictions[mood] = 0.0  # Default value in case of error
    return predictions


def analyze_track(file_path, embedding_model_path, prediction_model_path, mood_labels_list):
    """Analyzes a single track for tempo, key, scale, moods, and other models."""
    # Load audio once at 16000 Hz for all analyses.
    audio = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()

    # Generate MusiCNN embeddings once
    musicnn_embedding_model = TensorflowPredictMusiCNN(
        graphFilename=embedding_model_path, output="model/dense/BiasAdd" # Main embedding model
    )
    raw_musicnn_embeddings = musicnn_embedding_model(audio) # Raw output, potentially 2D

    processed_musicnn_embeddings = np.array([]) # Initialize as empty 1D array

    if isinstance(raw_musicnn_embeddings, np.ndarray):
        if raw_musicnn_embeddings.ndim == 1:
            processed_musicnn_embeddings = raw_musicnn_embeddings
        elif raw_musicnn_embeddings.ndim == 2:
            if raw_musicnn_embeddings.shape[0] > 0: # If there are rows (segments or batch_size=1)
                # Average across the first dimension
                processed_musicnn_embeddings = np.mean(raw_musicnn_embeddings, axis=0)
            else: # Shape is (0, D) - no rows
                print(f"Warning: Raw MusicNN embeddings are 2D with no rows (shape: {raw_musicnn_embeddings.shape}). Resulting in empty 1D embedding.")
                # processed_musicnn_embeddings remains np.array([])
        else: # ndim > 2 or ndim == 0 (scalar, unlikely)
            print(f"Warning: Raw MusicNN embeddings have unexpected ndim: {raw_musicnn_embeddings.ndim} (shape: {raw_musicnn_embeddings.shape}). Resulting in empty 1D embedding.")
            # processed_musicnn_embeddings remains np.array([])
    else:
        print(f"Warning: Raw MusicNN output is not a NumPy array. Type: {type(raw_musicnn_embeddings)}. Resulting in empty 1D embedding.")
        # processed_musicnn_embeddings remains np.array([])

    tempo, _, _, _, _ = RhythmExtractor2013()(audio)
    key, scale, _ = KeyExtractor()(audio)

    # Calculate raw total energy
    raw_total_energy = Energy()(audio)
    num_samples = len(audio)

    # Calculate average energy per sample
    if num_samples > 0:
        average_energy_per_sample = raw_total_energy / float(num_samples)
    else:
        average_energy_per_sample = 0.0 # Should not happen for valid audio

    # Initialize moods and other_predictions with defaults
    moods = {label: 0.0 for label in mood_labels_list}
    other_predictions = {
        "danceable": 0.0, "aggressive": 0.0, "happy": 0.0,
        "party": 0.0, "relaxed": 0.0, "sad": 0.0
    }

    if processed_musicnn_embeddings.size > 0 and np.all(np.isfinite(processed_musicnn_embeddings)):
        try:
            moods = predict_moods(processed_musicnn_embeddings, PREDICTION_MODEL_PATH, MOOD_LABELS)
        except Exception as e_mood:
            print(f"Error during predict_moods: {e_mood}. Using default moods.")
        try:
            other_predictions = predict_other_models(processed_musicnn_embeddings)
        except Exception as e_other:
            print(f"Error during predict_other_models: {e_other}. Using default other_predictions.")
    else:
        print(f"Warning: Processed MusicNN embeddings are empty or invalid. Skipping mood/other predictions. Shape: {processed_musicnn_embeddings.shape if isinstance(processed_musicnn_embeddings, np.ndarray) else 'N/A'}")

    # Combine all predictions into a single dictionary
    all_predictions = {
        "tempo": tempo,
        "key": key,
        "scale": scale,
        "moods": moods,
        "energy": float(average_energy_per_sample), # Store average energy per sample
        **other_predictions  # Include danceable, aggressive, etc. directly
    }
    # Return the predictions dictionary and the PROCESSED (1D) embeddings numpy array
    return all_predictions, processed_musicnn_embeddings

# --- RQ Task Definitions ---


def analyze_album_task(album_id, album_name, jellyfin_url, jellyfin_user_id, jellyfin_token, top_n_moods, parent_task_id):
    """RQ task to analyze a single album."""
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        from app import save_track_embedding # Import here
        initial_details = {"album_name": album_name, "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Album analysis task started."]}
        save_task_status(current_task_id, "album_analysis", TASK_STATUS_STARTED, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=0, details=initial_details)
        headers = {"X-Emby-Token": jellyfin_token}
        tracks_analyzed_count = 0
        tracks_skipped_count = 0
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
                    log_and_update_album_task(f"Skipping '{track_name_full}' (already fully analyzed with embedding)", current_progress_val, current_track_name=track_name_full)
                    tracks_skipped_count +=1
                    continue

                path = download_track(jellyfin_url, headers, TEMP_DIR, item)
                log_and_update_album_task(f"Download attempt for '{track_name_full}': {'Success' if path else 'Failed'}", current_progress_val, current_track_name=track_name_full)
                if not path:
                    log_and_update_album_task(f"Failed to download '{track_name_full}'. Skipping.", current_progress_val, current_track_name=track_name_full)
                    continue

                try:
                    # analyze_track now returns two values: results dict and the embedding
                    analysis_results, track_embedding = analyze_track(path, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, MOOD_LABELS)
                    tempo = analysis_results.get("tempo", 0.0)
                    key = analysis_results.get("key", "")
                    scale = analysis_results.get("scale", "")
                    all_moods_predicted = analysis_results.get("moods", {}) # All moods initially
                    # Include new predictions directly
                    danceable = analysis_results.get("danceable", 0.0)
                    energy = analysis_results.get("energy", 0.0) # Get energy
                    aggressive = analysis_results.get("aggressive", 0.0)
                    happy = analysis_results.get("happy", 0.0)
                    party = analysis_results.get("party", 0.0)
                    relaxed = analysis_results.get("relaxed", 0.0)
                    sad = analysis_results.get("sad", 0.0)

                    # Select only the top_n_moods for saving and detailed logging
                    sorted_moods_for_saving = sorted(all_moods_predicted.items(), key=lambda item: item[1], reverse=True)
                    top_moods_to_save_dict = dict(sorted_moods_for_saving[:top_n_moods])

                    # Prepare for database storage
                    other_features_str = f"danceable:{danceable:.2f},aggressive:{aggressive:.2f},happy:{happy:.2f},party:{party:.2f},relaxed:{relaxed:.2f},sad:{sad:.2f}" # Ensure no spaces for easier parsing
                    current_track_details_for_api = {
                        "name": track_name_full,
                        "tempo": round(tempo, 2),
                        "key": key,
                        "scale": scale,
                        "moods": {k: round(v, 2) for k, v in all_moods_predicted.items()}
                    }
                    # Save only the top_n_moods
                    save_track_analysis(item['Id'], item['Name'], item.get('AlbumArtist', 'Unknown'), tempo, key, scale, top_moods_to_save_dict, energy=energy, other_features=other_features_str)
                    # Save the embedding
                    save_track_embedding(item['Id'], track_embedding)
                    tracks_analyzed_count += 1

                    # For logging, we already have the top_moods_to_save_dict
                    # If top_n_moods passed to task is different from a global TOP_N_MOODS for logging, adjust here.
                    # Assuming top_n_moods arg is the definitive count for both saving and logging summary.
                    top_n_moods_for_log = list(top_moods_to_save_dict.items()) # Already top N
                    mood_details_str_log = ', '.join(f'{k}:{v:.2f}' for k,v in top_n_moods_for_log)

                    # Prepare for logging: New features
                    other_features_log_str = f"Energy: {energy:.2f}, Danceable: {danceable:.2f}, Aggressive: {aggressive:.2f}, Happy: {happy:.2f}, Party: {party:.2f}, Relaxed: {relaxed:.2f}, Sad: {sad:.2f}" # Include energy in log

                    analysis_summary_msg = f"Tempo: {tempo:.2f}, Key: {key} {scale}."
                    analysis_summary_msg += f" Top {top_n_moods} Moods: {mood_details_str_log}. Other Features: {other_features_log_str}"
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
                "tracks_skipped": tracks_skipped_count,
                "total_tracks_in_album": total_tracks_in_album,
                "message": f"Album '{album_name}' analysis complete."
            }
            log_and_update_album_task(f"Album '{album_name}' analysis complete. Analyzed {tracks_analyzed_count}, Skipped {tracks_skipped_count} (out of {total_tracks_in_album} total).",
                                      100, task_state=TASK_STATUS_SUCCESS, final_summary_details=success_summary)
            return {"status": "SUCCESS", "message": f"Album '{album_name}' analysis complete.", "tracks_analyzed": tracks_analyzed_count, "tracks_skipped": tracks_skipped_count, "total_tracks": total_tracks_in_album}
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
            active_album_jobs_map = {}
            from app import rq_queue as main_rq_queue
            
            albums_to_process_queue = list(albums) # Create a queue of albums to process
            print(f"[MainAnalysisTask-{current_task_id}] DEBUG: Initial albums_to_process_queue size: {len(albums_to_process_queue)}")
            albums_launched_count = 0
            albums_completed_count = 0

            while albums_completed_count < total_albums:
                # === Cooperative Cancellation Check for Main Analysis Task ===
                if current_job: # Check if running as an RQ job
                    with app.app_context(): # Ensure DB access
                        # print(f"[MainAnalysisTask-{current_task_id}] DEBUG: Checking for cooperative cancellation.")
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

                # Check status of active jobs and remove completed ones
                completed_in_this_cycle_ids = []
                # print(f"[MainAnalysisTask-{current_task_id}] DEBUG: Checking {len(active_album_jobs_map)} active jobs.")
                for job_id_active, job_instance_active in list(active_album_jobs_map.items()): # Iterate over a copy
                    try:
                        is_child_completed = False

                        try:
                            job_instance_active.refresh() # Get latest status from Redis
                            child_status_rq = job_instance_active.get_status()
                            # print(f"[MainAnalysisTask-{current_task_id}] DEBUG: Job {job_id_active} RQ status: {child_status_rq}")

                            if job_instance_active.is_finished or job_instance_active.is_failed or child_status_rq == JobStatus.CANCELED:
                                is_child_completed = True
                            else: # Still active in RQ
                                pass # Child is active, all_done will be set to False later

                        except NoSuchJobError:
                            #print(f"[MainAnalysisTask-{current_task_id}] Warning: Child job {job_instance.id} not found in Redis. Checking DB.")
                            with app.app_context(): # Ensure DB context
                                db_task_info = get_task_info_from_db(job_id_active)

                            # print(f"[MainAnalysisTask-{current_task_id}] DEBUG: Job {job_id_active} not in RQ. DB info: {db_task_info}")
                            if db_task_info:
                                db_status = db_task_info.get('status')
                                if db_status in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED, JobStatus.CANCELED, JobStatus.FINISHED, JobStatus.FAILED]:
                                    is_child_completed = True
                                    if db_status in [TASK_STATUS_FAILURE, TASK_STATUS_REVOKED, JobStatus.CANCELED, JobStatus.FAILED]:
                                        print(f"[MainAnalysisTask-{current_task_id}] Info: Child job {job_id_active} (from DB) has status: {db_status}.")
                                    # If SUCCESS/FINISHED, it's completed, no special print needed here.
                                else:
                                    print(f"[MainAnalysisTask-{current_task_id}] Warning: Child job {job_id_active} (from DB) has non-terminal status '{db_status}' but missing from RQ. Treating as completed.")
                                    is_child_completed = True
                            else:
                                print(f"[MainAnalysisTask-{current_task_id}] CRITICAL: Child job {job_id_active} not found in Redis or DB. Treating as completed.")
                                is_child_completed = True

                        if is_child_completed:
                            completed_in_this_cycle_ids.append(job_id_active)
                            albums_completed_count += 1
                    except Exception as e_monitor_child:
                        print(f"[MainAnalysisTask-{current_task_id}] ERROR monitoring child job {job_id_active}: {e_monitor_child}. Treating as completed to avoid hanging.")
                        traceback.print_exc()
                        completed_in_this_cycle_ids.append(job_id_active) # Remove from active map
                        albums_completed_count += 1 # Ensure progress
                
                for j_id in completed_in_this_cycle_ids:
                    if j_id in active_album_jobs_map:
                        # print(f"[MainAnalysisTask-{current_task_id}] DEBUG: Removing completed job {j_id} from active map.")
                        del active_album_jobs_map[j_id]

                # Enqueue new jobs if slots are available and albums are pending
                # print(f"[MainAnalysisTask-{current_task_id}] DEBUG: Checking to enqueue new jobs. Active: {len(active_album_jobs_map)}, Max Queued: {MAX_QUEUED_ANALYSIS_JOBS}, Albums in queue: {len(albums_to_process_queue)}")
                while len(active_album_jobs_map) < MAX_QUEUED_ANALYSIS_JOBS and albums_to_process_queue:
                    album = albums_to_process_queue.pop(0) # Get next album
                    # print(f"[MainAnalysisTask-{current_task_id}] DEBUG: Enqueuing job for album: {album.get('Name')}")
                    sub_task_id = str(uuid.uuid4())
                    save_task_status(sub_task_id, "album_analysis", "PENDING", parent_task_id=current_task_id, sub_type_identifier=album['Id'], details={"album_name": album['Name']})
                    job = main_rq_queue.enqueue(
                        analyze_album_task,
                        args=(album['Id'], album['Name'], jellyfin_url, jellyfin_user_id, jellyfin_token, top_n_moods, current_task_id),
                        job_id=sub_task_id,
                        description=f"Analyzing album: {album['Name']}",
                        job_timeout=-1, 
                        meta={'parent_task_id': current_task_id}
                    )
                    album_jobs.append(job) # Keep original list for final summary
                    album_job_ids.append(job.id) # Keep original list for final summary
                    active_album_jobs_map[job.id] = job
                    albums_launched_count += 1

                status_message_for_log = f"Albums launched: {albums_launched_count}/{total_albums}. Completed: {albums_completed_count}/{total_albums}. Active/Queued: {len(active_album_jobs_map)}."
                progress_while_waiting = 10 + int(80 * (albums_completed_count / float(total_albums))) if total_albums > 0 else 10
                log_and_update_main_analysis(
                    status_message_for_log,
                    progress_while_waiting,
                    {"albums_completed": albums_completed_count, "albums_launched": albums_launched_count, "total_albums": total_albums})
                
                # print(f"[MainAnalysisTask-{current_task_id}] DEBUG: End of main while loop iteration. albums_completed_count: {albums_completed_count}, total_albums: {total_albums}, len(active_album_jobs_map): {len(active_album_jobs_map)}")
                if albums_completed_count >= total_albums and not active_album_jobs_map: # All completed and active map empty
                    print(f"[MainAnalysisTask-{current_task_id}] DEBUG: All albums processed and active map empty. Breaking main loop.")
                    break
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
            # total_tracks_analyzed_all_albums will now reflect only newly/re-analyzed tracks
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
