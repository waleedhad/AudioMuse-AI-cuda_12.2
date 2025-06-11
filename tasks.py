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
from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor, TensorflowPredictMusiCNN, TensorflowPredict2D, Energy
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture # type: ignore
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score # type: ignore
from sklearn.preprocessing import StandardScaler

# RQ import
from rq import get_current_job
from rq.job import Job # Import Job class
from rq.exceptions import NoSuchJobError, InvalidJobOperation

# Import necessary components from the main app.py file (ensure these are available)
from app import (app, redis_conn, get_db, save_task_status, get_task_info_from_db,
                track_exists, save_track_analysis, get_all_tracks, update_playlist_table, JobStatus,
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
    STRATIFIED_GENRES, MIN_SONGS_PER_GENRE_FOR_STRATIFICATION, SAMPLING_PERCENTAGE_CHANGE_PER_RUN, ITERATIONS_PER_BATCH_JOB, MAX_CONCURRENT_BATCH_JOBS,
    TOP_K_MOODS_FOR_PURITY_CALCULATION, RAW_MOOD_DIVERSITY_STATS, RAW_MOOD_PURITY_STATS, # Import new stats
    STRATIFIED_SAMPLING_TARGET_PERCENTILE) # Import new config

# Import AI naming function and prompt template
from ai import get_ai_playlist_name, creative_prompt_template


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
    musicnn_embeddings = musicnn_embedding_model(audio)

    tempo, _, _, _, _ = RhythmExtractor2013()(audio)
    key, scale, _ = KeyExtractor()(audio)
    moods = predict_moods(musicnn_embeddings, PREDICTION_MODEL_PATH, MOOD_LABELS) # Pass embeddings and correct model path

    # Calculate raw total energy
    raw_total_energy = Energy()(audio)
    num_samples = len(audio)

    # Calculate average energy per sample
    if num_samples > 0:
        average_energy_per_sample = raw_total_energy / float(num_samples)
    else:
        average_energy_per_sample = 0.0 # Should not happen for valid audio

    other_predictions = predict_other_models(musicnn_embeddings) # Pass the same MusiCNN embeddings

    # Combine all predictions into a single dictionary
    all_predictions = {
        "tempo": tempo,
        "key": key,
        "scale": scale,
        "moods": moods,
        "energy": float(average_energy_per_sample), # Store average energy per sample
        **other_predictions  # Include danceable, aggressive, etc. directly
    }

    return all_predictions

def score_vector(row, mood_labels_list, other_feature_labels_list): # other_feature_labels_list is now passed
    """Converts a database row into a numerical feature vector for clustering."""
    # Extract features from the database row
    tempo = float(row['tempo']) if row['tempo'] is not None else 0.0
    energy = float(row['energy']) if row['energy'] is not None else 0.0 # Get energy
    mood_str = row['mood_vector'] or ""
    
    # Normalize tempo to 0-1 range
    tempo_range = TEMPO_MAX_BPM - TEMPO_MIN_BPM
    tempo_norm = (tempo - TEMPO_MIN_BPM) / tempo_range if tempo_range > 0 else 0.0
    tempo_norm = np.clip(tempo_norm, 0.0, 1.0)

    # Normalize energy to 0-1 range
    energy_range = ENERGY_MAX - ENERGY_MIN
    energy_norm = (energy - ENERGY_MIN) / energy_range if energy_range > 0 else 0.0
    energy_norm = np.clip(energy_norm, 0.0, 1.0)

    tempo_val = tempo_norm
    energy_val = energy_norm

    mood_scores_for_vector = np.zeros(len(mood_labels_list)) # Initialize vector for all moods
    if mood_str:
        for pair in mood_str.split(","):
            if ":" not in pair:
                continue
            label, score_str = pair.split(":")
            if label in mood_labels_list:
                try:
                    mood_scores_for_vector[mood_labels_list.index(label)] = float(score_str) # Populate all mood scores
                except ValueError:
                    continue

    # Parse and prepare other features
    other_feature_scores_for_vector = np.zeros(len(other_feature_labels_list)) # Initialize vector for all other features
    other_features_str = row.get('other_features', "") # Get the new field
    if other_features_str:
        for pair in other_features_str.split(","):
            if ":" not in pair:
                continue
            label, score_str = pair.split(":")
            if label in other_feature_labels_list:
                try:
                    other_feature_scores_for_vector[other_feature_labels_list.index(label)] = float(score_str) # Populate all other feature scores
                except ValueError:
                    continue


    # Construct the full feature vector: [tempo, energy, moods..., other_features...]
    # Tempo and energy are now normalized (0-1) before being passed to StandardScaler.
    full_vector = [tempo_val, energy_val] + list(mood_scores_for_vector) + list(other_feature_scores_for_vector)
    return full_vector

# OTHER_FEATURE_LABELS is imported from config.py
def name_cluster(centroid_scaled_vector, pca_model, pca_enabled, mood_labels_list, scaler_details=None):
    """Generates a human-readable name for a cluster. Now includes standardization inverse transform."""

    interpreted_vector = centroid_scaled_vector # This is from the clustered space (might be PCA-transformed)

    # Step 1: Inverse transform from PCA space to standardized space (if PCA enabled)
    if pca_enabled and pca_model is not None:
        try:
            interpreted_vector = pca_model.inverse_transform(interpreted_vector.reshape(1, -1))[0]
        except ValueError:
            print("Warning: PCA inverse_transform failed. Cannot fully inverse transform. Using PCA space for interpretation.")
            # If PCA inverse fails, further inverse transformation (scaler) will likely be incorrect.
            # We'll proceed with the PCA-transformed vector, but the interpretation of raw values might be off.
            pass # interpreted_vector remains in PCA space if inverse fails

    # Step 2: Inverse transform from standardized space to original raw space (if scaler details provided)
    if scaler_details:
        try:
            temp_scaler = StandardScaler()
            temp_scaler.mean_ = np.array(scaler_details["mean"])
            temp_scaler.scale_ = np.array(scaler_details["scale"])
            # Essential for `inverse_transform` to know the number of features it was fitted on
            temp_scaler.n_features_in_ = len(temp_scaler.mean_)

            if len(interpreted_vector) != len(temp_scaler.mean_):
                print(f"Warning: Dimension mismatch for scaler inverse transform. Expected {len(temp_scaler.mean_)} features, got {len(interpreted_vector)}. Skipping scaler inverse.")
                # If dimensions don't match, we cannot reliably inverse scale. Use the vector as is.
                final_interpreted_raw_vector = interpreted_vector
            else:
                final_interpreted_raw_vector = temp_scaler.inverse_transform(interpreted_vector.reshape(1, -1))[0]
        except Exception as e:
            print(f"Warning: StandardScaler inverse_transform failed: {e}. Using previously interpreted vector.")
            traceback.print_exc()
            final_interpreted_raw_vector = interpreted_vector # Fallback if scaler inverse fails
    else:
        # If no scaler details, the vector is either from PCA inverse (if PCA was enabled)
        # or the original (unscaled) vector passed directly (if no PCA and no scaler).
        final_interpreted_raw_vector = interpreted_vector


    # Now use the final_interpreted_raw_vector for naming logic
    # tempo_val and energy_val are now the 0-1 normalized values because they were normalized
    # *before* standardization, and scaler.inverse_transform gives us back those normalized values.
    tempo_val = final_interpreted_raw_vector[0]
    energy_val = final_interpreted_raw_vector[1]
    mood_values = final_interpreted_raw_vector[2 : 2 + len(mood_labels_list)]

    # For AI naming, we might want to pass the original scale or interpreted labels.
    # For now, let's pass the normalized values and let the AI prompt guide interpretation.
    # Or, we can reconstruct raw values if needed, but let's adjust labels first.

    # Extract scores for the other features (danceable, etc.)
    other_feature_values = final_interpreted_raw_vector[2 + len(mood_labels_list):]
    other_feature_scores_dict = {}
    if len(other_feature_values) == len(OTHER_FEATURE_LABELS):
        other_feature_scores_dict = dict(zip(OTHER_FEATURE_LABELS, other_feature_values))
    else:
        print(f"Warning: Mismatch between other_feature_values length ({len(other_feature_values)}) and OTHER_FEATURE_LABELS length ({len(OTHER_FEATURE_LABELS)}). Cannot map scores for AI naming.")

    # Convert normalized tempo (0-1) to label
    if tempo_val < 0.33: tempo_label = "Slow"
    elif tempo_val < 0.66: tempo_label = "Medium"
    else: tempo_label = "Fast"

    # Convert normalized energy (0-1) to label
    # energy_val is already the normalized energy (0-1 range)
    energy_label = "Low Energy" if energy_val < 0.3 else \
                   ("Medium Energy" if energy_val < 0.7 else \
                    "High Energy")


    # Determine top moods
    if len(mood_values) == 0 or np.sum(mood_values) == 0: top_indices = []
    else: top_indices = np.argsort(mood_values)[::-1][:3]

    mood_names = [mood_labels_list[i] for i in top_indices if i < len(mood_labels_list)]
    mood_part = "_".join(mood_names).title() if mood_names else "Mixed"
    base_name = f"{mood_part}_{tempo_label}"

    # Add top "other features" to the name if they are prominent
    OTHER_FEATURE_THRESHOLD_FOR_NAME = 0.5  # Threshold for a feature to be included in the name
    MAX_OTHER_FEATURES_IN_NAME = 2      # Max number of "other features" to append

    appended_other_features_str = ""
    if other_feature_scores_dict:
        # Filter features above threshold and sort them by score
        prominent_other_features = sorted(
            [(feature, score) for feature, score in other_feature_scores_dict.items() if score >= OTHER_FEATURE_THRESHOLD_FOR_NAME],
            key=lambda item: item[1],
            reverse=True
        )
        # Select top N features to add to the name, capitalized
        other_features_to_add_to_name_list = [feature.title() for feature, score in prominent_other_features[:MAX_OTHER_FEATURES_IN_NAME]]
        if other_features_to_add_to_name_list:
            appended_other_features_str = "_" + "_".join(other_features_to_add_to_name_list)
    full_name = f"{base_name}{appended_other_features_str}"

    top_mood_scores = {mood_labels_list[i]: mood_values[i] for i in top_indices if i < len(mood_labels_list)}
    # Store normalized tempo and energy in extra_info for the centroid
    # If original scale is needed for AI, it would require passing original values or inverse transforming the normalization.
    # For now, AI will receive normalized values if it uses this part of centroid_details.
    extra_info = {"tempo_normalized": round(tempo_val, 2), "energy_normalized": round(energy_val, 2)}

    # Combine all relevant centroid features for AI naming and general info
    # other_feature_scores_dict already contains 'danceable', 'aggressive', etc.
    # extra_info contains 'tempo' and 'energy'
    comprehensive_centroid_details = {**top_mood_scores, **extra_info, **other_feature_scores_dict}
    return full_name, comprehensive_centroid_details

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
            playlist_name_on_jellyfin = f"{base_name} ({idx})" if len(chunks) > 1 else base_name
            item_ids = [item_id for item_id, _, _ in chunk]
            if not item_ids: continue
            body = {"Name": playlist_name_on_jellyfin, "Ids": item_ids, "UserId": jellyfin_user_id_param}
            try:
                r = requests.post(f"{jellyfin_url_param}/Playlists", headers=headers_param, json=body, timeout=60) # Increased timeout
                if r.ok:
                    centroid_info = cluster_centers.get(base_name, {})
                    top_moods = {k: v for k, v in centroid_info.items() if k in mood_labels_list}
                    extra_info = {k: v for k, v in centroid_info.items() if k not in mood_labels_list}
                    centroid_str = ", ".join(f"{k}:{v:.2f}" for k, v in top_moods.items())
                    extras_str = ", ".join(f"{k}:{v:.2f}" for k, v in extra_info.items())
                    print(f"✅ Created playlist '{playlist_name_on_jellyfin}' with {len(item_ids)} tracks (Centroid for '{base_name}': {centroid_str} | {extras_str})")
            except Exception as e:
                print(f"Exception creating '{playlist_name_on_jellyfin}': {e}")

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
                    analysis_results = analyze_track(path, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, MOOD_LABELS)
                    tempo = analysis_results.get("tempo", 0.0)
                    key = analysis_results.get("key", "")
                    scale = analysis_results.get("scale", "")
                    moods = analysis_results.get("moods", {})
                    # Include new predictions directly
                    danceable = analysis_results.get("danceable", 0.0)
                    energy = analysis_results.get("energy", 0.0) # Get energy
                    aggressive = analysis_results.get("aggressive", 0.0)
                    happy = analysis_results.get("happy", 0.0)
                    party = analysis_results.get("party", 0.0)
                    relaxed = analysis_results.get("relaxed", 0.0)
                    sad = analysis_results.get("sad", 0.0)

                    # Prepare for database storage
                    other_features_str = f"danceable:{danceable:.2f},aggressive:{aggressive:.2f},happy:{happy:.2f},party:{party:.2f},relaxed:{relaxed:.2f},sad:{sad:.2f}" # Ensure no spaces for easier parsing

                    current_track_details_for_api = {
                        "name": track_name_full,
                        "tempo": round(tempo, 2),
                        "key": key,
                        "scale": scale,
                        "moods": {k: round(v, 2) for k, v in moods.items()}
                    }

                    # Save ALL mood scores, not just top N, to allow for more accurate purity calculations
                    save_track_analysis(item['Id'], item['Name'], item.get('AlbumArtist', 'Unknown'), tempo, key, scale, moods, energy=energy, other_features=other_features_str) # Pass full moods dict
                    tracks_analyzed_count += 1
                    mood_details_str = ', '.join(f'{k}:{v:.2f}' for k,v in moods.items())

                    # Prepare for logging: Get top N moods
                    sorted_moods = sorted(moods.items(), key=lambda item: item[1], reverse=True)
                    top_n_moods_for_log = sorted_moods[:top_n_moods] # Use the function argument
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
                            #print(f"[MainAnalysisTask-{current_task_id}] Warning: Child job {job_instance.id} not found in Redis. Checking DB.")
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
    run_idx, data_subset_for_clustering, # Changed from all_tracks_data_parsed
    clustering_method, num_clusters_min_max, dbscan_params_ranges, gmm_params_ranges, pca_params_ranges,
    max_songs_per_cluster, log_prefix="",
    elite_solutions_params_list=None, exploitation_probability=0.0, mutation_config=None,
    score_weight_diversity_override=None, score_weight_silhouette_override=None, # Existing weight overrides
    score_weight_davies_bouldin_override=None, score_weight_calinski_harabasz_override=None): # New weight overrides for DB and CH
    """
    Internal helper to perform a single clustering iteration. Not an RQ task.
    Receives a subset of track data (rows) for clustering.
    Returns a result dictionary or None on failure.
    `num_clusters_min_max` is a tuple (min, max)
    `dbscan_params_ranges` is a dict like {"eps_min": ..., "eps_max": ..., "samples_min": ..., "samples_max": ...}
    `gmm_params_ranges` is a dict like {"n_components_min": ..., "n_components_max": ...}
    `pca_params_ranges` is a dict like {"components_min": ..., "components_max": ...}
    `elite_solutions_params_list`: A list of 'parameters' dicts from previous best runs.
    `score_weight_diversity_override`: Specific weight for diversity for this run.
    `score_weight_silhouette_override`: Specific weight for silhouette for this run.
    `score_weight_davies_bouldin_override`: Specific weight for Davies-Bouldin for this run.
    `score_weight_calinski_harabasz_override`: Specific weight for Calinski-Harabasz for this run.
    `exploitation_probability`: Chance to use an elite solution for parameter generation.
    `mutation_config`: Dict with mutation strengths, e.g., {"int_abs_delta": 2, "float_abs_delta": 0.05}.
    """
    try:
        elite_solutions_params_list = elite_solutions_params_list or []
        mutation_config = mutation_config or {"int_abs_delta": MUTATION_INT_ABS_DELTA, "float_abs_delta": MUTATION_FLOAT_ABS_DELTA, "coord_mutation_fraction": MUTATION_KMEANS_COORD_FRACTION}
        if "coord_mutation_fraction" not in mutation_config: # Ensure default if not passed
            mutation_config["coord_mutation_fraction"] = MUTATION_KMEANS_COORD_FRACTION

        # Use override if provided, else use global config
        current_score_weight_diversity = score_weight_diversity_override if score_weight_diversity_override is not None else SCORE_WEIGHT_DIVERSITY
        current_score_weight_silhouette = score_weight_silhouette_override if score_weight_silhouette_override is not None else SCORE_WEIGHT_SILHOUETTE
        current_score_weight_davies_bouldin = score_weight_davies_bouldin_override if score_weight_davies_bouldin_override is not None else SCORE_WEIGHT_DAVIES_BOULDIN
        current_score_weight_calinski_harabasz = score_weight_calinski_harabasz_override if score_weight_calinski_harabasz_override is not None else SCORE_WEIGHT_CALINSKI_HARABASZ

        # --- Data Preparation ---
        # X_original is now derived from the passed data_subset_for_clustering
        X_original = np.array([score_vector(row, MOOD_LABELS, OTHER_FEATURE_LABELS) for row in data_subset_for_clustering])
        if X_original.shape[0] == 0:
            print(f"{log_prefix} Iteration {run_idx}: No data in subset to cluster.")
            return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": None, "parameters": {}} # Add scaler_details

        # Standardize the data
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X_original)
        scaler_details = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}

        data_for_clustering_current = X_standardized # Default if PCA is off or fails
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
            elite_scaler_details_original = chosen_elite_params_set.get("scaler_details") # Get elite scaler details

            # A simple check for scaler compatibility for mutation: if it was used in elite, it must be used now.
            # This is a heuristic, full compatibility check would be complex.
            scaler_compatible = (elite_scaler_details_original is not None) == (scaler_details is not None)

            if elite_method_config_original and elite_pca_config_original and scaler_compatible and \
               elite_method_config_original.get("method") == clustering_method:
                try:
                    # 1. Mutate PCA components first
                    elite_pca_comps = elite_pca_config_original.get("components", pca_params_ranges["components_min"])
                    mutated_pca_comps = _mutate_param(
                        elite_pca_comps,
                        pca_params_ranges["components_min"], pca_params_ranges["components_max"],
                        mutation_config.get("int_abs_delta", MUTATION_INT_ABS_DELTA)
                    )
                    # Max PCA components also limited by number of features in X_standardized and number of samples
                    max_pca_by_features = X_standardized.shape[1]
                    max_pca_by_samples = (X_standardized.shape[0] - 1) if X_standardized.shape[0] > 1 else 1

                    max_allowable_pca_mutated = min(mutated_pca_comps, max_pca_by_features, max_pca_by_samples)
                    max_allowable_pca_mutated = max(0, max_allowable_pca_mutated) # Ensure not negative
                    temp_pca_config = {"enabled": max_allowable_pca_mutated > 0, "components": max_allowable_pca_mutated}

                    # 2. Apply this new PCA config to get data for this iteration
                    if temp_pca_config["enabled"]:
                        if temp_pca_config["components"] > 0:
                            pca_model_for_this_iteration = PCA(n_components=temp_pca_config["components"])
                            data_for_clustering_current = pca_model_for_this_iteration.fit_transform(X_standardized)
                            temp_pca_config["components"] = pca_model_for_this_iteration.n_components_ # Update with actual components used
                        else: # Components somehow became 0
                            temp_pca_config["enabled"] = False
                            data_for_clustering_current = X_standardized # Fallback
                    else:
                        data_for_clustering_current = X_standardized # PCA not enabled

                    # 3. Mutate clustering method parameters (e.g., n_clusters)
                    #    This must happen AFTER PCA, as n_clusters can depend on data_for_clustering_current.shape[0]
                    temp_method_params_config = None
                    max_clusters_or_components = data_for_clustering_current.shape[0]
                    if max_clusters_or_components == 0: # No data points after PCA (should be rare)
                        raise ValueError("No data points available after PCA to determine cluster parameters.")

                    if clustering_method == "kmeans":
                        elite_n_clusters = elite_method_config_original.get("params", {}).get("n_clusters", num_clusters_min_max[0])
                        mutated_n_clusters = _mutate_param(
                            elite_n_clusters,
                            max(1, num_clusters_min_max[0]), # Ensure min clusters is at least 1
                            min(num_clusters_min_max[1], max_clusters_or_components), # Max clusters capped by available points
                            mutation_config.get("int_abs_delta", 2)
                        )
                        temp_method_params_config = {"method": "kmeans", "params": {"n_clusters": max(1, mutated_n_clusters)}}
                    elif clustering_method == "dbscan":
                        elite_eps = elite_method_config_original.get("params", {}).get("eps", dbscan_params_ranges["eps_min"])
                        elite_min_samples = elite_method_config_original.get("params", {}).get("min_samples", dbscan_params_ranges["samples_min"])
                        mutated_eps = _mutate_param(
                            elite_eps, dbscan_params_ranges["eps_min"], dbscan_params_ranges["eps_max"],
                            mutation_config.get("float_abs_delta", MUTATION_FLOAT_ABS_DELTA), is_float=True, round_digits=2
                        )
                        mutated_min_samples = _mutate_param(
                            elite_min_samples, dbscan_params_ranges["samples_min"], dbscan_params_ranges["samples_max"],
                            mutation_config.get("int_abs_delta", MUTATION_INT_ABS_DELTA)
                        ) # min_samples should be at least 1
                        temp_method_params_config = {"method": "dbscan", "params": {"eps": mutated_eps, "min_samples": mutated_min_samples}}
                    elif clustering_method == "gmm":
                        elite_n_components = elite_method_config_original.get("params", {}).get("n_components", gmm_params_ranges["n_components_min"])
                        mutated_n_components = _mutate_param(
                            elite_n_components,
                            gmm_params_ranges["n_components_min"],
                            min(gmm_params_ranges["n_components_max"], max_clusters_or_components), # Max components capped
                            mutation_config.get("int_abs_delta", MUTATION_INT_ABS_DELTA)
                        )
                        temp_method_params_config = {"method": "gmm", "params": {"n_components": max(1, mutated_n_components)}}

                    if temp_method_params_config and temp_pca_config is not None:
                        # 4. For KMeans, generate/mutate initial_centroids
                        if clustering_method == "kmeans":
                            kmeans_initial_centroids = _generate_or_mutate_kmeans_initial_centroids(
                                temp_method_params_config["params"]["n_clusters"],
                                data_for_clustering_current,
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
                    traceback.print_exc()
                    params_generated_by_mutation = False
        if not params_generated_by_mutation:
            # print(f"{log_prefix} Iteration {run_idx}: Using random parameters.")
            # Original random parameter generation
            sampled_pca_components_rand = random.randint(pca_params_ranges["components_min"], pca_params_ranges["components_max"])
            # Max PCA components also limited by number of features in X_standardized and number of samples
            max_allowable_pca_rand = min(sampled_pca_components_rand, X_standardized.shape[1], (X_standardized.shape[0] - 1) if X_standardized.shape[0] > 1 else 1)
            max_allowable_pca_rand = max(0, max_allowable_pca_rand) # Ensure not negative
            pca_config = {"enabled": max_allowable_pca_rand > 0, "components": max_allowable_pca_rand}

            # Apply PCA for random generation path to get data_for_clustering_current
            if pca_config["enabled"]:
                n_comps_rand = min(pca_config["components"], X_standardized.shape[1], (X_standardized.shape[0] - 1) if X_standardized.shape[0] > 1 else 1)
                if n_comps_rand > 0:
                    pca_model_for_this_iteration = PCA(n_components=n_comps_rand)
                    data_for_clustering_current = pca_model_for_this_iteration.fit_transform(X_standardized)
                    pca_config["components"] = pca_model_for_this_iteration.n_components_ # Update with actual
                else:
                    pca_config["enabled"] = False
                    data_for_clustering_current = X_standardized
            else:
                data_for_clustering_current = X_standardized

            max_clusters_or_components_rand = data_for_clustering_current.shape[0]
            if max_clusters_or_components_rand == 0: # No data points after PCA
                 print(f"{log_prefix} Iteration {run_idx}: No data points available after PCA for random parameter generation.")
                 return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details, "parameters": {"pca_config": pca_config}} # Add scaler_details

            if clustering_method == "kmeans":
                k_rand = random.randint(max(1, num_clusters_min_max[0]), min(num_clusters_min_max[1], max_clusters_or_components_rand)) # Ensure min clusters is at least 1
                k_rand = max(1, k_rand)
                kmeans_initial_centroids_rand = _generate_or_mutate_kmeans_initial_centroids(
                    k_rand, data_for_clustering_current, None, None, pca_config, # No elite for random
                    log_prefix=f"{log_prefix} Iteration {run_idx} (random)"
                )
                method_params_config = {"method": "kmeans", "params": {"n_clusters": k_rand, "initial_centroids": kmeans_initial_centroids_rand}}
            elif clustering_method == "dbscan":
                current_dbscan_eps_rand = round(random.uniform(dbscan_params_ranges["eps_min"], dbscan_params_ranges["eps_max"]), 2)
                current_dbscan_min_samples_rand = random.randint(dbscan_params_ranges["samples_min"], dbscan_params_ranges["samples_max"])
                method_params_config = {"method": "dbscan", "params": {"eps": current_dbscan_eps_rand, "min_samples": current_dbscan_min_samples_rand}}
            elif clustering_method == "gmm": # GMM also needs at least 1 component
                gmm_n_rand = random.randint(gmm_params_ranges["n_components_min"], min(gmm_params_ranges["n_components_max"], max_clusters_or_components_rand))
                method_params_config = {"method": "gmm", "params": {"n_components": max(1, gmm_n_rand)}}
            else:
                print(f"{log_prefix} Iteration {run_idx}: Unsupported clustering method {clustering_method}")
                return None

        # Ensure pca_model_for_this_iteration is set if pca_config is enabled but model wasn't created during mutation path
        if pca_config["enabled"] and pca_model_for_this_iteration is None and pca_config["components"] > 0:
            # Re-create PCA model and transform data if it wasn't done in the mutation path
            try:
                pca_model_for_this_iteration = PCA(n_components=pca_config["components"])
                data_for_clustering_current = pca_model_for_this_iteration.fit_transform(X_standardized)
                pca_config["components"] = pca_model_for_this_iteration.n_components_ # Update with actual
            except Exception as e_refit_pca:
                 print(f"{log_prefix} Iteration {run_idx}: Error re-fitting PCA: {e_refit_pca}. Disabling PCA for this run.")
                 traceback.print_exc()
                 pca_config["enabled"] = False
                 pca_config["components"] = 0
                 pca_model_for_this_iteration = None
                 data_for_clustering_current = X_standardized # Fallback to standardized data


        if not method_params_config or pca_config is None:
            print(f"{log_prefix} Iteration {run_idx}: Critical error: parameters not configured.")
            return None

        # --- Start of core logic from original run_single_clustering_iteration_task ---
        # Use data_for_clustering_current (which is either standardized or PCA-transformed standardized data) for clustering.

        labels = None
        cluster_centers_map = {}
        raw_distances = np.zeros(data_for_clustering_current.shape[0])
        method_from_config = method_params_config["method"]
        params_from_config = method_params_config["params"]

        if method_from_config == "kmeans":
            if not params_from_config.get("initial_centroids") or not isinstance(params_from_config["initial_centroids"], list) or len(params_from_config["initial_centroids"]) == 0:
                print(f"{log_prefix} Iteration {run_idx}: KMeans initial_centroids missing or empty. Cannot cluster with KMeans.")
                return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}} # Add scaler_details

            initial_centroids_np = np.array(params_from_config["initial_centroids"])

            # Ensure n_clusters matches the number of initial centroids provided
            if initial_centroids_np.ndim == 1 and initial_centroids_np.shape[0] == 0: # Empty array from empty list
                 print(f"{log_prefix} Iteration {run_idx}: KMeans initial_centroids resulted in empty numpy array. Cannot cluster.")
                 return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}} # Add scaler_details

            if initial_centroids_np.shape[0] != params_from_config["n_clusters"]:
                # print(f"{log_prefix} Iteration {run_idx}: Mismatch n_clusters ({params_from_config['n_clusters']}) and num initial_centroids ({initial_centroids_np.shape[0]}). Adjusting n_clusters.")
                params_from_config["n_clusters"] = initial_centroids_np.shape[0]

            if params_from_config["n_clusters"] == 0:
                print(f"{log_prefix} Iteration {run_idx}: n_clusters is 0 for KMeans after adjustment. Cannot cluster.")
                return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}} # Add scaler_details

            kmeans = KMeans(n_clusters=params_from_config["n_clusters"], init=initial_centroids_np, n_init=1)
            labels = kmeans.fit_predict(data_for_clustering_current)
            cluster_centers_map = {i: kmeans.cluster_centers_[i] for i in range(params_from_config["n_clusters"])}
            centers_for_points = kmeans.cluster_centers_[labels]
            raw_distances = np.linalg.norm(data_for_clustering_current - centers_for_points, axis=1)
        elif method_from_config == "dbscan":
            dbscan = DBSCAN(eps=params_from_config["eps"], min_samples=params_from_config["min_samples"])
            labels = dbscan.fit_predict(data_for_clustering_current)
            for cluster_id_val in set(labels):
                if cluster_id_val == -1: continue
                indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id_val]
                cluster_points = data_for_clustering_current[indices]
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    for i_idx in indices: raw_distances[i_idx] = np.linalg.norm(data_for_clustering_current[i_idx] - center)
                    cluster_centers_map[cluster_id_val] = center
        elif method_from_config == "gmm":
            gmm = GaussianMixture(n_components=params_from_config["n_components"], covariance_type=GMM_COVARIANCE_TYPE, random_state=None, max_iter=1000)
            gmm.fit(data_for_clustering_current)
            labels = gmm.predict(data_for_clustering_current)
            cluster_centers_map = {i: gmm.means_[i] for i in range(params_from_config["n_components"])}
            centers_for_points = gmm.means_[labels] # type: ignore
            raw_distances = np.linalg.norm(data_for_clustering_current - centers_for_points, axis=1)

        # --- Calculate Internal Validation Metrics ---
        silhouette_metric_value = 0.0 # Will store the raw Silhouette score, or 0.0 if not applicable
        davies_bouldin_metric_value = 0.0 # Initialize Davies-Bouldin metric
        calinski_harabasz_metric_value = 0.0 # Initialize Calinski-Harabasz metric

        num_actual_clusters = len(set(labels) - {-1}) # Number of clusters, excluding noise
        num_samples_for_metrics = data_for_clustering_current.shape[0]

        if num_actual_clusters >= 2 and num_actual_clusters < num_samples_for_metrics:
            if current_score_weight_silhouette > 0: # Use current_score_weight_silhouette
                try:
                    s_score = silhouette_score(data_for_clustering_current, labels, metric='euclidean')
                    # Normalize Silhouette score from [-1, 1] to [0, 1]
                    silhouette_metric_value = (s_score + 1) / 2.0
                except ValueError as e_sil:
                    print(f"{log_prefix} Iteration {run_idx}: Silhouette score error: {e_sil}") # e.g. if all points in one cluster after filtering noise
                    silhouette_metric_value = 0.0 # Default on error

            if current_score_weight_davies_bouldin > 0: # Use current_score_weight_davies_bouldin
                try:
                    db_score_raw = davies_bouldin_score(data_for_clustering_current, labels)
                    # Normalize Davies-Bouldin: lower is better (0 is best).
                    # Transform to make higher better, roughly in [0, 1] range.
                    # (1 / (1 + davies_bouldin_score))
                    davies_bouldin_metric_value = 1.0 / (1.0 + db_score_raw)
                except ValueError as e_db:
                    print(f"{log_prefix} Iteration {run_idx}: Davies-Bouldin score error: {e_db}")
                    davies_bouldin_metric_value = 0.0 # Ensure it's 0 on error

            if current_score_weight_calinski_harabasz > 0: # Use current_score_weight_calinski_harabasz
                try:
                    ch_score_raw = calinski_harabasz_score(data_for_clustering_current, labels)
                    # Normalize Calinski-Harabasz using diminishing returns.
                    # The scaling factor (previously 5.0) needs to be larger to prevent
                    # exp(-ch_score_raw / factor) from becoming 0 too quickly for typical CH scores.
                    calinski_harabasz_metric_value = 1.0 - np.exp(-ch_score_raw / 500.0)  # Increased scaling factor
                except ValueError as e_ch:
                    print(f"{log_prefix} Iteration {run_idx}: Calinski-Harabasz score error: {e_ch}")
                    calinski_harabasz_metric_value = 0.0 # Ensure it's 0 on error
        del data_for_clustering_current # Free memory

        if labels is None or len(set(labels) - {-1}) == 0:
            # print(f"{log_prefix} Iteration {run_idx}: No valid clusters found.")
            return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}} # Add scaler_details

        max_dist = raw_distances.max()
        normalized_distances = raw_distances / max_dist if max_dist > 0 else raw_distances
        # track_info_list now uses data_subset_for_clustering
        track_info_list = [{"row": data_subset_for_clustering[i], "label": labels[i], "distance": normalized_distances[i]} for i in range(len(data_subset_for_clustering))]

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
        unique_predominant_other_feature_scores = {} # New: For other feature diversity

        for label_val, songs_list in filtered_clusters.items():
            if songs_list:
                center_val = cluster_centers_map.get(label_val)
                if center_val is None or len(center_val) == 0: continue # Ensure centroid exists and is not empty
                # Pass scaler_details to name_cluster
                name, top_scores = name_cluster(center_val, pca_model_for_this_iteration, pca_config["enabled"], MOOD_LABELS, scaler_details)
                if top_scores and any(mood in MOOD_LABELS for mood in top_scores.keys()):
                    predominant_mood_key = max((k for k in top_scores if k in MOOD_LABELS), key=top_scores.get, default=None)
                    if predominant_mood_key:
                        current_mood_score = top_scores.get(predominant_mood_key, 0.0)
                        unique_predominant_mood_scores[predominant_mood_key] = max(unique_predominant_mood_scores.get(predominant_mood_key, 0.0), current_mood_score)

                # New: Extract predominant other feature for diversity score
                centroid_other_features_from_top_scores = {
                    label: top_scores.get(label, 0.0)
                    for label in OTHER_FEATURE_LABELS if label in top_scores
                } # Semicolon was here, removed for consistency, though not a functional error.
                predominant_other_feature_key_for_diversity = None
                max_other_feature_score_for_diversity = 0.3 # Threshold for considering a feature predominant
                if centroid_other_features_from_top_scores: # Check for other features
                    for feature_label, feature_score in centroid_other_features_from_top_scores.items():
                        if feature_score > max_other_feature_score_for_diversity:
                            max_other_feature_score_for_diversity = feature_score
                            predominant_other_feature_key_for_diversity = feature_label
                
                if predominant_other_feature_key_for_diversity:
                    unique_predominant_other_feature_scores[predominant_other_feature_key_for_diversity] = max(
                        unique_predominant_other_feature_scores.get(predominant_other_feature_key_for_diversity, 0.0),
                        max_other_feature_score_for_diversity
                    )
                current_named_playlists[name].extend(songs_list)
                current_playlist_centroids[name] = top_scores

        # --- Enhanced Score Calculation ---
        # Mood Diversity Score (raw_mood_diversity_score -> base_diversity_score after LN and scaling)
        raw_mood_diversity_score = sum(unique_predominant_mood_scores.values())
        base_diversity_score = 0.0  # This will be the final scaled score

        if len(MOOD_LABELS) > 0: # Ensure MOOD_LABELS is not empty
            # 1. Apply LN transformation
            ln_mood_diversity = np.log1p(raw_mood_diversity_score) # log1p(x) = log(1+x)

            # 2. Apply Z-score standardization using configured stats
            # ASSUMPTION: RAW_MOOD_DIVERSITY_STATS in config.py will be updated to include a "mean" key.
            # E.g., RAW_MOOD_DIVERSITY_STATS = {"min": ..., "max": ..., "mean": ..., "sd": ...}
            # The "sd" from config is used directly. For a more conventional Z-score of log-transformed data,
            # the SD of the log-transformed data itself would be used.
            raw_mean_diversity = RAW_MOOD_DIVERSITY_STATS.get("mean")
            raw_sd_diversity = RAW_MOOD_DIVERSITY_STATS.get("sd")

            if raw_mean_diversity is None or raw_sd_diversity is None:
                print(f"{log_prefix} Iteration {run_idx}: 'mean' or 'sd' missing in RAW_MOOD_DIVERSITY_STATS. Mood diversity score set to 0.")
                base_diversity_score = 0.0
            else:
                log_transformed_mean_diversity = np.log1p(raw_mean_diversity)
                if abs(raw_sd_diversity) < 1e-9: # Check if SD is effectively zero
                    # If SD is zero, and value is at the mean, Z-score is 0. Otherwise, it's undefined/infinity.
                    # Setting to 0 to prevent extreme score impact.
                    base_diversity_score = 0.0
                else:
                    base_diversity_score = (ln_mood_diversity - log_transformed_mean_diversity) / raw_sd_diversity
                    # Note: Z-scores are not typically clipped to [0,1]. Previous clip removed.
        else:
            base_diversity_score = 0.0

        # Mood Purity Score (raw_playlist_purity_component -> playlist_purity_component after LN and scaling)
        raw_playlist_purity_component = 0.0
        all_individual_playlist_purities = []
        # item_id_to_song_index_map must now refer to the current data_subset_for_clustering
        item_id_to_song_index_map = {row['item_id']: i for i, row in enumerate(data_subset_for_clustering)}

        if current_named_playlists:
            for playlist_name_key, songs_in_playlist_info_list in current_named_playlists.items():
                # playlist_name_key is the generated name like "Rock_Fast"
                # songs_in_playlist_info_list is list of (item_id, title, author)

                # Get the centroid data for this named playlist
                playlist_centroid_mood_data = current_playlist_centroids.get(playlist_name_key)
                if not playlist_centroid_mood_data or not songs_in_playlist_info_list:
                    continue

                # Get all mood scores from the centroid for this playlist
                centroid_mood_scores_for_purity_calc = {
                    m_label: playlist_centroid_mood_data.get(m_label, 0.0)
                    for m_label in MOOD_LABELS # Ensure we consider all possible moods
                    if m_label in playlist_centroid_mood_data # And that the centroid has a score for it
                }

                if not centroid_mood_scores_for_purity_calc: # No mood data in centroid
                    continue

                # Sort centroid moods by score to get the top K (TOP_K_MOODS_FOR_PURITY_CALCULATION)
                sorted_centroid_moods_for_purity = sorted(
                    centroid_mood_scores_for_purity_calc.items(),
                    key=lambda item: item[1],
                    reverse=True
                )
                
                # Consider moods with at least some score (e.g., > 0.01)
                top_k_centroid_mood_labels_for_purity = [
                    mood_label for mood_label, score in sorted_centroid_moods_for_purity[:TOP_K_MOODS_FOR_PURITY_CALCULATION] if score > 0.01
                ]

                if not top_k_centroid_mood_labels_for_purity: # No significant moods in centroid's top K
                    all_individual_playlist_purities.append(0.0)
                    continue
                
                # For each song, find its max score among these top_k_centroid_mood_labels_for_purity
                current_playlist_song_purity_scores = []
                for item_id, _, _ in songs_in_playlist_info_list:
                    song_original_index = item_id_to_song_index_map.get(item_id)
                    if song_original_index is not None:
                        # Corrected slicing for mood scores: X_original = [tempo, energy, moods..., other_features...]
                        song_mood_scores_vector = X_original[song_original_index][2 : 2 + len(MOOD_LABELS)]
                        
                        max_score_for_song_among_top_centroid_moods = 0.0
                        for centroid_mood_label in top_k_centroid_mood_labels_for_purity:
                            try:
                                mood_idx = MOOD_LABELS.index(centroid_mood_label)
                                if mood_idx < len(song_mood_scores_vector):
                                    song_score_for_this_centroid_mood = song_mood_scores_vector[mood_idx]
                                    if song_score_for_this_centroid_mood > max_score_for_song_among_top_centroid_moods:
                                        max_score_for_song_among_top_centroid_moods = song_score_for_this_centroid_mood
                            except ValueError: # Should not happen if MOOD_LABELS is consistent
                                pass 
                        current_playlist_song_purity_scores.append(max_score_for_song_among_top_centroid_moods)

                if current_playlist_song_purity_scores:
                    sum_purity_for_this_playlist = sum(current_playlist_song_purity_scores) # Changed from average to sum
                    all_individual_playlist_purities.append(sum_purity_for_this_playlist) # Add sum to list

        if all_individual_playlist_purities:
            raw_playlist_purity_component = sum(all_individual_playlist_purities)

        # 1. Apply LN transformation to the raw summed purity
        ln_mood_purity = np.log1p(raw_playlist_purity_component)

        # 2. Apply Z-score standardization using configured stats
        # ASSUMPTION: RAW_MOOD_PURITY_STATS in config.py will be updated to include a "mean" key.
        # E.g., RAW_MOOD_PURITY_STATS = {"min": ..., "max": ..., "mean": ..., "sd": ...}
        # The "sd" from config is used directly.
        raw_mean_purity = RAW_MOOD_PURITY_STATS.get("mean")
        raw_sd_purity = RAW_MOOD_PURITY_STATS.get("sd")

        if raw_mean_purity is None or raw_sd_purity is None:
            print(f"{log_prefix} Iteration {run_idx}: 'mean' or 'sd' missing in RAW_MOOD_PURITY_STATS. Mood purity score set to 0.")
            playlist_purity_component = 0.0
        else:
            log_transformed_mean_purity = np.log1p(raw_mean_purity)
            if abs(raw_sd_purity) < 1e-9: # Check if SD is effectively zero
                playlist_purity_component = 0.0
            else:
                playlist_purity_component = (ln_mood_purity - log_transformed_mean_purity) / raw_sd_purity
                # Note: Z-scores are not typically clipped to [0,1]. Previous clip removed.

        # If raw_playlist_purity_component was 0 (e.g., no playlists or all songs had 0 relevant mood scores),
        # then ln_mood_purity will be 0.
        # If raw_sd_purity is not zero, playlist_purity_component will be -log_transformed_mean_purity / raw_sd_purity.
        # This is the expected behavior for Z-scores when the value is 0.


        # New: Calculate other_features_diversity_score
        raw_other_features_diversity_score = sum(unique_predominant_other_feature_scores.values())
        # Normalize other feature diversity score
        if len(OTHER_FEATURE_LABELS) > 0:
            other_features_diversity_score = raw_other_features_diversity_score / len(OTHER_FEATURE_LABELS)
        else:
            other_features_diversity_score = 0.0

        # New: Calculate other_feature_purity_component
        all_individual_playlist_other_feature_purities = []
        OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY = 0.3 # Can be tuned

        if current_named_playlists and OTHER_FEATURE_LABELS: # Ensure there are other features defined
            for playlist_name_key, songs_in_playlist_info_list in current_named_playlists.items():
                playlist_centroid_data = current_playlist_centroids.get(playlist_name_key)
                if not playlist_centroid_data or not songs_in_playlist_info_list:
                    continue

                centroid_other_features_for_purity = {
                    label: playlist_centroid_data.get(label, 0.0)
                    for label in OTHER_FEATURE_LABELS if label in playlist_centroid_data
                }

                predominant_other_feature_for_this_playlist = None
                max_score_for_predominant_other = OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY
                if centroid_other_features_for_purity:
                    for feature_label, feature_score in centroid_other_features_for_purity.items():
                        if feature_score > max_score_for_predominant_other:
                            max_score_for_predominant_other = feature_score
                            predominant_other_feature_for_this_playlist = feature_label
                
                if not predominant_other_feature_for_this_playlist:
                    continue
                
                try:
                    predominant_other_feature_index_in_labels = OTHER_FEATURE_LABELS.index(predominant_other_feature_for_this_playlist)
                except ValueError:
                    print(f"{log_prefix} Iteration {run_idx}: Warning: Predominant other feature '{predominant_other_feature_for_this_playlist}' for playlist '{playlist_name_key}' not in OTHER_FEATURE_LABELS list.")
                    continue

                scores_of_predominant_other_feature_for_songs = []
                for item_id, _, _ in songs_in_playlist_info_list:
                    song_original_index = item_id_to_song_index_map.get(item_id)
                    if song_original_index is not None:
                        other_features_start_index = 2 + len(MOOD_LABELS)
                        song_other_features_vector = X_original[song_original_index][other_features_start_index:]
                        if predominant_other_feature_index_in_labels < len(song_other_features_vector):
                            song_specific_score = song_other_features_vector[predominant_other_feature_index_in_labels]
                            scores_of_predominant_other_feature_for_songs.append(song_specific_score)
                
                if scores_of_predominant_other_feature_for_songs:
                    avg_other_feature_purity = sum(scores_of_predominant_other_feature_for_songs) / len(scores_of_predominant_other_feature_for_songs)
                    all_individual_playlist_other_feature_purities.append(avg_other_feature_purity)

        other_feature_purity_component = 0.0
        if all_individual_playlist_other_feature_purities:
            other_feature_purity_component = sum(all_individual_playlist_other_feature_purities) / len(all_individual_playlist_other_feature_purities)

        final_enhanced_score = (current_score_weight_diversity * base_diversity_score) + \
                               (SCORE_WEIGHT_PURITY * playlist_purity_component) + \
                               (SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY * other_features_diversity_score) + \
                               (SCORE_WEIGHT_OTHER_FEATURE_PURITY * other_feature_purity_component) + \
                               (current_score_weight_silhouette * silhouette_metric_value) + \
                               (current_score_weight_davies_bouldin * davies_bouldin_metric_value) + \
                               (current_score_weight_calinski_harabasz * calinski_harabasz_metric_value)

        print(f"{log_prefix} Iteration {run_idx}: "
              f"Scores -> MoodDiv: {base_diversity_score:.2f}, MoodPur: {playlist_purity_component:.2f}, "
              f"OtherFeatDiv: {other_features_diversity_score:.2f}, OtherFeatPur: {other_feature_purity_component:.2f}, "
              f"Sil: {silhouette_metric_value:.2f}, DB: {davies_bouldin_metric_value:.2f}, CH: {calinski_harabasz_metric_value:.2f}, "
              f"FinalScore: {final_enhanced_score:.2f} (Weights: MoodDiv={current_score_weight_diversity}, "
              f"MoodPur={SCORE_WEIGHT_PURITY}, OtherFeatDiv={SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY}, "
              f"OtherFeatPur={SCORE_WEIGHT_OTHER_FEATURE_PURITY}, Sil={current_score_weight_silhouette}, "
              f"DB={current_score_weight_davies_bouldin}, CH={current_score_weight_calinski_harabasz})")

        pca_model_details = {"n_components": pca_model_for_this_iteration.n_components_, "explained_variance_ratio": pca_model_for_this_iteration.explained_variance_ratio_.tolist(), "mean": pca_model_for_this_iteration.mean_.tolist()} if pca_model_for_this_iteration and pca_config["enabled"] else None
        result = {
            "diversity_score": float(final_enhanced_score), # Use the new enhanced score
            "named_playlists": dict(current_named_playlists),
            "playlist_centroids": current_playlist_centroids,
            "pca_model_details": pca_model_details,
            "scaler_details": scaler_details, # Store scaler details here
            "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}
        }
        return result
    except Exception as e_iter:
        print(f"{log_prefix} Iteration {run_idx} failed: {e_iter}")
        traceback.print_exc()
        return None # Indicate failure for this iteration

# --- New Stratified Sampling Helper Function ---
def _get_stratified_song_subset(
    genre_to_full_track_data_map, # All available tracks, categorized by genre
    target_songs_per_genre,       # Desired count per genre
    previous_subset_track_ids=None, # List of item_ids from the previous subset
    percentage_change=0.0         # Percentage of songs to swap out for mutation
):
    """
    Generates a stratified sample of songs for clustering.
    If previous_subset_track_ids is provided and percentage_change > 0,
    it perturbs the existing subset while maintaining stratification.
    Otherwise, it generates a fresh stratified sample.
    """
    new_subset_tracks_list = [] # Use a list to build the subset
    # Use a set for efficient addition and uniqueness check for item_ids in the current iteration
    current_ids_in_new_subset_for_this_iteration = set()

    current_subset_track_ids_set = set(previous_subset_track_ids) if previous_subset_track_ids else set()
    current_subset_tracks_by_genre = defaultdict(list)

    if previous_subset_track_ids:
        all_tracks_flat = [track for genre_list in genre_to_full_track_data_map.values() for track in genre_list]
        id_to_track_map = {track['item_id']: track for track in all_tracks_flat}

        for track_id in current_subset_track_ids_set:
            track_data = id_to_track_map.get(track_id)
            if track_data:
                # Find the primary stratified genre for this track
                top_mood_found = None
                if 'mood_vector' in track_data and track_data['mood_vector']:
                    mood_scores = {}
                    for pair in track_data['mood_vector'].split(','):
                        if ':' in pair:
                            label, score = pair.split(':')
                            mood_scores[label] = float(score)
                    
                    # Find the highest scoring mood that is also a stratified genre
                    top_mood_in_stratified = None
                    max_mood_score = -1
                    for genre in STRATIFIED_GENRES:
                        if genre in mood_scores and mood_scores[genre] > max_mood_score:
                            max_mood_score = mood_scores[genre]
                            top_mood_in_stratified = genre
                    
                    if top_mood_in_stratified:
                        current_subset_tracks_by_genre[top_mood_in_stratified].append(track_data)
                    else:
                        # If no stratified genre is predominant, add to a 'misc' category for now
                        current_subset_tracks_by_genre['__misc__'].append(track_data)
                else:
                    current_subset_tracks_by_genre['__misc__'].append(track_data)

    for genre in STRATIFIED_GENRES:
        available_tracks_for_genre = genre_to_full_track_data_map.get(genre, [])
        num_available = len(available_tracks_for_genre)
        songs_added_for_this_genre_count = 0

        # 1. Keep songs from the previous subset for this genre (if applicable)
        if previous_subset_track_ids and percentage_change > 0:
            tracks_from_previous_for_this_genre = current_subset_tracks_by_genre.get(genre, [])
            num_to_keep_from_previous = int(len(tracks_from_previous_for_this_genre) * (1.0 - percentage_change))
            
            # Ensure we don't try to keep more than available or more than the target
            num_to_keep_from_previous = min(num_to_keep_from_previous, len(tracks_from_previous_for_this_genre), target_songs_per_genre)

            if num_to_keep_from_previous > 0:
                kept_tracks_for_genre = random.sample(tracks_from_previous_for_this_genre, num_to_keep_from_previous)
                for track_to_keep in kept_tracks_for_genre:
                    if track_to_keep['item_id'] not in current_ids_in_new_subset_for_this_iteration:
                        new_subset_tracks_list.append(track_to_keep)
                        current_ids_in_new_subset_for_this_iteration.add(track_to_keep['item_id'])
                        songs_added_for_this_genre_count += 1
                        if songs_added_for_this_genre_count >= target_songs_per_genre:
                            break # Reached target for this genre
            # print(f"  Genre {genre}: Kept {songs_added_for_this_genre_count} tracks from previous subset.")

        # 2. Add new songs if still needed for this genre to reach its target
        num_still_needed_for_genre = target_songs_per_genre - songs_added_for_this_genre_count

        if num_still_needed_for_genre > 0 and num_available > 0:
            # Get candidates from the full pool for this genre, excluding those already added in this iteration
            new_candidates = [
                track for track in available_tracks_for_genre
                if track['item_id'] not in current_ids_in_new_subset_for_this_iteration
            ]

            num_new_to_add_for_genre = min(num_still_needed_for_genre, len(new_candidates))

            if num_new_to_add_for_genre > 0:
                selected_new_for_genre = random.sample(new_candidates, num_new_to_add_for_genre)
                for track_to_add in selected_new_for_genre:
                    # Double check uniqueness, though `new_candidates` should already handle it
                    if track_to_add['item_id'] not in current_ids_in_new_subset_for_this_iteration:
                        new_subset_tracks_list.append(track_to_add)
                        current_ids_in_new_subset_for_this_iteration.add(track_to_add['item_id'])
                        songs_added_for_this_genre_count += 1 # This count is per-genre, not strictly needed here
                                                            # as we are just filling up to num_new_to_add_for_genre
                # print(f"  Genre {genre}: Added {num_new_to_add_for_genre} new tracks.")

    # If, after stratified sampling, the total number of tracks is less than expected
    # (e.g., due to very few tracks in some genres), fill with random tracks from anywhere.
    # This ensures a minimum dataset size for clustering, but might break perfect stratification.
    # This step is a fallback. The user might want strictly stratified, even if it means fewer total songs.
    # For now, I will prioritize strict stratification for the target genres.
    # If a genre has fewer than target_songs_per_genre, it will simply contribute all its available songs.

    # Shuffle the final subset to ensure random order for clustering
    random.shuffle(new_subset_tracks_list)
    return new_subset_tracks_list


def run_clustering_batch_task(
    batch_id_str, start_run_idx, num_iterations_in_batch,
    all_tracks_data_json, # All tracks, used for initial categorization and sampling
    genre_to_full_track_data_map_json, # Pre-categorized full track data
    target_songs_per_genre, # The dynamically determined target count
    sampling_percentage_change_per_run,
    clustering_method,
    num_clusters_min_max_tuple,
    dbscan_params_ranges_dict,
    gmm_params_ranges_dict,
    pca_params_ranges_dict,
    max_songs_per_cluster,
    parent_task_id,
    score_weight_diversity_param,
    score_weight_silhouette_param,
    score_weight_davies_bouldin_param, # Added Davies-Bouldin weight
    score_weight_calinski_harabasz_param, # Added Calinski-Harabasz weight
    elite_solutions_params_list_json, # No default, will be passed positionally
    exploitation_probability,         # No default, will be passed positionally
    mutation_config_json,             # No default, will be passed positionally
    initial_subset_track_ids_json     # No default, will be passed positionally
    ):
    """RQ task to run a batch of clustering iterations with stratified sampling."""
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

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
            # all_tracks_data is now actually full track objects
            all_tracks_data_parsed = json.loads(all_tracks_data_json)
            genre_to_full_track_data_map = json.loads(genre_to_full_track_data_map_json)

            current_sampled_track_ids_in_batch = []
            if initial_subset_track_ids_json:
                current_sampled_track_ids_in_batch = json.loads(initial_subset_track_ids_json)
            
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

            # Get initial subset for the first run of this batch
            current_subset_data = _get_stratified_song_subset(
                genre_to_full_track_data_map,
                target_songs_per_genre,
                previous_subset_track_ids=current_sampled_track_ids_in_batch, # Use the passed initial subset for the first run
                percentage_change=0.0 if start_run_idx == 0 else SAMPLING_PERCENTAGE_CHANGE_PER_RUN # No change for first run of the very first batch
            )
            # Update current_sampled_track_ids_in_batch for subsequent iterations within this batch
            current_sampled_track_ids_in_batch = [t['item_id'] for t in current_subset_data]


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
                            return {"status": "REVOKED", "message": "Batch task revoked mid-process.", "iterations_completed_in_batch": iterations_actually_completed, "best_result_from_batch": best_result_in_this_batch, "final_subset_track_ids": current_sampled_track_ids_in_batch} # Return final subset IDs

                # Get the song subset for this specific iteration (perturb if not the very first run of first batch)
                # For subsequent iterations within the same batch, perturb the current_sampled_track_ids_in_batch
                if i > 0 or start_run_idx > 0: # If it's not the very first iteration of the very first batch
                    current_subset_data = _get_stratified_song_subset(
                        genre_to_full_track_data_map,
                        target_songs_per_genre,
                        previous_subset_track_ids=current_sampled_track_ids_in_batch, # Use the last sampled IDs
                        percentage_change=SAMPLING_PERCENTAGE_CHANGE_PER_RUN
                    )
                    current_sampled_track_ids_in_batch = [t['item_id'] for t in current_subset_data] # Update IDs for next iteration

                iteration_result = _perform_single_clustering_iteration(
                    current_run_global_idx, current_subset_data, # Pass the current subset
                    clustering_method, num_clusters_min_max_tuple, dbscan_params_ranges_dict, gmm_params_ranges_dict, pca_params_ranges_dict,
                    max_songs_per_cluster, log_prefix=log_prefix_for_iter,
                    elite_solutions_params_list=(elite_solutions_params_list_for_iter if elite_solutions_params_list_for_iter else []),
                    exploitation_probability=exploitation_probability,
                    mutation_config=(mutation_config_for_iter if mutation_config_for_iter else {}),
                    score_weight_diversity_override=score_weight_diversity_param,
                    score_weight_silhouette_override=score_weight_silhouette_param,
                    score_weight_davies_bouldin_override=score_weight_davies_bouldin_param,       # Pass down DB weight
                    score_weight_calinski_harabasz_override=score_weight_calinski_harabasz_param # Pass down CH weight
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
                "full_best_result_from_batch": best_result_in_this_batch,
                "final_subset_track_ids": current_sampled_track_ids_in_batch # Return the last subset used for the batch
            }
            log_and_update_batch_task(f"Batch {batch_id_str} complete. Best score in batch: {best_score_in_this_batch:.2f}", 100, details_extra=final_batch_summary, task_state=TASK_STATUS_SUCCESS)
            return {"status": "SUCCESS", "iterations_completed_in_batch": iterations_actually_completed, "best_result_from_batch": best_result_in_this_batch, "final_subset_track_ids": current_sampled_track_ids_in_batch}

        except Exception as e:
            error_tb = traceback.format_exc()
            failure_details = {"error": str(e), "traceback": error_tb, "batch_id": batch_id_str}
            log_and_update_batch_task(f"Failed clustering batch {batch_id_str}: {e}", current_progress_batch, details_extra=failure_details, task_state=TASK_STATUS_FAILURE)
            print(f"ERROR: Clustering batch {batch_id_str} failed: {e}\n{error_tb}")
            # Ensure final_subset_track_ids is returned even on failure for subsequent batches to pick up
            return {"status": "FAILURE", "iterations_completed_in_batch": iterations_actually_completed, "best_result_from_batch": None, "final_subset_track_ids": current_sampled_track_ids_in_batch}


def run_clustering_task(
    clustering_method, num_clusters_min, num_clusters_max,
    dbscan_eps_min, dbscan_eps_max, dbscan_min_samples_min, dbscan_min_samples_max, # Keep these
    pca_components_min, pca_components_max, num_clustering_runs, max_songs_per_cluster,
    gmm_n_components_min, gmm_n_components_max, # GMM params
    score_weight_diversity_param, score_weight_silhouette_param, # Existing score weights
    score_weight_davies_bouldin_param, score_weight_calinski_harabasz_param, # New score weights for DB and CH
    ai_model_provider_param, ollama_server_url_param, ollama_model_name_param, # AI params must be after new score weights
    gemini_api_key_param, gemini_model_name_param):
    """Main RQ task for clustering and playlist generation, including AI naming options."""
    current_job = get_current_job(redis_conn)
    # Use job ID if available, otherwise generate one (though it should always be available for an RQ task)
    current_task_id = current_job.id if current_job else str(uuid.uuid4()) # type: ignore

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
            "clustering_run_job_ids": [],
            "score_weight_diversity_for_run": score_weight_diversity_param,
            "score_weight_silhouette_for_run": score_weight_silhouette_param,
            "score_weight_davies_bouldin_for_run": score_weight_davies_bouldin_param,     # Log DB weight
            "score_weight_calinski_harabasz_for_run": score_weight_calinski_harabasz_param, # Log CH weight
            # Add AI config to initial details for logging/status
            "ai_model_provider_for_run": ai_model_provider_param,
            "ollama_model_name_for_run": ollama_model_name_param,
            "gemini_model_name_for_run": gemini_model_name_param,
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
                # Only include keys that are actually present in the accumulated details
                if key in _main_task_accumulated_details and _main_task_accumulated_details[key] is not None:
                    meta_details[key] = _main_task_accumulated_details[key]
            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.meta['details'] = meta_details
                current_job.save_meta()
            save_task_status(current_task_id, "main_clustering", task_state, progress=progress, details=_main_task_accumulated_details)

        try:
            # --- Assign AI Config from Parameters for this run ---
            ai_model_provider_for_run = ai_model_provider_param
            ollama_server_url_for_run = ollama_server_url_param
            ollama_model_name_for_run = ollama_model_name_param
            gemini_api_key_for_run = gemini_api_key_param
            gemini_model_name_for_run = gemini_model_name_param


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

            # --- Stratified Sampling Preparation ---
            genre_to_full_track_data_map = defaultdict(list)
            for row in rows:
                if 'mood_vector' in row and row['mood_vector']:
                    mood_scores = {}
                    for pair in row['mood_vector'].split(','):
                        if ':' in pair:
                            label, score_str = pair.split(':')
                            mood_scores[label] = float(score_str)
                    
                    # Find the top mood for this track among the stratified genres
                    top_stratified_genre = None
                    max_score = -1.0
                    for genre in STRATIFIED_GENRES:
                        if genre in mood_scores and mood_scores[genre] > max_score:
                            max_score = mood_scores[genre]
                            top_stratified_genre = genre
                    
                    if top_stratified_genre:
                        genre_to_full_track_data_map[top_stratified_genre].append(row)
                    else:
                        # Fallback for tracks that don't strongly belong to a stratified genre
                        # We still want to include them in the overall pool, but not explicitly sample by genre
                        genre_to_full_track_data_map['__other__'].append(row)
                else:
                    genre_to_full_track_data_map['__other__'].append(row)

            # --- Determine the dynamic target_songs_per_genre based on percentile ---
            songs_counts_for_stratified_genres = []
            for genre in STRATIFIED_GENRES:
                if genre in genre_to_full_track_data_map:
                    songs_counts_for_stratified_genres.append(len(genre_to_full_track_data_map[genre]))

            calculated_target_based_on_percentile = 0
            if songs_counts_for_stratified_genres: # Avoid division by zero if list is empty
                # Use np.percentile. Ensure percentile is within [0, 100]
                percentile_to_use = np.clip(STRATIFIED_SAMPLING_TARGET_PERCENTILE, 0, 100)
                calculated_target_based_on_percentile = np.percentile(songs_counts_for_stratified_genres, percentile_to_use) # type: ignore
                log_and_update_main_clustering(
                    f"{percentile_to_use}th percentile of songs per stratified genre: {calculated_target_based_on_percentile:.2f}",
                    current_progress, print_console=False
                )
            else:
                log_and_update_main_clustering("No songs found for any stratified genres. Defaulting target.", current_progress, print_console=False)

            # Target is the max of the configured minimum and the calculated percentile value
            target_songs_per_genre = max(MIN_SONGS_PER_GENRE_FOR_STRATIFICATION, int(np.floor(calculated_target_based_on_percentile))) # type: ignore
            # If no stratified genres have enough songs, or total available is low, adjust target
            # Ensure target_songs_per_genre is not zero if there are any songs at all
            if target_songs_per_genre == 0 and len(rows) > 0:
                target_songs_per_genre = 1 # At least one song per genre if possible
            log_and_update_main_clustering(f"Determined target songs per genre for stratification: {target_songs_per_genre}", 7)

            # Store the raw track data in a serializable format for passing to batch jobs
            all_tracks_data_json = json.dumps([dict(row) for row in rows]) # This should be the original full data
            # Convert DictRow to dict for JSON serialization to prevent type errors in child tasks
            genre_to_full_track_data_map_serializable = defaultdict(list)
            for genre_key, track_list_val in genre_to_full_track_data_map.items():
                genre_to_full_track_data_map_serializable[genre_key] = [dict(track_item) for track_item in track_list_val]
            genre_to_full_track_data_map_json = json.dumps(genre_to_full_track_data_map_serializable)
            # --- End Stratified Sampling Preparation ---

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
            from app import rq_queue as main_rq_queue # Ensure we use the main queue
            active_jobs_map = {}
            total_iterations_completed_count = 0
            next_batch_job_idx_to_launch = 0
            num_total_batch_jobs = (num_clustering_runs + ITERATIONS_PER_BATCH_JOB - 1) // ITERATIONS_PER_BATCH_JOB
            _main_task_accumulated_details["total_batch_jobs"] = num_total_batch_jobs # Store total batches
            batches_completed_count = 0

            # Store the last subset IDs from the previous batch. This will be updated by batch tasks.
            last_subset_track_ids_for_batch_chaining = None

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
                            # Job missing from RQ, and DB status is not terminal.
                            # Treat as completed (abnormally) to prevent stall.
                            db_status_for_log = db_task_info_child.get('status') if db_task_info_child else "UNKNOWN (not in DB)"
                            print(f"[MainClusteringTask-{current_task_id}] Warning: Active batch job {job_id} missing from RQ, DB status '{db_status_for_log}' is not terminal. Treating as completed (abnormally) to prevent stall.")
                            is_child_truly_completed_this_cycle = True # Ensure it's marked completed
                    except Exception as e_monitor_child_active:
                        print(f"[MainClusteringTask-{current_task_id}] ERROR monitoring active batch job {job_id}: {e_monitor_child_active}. Treating as completed.")
                        # traceback.print_exc() # Potentially too verbose for this specific case
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
                            
                            # Update last_subset_track_ids_for_batch_chaining with the result from the *last* iteration of this batch
                            # This ensures continuity across batches
                            if "final_subset_track_ids" in batch_job_result and batch_job_result["final_subset_track_ids"] is not None:
                                last_subset_track_ids_for_batch_chaining = batch_job_result["final_subset_track_ids"]

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

                        # Pass the last_subset_track_ids_for_batch_chaining to the *new* batch job
                        # This ensures the new batch continues the sampling perturbation from where the previous left off
                        initial_subset_for_this_batch_json = json.dumps(last_subset_track_ids_for_batch_chaining) if last_subset_track_ids_for_batch_chaining else "[]"

                        new_job = main_rq_queue.enqueue(
                            run_clustering_batch_task,
                            args=(
                                batch_id_for_logging, current_batch_start_run_idx, num_iterations_for_this_batch,
                                all_tracks_data_json, # Original full data, used for initial sampling inside batch
                                genre_to_full_track_data_map_json, # Pre-categorized track data (JSON string)
                                target_songs_per_genre,
                                SAMPLING_PERCENTAGE_CHANGE_PER_RUN,
                                clustering_method, num_clusters_min_max_tuple_for_batch, dbscan_params_ranges_dict_for_batch,
                                gmm_params_ranges_dict_for_batch, pca_params_ranges_dict_for_batch,
                                max_songs_per_cluster, current_task_id,
                                score_weight_diversity_param, score_weight_silhouette_param, # Pass down to batch task
                                score_weight_davies_bouldin_param, score_weight_calinski_harabasz_param, # Pass DB & CH weights
                                current_elite_params_for_batch_json,
                                exploitation_prob_for_this_batch, # AI params are not passed to batch, but to main task
                                mutation_config_json,
                                initial_subset_for_this_batch_json # Pass the subset from the end of the previous batch
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
            final_pca_model_details = best_clustering_results["pca_model_details"] # Retrieve PCA details
            final_scaler_details = best_clustering_results["scaler_details"]     # Retrieve Scaler details

            # --- AI Playlist Naming for the BEST result ---
            log_prefix_main_task_ai = f"[MainClusteringTask-{current_task_id} AI Naming]"

            # Check if AI naming is enabled based on the provider parameter
            if ai_model_provider_for_run in ["OLLAMA", "GEMINI"]:
                print(f"{log_prefix_main_task_ai} AI Naming block entered. Attempting to import 'ai' module.")

                # Define the progress range for AI Naming
                # Adjust progress range based on total task progress (90-100 is finalization)
                # Let's use 90-95 for AI naming
                ai_naming_start_progress = 90
                ai_naming_end_progress = 95
                ai_naming_progress_range = ai_naming_end_progress - ai_naming_start_progress # 5%

                # Set initial progress for AI naming phase *only if* AI is enabled
                current_progress = ai_naming_start_progress
                log_and_update_main_clustering(f"Preparing for AI Naming...", current_progress, print_console=True) # Initial log for AI phase
                try:
                    # The ai module and creative_prompt_template are already imported at the top
                    # from ai import get_ai_playlist_name, creative_prompt_template # Redundant import

                    ai_renamed_playlists_final = defaultdict(list)
                    ai_renamed_centroids_final = {}
                    total_playlists_to_name = len(final_named_playlists)
                    playlists_named_count = 0

                    # Recreate PCA model if details are available for AI naming
                    pca_model_for_ai_naming = None
                    if final_pca_model_details and final_pca_model_details.get("n_components", 0) > 0:
                        try:
                            pca_model_for_ai_naming = PCA(n_components=final_pca_model_details["n_components"])
                            pca_model_for_ai_naming.mean_ = np.array(final_pca_model_details["mean"])

                            pass # No direct PCA model reconstruction needed here, name_cluster handles it via details.
                        except Exception as e_pca_details_ai:
                            print(f"{log_prefix_main_task_ai} Warning: Could not handle PCA model details for AI naming: {e_pca_details_ai}. Proceeding without specific PCA model for AI.")
                            # pca_model_for_ai_naming is already None by default if this block fails or isn't entered.
                            # No explicit assignment to None needed here unless it was modified within the try.

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
                        # The get_ai_playlist_name function will take this and append the song list.
                        # So, we pass the template itself, and the individual components to the function for Ollama, using passed-in params.
                        print(f"{log_prefix_main_task_ai} Generating AI name for '{original_name}' ({len(song_list_for_ai)} songs) using provider '{ai_model_provider_for_run}'. F1: '{feature1}', F2: '{feature2}', F3: '{feature3}'.")
                        # Retrieve the comprehensive centroid features for this playlist
                        centroid_features_for_ai = final_playlist_centroids.get(original_name, {})
                        ai_generated_name_str = get_ai_playlist_name(
                            ai_model_provider_for_run,
                            ollama_server_url_for_run, ollama_model_name_for_run,
                            gemini_api_key_for_run, gemini_model_name_for_run, # Pass Gemini params
                            creative_prompt_template,
                            feature1, feature2, feature3, # Pass all three features
                            song_list_for_ai,
                            centroid_features_for_ai) # Pass the comprehensive centroid features

                        # Debug print for the raw AI-generated name string
                        print(f"{log_prefix_main_task_ai} Raw AI output for '{original_name}': '{ai_generated_name_str}'")

                        current_playlist_final_name = original_name
                        # Check if the generated name is a valid name (not an error or skip message)
                        if ai_generated_name_str and not ai_generated_name_str.startswith("Error") and not ai_generated_name_str.startswith("AI Naming Skipped"):
                             clean_ai_name = ai_generated_name_str.strip().replace("\n", " ")
                             if clean_ai_name:
                                 # Check if the generated name is significantly different or just the original name
                                 # This helps avoid unnecessary logging if AI just returns the input name
                                 # Also ensure it's not empty after cleaning.
                                 if clean_ai_name.lower() != original_name.lower().strip().replace("_", " "):
                                      print(f"{log_prefix_main_task_ai} AI: '{original_name}' -> '{clean_ai_name}'")
                                      current_playlist_final_name = clean_ai_name
                                 else:
                                      # print(f"{log_prefix_main_task_ai} AI returned name similar to original or empty for '{original_name}'. Using original.")
                                      current_playlist_final_name = original_name # Explicitly keep original if AI name is same after cleaning
                             else:
                                 print(f"{log_prefix_main_task_ai} AI for '{original_name}' returned empty after cleaning. Raw: '{ai_generated_name_str}'. Using original.")
                        else:
                            print(f"{log_prefix_main_task_ai} AI naming for '{original_name}' failed or returned error/skip message: '{ai_generated_name_str}'. Using original.")

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

                    # CRITICAL FIX: Update the main variables with AI-generated names
                    if ai_renamed_playlists_final: # Ensure it's not empty if AI processing happened
                        final_named_playlists = ai_renamed_playlists_final
                        final_playlist_centroids = ai_renamed_centroids_final
                    log_and_update_main_clustering(f"{log_prefix_main_task_ai} AI Naming for best playlist set completed.", ai_naming_end_progress, print_console=True)
                except ImportError:
                    print(f"{log_prefix_main_task_ai} Could not import 'ai' module. Skipping AI naming for final playlists.")
                    traceback.print_exc()
                    # If AI fails, jump to the start of the next phase (DB update)
                    current_progress = ai_naming_end_progress # Set progress to 95
                    log_and_update_main_clustering(f"{log_prefix_main_task_ai} AI Naming skipped due to import error.", current_progress, print_console=True)
                except Exception as e_ai_final:
                    print(f"{log_prefix_main_task_ai} Error during final AI playlist naming: {e_ai_final}. Using original names.")
                    traceback.print_exc()
                    # If AI fails, jump to the start of the next phase (DB update)
                    current_progress = ai_naming_end_progress
                    log_and_update_main_clustering(f"{log_prefix_main_task_ai} AI Naming skipped due to error.", current_progress, print_console=True)
            else:
                # Log why AI naming was skipped if provider is NONE or invalid
                print(f"{log_prefix_main_task_ai} AI Naming skipped: Provider is '{ai_model_provider_for_run}'.")
            # --- End AI Playlist Naming for BEST result ---

            # --- Final Playlist Naming and Suffixing ---
            # Use the potentially AI-renamed playlists (or original if AI was skipped/failed)
            # Append the _automatic suffix to the final name before creating on Jellyfin
            playlists_to_create_on_jellyfin = {}
            centroids_for_jellyfin_playlists = {}

            for name, songs in final_named_playlists.items():
                 # Append the _automatic suffix to the final name before creating on Jellyfin
                 final_name_with_suffix = f"{name}_automatic"
                 playlists_to_create_on_jellyfin[final_name_with_suffix] = songs
                 # Keep the centroid info associated with the new name
                 if name in final_playlist_centroids:
                     centroids_for_jellyfin_playlists[final_name_with_suffix] = final_playlist_centroids[name]

            current_progress = 96 # Adjust progress slightly for this step
            log_and_update_main_clustering("Applying '_automatic' suffix to playlist names...", current_progress, print_console=True)
            # --- End Final Playlist Naming and Suffixing ---

            current_progress = 98
            log_and_update_main_clustering("Creating/Updating playlists on Jellyfin...", current_progress, print_console=False)
            # Use the suffixed names for Jellyfin creation
            create_or_update_playlists_on_jellyfin(JELLYFIN_URL, JELLYFIN_USER_ID, {"X-Emby-Token": JELLYFIN_TOKEN},
                                                    playlists_to_create_on_jellyfin, centroids_for_jellyfin_playlists,
                                                    MOOD_LABELS, final_max_songs_per_cluster)

            final_db_summary = {
                "best_score": best_diversity_score,
                "best_params": best_clustering_results.get("parameters"),
                "num_playlists_created": len(playlists_to_create_on_jellyfin) # Count suffixed playlists
            }
            current_progress = 100
            # Update playlist table AFTER AI naming, suffixing, and AFTER Jellyfin creation attempt
            log_and_update_main_clustering("Updating playlist database with final suffixed names...", current_progress, print_console=True, task_state=TASK_STATUS_PROGRESS)
            update_playlist_table(playlists_to_create_on_jellyfin) # Use suffixed names for DB update
            log_and_update_main_clustering("Playlist database updated.", current_progress, print_console=True, task_state=TASK_STATUS_PROGRESS)

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
            # Removed direct deletion of all_tracks_data_json as it's now handled by the function's scope.
            # If large data is processed in a temporary file, that cleanup would still be needed.
            pass

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
                                               "best_result_from_batch": details_dict.get("full_best_result_from_batch"),
                                               "final_subset_track_ids": details_dict.get("final_subset_track_ids")} # Include final subset IDs
                        elif 'full_result' in details_dict: # Fallback for old single run tasks if any
                            run_result_data = details_dict['full_result']
                        else:
                            print(f"[ParentTask-{parent_task_id_for_logging}] Warning: {task_type_for_logging} {job_id_for_result} (DB status: {db_status}) has no 'full_best_result_from_batch' or 'full_result' in details.")
                    except (json.JSONDecodeError, TypeError) as e_json:
                        print(f"[ParentTask-{parent_task_id_for_logging}] Warning: Could not parse details for {task_type_for_logging} {job_id_for_result} from DB: {e_json}")
            elif db_status in [TASK_STATUS_FAILURE, TASK_STATUS_REVOKED, JobStatus.CANCELED, JobStatus.FAILED]:
                print(f"[ParentTask-{parent_task_id_for_logging}] Info: {task_type_for_logging} {job_id_for_result} (DB status: {db_status}) did not succeed. No result to process.")
    return run_result_data