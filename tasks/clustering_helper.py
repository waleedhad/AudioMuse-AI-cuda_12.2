# tasks/clustering_helper.py

import json
import random
import traceback # Added for _perform_single_clustering_iteration

import numpy as np # type: ignore
import requests
from collections import defaultdict

# Sklearn imports for _perform_single_clustering_iteration
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture # type: ignore
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score # type: ignore


# RQ import (Needed for get_job_result_safely)
from rq.job import Job
from rq.exceptions import NoSuchJobError

# Import necessary constants from config
from config import STRATIFIED_GENRES, OTHER_FEATURE_LABELS, MOOD_LABELS
from config import (MAX_DISTANCE, MAX_SONGS_PER_CLUSTER, MAX_SONGS_PER_ARTIST,
                    GMM_COVARIANCE_TYPE, ENABLE_CLUSTERING_EMBEDDINGS,
                    SCORE_WEIGHT_SILHOUETTE, SCORE_WEIGHT_DAVIES_BOULDIN, SCORE_WEIGHT_CALINSKI_HARABASZ,
                    SCORE_WEIGHT_DIVERSITY, SCORE_WEIGHT_PURITY, SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY, SCORE_WEIGHT_OTHER_FEATURE_PURITY,
                    MUTATION_KMEANS_COORD_FRACTION, MUTATION_INT_ABS_DELTA, MUTATION_FLOAT_ABS_DELTA,
                    TOP_K_MOODS_FOR_PURITY_CALCULATION, LN_MOOD_DIVERSITY_STATS, LN_MOOD_PURITY_STATS,
                    LN_MOOD_DIVERSITY_EMBEDING_STATS, LN_MOOD_PURITY_EMBEDING_STATS, LN_OTHER_FEATURES_DIVERSITY_STATS, LN_OTHER_FEATURES_PURITY_STATS,
                    OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY as CONFIG_OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY,
                    USE_MINIBATCH_KMEANS, MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE, DB_FETCH_CHUNK_SIZE) # Added DB_FETCH_CHUNK_SIZE

# Import from app (for get_job_result_safely and potentially others if they evolve)
from app import (app, redis_conn, get_db, save_task_status, get_task_info_from_db,
                track_exists, save_track_analysis, get_all_tracks, get_tracks_by_ids, 
                get_score_data_by_ids, update_playlist_table, JobStatus, # Added get_score_data_by_ids
                TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS, TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)

# Import from commons for _perform_single_clustering_iteration
from .commons import score_vector

import logging

# Note: JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN will be passed as arguments if needed by functions here.

def _get_stratified_song_subset(
    genre_to_track_data_map, # Can be full or lightweight data
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
        all_tracks_flat = [track for genre_list in genre_to_track_data_map.values() for track in genre_list]
        id_to_track_map = {track['item_id']: track for track in all_tracks_flat}

        for track_id in current_subset_track_ids_set:
            track_data = id_to_track_map.get(track_id)
            if track_data:
                # Find the primary stratified genre for this track
                if 'mood_vector' in track_data and track_data['mood_vector']:
                    mood_scores = {}
                    for pair in track_data['mood_vector'].split(','):
                        if ':' in pair:
                            label, score_val_str = pair.split(':') # Renamed score to score_val_str
                            mood_scores[label] = float(score_val_str)
                    
                    top_mood_in_stratified = None
                    max_mood_score = -1
                    for genre in STRATIFIED_GENRES:
                        if genre in mood_scores and mood_scores[genre] > max_mood_score:
                            max_mood_score = mood_scores[genre]
                            top_mood_in_stratified = genre
                    
                    if top_mood_in_stratified:
                        current_subset_tracks_by_genre[top_mood_in_stratified].append(track_data)
                    else:
                        current_subset_tracks_by_genre['__misc__'].append(track_data)
                else:
                    current_subset_tracks_by_genre['__misc__'].append(track_data)

    for genre in STRATIFIED_GENRES:
        available_tracks_for_genre = genre_to_track_data_map.get(genre, [])
        num_available = len(available_tracks_for_genre)
        songs_added_for_this_genre_count = 0

        if previous_subset_track_ids and percentage_change > 0:
            tracks_from_previous_for_this_genre = current_subset_tracks_by_genre.get(genre, [])
            num_to_keep_from_previous = int(len(tracks_from_previous_for_this_genre) * (1.0 - percentage_change))
            num_to_keep_from_previous = min(num_to_keep_from_previous, len(tracks_from_previous_for_this_genre), target_songs_per_genre)

            if num_to_keep_from_previous > 0:
                kept_tracks_for_genre = random.sample(tracks_from_previous_for_this_genre, num_to_keep_from_previous)
                for track_to_keep in kept_tracks_for_genre:
                    if track_to_keep['item_id'] not in current_ids_in_new_subset_for_this_iteration:
                        new_subset_tracks_list.append(track_to_keep)
                        current_ids_in_new_subset_for_this_iteration.add(track_to_keep['item_id'])
                        songs_added_for_this_genre_count += 1
                        if songs_added_for_this_genre_count >= target_songs_per_genre:
                            break 

        num_still_needed_for_genre = target_songs_per_genre - songs_added_for_this_genre_count

        if num_still_needed_for_genre > 0 and num_available > 0:
            new_candidates = [
                track for track in available_tracks_for_genre
                if track['item_id'] not in current_ids_in_new_subset_for_this_iteration
            ]
            num_new_to_add_for_genre = min(num_still_needed_for_genre, len(new_candidates))

            if num_new_to_add_for_genre > 0:
                selected_new_for_genre = random.sample(new_candidates, num_new_to_add_for_genre)
                for track_to_add in selected_new_for_genre:
                    if track_to_add['item_id'] not in current_ids_in_new_subset_for_this_iteration:
                        new_subset_tracks_list.append(track_to_add)
                        current_ids_in_new_subset_for_this_iteration.add(track_to_add['item_id'])
    random.shuffle(new_subset_tracks_list)
    return new_subset_tracks_list

def name_cluster(centroid_scaled_vector, pca_model, pca_enabled, mood_labels_list, scaler_details=None):
    """Generates a human-readable name for a cluster. Now includes standardization inverse transform."""
    interpreted_vector = centroid_scaled_vector
    if pca_enabled and pca_model is not None:
        try:
            interpreted_vector = pca_model.inverse_transform(interpreted_vector.reshape(1, -1))[0]
        except ValueError:
            logging.warning("PCA inverse_transform failed. Using PCA space for interpretation.")
            pass
    if scaler_details:
        try:
            temp_scaler = StandardScaler()
            temp_scaler.mean_ = np.array(scaler_details["mean"])
            temp_scaler.scale_ = np.array(scaler_details["scale"])
            temp_scaler.n_features_in_ = len(temp_scaler.mean_)
            if len(interpreted_vector) != len(temp_scaler.mean_):
                final_interpreted_raw_vector = interpreted_vector
            else:
                final_interpreted_raw_vector = temp_scaler.inverse_transform(interpreted_vector.reshape(1, -1))[0]
        except Exception as e:
            logging.warning("StandardScaler inverse_transform failed: %s. Using previously interpreted vector.", e)
            final_interpreted_raw_vector = interpreted_vector
    else:
        final_interpreted_raw_vector = interpreted_vector

    tempo_val = final_interpreted_raw_vector[0]
    energy_val = final_interpreted_raw_vector[1]
    mood_values = final_interpreted_raw_vector[2 : 2 + len(mood_labels_list)]
    other_feature_values = final_interpreted_raw_vector[2 + len(mood_labels_list):]
    other_feature_scores_dict = {}
    if len(other_feature_values) == len(OTHER_FEATURE_LABELS):
        other_feature_scores_dict = dict(zip(OTHER_FEATURE_LABELS, other_feature_values))

    if tempo_val < 0.33: tempo_label = "Slow"
    elif tempo_val < 0.66: tempo_label = "Medium"
    else: tempo_label = "Fast"
    energy_label = "Low Energy" if energy_val < 0.3 else ("Medium Energy" if energy_val < 0.7 else "High Energy")

    if len(mood_values) == 0 or np.sum(mood_values) == 0: top_indices = []
    else: top_indices = np.argsort(mood_values)[::-1][:3]
    mood_names = [mood_labels_list[i] for i in top_indices if i < len(mood_labels_list)]
    mood_part = "_".join(mood_names).title() if mood_names else "Mixed"
    base_name = f"{mood_part}_{tempo_label}"

    OTHER_FEATURE_THRESHOLD_FOR_NAME = 0.5
    MAX_OTHER_FEATURES_IN_NAME = 2
    appended_other_features_str = ""
    if other_feature_scores_dict:
        prominent_other_features = sorted(
            [(feature, score) for feature, score in other_feature_scores_dict.items() if score >= OTHER_FEATURE_THRESHOLD_FOR_NAME],
            key=lambda item: item[1], reverse=True
        )
        other_features_to_add_to_name_list = [feature.title() for feature, score in prominent_other_features[:MAX_OTHER_FEATURES_IN_NAME]]
        if other_features_to_add_to_name_list:
            appended_other_features_str = "_" + "_".join(other_features_to_add_to_name_list)
    full_name = f"{base_name}{appended_other_features_str}"
    top_mood_scores = {mood_labels_list[i]: mood_values[i] for i in top_indices if i < len(mood_labels_list)}
    extra_info = {"tempo_normalized": round(tempo_val, 2), "energy_normalized": round(energy_val, 2)}
    comprehensive_centroid_details = {**top_mood_scores, **extra_info, **other_feature_scores_dict}
    return full_name, comprehensive_centroid_details

def delete_old_automatic_playlists(jellyfin_url, jellyfin_user_id, headers):
    url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        for item in r.json().get("Items", []):
            if "_automatic" in item.get("Name", ""):
                del_url = f"{jellyfin_url}/Items/{item['Id']}"
                del_resp = requests.delete(del_url, headers=headers, timeout=10) # nosec
                if del_resp.ok: logging.info("🗑️ Deleted old playlist: %s", item['Name'])
    except Exception as e: # nosec
        logging.error("Failed to clean old playlists: %s", e, exc_info=True)

def create_or_update_playlists_on_jellyfin(jellyfin_url_param, jellyfin_user_id_param, headers_param, playlists, cluster_centers, mood_labels_list, max_songs_per_cluster_param):
    delete_old_automatic_playlists(jellyfin_url_param, jellyfin_user_id_param, headers_param)
    for base_name, cluster in playlists.items():
        chunks = []
        if max_songs_per_cluster_param > 0:
            chunks = [cluster[i:i+max_songs_per_cluster_param] for i in range(0, len(cluster), max_songs_per_cluster_param)]
        else:
            if cluster: chunks = [cluster]
        for idx, chunk in enumerate(chunks, 1):
            playlist_name_on_jellyfin = f"{base_name} ({idx})" if len(chunks) > 1 else base_name
            item_ids = [item_id for item_id, _, _ in chunk]
            if not item_ids: continue
            body = {"Name": playlist_name_on_jellyfin, "Ids": item_ids, "UserId": jellyfin_user_id_param}
            try:
                r = requests.post(f"{jellyfin_url_param}/Playlists", headers=headers_param, json=body, timeout=60)
                if r.ok:
                    centroid_info = cluster_centers.get(base_name, {})
                    top_moods = {k: v for k, v in centroid_info.items() if k in mood_labels_list} # Use full MOOD_LABELS for checking
                    extra_info = {k: v for k, v in centroid_info.items() if k not in mood_labels_list}
                    centroid_str = ", ".join(f"{k}:{v:.2f}" for k, v in top_moods.items())
                    extras_str = ", ".join(f"{k}:{v:.2f}" for k, v in extra_info.items()) # type: ignore
                    logging.info("✅ Created playlist '%s' with %s tracks (Centroid for '%s': %s | %s)", playlist_name_on_jellyfin, len(item_ids), base_name, centroid_str, extras_str)
            except Exception as e: # nosec
                logging.error("Exception creating '%s': %s", playlist_name_on_jellyfin, e, exc_info=True)

def _mutate_param(value, min_val, max_val, delta, is_float=False, round_digits=None):
    if is_float:
        mutation = random.uniform(-delta, delta)
        new_value = value + mutation
        if round_digits is not None: new_value = round(new_value, round_digits)
    else:
        int_delta = max(1, int(delta)) if isinstance(delta, float) else int(delta)
        mutation = random.randint(-int_delta, int_delta)
        new_value = value + mutation
    new_value = np.clip(new_value, min_val, max_val)
    return int(new_value) if not is_float else new_value

def _generate_or_mutate_kmeans_initial_centroids(n_clusters_current, data_for_clustering_current, elite_kmeans_params_original, elite_pca_config_original, new_pca_config_current, mutation_coord_fraction=0.05, log_prefix=""):
    use_mutated_elite_centroids = False
    if elite_kmeans_params_original and elite_pca_config_original:
        elite_initial_centroids_list = elite_kmeans_params_original.get("initial_centroids")
        elite_n_clusters = elite_kmeans_params_original.get("n_clusters")
        pca_compatible = (elite_pca_config_original.get("enabled") == new_pca_config_current.get("enabled"))
        if pca_compatible and elite_pca_config_original.get("enabled"): pca_compatible = (elite_pca_config_original.get("components") == new_pca_config_current.get("components"))
        if elite_initial_centroids_list and isinstance(elite_initial_centroids_list, list) and pca_compatible and elite_n_clusters == n_clusters_current:
            use_mutated_elite_centroids = True
    if use_mutated_elite_centroids:
        mutated_centroids = []
        elite_centroids_arr = np.array(elite_initial_centroids_list)
        if data_for_clustering_current.shape[0] > 0 and data_for_clustering_current.shape[1] > 0 and elite_centroids_arr.ndim == 2 and elite_centroids_arr.shape[1] == data_for_clustering_current.shape[1]:
            data_min, data_max = np.min(data_for_clustering_current, axis=0), np.max(data_for_clustering_current, axis=0)
            coord_mutation_deltas = (data_max - data_min) * mutation_coord_fraction
            for centroid_coords in elite_centroids_arr:
                mutated_coord = np.clip(centroid_coords + np.random.uniform(-coord_mutation_deltas, coord_mutation_deltas, size=centroid_coords.shape), data_min, data_max)
                mutated_centroids.append(mutated_coord.tolist())
            return mutated_centroids
    if data_for_clustering_current.shape[0] == 0 or n_clusters_current == 0: return []
    num_available_points = data_for_clustering_current.shape[0]
    actual_n_clusters = min(n_clusters_current, num_available_points)
    if actual_n_clusters == 0 and num_available_points > 0: actual_n_clusters = 1
    if actual_n_clusters == 0: return []
    indices = np.random.choice(num_available_points, actual_n_clusters, replace=(actual_n_clusters > num_available_points))
    return data_for_clustering_current[indices].tolist()

def get_job_result_safely(job_id_for_result, parent_task_id_for_logging, task_type_for_logging="child task"):
    """
    Safely retrieves the result of an RQ job, checking both RQ and the database.
    Useful for long-running tasks where the job might expire from RQ but its status
    and result details are saved in the database.
    """
    run_result_data = None
    job_instance = None
    try:
        job_instance = Job.fetch(job_id_for_result, connection=redis_conn)
        job_instance.refresh()
        if job_instance.is_finished and isinstance(job_instance.result, dict):
            run_result_data = job_instance.result
    except NoSuchJobError:
        logging.warning("[ParentTask-%s] Warning: %s %s not found in RQ. Checking DB.", parent_task_id_for_logging, task_type_for_logging, job_id_for_result)
    except Exception as e_rq_fetch:
        logging.error("[ParentTask-%s] Error fetching %s %s from RQ: %s. Will check DB.", parent_task_id_for_logging, task_type_for_logging, job_id_for_result, e_rq_fetch)

    if run_result_data is None:
        with app.app_context():
            db_task_info = get_task_info_from_db(job_id_for_result)
        if db_task_info:
            db_status = db_task_info.get('status')
            if db_status in [TASK_STATUS_SUCCESS, JobStatus.FINISHED]:
                if db_task_info.get('details'):
                    try:
                        details_dict = json.loads(db_task_info.get('details'))
                        if 'full_best_result_from_batch' in details_dict:
                            run_result_data = {"status": "SUCCESS",
                                               "iterations_completed_in_batch": details_dict.get("iterations_completed_in_batch", 0),
                                               "best_result_from_batch": details_dict.get("full_best_result_from_batch"),
                                               "final_subset_track_ids": details_dict.get("final_subset_track_ids")}
                        elif 'full_result' in details_dict: # For older single iteration tasks if any
                            run_result_data = details_dict['full_result'] # type: ignore
                        else:
                            logging.warning("[ParentTask-%s] Warning: %s %s (DB status: %s) has no 'full_best_result_from_batch' or 'full_result' in details.", parent_task_id_for_logging, task_type_for_logging, job_id_for_result, db_status)
                    except (json.JSONDecodeError, TypeError) as e_json:
                        logging.warning("[ParentTask-%s] Warning: Could not parse details for %s %s from DB: %s", parent_task_id_for_logging, task_type_for_logging, job_id_for_result, e_json)
            elif db_status in [TASK_STATUS_FAILURE, TASK_STATUS_REVOKED, JobStatus.CANCELED, JobStatus.FAILED]:
                logging.info("[ParentTask-%s] Info: %s %s (DB status: %s) did not succeed. No result to process.", parent_task_id_for_logging, task_type_for_logging, job_id_for_result, db_status)
    return run_result_data

def _perform_single_clustering_iteration(
    run_idx, item_ids_for_subset, 
    clustering_method, num_clusters_min_max, dbscan_params_ranges, gmm_params_ranges, pca_params_ranges, active_mood_labels, 
    max_songs_per_cluster, log_prefix="",
    elite_solutions_params_list=None, exploitation_probability=0.0, mutation_config=None,
    score_weight_diversity_override=None, score_weight_silhouette_override=None, 
    score_weight_davies_bouldin_override=None, score_weight_calinski_harabasz_override=None, 
    score_weight_purity_override=None,
    score_weight_other_feature_diversity_override=None, 
    score_weight_other_feature_purity_override=None,
    enable_clustering_embeddings_param=None):
    try:
        elite_solutions_params_list = elite_solutions_params_list or []
        mutation_config = mutation_config or {"int_abs_delta": MUTATION_INT_ABS_DELTA, "float_abs_delta": MUTATION_FLOAT_ABS_DELTA, "coord_mutation_fraction": MUTATION_KMEANS_COORD_FRACTION}
        if "coord_mutation_fraction" not in mutation_config: 
            mutation_config["coord_mutation_fraction"] = MUTATION_KMEANS_COORD_FRACTION

        current_score_weight_diversity = score_weight_diversity_override if score_weight_diversity_override is not None else SCORE_WEIGHT_DIVERSITY
        current_score_weight_silhouette = score_weight_silhouette_override if score_weight_silhouette_override is not None else SCORE_WEIGHT_SILHOUETTE
        current_score_weight_davies_bouldin = score_weight_davies_bouldin_override if score_weight_davies_bouldin_override is not None else SCORE_WEIGHT_DAVIES_BOULDIN
        current_score_weight_calinski_harabasz = score_weight_calinski_harabasz_override if score_weight_calinski_harabasz_override is not None else SCORE_WEIGHT_CALINSKI_HARABASZ
        current_score_weight_purity = score_weight_purity_override if score_weight_purity_override is not None else SCORE_WEIGHT_PURITY
        current_score_weight_other_feature_diversity = score_weight_other_feature_diversity_override if score_weight_other_feature_diversity_override is not None else SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY 
        current_score_weight_other_feature_purity = score_weight_other_feature_purity_override if score_weight_other_feature_purity_override is not None else SCORE_WEIGHT_OTHER_FEATURE_PURITY 
        
        use_embeddings_for_this_iteration = enable_clustering_embeddings_param if enable_clustering_embeddings_param is not None else ENABLE_CLUSTERING_EMBEDDINGS

        if not item_ids_for_subset:
            logging.warning("%s Iteration %s: Received empty item_ids_for_subset. Cannot cluster.", log_prefix, run_idx)
            return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": None, "parameters": {}}

        # Determine if embeddings should be fetched in the initial comprehensive data load.
        # This is true if embeddings are used AND it's NOT MiniBatchKMeans.
        fetch_embeddings_in_initial_load = use_embeddings_for_this_iteration and not USE_MINIBATCH_KMEANS

        logging.info("%s Iteration %s: Fetching initial data for %s tracks. Will fetch embeddings in this initial load: %s.", log_prefix, run_idx, len(item_ids_for_subset), fetch_embeddings_in_initial_load)
        all_track_data_for_subset_rows_list = []
        with app.app_context():
            for i_chunk in range(0, len(item_ids_for_subset), DB_FETCH_CHUNK_SIZE):
                chunk_ids = item_ids_for_subset[i_chunk : i_chunk + DB_FETCH_CHUNK_SIZE]
                if chunk_ids:
                    if fetch_embeddings_in_initial_load:
                        # Fetches score AND embedding columns
                        chunk_rows = get_tracks_by_ids(chunk_ids) 
                    else:
                        # Fetches ONLY score columns
                        chunk_rows = get_score_data_by_ids(chunk_ids)
                    all_track_data_for_subset_rows_list.extend(chunk_rows)
        
        valid_tracks_for_processing = [dict(row) for row in all_track_data_for_subset_rows_list if row]
        logging.info("%s Iteration %s: Retrieved %s valid track records from DB for initial processing.", log_prefix, run_idx, len(valid_tracks_for_processing))

        X_feat_orig_list = []
        # X_embed_raw_list will only be populated if using embeddings AND NOT MiniBatchKMeans
        X_embed_raw_list_for_non_mbk = [] 

        for track_row_data in valid_tracks_for_processing:
            try:
                feature_vec = score_vector(track_row_data, active_mood_labels, OTHER_FEATURE_LABELS)
                X_feat_orig_list.append(feature_vec)
                
                # Populate X_embed_raw_list_for_non_mbk only if embeddings were fetched in the initial load
                if fetch_embeddings_in_initial_load: # This implies use_embeddings_for_this_iteration and not USE_MINIBATCH_KMEANS
                    raw_embedding_value = track_row_data.get('embedding_vector')
                    parsed_list_for_numpy_non_mbk = None
                    if isinstance(raw_embedding_value, list):
                        parsed_list_for_numpy_non_mbk = raw_embedding_value
                    elif isinstance(raw_embedding_value, str):
                        if raw_embedding_value.strip() and raw_embedding_value.lower() != 'null':
                            try:
                                loaded_from_string = json.loads(raw_embedding_value)
                                if isinstance(loaded_from_string, list):
                                    parsed_list_for_numpy_non_mbk = loaded_from_string
                            except (json.JSONDecodeError, TypeError):
                                logging.warning("%s Iteration %s: Non-MBK - Failed to parse embedding string for %s. Skipping embedding for this track.", log_prefix, run_idx, track_row_data.get('item_id'))
                    
                    if parsed_list_for_numpy_non_mbk is not None:
                        try:
                            # Embedding is expected to be 1D from the DB now
                            emb_vec = np.array(parsed_list_for_numpy_non_mbk) 
                            if isinstance(emb_vec, np.ndarray) and emb_vec.ndim == 1 and np.issubdtype(emb_vec.dtype, np.number) and emb_vec.size > 0:
                                X_embed_raw_list_for_non_mbk.append(emb_vec)
                        except Exception as e_np_non_mbk: # Catch numpy conversion errors
                            logging.warning("%s Iteration %s: Non-MBK - NumPy error processing embedding for %s. Skipping embedding for this track.", log_prefix, run_idx, track_row_data.get('item_id'))
            except (json.JSONDecodeError, TypeError) as e: # Error from score_vector
                logging.warning("%s Iteration %s: Skipping track %s in initial processing due to data parsing error: %s", log_prefix, run_idx, track_row_data.get('item_id', 'Unknown ID'), e)
                continue
        
        if not X_feat_orig_list:
            logging.error("%s Iteration %s: No valid feature vectors (X_feat_orig) could be constructed. Cannot cluster.", log_prefix, run_idx)
            return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": None, "parameters": {}}
        X_feat_orig = np.array(X_feat_orig_list)

        temp_aligned_valid_tracks = []
        temp_aligned_X_feat_orig_list = []
        temp_aligned_X_embed_raw_list_for_non_mbk = []
        processed_ids_for_alignment = set()

        for track_data_align in valid_tracks_for_processing:
            item_id_align = track_data_align.get('item_id')
            if not item_id_align or item_id_align in processed_ids_for_alignment:
                continue
            try:
                f_vec = score_vector(track_data_align, active_mood_labels, OTHER_FEATURE_LABELS)
                temp_aligned_X_feat_orig_list.append(f_vec)
                temp_aligned_valid_tracks.append(track_data_align)
                processed_ids_for_alignment.add(item_id_align)

                if fetch_embeddings_in_initial_load:
                    raw_embedding_value_align = track_data_align.get('embedding_vector')
                    parsed_list_align = None
                    if isinstance(raw_embedding_value_align, list):
                        parsed_list_align = raw_embedding_value_align
                    elif isinstance(raw_embedding_value_align, str):
                        if raw_embedding_value_align.strip() and raw_embedding_value_align.lower() != 'null':
                            try: 
                                loaded_str_align = json.loads(raw_embedding_value_align)
                                if isinstance(loaded_str_align, list): parsed_list_align = loaded_str_align
                            except: pass # Ignore parsing errors here, will lead to mismatch if critical
                    if parsed_list_align is not None:
                        try:
                            # Embedding is expected to be 1D
                            emb_vec_align = np.array(parsed_list_align) 
                            if isinstance(emb_vec_align, np.ndarray) and emb_vec_align.ndim == 1 and np.issubdtype(emb_vec_align.dtype, np.number) and emb_vec_align.size > 0:
                                temp_aligned_X_embed_raw_list_for_non_mbk.append(emb_vec_align)
                        except Exception: pass # Ignore numpy errors here during alignment
            except:
                pass 
        
        X_feat_orig = np.array(temp_aligned_X_feat_orig_list)
        valid_tracks_for_processing = temp_aligned_valid_tracks 
        X_embed_raw_list_for_non_mbk = temp_aligned_X_embed_raw_list_for_non_mbk

        if X_feat_orig.shape[0] == 0:
             logging.error("%s Iteration %s: No valid feature vectors (X_feat_orig) after alignment. Cannot cluster.", log_prefix, run_idx)
             return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": None, "parameters": {}}

        scaler_details_for_run = None
        X_to_cluster_standardized = None 

        if use_embeddings_for_this_iteration:
            if not USE_MINIBATCH_KMEANS: # type: ignore
                if not X_embed_raw_list_for_non_mbk or len(X_embed_raw_list_for_non_mbk) != X_feat_orig.shape[0]:
                    logging.error("%s Iteration %s: No/mismatched embedding data for non-MiniBatchKMeans. Expected %s based on X_feat_orig, got %s. Cannot cluster with embeddings.", log_prefix, run_idx, X_feat_orig.shape[0], len(X_embed_raw_list_for_non_mbk))
                    return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": None, "parameters": {}}
                X_embed_raw = np.array(X_embed_raw_list_for_non_mbk)
                scaler_embed = StandardScaler()
                X_to_cluster_standardized = scaler_embed.fit_transform(X_embed_raw)
                scaler_details_for_run = {"mean": scaler_embed.mean_.tolist(), "scale": scaler_embed.scale_.tolist(), "type": "embedding"}
        else: 
            scaler_feat = StandardScaler()
            X_to_cluster_standardized = scaler_feat.fit_transform(X_feat_orig) 
            scaler_details_for_run = {"mean": scaler_feat.mean_.tolist(), "scale": scaler_feat.scale_.tolist(), "type": "feature"}

        if not (use_embeddings_for_this_iteration and USE_MINIBATCH_KMEANS):
            if X_to_cluster_standardized is None or X_to_cluster_standardized.shape[0] == 0:
                logging.error("%s Iteration %s: Data for clustering (X_to_cluster_standardized) is empty. Cannot proceed.", log_prefix, run_idx)
                return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details_for_run, "parameters": {}}

        data_for_clustering_current = X_to_cluster_standardized 
        
        data_for_param_gen_shape_ref = X_to_cluster_standardized if X_to_cluster_standardized is not None else X_feat_orig
        pca_model_for_this_iteration = None
        method_params_config = {}
        pca_config = {} 
        params_generated_by_mutation = False

        if elite_solutions_params_list and random.random() < exploitation_probability:
            chosen_elite_params_set = random.choice(elite_solutions_params_list)
            elite_method_config_original = chosen_elite_params_set.get("clustering_method_config")
            elite_pca_config_original = chosen_elite_params_set.get("pca_config", {"enabled": False, "components": 0})
            elite_scaler_details_original = chosen_elite_params_set.get("scaler_details") 
            scaler_compatible = (elite_scaler_details_original is not None) == (scaler_details_for_run is not None)
            if scaler_compatible and elite_scaler_details_original and scaler_details_for_run:
                scaler_compatible = elite_scaler_details_original.get("type") == scaler_details_for_run.get("type")
            
            if elite_method_config_original and elite_pca_config_original and scaler_compatible and \
               elite_method_config_original.get("method") == clustering_method:
                try:
                    pca_input_data_for_mutation = data_for_clustering_current if data_for_clustering_current is not None else data_for_param_gen_shape_ref
                    if pca_input_data_for_mutation is None or pca_input_data_for_mutation.shape[0] == 0:
                        if not (use_embeddings_for_this_iteration and USE_MINIBATCH_KMEANS):
                           raise ValueError("PCA input data for mutation is empty and not MiniBatchKMeans path.")
                    
                    data_after_pca_mutation = pca_input_data_for_mutation
                    temp_pca_config = dict(elite_pca_config_original)

                    if pca_input_data_for_mutation is not None and pca_input_data_for_mutation.shape[0] > 0 :
                        elite_pca_comps = elite_pca_config_original.get("components", pca_params_ranges["components_min"])
                        mutated_pca_comps = _mutate_param(elite_pca_comps, pca_params_ranges["components_min"], pca_params_ranges["components_max"], mutation_config.get("int_abs_delta", MUTATION_INT_ABS_DELTA))
                        max_pca_by_features = pca_input_data_for_mutation.shape[1]
                        max_pca_by_samples = (pca_input_data_for_mutation.shape[0] - 1) if pca_input_data_for_mutation.shape[0] > 1 else 1
                        max_allowable_pca_mutated = min(mutated_pca_comps, max_pca_by_features, max_pca_by_samples)
                        max_allowable_pca_mutated = max(0, max_allowable_pca_mutated)
                        temp_pca_config = {"enabled": max_allowable_pca_mutated > 0, "components": max_allowable_pca_mutated}
                        if temp_pca_config["enabled"] and temp_pca_config["components"] > 0:
                            pca_model_for_this_iteration = PCA(n_components=temp_pca_config["components"])
                            data_after_pca_mutation = pca_model_for_this_iteration.fit_transform(pca_input_data_for_mutation)
                            temp_pca_config["components"] = pca_model_for_this_iteration.n_components_
                    
                    temp_method_params_config = None
                    max_clusters_or_components = data_after_pca_mutation.shape[0] if data_after_pca_mutation is not None else 0
                    if max_clusters_or_components == 0 and not (use_embeddings_for_this_iteration and USE_MINIBATCH_KMEANS and clustering_method == "kmeans"):
                        raise ValueError("No data points available after PCA (mutation path) to determine cluster parameters.")

                    if clustering_method == "kmeans":
                        elite_n_clusters = elite_method_config_original.get("params", {}).get("n_clusters", num_clusters_min_max[0])
                        upper_bound_n_clusters = min(num_clusters_min_max[1], max_clusters_or_components) if max_clusters_or_components > 0 else num_clusters_min_max[1]
                        mutated_n_clusters = _mutate_param(elite_n_clusters, max(1, num_clusters_min_max[0]), upper_bound_n_clusters, mutation_config.get("int_abs_delta", 2))
                        temp_method_params_config = {"method": "kmeans", "params": {"n_clusters": max(1, mutated_n_clusters)}}
                    elif clustering_method == "dbscan":
                        elite_eps = elite_method_config_original.get("params", {}).get("eps", dbscan_params_ranges["eps_min"])
                        elite_min_samples = elite_method_config_original.get("params", {}).get("min_samples", dbscan_params_ranges["samples_min"])
                        mutated_eps = _mutate_param(elite_eps, dbscan_params_ranges["eps_min"], dbscan_params_ranges["eps_max"], mutation_config.get("float_abs_delta", MUTATION_FLOAT_ABS_DELTA), is_float=True, round_digits=2)
                        mutated_min_samples = _mutate_param(elite_min_samples, dbscan_params_ranges["samples_min"], dbscan_params_ranges["samples_max"], mutation_config.get("int_abs_delta", MUTATION_INT_ABS_DELTA))
                        temp_method_params_config = {"method": "dbscan", "params": {"eps": mutated_eps, "min_samples": mutated_min_samples}}
                    elif clustering_method == "gmm":
                        elite_n_components = elite_method_config_original.get("params", {}).get("n_components", gmm_params_ranges["n_components_min"])
                        upper_bound_gmm_comps = min(gmm_params_ranges["n_components_max"], max_clusters_or_components) if max_clusters_or_components > 0 else gmm_params_ranges["n_components_max"]
                        mutated_n_components = _mutate_param(elite_n_components, gmm_params_ranges["n_components_min"], upper_bound_gmm_comps, mutation_config.get("int_abs_delta", MUTATION_INT_ABS_DELTA))
                        temp_method_params_config = {"method": "gmm", "params": {"n_components": max(1, mutated_n_components)}}

                    if temp_method_params_config and temp_pca_config is not None:
                        if clustering_method == "kmeans": # type: ignore
                            centroid_gen_data = data_after_pca_mutation if data_after_pca_mutation is not None and data_after_pca_mutation.shape[0] > 0 else np.array([]) # type: ignore
                            kmeans_initial_centroids = _generate_or_mutate_kmeans_initial_centroids(
                                temp_method_params_config["params"]["n_clusters"], centroid_gen_data,
                                elite_method_config_original.get("params"), elite_pca_config_original, temp_pca_config,           
                                mutation_config.get("coord_mutation_fraction"), log_prefix=f"{log_prefix} Iteration {run_idx} (mutation)") # type: ignore
                            temp_method_params_config["params"]["initial_centroids"] = kmeans_initial_centroids
                        method_params_config = temp_method_params_config
                        pca_config = temp_pca_config
                        data_for_clustering_current = data_after_pca_mutation
                        params_generated_by_mutation = True
                except Exception as e_mutate:
                    logging.error("%s Iteration %s: Error mutating elite params: %s. Falling back to random.", log_prefix, run_idx, e_mutate, exc_info=True)
                    params_generated_by_mutation = False
                    pca_model_for_this_iteration = None 
                    data_for_clustering_current = X_to_cluster_standardized

        if not params_generated_by_mutation:
            pca_input_data_for_random = X_to_cluster_standardized if X_to_cluster_standardized is not None else data_for_param_gen_shape_ref
            data_after_pca_random = pca_input_data_for_random

            if pca_input_data_for_random is not None and pca_input_data_for_random.shape[0] > 0:
                sampled_pca_components_rand = random.randint(pca_params_ranges["components_min"], pca_params_ranges["components_max"])
                max_allowable_pca_rand = min(sampled_pca_components_rand, pca_input_data_for_random.shape[1], (pca_input_data_for_random.shape[0] - 1) if pca_input_data_for_random.shape[0] > 1 else 1)
                max_allowable_pca_rand = max(0, max_allowable_pca_rand) 
                pca_config = {"enabled": max_allowable_pca_rand > 0, "components": max_allowable_pca_rand}
                if pca_config["enabled"] and pca_config["components"] > 0:
                    n_comps_rand = min(pca_config["components"], pca_input_data_for_random.shape[1], (pca_input_data_for_random.shape[0] - 1) if pca_input_data_for_random.shape[0] > 1 else 1)
                    if n_comps_rand > 0:
                        pca_model_for_this_iteration = PCA(n_components=n_comps_rand)
                        data_after_pca_random = pca_model_for_this_iteration.fit_transform(pca_input_data_for_random)
                        pca_config["components"] = pca_model_for_this_iteration.n_components_
                    else: pca_config["enabled"] = False
            else: 
                pca_config = {"enabled": False, "components": 0}
            
            data_for_clustering_current = data_after_pca_random
            max_clusters_or_components_rand = data_for_clustering_current.shape[0] if data_for_clustering_current is not None else 0

            if max_clusters_or_components_rand == 0 and not (use_embeddings_for_this_iteration and USE_MINIBATCH_KMEANS and clustering_method == "kmeans"):
                 logging.warning("%s Iteration %s: No data points for random parameter generation.", log_prefix, run_idx)
                 return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details_for_run, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}}

            if clustering_method == "kmeans":
                upper_bound_k_rand = min(num_clusters_min_max[1], max_clusters_or_components_rand) if max_clusters_or_components_rand > 0 else num_clusters_min_max[1]
                k_rand = random.randint(max(1, num_clusters_min_max[0]), upper_bound_k_rand) 
                k_rand = max(1, k_rand)
                centroid_gen_data_rand = data_for_clustering_current if data_for_clustering_current is not None and data_for_clustering_current.shape[0] > 0 else np.array([])
                kmeans_initial_centroids_rand = _generate_or_mutate_kmeans_initial_centroids(k_rand, centroid_gen_data_rand, None, None, pca_config, log_prefix=f"{log_prefix} Iteration {run_idx} (random)")
                method_params_config = {"method": "kmeans", "params": {"n_clusters": k_rand, "initial_centroids": kmeans_initial_centroids_rand}}
            elif clustering_method == "dbscan":
                current_dbscan_eps_rand = round(random.uniform(dbscan_params_ranges["eps_min"], dbscan_params_ranges["eps_max"]), 2)
                current_dbscan_min_samples_rand = random.randint(dbscan_params_ranges["samples_min"], dbscan_params_ranges["samples_max"])
                method_params_config = {"method": "dbscan", "params": {"eps": current_dbscan_eps_rand, "min_samples": current_dbscan_min_samples_rand}}
            elif clustering_method == "gmm": 
                upper_bound_gmm_rand = min(gmm_params_ranges["n_components_max"], max_clusters_or_components_rand) if max_clusters_or_components_rand > 0 else gmm_params_ranges["n_components_max"]
                gmm_n_rand = random.randint(gmm_params_ranges["n_components_min"], upper_bound_gmm_rand)
                method_params_config = {"method": "gmm", "params": {"n_components": max(1, gmm_n_rand)}}
            else:
                logging.error("%s Iteration %s: Unsupported clustering method %s", log_prefix, run_idx, clustering_method)
                return None # type: ignore
        
        if data_for_clustering_current is None and not (use_embeddings_for_this_iteration and USE_MINIBATCH_KMEANS and method_params_config.get("method") == "kmeans"):
            data_for_clustering_current = X_to_cluster_standardized

        if pca_config.get("enabled") and pca_model_for_this_iteration is None and pca_config.get("components", 0) > 0:
            if X_to_cluster_standardized is not None and X_to_cluster_standardized.shape[0] > 0 and X_to_cluster_standardized.shape[1] > 0:
                try:
                    n_comps_refit = min(pca_config["components"], X_to_cluster_standardized.shape[1], (X_to_cluster_standardized.shape[0] - 1) if X_to_cluster_standardized.shape[0] > 1 else 1)
                    if n_comps_refit > 0:
                        pca_model_for_this_iteration = PCA(n_components=n_comps_refit)
                        data_for_clustering_current = pca_model_for_this_iteration.fit_transform(X_to_cluster_standardized)
                        pca_config["components"] = pca_model_for_this_iteration.n_components_
                    else: # nosec
                        pca_config["enabled"] = False; pca_config["components"] = 0 # type: ignore
                        data_for_clustering_current = X_to_cluster_standardized
                except Exception as e_refit_pca_guard:
                    logging.error("%s Iteration %s: Error re-fitting PCA (guard): %s. Disabling PCA.", log_prefix, run_idx, e_refit_pca_guard)
                    pca_config["enabled"] = False; pca_config["components"] = 0; pca_model_for_this_iteration = None
                    data_for_clustering_current = X_to_cluster_standardized
            else: 
                pca_config["enabled"] = False; pca_config["components"] = 0

        if not method_params_config or pca_config is None:
            logging.error("%s Iteration %s: Critical error: parameters not configured.", log_prefix, run_idx)
            return None # type: ignore
        
        labels = None
        cluster_centers_map = {}
        num_points_for_distances = len(item_ids_for_subset) if (use_embeddings_for_this_iteration and USE_MINIBATCH_KMEANS and method_params_config.get("method") == "kmeans") else (data_for_clustering_current.shape[0] if data_for_clustering_current is not None else 0)
        raw_distances = np.full(num_points_for_distances, np.inf) 

        method_from_config = method_params_config["method"]
        params_from_config = method_params_config["params"]

        if method_from_config == "kmeans":
            initial_centroids_np = None
            if params_from_config.get("initial_centroids"):
                initial_centroids_np = np.array(params_from_config["initial_centroids"])
                if initial_centroids_np.ndim == 1 and initial_centroids_np.shape[0] == 0: initial_centroids_np = None 
                elif initial_centroids_np.shape[0] != params_from_config["n_clusters"]: params_from_config["n_clusters"] = initial_centroids_np.shape[0]

            if params_from_config["n_clusters"] == 0:
                logging.warning("%s Iteration %s: n_clusters is 0 for KMeans.", log_prefix, run_idx) # type: ignore
                return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details_for_run, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}}

            if use_embeddings_for_this_iteration and USE_MINIBATCH_KMEANS:
                logging.info("%s Iteration %s: Using MiniBatchKMeans with n_clusters=%s.", log_prefix, run_idx, params_from_config['n_clusters'])
                
                # CRITICAL FIX for MiniBatchKMeans with embeddings:
                # The initial_centroids_np (if provided from params_from_config) would be based on X_feat_orig (e.g., 13 features).
                # However, the data for partial_fit will be the actual embeddings (e.g., 200 features).
                # This causes a shape mismatch.
                # Therefore, for MBK with embeddings, we MUST let it initialize its own centroids from the embedding data.
                mbk_init_strategy = 'k-means++'
                
                mbk = MiniBatchKMeans(
                    n_clusters=params_from_config["n_clusters"], 
                    init=mbk_init_strategy, # Force 'k-means++' or 'random'
                    n_init='auto', # n_init is ignored if init is callable, but good to keep for other cases
                    random_state=None, 
                    max_no_improvement=10, 
                    batch_size=MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE
                )
                
                all_standardized_embeddings_for_predict = []
                scaler_embed_mbk = StandardScaler()
                scaler_fitted_mbk = False
                num_total_tracks_for_iter = len(item_ids_for_subset)
                processed_tracks_count_mbk = 0
                original_indices_of_embedded_tracks = [] 

                # Buffer for accumulating embeddings for the first partial_fit call
                temp_buffer_for_first_fit = []
                # No need to buffer original_indices_for_first_fit, as they are globally collected
                # in original_indices_of_embedded_tracks for final label mapping.

                for chunk_start_idx in range(0, num_total_tracks_for_iter, MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE):
                    chunk_end_idx = min(chunk_start_idx + MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE, num_total_tracks_for_iter)
                    current_chunk_ids = item_ids_for_subset[chunk_start_idx:chunk_end_idx]
                    if not current_chunk_ids: continue

                    # This log remains the same, indicating the DB fetch for the small chunk
                    logging.info("%s Iteration %s: MBK - Fetching embeddings for chunk %s (IDs: %s...).", log_prefix, run_idx, chunk_start_idx // MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE + 1, current_chunk_ids[:3]) # type: ignore
                    with app.app_context():
                        chunk_db_data_rows = get_tracks_by_ids(current_chunk_ids) 
                    
                    chunk_embeddings_list = []
                    chunk_original_indices_map = []
                    for original_idx_in_subset_loop in range(chunk_start_idx, chunk_end_idx):
                        track_id_for_chunk = item_ids_for_subset[original_idx_in_subset_loop]
                        track_data_for_chunk = next((row for row in chunk_db_data_rows if row['item_id'] == track_id_for_chunk), None)
                        
                        if track_data_for_chunk:
                            raw_embedding_value = track_data_for_chunk.get('embedding_vector')
                            parsed_list_for_numpy = None

                            if isinstance(raw_embedding_value, list):
                                parsed_list_for_numpy = raw_embedding_value
                            elif isinstance(raw_embedding_value, str):
                                if raw_embedding_value.strip() and raw_embedding_value.lower() != 'null':
                                    try:
                                        loaded_from_string = json.loads(raw_embedding_value)
                                        if isinstance(loaded_from_string, list):
                                            parsed_list_for_numpy = loaded_from_string # type: ignore
                                        else:
                                            logging.warning("%s Iteration %s: MBK - Embedding string for %s parsed to JSON, but not a list. Type: %s. Skipping.", log_prefix, run_idx, track_id_for_chunk, type(loaded_from_string))
                                    except (json.JSONDecodeError, TypeError) as e_parse:
                                        logging.warning("%s Iteration %s: MBK - Failed to parse embedding JSON string for %s. Error: %s. Raw string (first 100 chars): '%s'. Skipping.", log_prefix, run_idx, track_id_for_chunk, e_parse, raw_embedding_value[:100])
                            elif raw_embedding_value is None:
                                logging.warning("%s Iteration %s: MBK - Embedding for %s is None. This was unexpected after initial filtering. Skipping.", log_prefix, run_idx, track_id_for_chunk)
                            
                            if parsed_list_for_numpy is not None:
                                try:
                                    # Embedding is expected to be 1D
                                    emb_vec = np.array(parsed_list_for_numpy) 
                                    if isinstance(emb_vec, np.ndarray) and emb_vec.ndim == 1 and np.issubdtype(emb_vec.dtype, np.number) and emb_vec.size > 0:
                                        chunk_embeddings_list.append(emb_vec)
                                        chunk_original_indices_map.append(original_idx_in_subset_loop)
                                    else:
                                        logging.warning("%s Iteration %s: MBK - Parsed embedding for %s is not a valid 1D numerical array or is empty. Type: %s, Shape: %s. Skipping.", log_prefix, run_idx, track_id_for_chunk, type(emb_vec), str(emb_vec.shape) if isinstance(emb_vec, np.ndarray) else 'Not a NumPy array')
                                except Exception as e_np:
                                    logging.warning("%s Iteration %s: MBK - NumPy error processing embedding for %s. Error: %s. Parsed list (first 100 chars): '%s'. Skipping.", log_prefix, run_idx, track_id_for_chunk, e_np, str(parsed_list_for_numpy)[:100])
                        else:
                            logging.warning("%s Iteration %s: MBK - Track ID %s from item_ids_for_subset not found in chunk_db_data_rows. This is unexpected.", log_prefix, run_idx, track_id_for_chunk)
                    
                    if not chunk_embeddings_list:
                        logging.warning("%s Iteration %s: MBK - No valid embeddings successfully processed for chunk %s (IDs: %s).", log_prefix, run_idx, chunk_start_idx // MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE + 1, current_chunk_ids)
                        processed_tracks_count_mbk += len(current_chunk_ids)
                        continue
                    
                    chunk_embeddings_array = np.array(chunk_embeddings_list)
                    if not scaler_fitted_mbk:
                        standardized_chunk_embeddings = scaler_embed_mbk.fit_transform(chunk_embeddings_array)
                        scaler_fitted_mbk = True
                        scaler_details_for_run = {"mean": scaler_embed_mbk.mean_.tolist(), "scale": scaler_embed_mbk.scale_.tolist(), "type": "embedding_mbk"}
                    else:
                        standardized_chunk_embeddings = scaler_embed_mbk.transform(chunk_embeddings_array)
                    
                    all_standardized_embeddings_for_predict.append(standardized_chunk_embeddings)
                    original_indices_of_embedded_tracks.extend(chunk_original_indices_map)

                    # Logic for handling partial_fit:
                    # hasattr check is a robust way to see if centroids are initialized.
                    # MiniBatchKMeans initializes self.cluster_centers_ after the first successful partial_fit or fit.
                    if not hasattr(mbk, "cluster_centers_") or mbk.cluster_centers_ is None:
                        temp_buffer_for_first_fit.append(standardized_chunk_embeddings)
                        current_buffered_samples = sum(arr.shape[0] for arr in temp_buffer_for_first_fit)
                        is_last_db_chunk = (chunk_end_idx >= num_total_tracks_for_iter)

                        # Trigger first partial_fit if buffer is large enough OR it's the last chunk and we have some data
                        if current_buffered_samples >= mbk.n_clusters or \
                           (is_last_db_chunk and current_buffered_samples > 0):
                            
                            combined_first_batch = np.vstack(temp_buffer_for_first_fit)
                            
                            if combined_first_batch.shape[0] < mbk.n_clusters:
                                logging.warning("%s Iteration %s: MBK - Total valid embeddings (%s) is less than target n_clusters (%s). Skipping MBK for this iteration as it would fail.", log_prefix, run_idx, combined_first_batch.shape[0], mbk.n_clusters)
                                all_standardized_embeddings_for_predict = [] # Signal failure to get valid labels
                                break # Exit the chunk processing loop for MBK
                            
                            logging.info("%s Iteration %s: MBK - Initializing centroids with first partial_fit using %s accumulated samples.", log_prefix, run_idx, combined_first_batch.shape[0])
                            mbk.partial_fit(combined_first_batch) # This initializes cluster_centers_
                            temp_buffer_for_first_fit = [] # Clear the buffer
                    else:
                        # Centroids are already initialized, proceed with normal partial_fit on the current small chunk
                        if standardized_chunk_embeddings.shape[0] > 0: # Ensure chunk is not empty
                            # print(f"{log_prefix} Iteration {run_idx}: MBK - Subsequent partial_fit with {standardized_chunk_embeddings.shape[0]} samples.")
                            mbk.partial_fit(standardized_chunk_embeddings)

                    processed_tracks_count_mbk += len(current_chunk_ids)
                    # Reduced verbosity for this line, as the fit calls are now more descriptive
                    if (chunk_start_idx // MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE + 1) % 10 == 0 or is_last_db_chunk : # Log every 10 chunks or on last # nosec
                        logging.info("%s Iteration %s: MBK - Processed DB chunk %s. Total tracks attempted: %s/%s.", log_prefix, run_idx, chunk_start_idx // MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE + 1, processed_tracks_count_mbk, num_total_tracks_for_iter)

                if not all_standardized_embeddings_for_predict:
                    logging.warning("%s Iteration %s: MBK - No embeddings processed. Defaulting to no clusters.", log_prefix, run_idx)
                    labels = np.full(len(item_ids_for_subset), -1, dtype=int)
                    cluster_centers_final = np.array([])
                else:
                    data_for_clustering_current = np.vstack(all_standardized_embeddings_for_predict) 
                    if hasattr(mbk, 'cluster_centers_') and mbk.cluster_centers_ is not None and data_for_clustering_current.shape[0] > 0:
                        labels_for_embedded_tracks = mbk.predict(data_for_clustering_current)
                        cluster_centers_final = mbk.cluster_centers_
                        labels = np.full(len(item_ids_for_subset), -1, dtype=int)
                        for i, original_idx in enumerate(original_indices_of_embedded_tracks):
                            labels[original_idx] = labels_for_embedded_tracks[i]
                        
                        temp_raw_distances = np.full(len(item_ids_for_subset), np.inf)
                        valid_labels_for_dist_calc = labels_for_embedded_tracks[labels_for_embedded_tracks != -1]
                        if len(valid_labels_for_dist_calc) > 0:
                            centers_for_embedded_points = cluster_centers_final[valid_labels_for_dist_calc]
                            embedded_points_for_dist_calc = data_for_clustering_current[labels_for_embedded_tracks != -1]
                            if embedded_points_for_dist_calc.shape[0] > 0 and centers_for_embedded_points.shape[0] == embedded_points_for_dist_calc.shape[0]:
                               calculated_distances = np.linalg.norm(embedded_points_for_dist_calc - centers_for_embedded_points, axis=1)
                               dist_idx = 0
                               for i, original_idx in enumerate(original_indices_of_embedded_tracks):
                                   if labels_for_embedded_tracks[i] != -1:
                                       if dist_idx < len(calculated_distances):
                                           temp_raw_distances[original_idx] = calculated_distances[dist_idx]
                                           dist_idx +=1
                                       else:
                                           logging.warning("%s Iteration %s: MBK - Distance calculation index out of bounds.", log_prefix, run_idx) # type: ignore
                        raw_distances = temp_raw_distances
                    else: 
                        logging.warning("%s Iteration %s: MBK - No cluster centers or no data. Defaulting to no clusters.", log_prefix, run_idx)
                        labels = np.full(len(item_ids_for_subset), -1, dtype=int)
                        cluster_centers_final = np.array([])
                if cluster_centers_final.size > 0:
                    cluster_centers_map = {i: cluster_centers_final[i] for i in range(cluster_centers_final.shape[0])}
            
            else: # Standard KMeans
                logging.info("%s Iteration %s: Using standard KMeans with n_clusters=%s.", log_prefix, run_idx, params_from_config['n_clusters'])
                if initial_centroids_np is None: # nosec
                     logging.error("%s Iteration %s: Standard KMeans requires initial_centroids. Error.", log_prefix, run_idx)
                     return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details_for_run, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}}
                if data_for_clustering_current is None or data_for_clustering_current.shape[0] == 0:
                    logging.warning("%s Iteration %s: Data for standard KMeans is empty.", log_prefix, run_idx)
                    return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details_for_run, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}}
                kmeans = KMeans(n_clusters=params_from_config["n_clusters"], init=initial_centroids_np, n_init=1, random_state=None)
                labels = kmeans.fit_predict(data_for_clustering_current)
                cluster_centers_map = {i: kmeans.cluster_centers_[i] for i in range(params_from_config["n_clusters"])}
                if data_for_clustering_current.shape[0] > 0 and labels.max() < kmeans.cluster_centers_.shape[0] and len(labels) == data_for_clustering_current.shape[0]:
                    centers_for_points = kmeans.cluster_centers_[labels]
                    raw_distances = np.linalg.norm(data_for_clustering_current - centers_for_points, axis=1)
        
        elif method_from_config == "dbscan":
            if data_for_clustering_current is None or data_for_clustering_current.shape[0] == 0: # nosec # type: ignore
                logging.warning("%s Iteration %s: Data for DBSCAN is empty. Cannot fit.", log_prefix, run_idx)
                return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details_for_run, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}}
            dbscan = DBSCAN(eps=params_from_config["eps"], min_samples=params_from_config["min_samples"])
            labels = dbscan.fit_predict(data_for_clustering_current)
            raw_distances = np.full(data_for_clustering_current.shape[0], np.inf) 
            for cluster_id_val in set(labels):
                if cluster_id_val == -1: continue
                indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id_val]
                if not indices: continue
                cluster_points = data_for_clustering_current[indices]
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    for i_idx in indices: 
                        raw_distances[i_idx] = np.linalg.norm(data_for_clustering_current[i_idx] - center)
                    cluster_centers_map[cluster_id_val] = center

        elif method_from_config == "gmm":
            if data_for_clustering_current is None or data_for_clustering_current.shape[0] == 0: # nosec # type: ignore
                logging.warning("%s Iteration %s: Data for GMM is empty. Cannot fit.", log_prefix, run_idx)
                return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details_for_run, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}}
            gmm = GaussianMixture(n_components=params_from_config["n_components"], covariance_type=GMM_COVARIANCE_TYPE, random_state=None, max_iter=1000)
            gmm.fit(data_for_clustering_current)
            labels = gmm.predict(data_for_clustering_current)
            cluster_centers_map = {i: gmm.means_[i] for i in range(params_from_config["n_components"])}
            if data_for_clustering_current.shape[0] > 0 and labels.max() < gmm.means_.shape[0] and len(labels) == data_for_clustering_current.shape[0]:
                centers_for_points = gmm.means_[labels] 
                raw_distances = np.linalg.norm(data_for_clustering_current - centers_for_points, axis=1)

        silhouette_metric_value = 0.0; davies_bouldin_metric_value = 0.0; calinski_harabasz_metric_value = 0.0
        s_score_raw_val_for_log = 0.0; db_score_raw_val_for_log = 0.0; ch_score_raw_val_for_log = 0.0 # nosec
        
        if labels is None:
            logging.warning("%s Iteration %s: Labels are None. Cannot calculate metrics.", log_prefix, run_idx) # type: ignore
            return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details_for_run, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}} # type: ignore

        num_actual_clusters = len(set(labels) - {-1})
        
        if data_for_clustering_current is None or data_for_clustering_current.shape[0] == 0:
             logging.error("%s Iteration %s: data_for_clustering_current is empty before metric calculation.", log_prefix, run_idx)
             return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details_for_run, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}}

        num_samples_for_metrics = data_for_clustering_current.shape[0]
        
        labels_for_metrics = labels
        if use_embeddings_for_this_iteration and USE_MINIBATCH_KMEANS:
            if 'mbk' in locals() and hasattr(mbk, 'cluster_centers_') and data_for_clustering_current.shape[0] > 0:
                labels_for_metrics = mbk.predict(data_for_clustering_current) # type: ignore # type: ignore
            else: 
                logging.warning("%s Iteration %s: MBK - Cannot get specific labels for metrics. Metrics might be inaccurate.", log_prefix, run_idx)
                labels_for_metrics = np.array([])

        if num_actual_clusters >= 2 and num_actual_clusters < num_samples_for_metrics and labels_for_metrics.size > 0 and len(np.unique(labels_for_metrics)) >= 2:
            # Ensure labels_for_metrics corresponds to data_for_clustering_current
            if len(labels_for_metrics) != data_for_clustering_current.shape[0]:
                logging.warning("%s Iteration %s: Mismatch between metric labels (%s) and data points (%s) for metrics. Skipping metrics.", log_prefix, run_idx, len(labels_for_metrics), data_for_clustering_current.shape[0]) # type: ignore
            else:
                if current_score_weight_silhouette > 0: 
                    try:
                        s_score = silhouette_score(data_for_clustering_current, labels_for_metrics, metric='euclidean')
                        s_score_raw_val_for_log = s_score; silhouette_metric_value = (s_score + 1) / 2.0
                    except ValueError as e_sil: logging.warning("%s Iteration %s: Silhouette error: %s", log_prefix, run_idx, e_sil); silhouette_metric_value = 0.0
                if current_score_weight_davies_bouldin != 0: # Check if weight is non-zero (positive weight is expected now)
                    try:
                        db_score_raw = davies_bouldin_score(data_for_clustering_current, labels_for_metrics)
                        db_score_raw_val_for_log = db_score_raw
                        davies_bouldin_metric_value = 1.0 / (1.0 + db_score_raw) # Higher is better, maps to (0, 1]
                    except ValueError as e_db: logging.warning("%s Iteration %s: Davies-Bouldin error: %s", log_prefix, run_idx, e_db); davies_bouldin_metric_value = 0.0 
                if current_score_weight_calinski_harabasz > 0: 
                    try:
                        ch_score_raw = calinski_harabasz_score(data_for_clustering_current, labels_for_metrics)
                        ch_score_raw_val_for_log = ch_score_raw; calinski_harabasz_metric_value = 1.0 - np.exp(-ch_score_raw / 500.0)  
                    except ValueError as e_ch: logging.warning("%s Iteration %s: Calinski-Harabasz error: %s", log_prefix, run_idx, e_ch); calinski_harabasz_metric_value = 0.0 
        
        if len(set(labels) - {-1}) == 0:
            logging.info("%s Iteration %s: No actual clusters formed.", log_prefix, run_idx)
            return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details_for_run, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}}
        
        if len(labels) != len(valid_tracks_for_processing) or len(raw_distances) != len(valid_tracks_for_processing):
            logging.error("%s Iteration %s: CRITICAL MISMATCH for playlist formatting. Labels: %s, Distances: %s, Valid Tracks: %s. Aborting iteration result.", log_prefix, run_idx, len(labels), len(raw_distances), len(valid_tracks_for_processing))
            return {"diversity_score": -1.0, "named_playlists": {}, "playlist_centroids": {}, "pca_model_details": None, "scaler_details": scaler_details_for_run, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}}
        max_dist_val = raw_distances[raw_distances != np.inf].max() if np.any(raw_distances != np.inf) else 0
        normalized_distances = raw_distances / max_dist_val if max_dist_val > 0 else raw_distances
        track_info_list = [{"row": valid_tracks_for_processing[i], "label": labels[i], "distance": normalized_distances[i]} for i in range(len(valid_tracks_for_processing))]

        filtered_clusters = defaultdict(list)
        for cid in set(labels): 
            if cid == -1: continue 
            cluster_tracks_info = [t_info for t_info in track_info_list if t_info["label"] == cid and t_info["distance"] <= MAX_DISTANCE]
            if not cluster_tracks_info: continue
            cluster_tracks_info.sort(key=lambda x: x["distance"])
            count_per_artist = defaultdict(int)
            selected_tracks_for_playlist = []
            for t_item_info in cluster_tracks_info:
                author = t_item_info["row"]["author"]
                if count_per_artist[author] < MAX_SONGS_PER_ARTIST:
                    selected_tracks_for_playlist.append(t_item_info)
                    count_per_artist[author] += 1
                if max_songs_per_cluster > 0 and len(selected_tracks_for_playlist) >= max_songs_per_cluster: break
            for t_item_info_final in selected_tracks_for_playlist:
                item_id_val, title_val, author_val = t_item_info_final["row"]["item_id"], t_item_info_final["row"]["title"], t_item_info_final["row"]["author"]
                filtered_clusters[cid].append((item_id_val, title_val, author_val))

        current_named_playlists = {} # Changed from defaultdict(list)
        current_playlist_centroids = {} # Changed from regular dict
        unique_predominant_mood_scores = {}
        raw_cluster_base_name_generation_count = defaultdict(int) # To track base name generations
        unique_predominant_other_feature_scores = {} 
        item_id_to_song_index_map = {track_data['item_id']: i for i, track_data in enumerate(valid_tracks_for_processing)}

        for label_val, songs_list in filtered_clusters.items():
            if songs_list: 
                base_name = ""; top_scores = {} # Initialize base_name
                if use_embeddings_for_this_iteration:
                    song_feat_indices_in_cluster = [item_id_to_song_index_map[item_id_in_song] for item_id_in_song, _, _ in songs_list if item_id_in_song in item_id_to_song_index_map]
                    if not song_feat_indices_in_cluster: continue
                    song_feat_vectors_in_cluster = X_feat_orig[song_feat_indices_in_cluster]
                    if song_feat_vectors_in_cluster.shape[0] == 0: continue
                    mean_original_feature_vector = np.mean(song_feat_vectors_in_cluster, axis=0)
                    base_name, top_scores = name_cluster(mean_original_feature_vector, None, False, active_mood_labels, None)
                else: 
                    centroid_val = cluster_centers_map.get(label_val)
                    if centroid_val is None or len(centroid_val) == 0: continue
                    base_name, top_scores = name_cluster(centroid_val, pca_model_for_this_iteration, pca_config.get("enabled", False), active_mood_labels, scaler_details_for_run)
                
                raw_cluster_base_name_generation_count[base_name] += 1
                if top_scores and any(mood in active_mood_labels for mood in top_scores.keys()):
                    predominant_mood_key = max((k for k in top_scores if k in MOOD_LABELS), key=top_scores.get, default=None)
                    if predominant_mood_key:
                        current_mood_score = top_scores.get(predominant_mood_key, 0.0)
                        unique_predominant_mood_scores[predominant_mood_key] = max(unique_predominant_mood_scores.get(predominant_mood_key, 0.0), current_mood_score) # type: ignore
                centroid_other_features_for_diversity_evaluation = {lk: top_scores.get(lk, 0.0) for lk in OTHER_FEATURE_LABELS if lk in top_scores}
                predominant_other_feature_key_for_diversity_score_calc = None 
                highest_predominant_other_feature_score_this_centroid = 0.3 
                if centroid_other_features_for_diversity_evaluation: 
                    for feature_label, feature_score in centroid_other_features_for_diversity_evaluation.items():
                        if feature_score > highest_predominant_other_feature_score_this_centroid:
                            highest_predominant_other_feature_score_this_centroid = feature_score 
                            predominant_other_feature_key_for_diversity_score_calc = feature_label 
                if predominant_other_feature_key_for_diversity_score_calc:
                     unique_predominant_other_feature_scores[predominant_other_feature_key_for_diversity_score_calc] = max(unique_predominant_other_feature_scores.get(predominant_other_feature_key_for_diversity_score_calc, 0.0), highest_predominant_other_feature_score_this_centroid)
                
                # Use the base_name as the key. If multiple raw clusters generate the same base_name,
                # the last one processed in this loop will overwrite previous ones for this iteration.
                playlist_key = base_name

                current_named_playlists[playlist_key] = songs_list # Assign directly, do not extend
                current_playlist_centroids[playlist_key] = top_scores

        # Log if any base names were generated by multiple raw clusters, noting potential overwrites.
        for name, count in raw_cluster_base_name_generation_count.items():
            if count > 1:
                logging.info("%s Iteration %s: Playlist base name '%s' was generated by %s raw clusters. The last processed cluster with this name was used.", log_prefix, run_idx, name, count)

        raw_mood_diversity_score = sum(unique_predominant_mood_scores.values())
        # print(f"{log_prefix} Iteration {run_idx}: Raw Mood Diversity Score: {raw_mood_diversity_score}, Unique Moods: {len(unique_predominant_mood_scores)}")
        base_diversity_score = 0.0  
        if len(active_mood_labels) > 0: 
            ln_mood_diversity = np.log1p(raw_mood_diversity_score) # type: ignore
            # Select mood diversity stats based on whether embeddings are used for clustering
            if use_embeddings_for_this_iteration:
                diversity_stats_to_use = LN_MOOD_DIVERSITY_EMBEDING_STATS
                # print(f"{log_prefix} Iteration {run_idx}: Using EMBEDDING mood diversity stats.")
            else:
                diversity_stats_to_use = LN_MOOD_DIVERSITY_STATS
                logging.debug("%s Iteration %s: Using REGULAR mood diversity stats.", log_prefix, run_idx)
            config_mean_ln_diversity = diversity_stats_to_use.get("mean"); config_sd_ln_diversity = diversity_stats_to_use.get("sd")
            if config_mean_ln_diversity is None or config_sd_ln_diversity is None: logging.warning("%s Iteration %s: LN_MOOD_DIVERSITY_STATS missing mean/sd.", log_prefix, run_idx)
            elif abs(config_sd_ln_diversity) < 1e-9: base_diversity_score = 0.0 # Avoid division by zero or very small SD
            else: base_diversity_score = (ln_mood_diversity - config_mean_ln_diversity) / config_sd_ln_diversity
        
        raw_playlist_purity_component = 0.0; all_individual_playlist_purities = []
        if current_named_playlists:
            for playlist_name_key, songs_in_playlist_info_list in current_named_playlists.items():
                playlist_centroid_mood_data = current_playlist_centroids.get(playlist_name_key)
                if not playlist_centroid_mood_data or not songs_in_playlist_info_list: continue
                centroid_mood_scores_for_purity_calc = { m_label: playlist_centroid_mood_data.get(m_label, 0.0) for m_label in MOOD_LABELS if m_label in playlist_centroid_mood_data }
                if not centroid_mood_scores_for_purity_calc: continue
                sorted_centroid_moods_for_purity = sorted(centroid_mood_scores_for_purity_calc.items(), key=lambda item: item[1], reverse=True)
                top_k_centroid_mood_labels_for_purity = [ml for ml, sv in sorted_centroid_moods_for_purity[:TOP_K_MOODS_FOR_PURITY_CALCULATION] if sv > 0.01]
                if not top_k_centroid_mood_labels_for_purity: all_individual_playlist_purities.append(0.0); continue
                current_playlist_song_purity_scores = []
                for item_id_in_playlist, _, _ in songs_in_playlist_info_list:
                    song_original_index = item_id_to_song_index_map.get(item_id_in_playlist)
                    if song_original_index is not None and song_original_index < X_feat_orig.shape[0]:
                        mood_vector_end_index = 2 + len(active_mood_labels)
                        if X_feat_orig.shape[1] >= mood_vector_end_index:
                            song_mood_scores_vector = X_feat_orig[song_original_index][2 : mood_vector_end_index]
                            max_score_for_song_among_top_centroid_moods = 0.0
                            for centroid_mood_label in top_k_centroid_mood_labels_for_purity:
                                try:
                                    mood_idx = active_mood_labels.index(centroid_mood_label)
                                    if mood_idx < len(song_mood_scores_vector):
                                        song_score_for_this_centroid_mood = song_mood_scores_vector[mood_idx] 
                                        if song_score_for_this_centroid_mood > max_score_for_song_among_top_centroid_moods: max_score_for_song_among_top_centroid_moods = song_mood_scores_vector[mood_idx] 
                                except ValueError: pass 
                            if max_score_for_song_among_top_centroid_moods > 0: current_playlist_song_purity_scores.append(max_score_for_song_among_top_centroid_moods)
                if current_playlist_song_purity_scores: all_individual_playlist_purities.append(sum(current_playlist_song_purity_scores))
        if all_individual_playlist_purities: raw_playlist_purity_component = sum(all_individual_playlist_purities)
        ln_mood_purity = np.log1p(raw_playlist_purity_component)
        
        # Select purity stats based on whether embeddings are used for clustering
        if use_embeddings_for_this_iteration:
            purity_stats_to_use = LN_MOOD_PURITY_EMBEDING_STATS
            # print(f"{log_prefix} Iteration {run_idx}: Using EMBEDDING mood purity stats.")
        else:
            purity_stats_to_use = LN_MOOD_PURITY_STATS # type: ignore
            logging.debug("%s Iteration %s: Using REGULAR mood purity stats.", log_prefix, run_idx)
        config_mean_ln_purity = purity_stats_to_use.get("mean"); config_sd_ln_purity = purity_stats_to_use.get("sd")
        playlist_purity_component = 0.0
        if config_mean_ln_purity is None or config_sd_ln_purity is None: logging.warning("%s Iteration %s: LN_MOOD_PURITY_STATS missing mean/sd.", log_prefix, run_idx)
        elif abs(config_sd_ln_purity) < 1e-9: playlist_purity_component = 0.0
        else: playlist_purity_component = (ln_mood_purity - config_mean_ln_purity) / config_sd_ln_purity
        raw_other_features_diversity_score = sum(unique_predominant_other_feature_scores.values()) 
        ln_other_features_diversity = np.log1p(raw_other_features_diversity_score) # type: ignore
        config_mean_ln_other_div = LN_OTHER_FEATURES_DIVERSITY_STATS.get("mean"); config_sd_ln_other_div = LN_OTHER_FEATURES_DIVERSITY_STATS.get("sd")
        other_features_diversity_score = 0.0
        if config_mean_ln_other_div is None or config_sd_ln_other_div is None: logging.warning("%s Iteration %s: LN_OTHER_FEATURES_DIVERSITY_STATS missing mean/sd.", log_prefix, run_idx)
        elif abs(config_sd_ln_other_div) < 1e-9: other_features_diversity_score = 0.0
        else: other_features_diversity_score = (ln_other_features_diversity - config_mean_ln_other_div) / config_sd_ln_other_div
        
        all_individual_playlist_other_feature_purities = []
        if current_named_playlists and OTHER_FEATURE_LABELS: 
            for playlist_name_key, songs_in_playlist_info_list in current_named_playlists.items():
                playlist_centroid_data = current_playlist_centroids.get(playlist_name_key)
                if not playlist_centroid_data or not songs_in_playlist_info_list: continue
                centroid_other_features_for_purity = {lk: playlist_centroid_data.get(lk, 0.0) for lk in OTHER_FEATURE_LABELS if lk in playlist_centroid_data}
                predominant_other_feature_for_this_playlist = None; max_score_for_predominant_other = CONFIG_OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY
                if centroid_other_features_for_purity:
                    for fl, fs in centroid_other_features_for_purity.items():
                        if fs > max_score_for_predominant_other: max_score_for_predominant_other = fs; predominant_other_feature_for_this_playlist = fl
                if not predominant_other_feature_for_this_playlist: continue
                try: predominant_other_feature_index_in_labels = OTHER_FEATURE_LABELS.index(predominant_other_feature_for_this_playlist)
                except ValueError: logging.warning("%s Iteration %s: Predominant other feature '%s' not in OTHER_FEATURE_LABELS.", log_prefix, run_idx, predominant_other_feature_for_this_playlist); continue
                scores_of_predominant_other_feature_for_songs = [] # nosec # type: ignore
                for item_id_in_playlist, _, _ in songs_in_playlist_info_list:
                    song_original_index = item_id_to_song_index_map.get(item_id_in_playlist)
                    if song_original_index is not None and song_original_index < X_feat_orig.shape[0]:
                        other_features_start_index = 2 + len(active_mood_labels)
                        if X_feat_orig.shape[1] > other_features_start_index + predominant_other_feature_index_in_labels:
                            song_other_features_vector = X_feat_orig[song_original_index][other_features_start_index:] 
                            if predominant_other_feature_index_in_labels < len(song_other_features_vector):
                                scores_of_predominant_other_feature_for_songs.append(song_other_features_vector[predominant_other_feature_index_in_labels])
                if scores_of_predominant_other_feature_for_songs: all_individual_playlist_other_feature_purities.append(sum(scores_of_predominant_other_feature_for_songs))
        raw_other_feature_purity_component = sum(all_individual_playlist_other_feature_purities) if all_individual_playlist_other_feature_purities else 0.0 # type: ignore
        ln_other_features_purity = np.log1p(raw_other_feature_purity_component)
        config_mean_ln_other_pur = LN_OTHER_FEATURES_PURITY_STATS.get("mean"); config_sd_ln_other_pur = LN_OTHER_FEATURES_PURITY_STATS.get("sd")
        other_feature_purity_component = 0.0
        if config_mean_ln_other_pur is None or config_sd_ln_other_pur is None: logging.warning("%s Iteration %s: LN_OTHER_FEATURES_PURITY_STATS missing mean/sd.", log_prefix, run_idx)
        elif abs(config_sd_ln_other_pur) < 1e-9: other_feature_purity_component = 0.0
        else: other_feature_purity_component = (ln_other_features_purity - config_mean_ln_other_pur) / config_sd_ln_other_pur

        final_enhanced_score = (current_score_weight_diversity * base_diversity_score) + (current_score_weight_purity * playlist_purity_component) + (current_score_weight_other_feature_diversity * other_features_diversity_score) + (current_score_weight_other_feature_purity * other_feature_purity_component) + (current_score_weight_silhouette * silhouette_metric_value) + (current_score_weight_davies_bouldin * davies_bouldin_metric_value) + (current_score_weight_calinski_harabasz * calinski_harabasz_metric_value)
        log_message = (
            f"{log_prefix} Iteration {run_idx}: Scores -> "
            f"MoodDiv: {raw_mood_diversity_score:.2f}/{base_diversity_score:.2f}, "
            f"MoodPur: {raw_playlist_purity_component:.2f}/{playlist_purity_component:.2f}, "
            f"OtherFeatDiv: {raw_other_features_diversity_score:.2f}/{other_features_diversity_score:.2f}, "
            f"OtherFeatPur: {raw_other_feature_purity_component:.2f}/{other_feature_purity_component:.2f}, "
            f"Sil: {s_score_raw_val_for_log:.2f}/{silhouette_metric_value:.2f}, "

            f"DB: {db_score_raw_val_for_log:.2f}/{davies_bouldin_metric_value:.2f}, "
            f"CH: {ch_score_raw_val_for_log:.2f}/{calinski_harabasz_metric_value:.2f}, "
            f"FinalScore: {final_enhanced_score:.2f} "
            f"(Weights: MoodDiv={current_score_weight_diversity}, MoodPur={current_score_weight_purity}, "
            f"OtherFeatDiv={current_score_weight_other_feature_diversity}, OtherFeatPur={current_score_weight_other_feature_purity}, "
            f"Sil={current_score_weight_silhouette}, DB={current_score_weight_davies_bouldin}, CH={current_score_weight_calinski_harabasz})"
        )

        logging.info(log_message)
        pca_model_details_to_return = None
        if pca_model_for_this_iteration and pca_config.get("enabled"):
            try: pca_model_details_to_return = {"n_components": pca_model_for_this_iteration.n_components_, "explained_variance_ratio": pca_model_for_this_iteration.explained_variance_ratio_.tolist(), "mean": pca_model_for_this_iteration.mean_.tolist()} # type: ignore
            except AttributeError as e_pca_attr: logging.warning("%s Iteration %s: Warning - PCA model details retrieval error: %s", log_prefix, run_idx, e_pca_attr)
        result = {"diversity_score": float(final_enhanced_score), "named_playlists": dict(current_named_playlists), "playlist_centroids": current_playlist_centroids, "pca_model_details": pca_model_details_to_return, "scaler_details": scaler_details_for_run, "parameters": {"clustering_method_config": method_params_config, "pca_config": pca_config, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx}}
        return result
    except Exception as e_iter:
        logging.error("%s Iteration %s failed: %s", log_prefix, run_idx, e_iter, exc_info=True)
        return None # type: ignore
