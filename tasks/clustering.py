# tasks/clustering.py

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

# RQ import
from rq import get_current_job
from rq.job import Job
from rq.exceptions import NoSuchJobError, InvalidJobOperation

# Import necessary components from the main app.py file (ensure these are available)
from app import (app, redis_conn, get_db, save_task_status, get_task_info_from_db,
                track_exists, save_track_analysis, get_all_tracks, get_tracks_by_ids, update_playlist_table, JobStatus,
                TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
                TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)
from psycopg2.extras import DictCursor


# Import configuration (ensure config.py is in PYTHONPATH or same directory)
from config import (TEMP_DIR, MAX_SONGS_PER_CLUSTER, # MAX_DISTANCE, GMM_COVARIANCE_TYPE etc. moved to helper
    MOOD_LABELS, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, ENERGY_MIN, ENERGY_MAX,
    TEMPO_MIN_BPM, TEMPO_MAX_BPM, JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, REDIS_URL, DATABASE_URL,
    OLLAMA_SERVER_URL, OLLAMA_MODEL_NAME, AI_MODEL_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL_NAME,
    DANCEABILITY_MODEL_PATH, AGGRESSIVE_MODEL_PATH, HAPPY_MODEL_PATH, PARTY_MODEL_PATH, RELAXED_MODEL_PATH, SAD_MODEL_PATH,
    MUTATION_KMEANS_COORD_FRACTION, MUTATION_INT_ABS_DELTA, MUTATION_FLOAT_ABS_DELTA,
    TOP_N_ELITES, EXPLOITATION_START_FRACTION, EXPLOITATION_PROBABILITY_CONFIG, TOP_N_MOODS, TOP_N_OTHER_FEATURES,
    STRATIFIED_GENRES, MIN_SONGS_PER_GENRE_FOR_STRATIFICATION, SAMPLING_PERCENTAGE_CHANGE_PER_RUN, ITERATIONS_PER_BATCH_JOB, MAX_CONCURRENT_BATCH_JOBS, ENABLE_CLUSTERING_EMBEDDINGS,  # type: ignore
    DB_FETCH_CHUNK_SIZE, STRATIFIED_SAMPLING_TARGET_PERCENTILE) # Score weights, purity/stats/MiniBatch configs etc. moved to helper

# Import AI naming function and prompt template
from ai import get_ai_playlist_name, creative_prompt_template
# from .commons import score_vector # Moved to clustering_helper.py
from .clustering_helper import (
    _get_stratified_song_subset,
    create_or_update_playlists_on_jellyfin,
    get_job_result_safely,
    _perform_single_clustering_iteration
)

# Task specific helper function



def run_clustering_batch_task(
    batch_id_str, start_run_idx, num_iterations_in_batch, # Batch control
    genre_to_lightweight_track_data_map_json, 
    target_songs_per_genre, 
    sampling_percentage_change_per_run,
    clustering_method,
    active_mood_labels_for_batch, 
    num_clusters_min_max_tuple,
    dbscan_params_ranges_dict,
    gmm_params_ranges_dict,
    pca_params_ranges_dict,
    max_songs_per_cluster,
    parent_task_id,
    score_weight_diversity_param,
    score_weight_silhouette_param,
    score_weight_davies_bouldin_param,
    score_weight_calinski_harabasz_param,
    score_weight_purity_param,                  # Corrected order
    score_weight_other_feature_diversity_param, # Corrected order
    score_weight_other_feature_purity_param,    # Corrected order
    elite_solutions_params_list_json, 
    exploitation_probability,         
    mutation_config_json,             
    initial_subset_track_ids_json,    
    enable_clustering_embeddings_param 
    ):
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

        def log_and_update_batch_task(message, progress, details_extra=None, task_state=TASK_STATUS_PROGRESS, print_console=True): 
            nonlocal current_progress_batch, current_task_logs_batch
            current_progress_batch = progress
            if print_console: 
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
            genre_to_lightweight_track_data_map = json.loads(genre_to_lightweight_track_data_map_json)
            current_sampled_track_ids_in_batch = []
            if initial_subset_track_ids_json:
                current_sampled_track_ids_in_batch = json.loads(initial_subset_track_ids_json)
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
                        return {"status": "REVOKED", "message": f"Batch task {batch_id_str} stopped.", "iterations_completed_in_batch": 0, "best_result_from_batch": None, "final_subset_track_ids": current_sampled_track_ids_in_batch}
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
                # Revocation check
                if current_job:
                    with app.app_context():
                        task_db_info_iter_check = get_task_info_from_db(current_task_id) 
                        parent_task_db_info_iter_check = get_task_info_from_db(parent_task_id) if parent_task_id else None
                        is_self_revoked_iter = task_db_info_iter_check and task_db_info_iter_check.get('status') == TASK_STATUS_REVOKED
                        is_parent_failed_or_revoked_iter = parent_task_db_info_iter_check and parent_task_db_info_iter_check.get('status') in [TASK_STATUS_REVOKED, TASK_STATUS_FAILURE]
                        if is_self_revoked_iter or is_parent_failed_or_revoked_iter:
                            revocation_reason_iter = "self was REVOKED" if is_self_revoked_iter else f"parent task {parent_task_id} status is {parent_task_db_info_iter_check.get('status') if parent_task_db_info_iter_check else 'N/A'}"
                            log_and_update_batch_task(f"🛑 Batch task {batch_id_str} stopping mid-batch (before iter {current_run_global_idx}) because {revocation_reason_iter}.", current_progress_batch, task_state=TASK_STATUS_REVOKED)
                            return {"status": "REVOKED", "message": "Batch task revoked mid-process.", "iterations_completed_in_batch": iterations_actually_completed, "best_result_from_batch": best_result_in_this_batch, "final_subset_track_ids": current_sampled_track_ids_in_batch}
                
                # Determine track IDs for this iteration using stratified sampling
                # For the first iteration (i=0), current_sampled_track_ids_in_batch is from initial_subset_track_ids_json
                # For subsequent iterations, it's the result of the previous _get_stratified_song_subset call
                percentage_change_for_this_iter = 0.0 if i == 0 else sampling_percentage_change_per_run
                current_subset_lightweight_data = _get_stratified_song_subset(
                    genre_to_lightweight_track_data_map,
                    target_songs_per_genre,
                    previous_subset_track_ids=current_sampled_track_ids_in_batch, 
                    percentage_change=percentage_change_for_this_iter
                )
                item_ids_for_this_iteration = [t['item_id'] for t in current_subset_lightweight_data]
                current_sampled_track_ids_in_batch = list(item_ids_for_this_iteration) # Update for next iteration's input

                if not item_ids_for_this_iteration:
                    log_and_update_batch_task(f"Subset for iteration {current_run_global_idx} is empty after stratification. Skipping.", current_progress_batch)
                    iterations_actually_completed += 1
                    continue

                iteration_result = _perform_single_clustering_iteration(
                    current_run_global_idx, item_ids_for_this_iteration, # Pass IDs
                    clustering_method, num_clusters_min_max_tuple, dbscan_params_ranges_dict, gmm_params_ranges_dict, pca_params_ranges_dict, active_mood_labels_for_batch,
                    max_songs_per_cluster, log_prefix=log_prefix_for_iter,
                    elite_solutions_params_list=(elite_solutions_params_list_for_iter if elite_solutions_params_list_for_iter else []),
                    exploitation_probability=exploitation_probability,
                    mutation_config=(mutation_config_for_iter if mutation_config_for_iter else {}),
                    score_weight_diversity_override=score_weight_diversity_param,
                    score_weight_silhouette_override=score_weight_silhouette_param,
                    score_weight_davies_bouldin_override=score_weight_davies_bouldin_param,       
                    score_weight_calinski_harabasz_override=score_weight_calinski_harabasz_param, 
                    score_weight_purity_override=score_weight_purity_param,
                    score_weight_other_feature_diversity_override=score_weight_other_feature_diversity_param,
                    score_weight_other_feature_purity_override=score_weight_other_feature_purity_param,
                    enable_clustering_embeddings_param=enable_clustering_embeddings_param # Pass this through
                ) 
                iterations_actually_completed += 1 
                if iteration_result and iteration_result.get("diversity_score", -1.0) > best_score_in_this_batch:
                    best_score_in_this_batch = iteration_result["diversity_score"]
                    best_result_in_this_batch = iteration_result
                iter_progress = 5 + int(90 * (iterations_actually_completed / float(num_iterations_in_batch)))
                log_and_update_batch_task(f"Iteration {current_run_global_idx} (in batch: {i+1}/{num_iterations_in_batch}) complete. Batch best score: {best_score_in_this_batch:.2f}", iter_progress)

            final_batch_summary = {
                "best_score_in_batch": best_score_in_this_batch,
                "iterations_completed_in_batch": iterations_actually_completed,
                "full_best_result_from_batch": best_result_in_this_batch,
                "final_subset_track_ids": current_sampled_track_ids_in_batch 
            }
            log_and_update_batch_task(f"Batch {batch_id_str} complete. Best score in batch: {best_score_in_this_batch:.2f}", 100, details_extra=final_batch_summary, task_state=TASK_STATUS_SUCCESS)
            return {"status": "SUCCESS", "iterations_completed_in_batch": iterations_actually_completed, "best_result_from_batch": best_result_in_this_batch, "final_subset_track_ids": current_sampled_track_ids_in_batch}
        except Exception as e:
            error_tb = traceback.format_exc()
            failure_details = {"error": str(e), "traceback": error_tb, "batch_id": batch_id_str}
            log_and_update_batch_task(f"Failed clustering batch {batch_id_str}: {e}", current_progress_batch, details_extra=failure_details, task_state=TASK_STATUS_FAILURE)
            print(f"ERROR: Clustering batch {batch_id_str} failed: {e}\n{error_tb}")
            return {"status": "FAILURE", "iterations_completed_in_batch": iterations_actually_completed, "best_result_from_batch": None, "final_subset_track_ids": current_sampled_track_ids_in_batch}


def run_clustering_task(
    clustering_method, num_clusters_min, num_clusters_max,
    dbscan_eps_min, dbscan_eps_max, dbscan_min_samples_min, dbscan_min_samples_max, 
    pca_components_min, pca_components_max, num_clustering_runs, max_songs_per_cluster, # type: ignore
    gmm_n_components_min, gmm_n_components_max, 
    min_songs_per_genre_for_stratification_param, # Added
    stratified_sampling_target_percentile_param,  # Added
    score_weight_diversity_param, score_weight_silhouette_param, 
    score_weight_davies_bouldin_param, score_weight_calinski_harabasz_param, 
    score_weight_purity_param, 
    score_weight_other_feature_diversity_param, 
    score_weight_other_feature_purity_param,    
    ai_model_provider_param, ollama_server_url_param, ollama_model_name_param, 
    gemini_api_key_param, gemini_model_name_param, top_n_moods_for_clustering_param,
    enable_clustering_embeddings_param): 
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
            "clustering_run_job_ids": [],
            "score_weight_diversity_for_run": score_weight_diversity_param,
            "score_weight_silhouette_for_run": score_weight_silhouette_param,
            "score_weight_davies_bouldin_for_run": score_weight_davies_bouldin_param,     
            "score_weight_calinski_harabasz_for_run": score_weight_calinski_harabasz_param, 
            "score_weight_purity_for_run": score_weight_purity_param, 
            "score_weight_other_feature_diversity_for_run": score_weight_other_feature_diversity_param, 
            "score_weight_other_feature_purity_for_run": score_weight_other_feature_purity_param, 
            "ai_model_provider_for_run": ai_model_provider_param,
            "ollama_model_name_for_run": ollama_model_name_param,
            "gemini_model_name_for_run": gemini_model_name_param,
            "embeddings_enabled_for_clustering": enable_clustering_embeddings_param 
        } # Add the new param here for logging
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
                if key in _main_task_accumulated_details and _main_task_accumulated_details[key] is not None:
                    meta_details[key] = _main_task_accumulated_details[key]
            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.meta['details'] = meta_details
                current_job.save_meta()
            save_task_status(current_task_id, "main_clustering", task_state, progress=progress, details=_main_task_accumulated_details)

        try:
            ai_model_provider_for_run = ai_model_provider_param
            ollama_server_url_for_run = ollama_server_url_param
            ollama_model_name_for_run = ollama_model_name_param 
            gemini_api_key_for_run = gemini_api_key_param
            gemini_model_name_for_run = gemini_model_name_param

            log_and_update_main_clustering("📊 Starting main clustering process...", 0)
            log_and_update_main_clustering("Fetching lightweight track data for stratification...", 1, print_console=False)
            db = get_db()
            cur = db.cursor(cursor_factory=DictCursor)
            if enable_clustering_embeddings_param:
                # If using embeddings, only select tracks that have score data AND a non-null embedding
                cur.execute("""
                    SELECT s.item_id, s.author, s.mood_vector
                    FROM score s
                    WHERE s.mood_vector IS NOT NULL AND s.mood_vector != ''
                """)
            else: # If not using embeddings, just need score data
                cur.execute("SELECT item_id, author, mood_vector FROM score WHERE mood_vector IS NOT NULL AND mood_vector != ''")
            lightweight_rows = cur.fetchall()
            cur.close()
            log_and_update_main_clustering(f"Retrieved {len(lightweight_rows)} lightweight track records from database.", 2, print_console=False)

            min_tracks_for_kmeans = num_clusters_min if clustering_method == "kmeans" else 2
            min_tracks_for_gmm = gmm_n_components_min if clustering_method == "gmm" else 2
            log_and_update_main_clustering(f"Calculated min_req_overall based on method '{clustering_method}' and PCA settings.", 2, print_console=False) 
            min_req_pca = (pca_components_min + 1) if pca_components_min > 0 else 2
            min_req_overall = max(2, min_tracks_for_kmeans, min_tracks_for_gmm, min_req_pca)
            if len(lightweight_rows) < min_req_overall:
                err_msg = f"Not enough tracks in DB ({len(lightweight_rows)}) for clustering. Minimum required: {min_req_overall}."
                log_and_update_main_clustering(err_msg, 100, details_to_add_or_update={"error": "Insufficient data"}, task_state=TASK_STATUS_FAILURE)
                return {"status": "FAILURE", "message": err_msg}
            if num_clustering_runs == 0:
                log_and_update_main_clustering("Number of clustering runs is 0. Nothing to do.", 100, task_state=TASK_STATUS_SUCCESS)
                app.logger.info(f"[MainClusteringTask-{current_task_id}] Number of clustering runs is 0.")
                return {"status": "SUCCESS", "message": "Number of clustering runs was 0."}
            active_mood_labels = MOOD_LABELS[:top_n_moods_for_clustering_param] if top_n_moods_for_clustering_param > 0 else MOOD_LABELS
            log_and_update_main_clustering(f"Active mood labels for clustering: {active_mood_labels}", 4, print_console=False)
            log_and_update_main_clustering("Starting stratified sampling preparation...", 4, print_console=False)
            genre_to_lightweight_track_data_map = defaultdict(list)
            for row in lightweight_rows: 
                if 'mood_vector' in row and row['mood_vector']:
                    mood_scores = {}
                    for pair in row['mood_vector'].split(','):
                        if ':' in pair:
                            label, score_str = pair.split(':')
                            mood_scores[label] = float(score_str)
                    top_stratified_genre = None
                    max_score = -1.0
                    for genre in STRATIFIED_GENRES:
                        if genre in mood_scores and mood_scores[genre] > max_score:
                            max_score = mood_scores[genre]
                            top_stratified_genre = genre
                    minimal_track_info = {'item_id': row['item_id'], 'mood_vector': row['mood_vector']} 
                    if top_stratified_genre:
                        genre_to_lightweight_track_data_map[top_stratified_genre].append(minimal_track_info)
                    else:
                        genre_to_lightweight_track_data_map['__other__'].append(minimal_track_info)
                else: # Should not happen due to WHERE clause, but defensive
                    genre_to_lightweight_track_data_map['__other__'].append({'item_id': row['item_id'], 'mood_vector': ''})


            songs_counts_for_stratified_genres = []
            for genre in STRATIFIED_GENRES:
                if genre in genre_to_lightweight_track_data_map:
                    songs_counts_for_stratified_genres.append(len(genre_to_lightweight_track_data_map[genre])) # type: ignore
            calculated_target_based_on_percentile = 0
            if songs_counts_for_stratified_genres:
                percentile_to_use = np.clip(stratified_sampling_target_percentile_param, 0, 100) # Use passed param
                calculated_target_based_on_percentile = np.percentile(songs_counts_for_stratified_genres, percentile_to_use) 
                log_and_update_main_clustering(
                    f"{percentile_to_use}th percentile of songs per stratified genre: {calculated_target_based_on_percentile:.2f}",
                    current_progress, print_console=False
                )
            else:
                log_and_update_main_clustering("No songs found for any stratified genres. Defaulting target.", current_progress, print_console=False) # type: ignore
            target_songs_per_genre = max(min_songs_per_genre_for_stratification_param, int(np.floor(calculated_target_based_on_percentile))) # Use passed param
            if target_songs_per_genre == 0 and len(lightweight_rows) > 0:
                target_songs_per_genre = 1 
            log_and_update_main_clustering(f"Determined target songs per genre for stratification: {target_songs_per_genre}", 7)
            genre_to_lightweight_track_data_map_json = json.dumps(genre_to_lightweight_track_data_map) # Already dicts
            log_and_update_main_clustering("Stratified sampling preparation complete.", 6, print_console=False)
            initial_subset_lightweight = _get_stratified_song_subset(
                genre_to_lightweight_track_data_map,
                target_songs_per_genre,
                previous_subset_track_ids=None,
                percentage_change=0.0
            )
            last_subset_track_ids_for_batch_chaining = [t['item_id'] for t in initial_subset_lightweight]

            best_diversity_score = _main_task_accumulated_details.get("best_score", -1.0)
            best_clustering_results = None 
            elite_solutions_list = []      
            mutation_config = {
                "int_abs_delta": MUTATION_INT_ABS_DELTA,
                "float_abs_delta": MUTATION_FLOAT_ABS_DELTA,
                "coord_mutation_fraction": MUTATION_KMEANS_COORD_FRACTION
            }
            mutation_config_json = json.dumps(mutation_config)
            exploitation_start_run_idx = int(num_clustering_runs * EXPLOITATION_START_FRACTION)
            all_launched_child_jobs_instances = []
            from app import rq_queue as main_rq_queue 
            active_jobs_map = {}
            total_iterations_completed_count = 0
            next_batch_job_idx_to_launch = 0
            num_total_batch_jobs = (num_clustering_runs + ITERATIONS_PER_BATCH_JOB - 1) // ITERATIONS_PER_BATCH_JOB
            _main_task_accumulated_details["total_batch_jobs"] = num_total_batch_jobs 
            batches_completed_count = 0
            data_source_for_clustering_log = "embeddings" if enable_clustering_embeddings_param else "score vectors"
            log_and_update_main_clustering(f"Processing {len(lightweight_rows)} tracks using {data_source_for_clustering_log}. Preparing {num_clustering_runs} runs in {num_total_batch_jobs} batches.", 8)

            while batches_completed_count < num_total_batch_jobs:
                if current_job:
                    with app.app_context():
                        main_task_db_info = get_task_info_from_db(current_task_id)
                        if main_task_db_info and main_task_db_info.get('status') == TASK_STATUS_REVOKED:
                            log_and_update_main_clustering(f"🛑 Main clustering task {current_task_id} REVOKED.", current_progress, task_state=TASK_STATUS_REVOKED)
                            app.logger.info(f"[MainClusteringTask-{current_task_id}] Task revoked, exiting batch loop.")
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
                            db_status_for_log = db_task_info_child.get('status') if db_task_info_child else "UNKNOWN (not in DB)"
                            print(f"[MainClusteringTask-{current_task_id}] Warning: Active batch job {job_id} missing from RQ, DB status '{db_status_for_log}' is not terminal. Treating as completed (abnormally) to prevent stall.")
                            is_child_truly_completed_this_cycle = True 
                    except Exception as e_monitor_child_active:
                        print(f"[MainClusteringTask-{current_task_id}] ERROR monitoring active batch job {job_id}: {e_monitor_child_active}. Treating as completed.")
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
                                    best_clustering_results = best_from_batch 
                                    print(f"[MainClusteringTask-{current_task_id}] Intermediate new best score: {best_diversity_score:.2f} from batch job {job_id_just_completed}")
                        else:
                            print(f"[MainClusteringTask-{current_task_id}] Warning: Batch job {job_id_just_completed} completed but no result.")
                if newly_completed_elite_candidates:
                    all_potential_elites = elite_solutions_list + newly_completed_elite_candidates
                    all_potential_elites.sort(key=lambda x: x["score"], reverse=True)
                    elite_solutions_list = all_potential_elites[:TOP_N_ELITES]
                can_launch_more_batch_jobs = next_batch_job_idx_to_launch < num_total_batch_jobs
                if can_launch_more_batch_jobs:
                    num_slots_to_fill = MAX_CONCURRENT_BATCH_JOBS - len(active_jobs_map)
                    for _ in range(num_slots_to_fill):
                        if next_batch_job_idx_to_launch >= num_total_batch_jobs: break
                        batch_job_task_id = str(uuid.uuid4())
                        current_batch_start_run_idx = next_batch_job_idx_to_launch * ITERATIONS_PER_BATCH_JOB
                        num_iterations_for_this_batch = min(ITERATIONS_PER_BATCH_JOB, num_clustering_runs - current_batch_start_run_idx)
                        if num_iterations_for_this_batch <= 0:
                            print(f"[MainClusteringTask-{current_task_id}] Warning: Calculated 0 iterations for batch {next_batch_job_idx_to_launch}. Skipping.")
                            next_batch_job_idx_to_launch +=1 
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
                        log_and_update_main_clustering(f"Preparing to enqueue batch job {batch_id_for_logging} (runs {current_batch_start_run_idx}-{current_batch_start_run_idx + num_iterations_for_this_batch -1}). Initial subset IDs for this batch: {'Provided' if last_subset_track_ids_for_batch_chaining else 'None'}", current_progress, print_console=False)
                        initial_subset_for_this_batch_json = json.dumps(last_subset_track_ids_for_batch_chaining) if last_subset_track_ids_for_batch_chaining else "[]"

                        new_job = main_rq_queue.enqueue(
                            run_clustering_batch_task,
                            args=(
                                batch_id_for_logging, current_batch_start_run_idx, num_iterations_for_this_batch,
                                genre_to_lightweight_track_data_map_json, 
                                target_songs_per_genre,
                                SAMPLING_PERCENTAGE_CHANGE_PER_RUN,
                                clustering_method, active_mood_labels,
                                num_clusters_min_max_tuple_for_batch, dbscan_params_ranges_dict_for_batch,
                                gmm_params_ranges_dict_for_batch, pca_params_ranges_dict_for_batch,
                                max_songs_per_cluster, current_task_id,
                                score_weight_diversity_param, score_weight_silhouette_param, 
                                score_weight_davies_bouldin_param, score_weight_calinski_harabasz_param, 
                                score_weight_purity_param, 
                                score_weight_other_feature_diversity_param, score_weight_other_feature_purity_param,
                                current_elite_params_for_batch_json,
                                exploitation_prob_for_this_batch,
                                mutation_config_json,
                                initial_subset_for_this_batch_json,
                                enable_clustering_embeddings_param
                            ),
                            job_id=batch_job_task_id,
                            description=f"Clustering Batch {next_batch_job_idx_to_launch} (Runs {current_batch_start_run_idx}-{current_batch_start_run_idx + num_iterations_for_this_batch -1})",
                            job_timeout=3600 * (ITERATIONS_PER_BATCH_JOB / 2),
                            meta={'parent_task_id': current_task_id}
                        )
                        active_jobs_map[new_job.id] = new_job
                        all_launched_child_jobs_instances.append(new_job)
                        _main_task_accumulated_details.setdefault("clustering_run_job_ids", []).append(new_job.id)
                        log_and_update_main_clustering(f"Enqueued batch job {new_job.id} ({batch_id_for_logging}).", current_progress, print_console=False)
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
                    f"Clustering with {data_source_for_clustering_log}. Batch Jobs Launched: {next_batch_job_idx_to_launch}/{num_total_batch_jobs}. Active: {len(active_jobs_map)}. Iterations: {total_iterations_completed_count}/{num_clustering_runs}. Best Score: {best_diversity_score:.2f}",
                    current_progress_val,
                    details_to_add_or_update={
                        "runs_completed": total_iterations_completed_count,
                        "batch_jobs_launched": next_batch_job_idx_to_launch,
                        "active_runs_count": len(active_jobs_map),
                        "best_score": best_diversity_score
                    }
                )
                if batches_completed_count >= num_total_batch_jobs and not active_jobs_map:
                    log_and_update_main_clustering(f"All {num_total_batch_jobs} batches completed and no active jobs. Exiting main loop.", current_progress, print_console=False)
                    break
                log_and_update_main_clustering(f"Waiting for 3 seconds before next cycle. Active jobs: {len(active_jobs_map)}", current_progress, print_console=False) 
                time.sleep(3)

            log_and_update_main_clustering("All clustering batch jobs completed. Finalizing best result...", 90,
                                           details_to_add_or_update={"best_score": best_diversity_score})
            if not best_clustering_results or best_diversity_score < 0:
                log_and_update_main_clustering("No valid clustering solution found after all runs.", 100, details_to_add_or_update={"error": "No suitable clustering found", "best_score": best_diversity_score}, task_state=TASK_STATUS_FAILURE)
                app.logger.error(f"[MainClusteringTask-{current_task_id}] No valid clustering solution found. Best score: {best_diversity_score}")
                return {"status": "FAILURE", "message": "No valid clusters found after multiple runs."}
            current_progress = 92
            log_and_update_main_clustering(f"Best clustering found with diversity score: {best_diversity_score:.2f}. Preparing to create playlists.", current_progress, details_to_add_or_update={"best_score": best_diversity_score, "best_params": best_clustering_results.get("parameters")})
            final_named_playlists = best_clustering_results["named_playlists"]
            final_playlist_centroids = best_clustering_results["playlist_centroids"]
            final_max_songs_per_cluster = best_clustering_results["parameters"]["max_songs_per_cluster"]
            final_pca_model_details = best_clustering_results["pca_model_details"] 
            final_scaler_details = best_clustering_results["scaler_details"]     
            log_prefix_main_task_ai = f"[MainClusteringTask-{current_task_id} AI Naming]"
            if ai_model_provider_for_run in ["OLLAMA", "GEMINI"]:
                print(f"{log_prefix_main_task_ai} AI Naming block entered. Attempting to import 'ai' module.")
                ai_naming_start_progress = 90
                ai_naming_end_progress = 95
                ai_naming_progress_range = ai_naming_end_progress - ai_naming_start_progress
                current_progress = ai_naming_start_progress
                log_and_update_main_clustering(f"Preparing for AI Naming...", current_progress, print_console=True)
                try:
                    ai_renamed_playlists_final = {} # Changed from defaultdict(list)
                    ai_renamed_centroids_final = {} # Changed from regular dict
                    ai_base_name_generation_count = defaultdict(int) # To track AI base name collisions
                    total_playlists_to_name = len(final_named_playlists)
                    playlists_named_count = 0
                    for original_name, songs_in_playlist in final_named_playlists.items():
                        if not songs_in_playlist:
                            ai_renamed_playlists_final[original_name] = songs_in_playlist
                            if original_name in final_playlist_centroids:
                                ai_renamed_centroids_final[original_name] = final_playlist_centroids[original_name]
                            continue
                        song_list_for_ai = [{'title': s_title, 'author': s_author} for _, s_title, s_author in songs_in_playlist]
                        
                        feature1_for_ai, feature2_for_ai, feature3_for_ai = "Unknown", "General", "Music"
                        
                        if enable_clustering_embeddings_param:
                            # If embeddings were used for clustering, use generic tags for the AI prompt
                            feature1_for_ai = "Vibe"
                            feature2_for_ai = "Focused"
                            feature3_for_ai = "Collection"
                            print(f"{log_prefix_main_task_ai} Embeddings used for clustering. Using generic tags for AI: '{feature1_for_ai}', '{feature2_for_ai}', '{feature3_for_ai}'.")
                        else:
                            # If not using embeddings, derive tags from the rule-based original_name
                            name_parts = original_name.split('_')
                            if len(name_parts) >= 1: feature1_for_ai = name_parts[0]
                            if len(name_parts) >= 2: feature2_for_ai = name_parts[1]
                            if len(name_parts) >= 3 and name_parts[2].lower() not in ["slow", "medium", "fast"]:
                                feature3_for_ai = name_parts[2]
                            elif len(name_parts) >= 2 and name_parts[1].lower() not in ["slow", "medium", "fast"]: # Check second part if third is tempo
                                feature3_for_ai = name_parts[1]
                        print(f"{log_prefix_main_task_ai} Generating AI name for '{original_name}' ({len(song_list_for_ai)} songs) using provider '{ai_model_provider_for_run}'. Tags for AI: F1: '{feature1_for_ai}', F2: '{feature2_for_ai}', F3: '{feature3_for_ai}'.")
                        centroid_features_for_ai = final_playlist_centroids.get(original_name, {})
                        ai_generated_name_str = get_ai_playlist_name(
                            ai_model_provider_for_run,
                            ollama_server_url_for_run, ollama_model_name_for_run,
                            gemini_api_key_for_run, gemini_model_name_for_run, 
                            creative_prompt_template,
                            feature1_for_ai, feature2_for_ai, feature3_for_ai, # Pass the potentially generic tags
                            song_list_for_ai,
                            centroid_features_for_ai)
                        print(f"{log_prefix_main_task_ai} Raw AI output for '{original_name}': '{ai_generated_name_str}'")
                        current_playlist_final_name = original_name
                        ai_generated_base_name = original_name # This will be the name before disambiguation

                        if ai_generated_name_str and not ai_generated_name_str.startswith("Error") and not ai_generated_name_str.startswith("AI Naming Skipped"):
                             clean_ai_name = ai_generated_name_str.strip().replace("\n", " ")
                             if clean_ai_name:
                                 if clean_ai_name.lower() != original_name.lower().strip().replace("_", " "):
                                      print(f"{log_prefix_main_task_ai} AI: '{original_name}' -> '{clean_ai_name}'")
                                      ai_generated_base_name = clean_ai_name # This is the name AI produced
                                 else:
                                      ai_generated_base_name = original_name # No change, effectively
                             else:
                                 print(f"{log_prefix_main_task_ai} AI for '{original_name}' returned empty after cleaning. Raw: '{ai_generated_name_str}'. Using original.")
                                 ai_generated_base_name = original_name
                        else:
                            print(f"{log_prefix_main_task_ai} AI naming for '{original_name}' failed or returned error/skip message: '{ai_generated_name_str}'. Using original.")
                            ai_generated_base_name = original_name
                        
                        # Disambiguate playlist names if the AI generates the same name for different clusters.
                        ai_base_name_generation_count[ai_generated_base_name] += 1
                        current_playlist_final_name = ai_generated_base_name
                        
                        # If this is not the first time we've seen this base name, it's a collision.
                        if ai_base_name_generation_count[ai_generated_base_name] > 1:
                            # Append a number, e.g., "Chill Vibes (2)"
                            suffix = ai_base_name_generation_count[ai_generated_base_name]
                            current_playlist_final_name = f"{ai_generated_base_name} ({suffix})"

                        # It's possible the AI generated "Chill Vibes (2)" directly, which might already be taken.
                        # Loop until we find a unique name.
                        while current_playlist_final_name in ai_renamed_playlists_final:
                            ai_base_name_generation_count[ai_generated_base_name] += 1
                            suffix = ai_base_name_generation_count[ai_generated_base_name]
                            current_playlist_final_name = f"{ai_generated_base_name} ({suffix})"
                        
                        ai_renamed_playlists_final[current_playlist_final_name] = songs_in_playlist # Assign directly
                        if original_name in final_playlist_centroids:
                            ai_renamed_centroids_final[current_playlist_final_name] = final_playlist_centroids[original_name]
                        playlists_named_count += 1
                        if total_playlists_to_name > 0:
                            current_ai_progress = ai_naming_start_progress + (playlists_named_count / total_playlists_to_name) * ai_naming_progress_range
                            log_and_update_main_clustering(
                                f"AI Naming: Processed {playlists_named_count}/{total_playlists_to_name} playlists.",
                                int(current_ai_progress),
                                print_console=False
                            )
                    # Log if any AI-generated base names were generated for multiple input playlists
                    for ai_name, contribution_count in ai_base_name_generation_count.items():
                        if contribution_count > 1:
                            print(f"{log_prefix_main_task_ai} AI-generated base name '{ai_name}' was generated for {contribution_count} input playlists. They are stored as distinct playlists with numerical suffixes (e.g., '{ai_name} (2)').")

                    if ai_renamed_playlists_final:
                        final_named_playlists = ai_renamed_playlists_final
                        final_playlist_centroids = ai_renamed_centroids_final
                    log_and_update_main_clustering(f"{log_prefix_main_task_ai} AI Naming for best playlist set completed.", ai_naming_end_progress, print_console=True)
                except ImportError:
                    print(f"{log_prefix_main_task_ai} Could not import 'ai' module. Skipping AI naming for final playlists.")
                    traceback.print_exc()
                    current_progress = ai_naming_end_progress
                    log_and_update_main_clustering(f"{log_prefix_main_task_ai} AI Naming skipped due to import error.", current_progress, print_console=True)
                except Exception as e_ai_final:
                    print(f"{log_prefix_main_task_ai} Error during final AI playlist naming: {e_ai_final}. Using original names.")
                    traceback.print_exc()
                    current_progress = ai_naming_end_progress
                    log_and_update_main_clustering(f"{log_prefix_main_task_ai} AI Naming skipped due to error.", current_progress, print_console=True)
            else:
                print(f"{log_prefix_main_task_ai} AI Naming skipped: Provider is '{ai_model_provider_for_run}'.")
            playlists_to_create_on_jellyfin = {}
            centroids_for_jellyfin_playlists = {}
            for name, songs in final_named_playlists.items():
                 final_name_with_suffix = f"{name}_automatic"
                 playlists_to_create_on_jellyfin[final_name_with_suffix] = songs
                 if name in final_playlist_centroids:
                     centroids_for_jellyfin_playlists[final_name_with_suffix] = final_playlist_centroids[name]
            current_progress = 96
            log_and_update_main_clustering("Applying '_automatic' suffix to playlist names...", current_progress, print_console=True)
            current_progress = 98
            log_and_update_main_clustering("Creating/Updating playlists on Jellyfin...", current_progress, print_console=False)
            create_or_update_playlists_on_jellyfin(JELLYFIN_URL, JELLYFIN_USER_ID, {"X-Emby-Token": JELLYFIN_TOKEN},
                                                    playlists_to_create_on_jellyfin, centroids_for_jellyfin_playlists,
                                                    active_mood_labels, final_max_songs_per_cluster)
            final_db_summary = {
                "best_score": best_diversity_score,
                "best_params": best_clustering_results.get("parameters"),
                "num_playlists_created": len(playlists_to_create_on_jellyfin)
            }
            current_progress = 100
            log_and_update_main_clustering("Updating playlist database with final suffixed names...", current_progress, print_console=True, task_state=TASK_STATUS_PROGRESS)
            update_playlist_table(playlists_to_create_on_jellyfin)
            log_and_update_main_clustering("Playlist database updated.", current_progress, print_console=True, task_state=TASK_STATUS_PROGRESS)
            log_and_update_main_clustering(f"Playlists generated and updated on Jellyfin! Best diversity score: {best_diversity_score:.2f}.", current_progress, details_to_add_or_update=final_db_summary, task_state=TASK_STATUS_SUCCESS)
            return {"status": "SUCCESS", "message": f"Playlists generated and updated on Jellyfin! Best run had diversity score of {best_diversity_score:.2f}."}
        except Exception as e:
            error_tb = traceback.format_exc()
            print(f"FATAL ERROR: Clustering failed: {e}\n{error_tb}")
            with app.app_context():
                log_and_update_main_clustering(f"❌ Main clustering failed: {e}", current_progress, details_to_add_or_update={"error_message": str(e), "traceback": error_tb}, task_state=TASK_STATUS_FAILURE)
