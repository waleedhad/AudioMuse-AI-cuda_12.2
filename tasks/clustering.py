# tasks/clustering.py

import os
import shutil
from collections import defaultdict
import numpy as np
import json
import time
import random
import logging
import uuid
import traceback

# RQ import
from rq import get_current_job, Retry
from rq.job import Job
from rq.exceptions import NoSuchJobError, InvalidJobOperation

# Import necessary components from the main app.py file (ensure these are available)
from app import (app, redis_conn, get_db, save_task_status, get_task_info_from_db,
                track_exists, save_track_analysis, get_all_tracks, get_tracks_by_ids, update_playlist_table, JobStatus,
                TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
                TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED,
                get_child_tasks_from_db)
from psycopg2.extras import DictCursor


# Import configuration (ensure config.py is in PYTHONPATH or same directory)
from config import (TEMP_DIR, MAX_SONGS_PER_CLUSTER,
    MOOD_LABELS, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, ENERGY_MIN, ENERGY_MAX,
    TEMPO_MIN_BPM, TEMPO_MAX_BPM, JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, REDIS_URL, DATABASE_URL,
    OLLAMA_SERVER_URL, OLLAMA_MODEL_NAME, AI_MODEL_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL_NAME,
    DANCEABILITY_MODEL_PATH, AGGRESSIVE_MODEL_PATH, HAPPY_MODEL_PATH, PARTY_MODEL_PATH, RELAXED_MODEL_PATH, SAD_MODEL_PATH,
    MUTATION_KMEANS_COORD_FRACTION, MUTATION_INT_ABS_DELTA, MUTATION_FLOAT_ABS_DELTA,
    TOP_N_ELITES, EXPLOITATION_START_FRACTION, EXPLOITATION_PROBABILITY_CONFIG, TOP_N_MOODS, TOP_N_OTHER_FEATURES,
    STRATIFIED_GENRES, MIN_SONGS_PER_GENRE_FOR_STRATIFICATION, SAMPLING_PERCENTAGE_CHANGE_PER_RUN, ITERATIONS_PER_BATCH_JOB, MAX_CONCURRENT_BATCH_JOBS, ENABLE_CLUSTERING_EMBEDDINGS,
    DB_FETCH_CHUNK_SIZE, STRATIFIED_SAMPLING_TARGET_PERCENTILE)

# Import AI naming function and prompt template
from ai import get_ai_playlist_name, creative_prompt_template
# MODIFIED: Assuming a generic function exists in mediaserver.py for creating/updating playlists.
# If this function name is different in your version of mediaserver.py, please update it here.
from .mediaserver import create_playlist, delete_automatic_playlists
from .clustering_helper import (
    _get_stratified_song_subset,
    get_job_result_safely,
    _perform_single_clustering_iteration
)

logger = logging.getLogger(__name__)

# Task specific helper function



def run_clustering_batch_task(
    batch_id_str, start_run_idx, num_iterations_in_batch,
    genre_to_lightweight_track_data_map_json,
    target_songs_per_genre,
    sampling_percentage_change_per_run,
    clustering_method,
    active_mood_labels_for_batch,
    num_clusters_min_max_tuple,
    dbscan_params_ranges_dict,
    gmm_params_ranges_dict,
    spectral_params_ranges_dict,
    pca_params_ranges_dict,
    max_songs_per_cluster,
    parent_task_id,
    score_weight_diversity_param,
    score_weight_silhouette_param,
    score_weight_davies_bouldin_param,
    score_weight_calinski_harabasz_param,
    score_weight_purity_param,
    score_weight_other_feature_diversity_param,
    score_weight_other_feature_purity_param,
    elite_solutions_params_list_json,
    exploitation_probability,
    mutation_config_json,
    initial_subset_track_ids_json,
    enable_clustering_embeddings_param
    ):
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())
    logger.info(f"Starting clustering batch task {current_task_id} (Batch: {batch_id_str}) from queue: {current_job.origin if current_job else 'N/A'}")

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
                logger.info("[ClusteringBatchTask-%s] %s", current_task_id, message)

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
                    logger.warning("%s Warning: Could not decode elite solutions JSON. Proceeding without elites for this batch.", log_prefix_for_iter)
            mutation_config_for_iter = {}
            if mutation_config_json:
                try:
                    mutation_config_for_iter = json.loads(mutation_config_json)
                except json.JSONDecodeError:
                    logger.warning("%s Warning: Could not decode mutation config JSON. Using default mutation behavior.", log_prefix_for_iter)

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
                    clustering_method, num_clusters_min_max_tuple, dbscan_params_ranges_dict, gmm_params_ranges_dict,
                    spectral_params_ranges_dict,
                    pca_params_ranges_dict, active_mood_labels_for_batch,
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
            # Log the full traceback on the server for debugging.
            logger.error("Clustering batch %s failed", batch_id_str, exc_info=True)
            # Save a generic error to the DB, without the stack trace.
            failure_details = {"error": "An unexpected error occurred in the clustering batch.", "batch_id": batch_id_str}
            log_and_update_batch_task(
                f"Failed clustering batch {batch_id_str}: An unexpected error occurred.",
                current_progress_batch,
                details_extra=failure_details,
                task_state=TASK_STATUS_FAILURE,
                print_console=False
            )
            return {"status": "FAILURE", "iterations_completed_in_batch": iterations_actually_completed, "best_result_from_batch": None, "final_subset_track_ids": current_sampled_track_ids_in_batch}


def run_clustering_task(
    clustering_method, num_clusters_min, num_clusters_max,
    dbscan_eps_min, dbscan_eps_max, dbscan_min_samples_min, dbscan_min_samples_max,
    pca_components_min, pca_components_max, num_clustering_runs, max_songs_per_cluster_val, # Renamed to max_songs_per_cluster_val
    gmm_n_components_min, gmm_n_components_max,
    spectral_n_clusters_min, spectral_n_clusters_max,
    min_songs_per_genre_for_stratification_param,
    stratified_sampling_target_percentile_param,
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
    logger.info(f"Starting main clustering task {current_task_id} from queue: {current_job.origin if current_job else 'N/A'}")

    # Initialize variables for safe access in log messages
    final_max_songs_per_cluster = max_songs_per_cluster_val
    best_clustering_results = None

    with app.app_context():
        # IDEMPOTENCY CHECK: If task is already in a terminal state, don't run again.
        task_info = get_task_info_from_db(current_task_id)
        if task_info and task_info.get('status') in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]:
            logger.info(f"Main clustering task {current_task_id} is already in a terminal state ('{task_info.get('status')}'). Skipping execution.")
            final_details = {}
            if task_info.get('details'):
                try:
                    final_details = json.loads(task_info.get('details'))
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse details JSON for already terminal task {current_task_id}.")
            # Return the existing final status and details
            return {"status": task_info.get('status'), "message": f"Task already in terminal state '{task_info.get('status')}'.", "details": final_details}

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
            "clustering_method": clustering_method,
            "num_clusters_min": num_clusters_min,
            "num_clusters_max": num_clusters_max,
            "dbscan_eps_min": dbscan_eps_min,
            "dbscan_eps_max": dbscan_eps_max,
            "dbscan_min_samples_min": dbscan_min_samples_min,
            "dbscan_min_samples_max": dbscan_min_samples_max,
            "gmm_n_components_min": gmm_n_components_min,
            "gmm_n_components_max": gmm_n_components_max,
            "spectral_n_clusters_min": spectral_n_clusters_min,
            "spectral_n_clusters_max": spectral_n_clusters_max,
            "pca_components_min": pca_components_min,
            "pca_components_max": pca_components_max,
            "max_songs_per_cluster": max_songs_per_cluster_val,
            "min_songs_per_genre_for_stratification": min_songs_per_genre_for_stratification_param,
            "stratified_sampling_target_percentile": stratified_sampling_target_percentile_param,
            "score_weight_diversity_for_run": score_weight_diversity_param,
            "score_weight_silhouette_for_run": score_weight_silhouette_param,
            "score_weight_davies_bouldin_for_run": score_weight_davies_bouldin_param,
            "score_weight_calinski_harabasz_for_run": score_weight_calinski_harabasz_param,
            "score_weight_purity_for_run": score_weight_purity_param,
            "score_weight_other_feature_diversity_for_run": score_weight_other_feature_diversity_param,
            "score_weight_other_feature_purity_for_run": score_weight_other_feature_purity_param,
            "ai_model_provider_for_run": ai_model_provider_param,
            "ollama_server_url_for_run": ollama_server_url_param,
            "ollama_model_name_for_run": ollama_model_name_param,
            "gemini_model_name_for_run": gemini_model_name_param,
            "gemini_api_key_for_run": "<API_KEY_HIDDEN>", # Do not log actual API key
            "top_n_moods_for_clustering": top_n_moods_for_clustering_param,
            "embeddings_enabled_for_clustering": enable_clustering_embeddings_param
        }

        # Define current_progress before the helper function that uses it
        current_progress = 0

        # Define the helper function BEFORE it's called
        def log_and_update_main_clustering(message, progress, details_to_add_or_update=None, task_state=TASK_STATUS_PROGRESS, print_console=True):
            nonlocal current_progress, _main_task_accumulated_details
            current_progress = progress
            if print_console: logger.info("[MainClusteringTask-%s] %s", current_task_id, message)
            if details_to_add_or_update: _main_task_accumulated_details.update(details_to_add_or_update)
            _main_task_accumulated_details["status_message"] = message
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            current_log_list = _main_task_accumulated_details.get("log", [])
            if task_state == TASK_STATUS_SUCCESS:
                _main_task_accumulated_details["log"] = [f"Task completed successfully. Final status: {message}"]
                _main_task_accumulated_details.pop('log_storage_info', None)
                # Remove specific keys for SUCCESS state
                _main_task_accumulated_details.pop('best_params', None)
                _main_task_accumulated_details.pop('clustering_run_job_ids', None)
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
                        "clustering_run_job_ids", "batch_jobs_launched", "total_batch_jobs", "active_runs_count",
                        "clustering_method", "num_clusters_min", "num_clusters_max", "dbscan_eps_min",
                        "dbscan_eps_max", "dbscan_min_samples_min", "dbscan_min_samples_max",
                        "gmm_n_components_min", "gmm_n_components_max", "spectral_n_clusters_min", "spectral_n_clusters_max",
                        "pca_components_min", "pca_components_max", "max_songs_per_cluster",
                        "min_songs_per_genre_for_stratification", "stratified_sampling_target_percentile",
                        "score_weight_diversity_for_run", "score_weight_silhouette_for_run",
                        "score_weight_davies_bouldin_for_run", "score_weight_calinski_harabasz_for_run",
                        "score_weight_purity_for_run", "score_weight_other_feature_diversity_for_run",
                        "score_weight_other_feature_purity_for_run", "ai_model_provider_for_run",
                        "ollama_server_url_for_run", "ollama_model_name_for_run",
                        "gemini_model_name_for_run", "top_n_moods_for_clustering", "embeddings_enabled_for_clustering"]:
                if key in _main_task_accumulated_details and _main_task_accumulated_details[key] is not None:
                    # Skip sensitive info for meta details if not explicitly needed
                    if key == "gemini_api_key_for_run":
                         meta_details[key] = "<API_KEY_HIDDEN>"
                    else:
                        meta_details[key] = _main_task_accumulated_details[key]
            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.meta['details'] = meta_details
                current_job.save_meta()
            save_task_status(current_task_id, "main_clustering", task_state, progress=progress, details=_main_task_accumulated_details)

        # --- STATE RECOVERY BLOCK ---
        log_and_update_main_clustering("Checking for existing state (recovering)...", 2, print_console=False)

        active_jobs_map = {}
        elite_solutions_list = []

        # Track if this task is recovering from a previous run
        is_recovering = False
        best_diversity_score = _main_task_accumulated_details.get("best_score", -1.0)
        batches_completed_count = 0
        total_iterations_completed_count = 0
        last_subset_track_ids_for_batch_chaining = []

        child_tasks_from_db = get_child_tasks_from_db(current_task_id)

        batch_index_to_task_map = {}
        if child_tasks_from_db:
            for child_row in child_tasks_from_db:
                batch_id_str = child_row['sub_type_identifier']
                if batch_id_str and batch_id_str.startswith("Batch_"):
                    try:
                        batch_index = int(batch_id_str.split('_')[1])
                        batch_index_to_task_map[batch_index] = child_row
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse batch index from sub_type_identifier: {batch_id_str}")

        max_processed_batch_idx = -1

        if batch_index_to_task_map:
            sorted_batch_indices = sorted(batch_index_to_task_map.keys())
            for batch_idx in sorted_batch_indices:
                child_task = batch_index_to_task_map[batch_idx]
                child_status = child_task['status']
                child_id = child_task['task_id']

                if child_status in [TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS]:
                    logger.info(f"Recovering and monitoring active batch job {child_id} (Batch_{batch_idx}).")
                    try:
                        job_instance = Job.fetch(child_id, connection=redis_conn)
                        active_jobs_map[child_id] = job_instance
                    except NoSuchJobError:
                        logger.warning(f"Job {child_id} was active in DB but not found in RQ. It will be ignored.")


                elif child_status in [TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]:
                    logger.info(f"Recovering from failed/revoked batch job {child_id} (Batch_{batch_idx}).")
                    batches_completed_count += 1
                    max_processed_batch_idx = batch_idx

                elif child_status == TASK_STATUS_SUCCESS:
                    logger.info(f"Recovering from successful batch job {child_id} (Batch_{batch_idx}).")
                    batch_job_result = get_job_result_safely(child_id, current_task_id, f"clustering_batch (recovery)")
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
                                elite_solutions_list.append({"score": batch_best_score, "params": batch_best_params})
                            if batch_best_score > best_diversity_score:
                                best_diversity_score = batch_best_score
                                _main_task_accumulated_details["best_score"] = best_diversity_score
                                best_clustering_results = best_from_batch
                    batches_completed_count += 1
                    max_processed_batch_idx = batch_idx

        elite_solutions_list.sort(key=lambda x: x["score"], reverse=True)
        elite_solutions_list = elite_solutions_list[:TOP_N_ELITES]
        next_batch_job_idx_to_launch = max_processed_batch_idx + 1

        logger.info(f"State recovery complete. Resuming. Batches completed: {batches_completed_count}, Iterations completed: {total_iterations_completed_count}, Next batch to launch: {next_batch_job_idx_to_launch}, Best score so far: {best_diversity_score:.2f}")
        # --- END OF STATE RECOVERY BLOCK ---

        if batches_completed_count > 0:
            is_recovering = True

        if is_recovering:
            log_and_update_main_clustering(
                f"Resuming clustering with existing state: {num_clustering_runs} total runs, {batches_completed_count} batches previously completed, next batch to launch: {next_batch_job_idx_to_launch}.",
                2, print_console=True
            )
        else:
             log_and_update_main_clustering(
                f"Starting fresh clustering with {num_clustering_runs} runs.",
                2, print_console=True
            )

        save_task_status(current_task_id, "main_clustering", TASK_STATUS_STARTED, progress=0, details=_main_task_accumulated_details)

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
                logger.info("Number of clustering runs is 0.")
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
                else:
                    genre_to_lightweight_track_data_map['__other__'].append({'item_id': row['item_id'], 'mood_vector': ''})


            songs_counts_for_stratified_genres = []
            for genre in STRATIFIED_GENRES:
                if genre in genre_to_lightweight_track_data_map:
                    songs_counts_for_stratified_genres.append(len(genre_to_lightweight_track_data_map[genre]))
            calculated_target_based_on_percentile = 0
            if songs_counts_for_stratified_genres:
                percentile_to_use = np.clip(stratified_sampling_target_percentile_param, 0, 100)
                calculated_target_based_on_percentile = np.percentile(songs_counts_for_stratified_genres, percentile_to_use)
                log_and_update_main_clustering(
                    f"{percentile_to_use}th percentile of songs per stratified genre: {calculated_target_based_on_percentile:.2f}",
                    current_progress, print_console=False
                )
            else:
                log_and_update_main_clustering("No songs found for any stratified genres. Defaulting target.", current_progress, print_console=False)
            target_songs_per_genre = max(min_songs_per_genre_for_stratification_param, int(np.floor(calculated_target_based_on_percentile)))
            if target_songs_per_genre == 0 and len(lightweight_rows) > 0: target_songs_per_genre = 1
            log_and_update_main_clustering(f"Determined target songs per genre for stratification: {target_songs_per_genre}", 7)
            genre_to_lightweight_track_data_map_json = json.dumps(genre_to_lightweight_track_data_map)
            log_and_update_main_clustering("Stratified sampling preparation complete.", 6, print_console=False)

            # If we are not recovering state, set the initial subset. Otherwise, recovery has already set it.
            if not last_subset_track_ids_for_batch_chaining:
                initial_subset_lightweight = _get_stratified_song_subset(
                    genre_to_lightweight_track_data_map,
                    target_songs_per_genre,
                    previous_subset_track_ids=None,
                    percentage_change=0.0
                )
                last_subset_track_ids_for_batch_chaining = [t['item_id'] for t in initial_subset_lightweight]

            mutation_config = {
                "int_abs_delta": MUTATION_INT_ABS_DELTA,
                "float_abs_delta": MUTATION_FLOAT_ABS_DELTA,
                "coord_mutation_fraction": MUTATION_KMEANS_COORD_FRACTION
            }
            mutation_config_json = json.dumps(mutation_config)
            exploitation_start_run_idx = int(num_clustering_runs * EXPLOITATION_START_FRACTION)
            all_launched_child_jobs_instances = []
            from app import rq_queue_default
            num_total_batch_jobs = (num_clustering_runs + ITERATIONS_PER_BATCH_JOB - 1) // ITERATIONS_PER_BATCH_JOB
            _main_task_accumulated_details["total_batch_jobs"] = num_total_batch_jobs
            data_source_for_clustering_log = "embeddings" if enable_clustering_embeddings_param else "score vectors"
            log_and_update_main_clustering(f"Processing {len(lightweight_rows)} tracks using {data_source_for_clustering_log}. Preparing {num_clustering_runs} runs in {num_total_batch_jobs} batches.", 8)

            while batches_completed_count < num_total_batch_jobs:
                # --- Worker Shutdown Check ---
                if current_job and current_job.is_stopped:
                    logger.warning(f"Main clustering task {current_task_id} received stop signal from worker. Raising exception to force re-queue.")
                    raise Exception("Worker shutdown detected, re-queueing task for graceful restart.")
                # --- End Worker Shutdown Check ---

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
                            logger.warning("[MainClusteringTask-%s] Warning: Active batch job %s missing from RQ, DB status '%s' is not terminal. Treating as completed (abnormally) to prevent stall.", current_task_id, job_id, db_status_for_log)
                            is_child_truly_completed_this_cycle = True
                    except Exception as e_monitor_child_active:
                        logger.error("[MainClusteringTask-%s] ERROR monitoring active batch job %s: %s. Treating as completed.", current_task_id, job_id, e_monitor_child_active, exc_info=True)
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
                                    elite_solutions_list.append({"score": batch_best_score, "params": batch_best_params})
                                if batch_best_score > best_diversity_score:
                                    best_diversity_score = batch_best_score
                                    _main_task_accumulated_details["best_score"] = best_diversity_score
                                    best_clustering_results = best_from_batch
                                    logger.info("[MainClusteringTask-%s] Intermediate new best score: %.2f from batch job %s", current_task_id, best_diversity_score, job_id_just_completed)
                        else:
                            logger.warning("[MainClusteringTask-%s] Warning: Batch job %s completed but no result.", current_task_id, job_id_just_completed)
                if newly_completed_elite_candidates:
                    all_potential_elites = elite_solutions_list + newly_completed_elite_candidates
                    all_potential_elites.sort(key=lambda x: x["score"], reverse=True)
                    elite_solutions_list = all_potential_elites[:TOP_N_ELITES]
                can_launch_more_batch_jobs = next_batch_job_idx_to_launch < num_total_batch_jobs
                if can_launch_more_batch_jobs:
                    num_slots_to_fill = MAX_CONCURRENT_BATCH_JOBS - len(active_jobs_map)
                    for _ in range(num_slots_to_fill):
                        if next_batch_job_idx_to_launch >= num_total_batch_jobs: break

                        # IDEMPOTENCY: Use a deterministic job ID for batch tasks
                        batch_job_task_id = f"{current_task_id}_batch_{next_batch_job_idx_to_launch}"

                        current_batch_start_run_idx = next_batch_job_idx_to_launch * ITERATIONS_PER_BATCH_JOB
                        num_iterations_for_this_batch = min(ITERATIONS_PER_BATCH_JOB, num_clustering_runs - current_batch_start_run_idx)
                        if num_iterations_for_this_batch <= 0:
                            logger.warning("[MainClusteringTask-%s] Warning: Calculated 0 iterations for batch %s. Skipping.", current_task_id, next_batch_job_idx_to_launch)
                            next_batch_job_idx_to_launch +=1
                            continue
                        batch_id_for_logging = f"Batch_{next_batch_job_idx_to_launch}"
                        save_task_status(batch_job_task_id, "clustering_batch", "PENDING", parent_task_id=current_task_id, sub_type_identifier=batch_id_for_logging,
                                         details={"start_run_idx": current_batch_start_run_idx, "num_iterations": num_iterations_for_this_batch})
                        num_clusters_min_max_tuple_for_batch = (num_clusters_min, num_clusters_max)
                        dbscan_params_ranges_dict_for_batch = {"eps_min": dbscan_eps_min, "eps_max": dbscan_eps_max, "samples_min": dbscan_min_samples_min, "samples_max": dbscan_min_samples_max}
                        gmm_params_ranges_dict_for_batch = {"n_components_min": gmm_n_components_min, "n_components_max": gmm_n_components_max}
                        spectral_params_ranges_dict_for_batch = {"n_clusters_min": spectral_n_clusters_min, "n_clusters_max": spectral_n_clusters_max}
                        pca_params_ranges_dict_for_batch = {"components_min": pca_components_min, "components_max": pca_components_max}
                        current_elite_params_for_batch_json = json.dumps([item["params"] for item in elite_solutions_list]) if elite_solutions_list else "[]"
                        should_exploit_for_this_batch = (current_batch_start_run_idx >= exploitation_start_run_idx) and elite_solutions_list
                        exploitation_prob_for_this_batch = EXPLOITATION_PROBABILITY_CONFIG if should_exploit_for_this_batch else 0.0
                        log_and_update_main_clustering(f"Preparing to enqueue batch job {batch_id_for_logging} (runs {current_batch_start_run_idx}-{current_batch_start_run_idx + num_iterations_for_this_batch -1}). Initial subset IDs for this batch: {'Provided' if last_subset_track_ids_for_batch_chaining else 'None'}", current_progress, print_console=False)
                        initial_subset_for_this_batch_json = json.dumps(last_subset_track_ids_for_batch_chaining) if last_subset_track_ids_for_batch_chaining else "[]"

                        new_job = rq_queue_default.enqueue(
                            run_clustering_batch_task,
                            args=(
                                batch_id_for_logging, current_batch_start_run_idx, num_iterations_for_this_batch,
                                genre_to_lightweight_track_data_map_json,
                                target_songs_per_genre,
                                SAMPLING_PERCENTAGE_CHANGE_PER_RUN,
                                clustering_method, active_mood_labels,
                                num_clusters_min_max_tuple_for_batch, dbscan_params_ranges_dict_for_batch,
                                gmm_params_ranges_dict_for_batch,
                                spectral_params_ranges_dict_for_batch,
                                pca_params_ranges_dict_for_batch,
                                max_songs_per_cluster_val, current_task_id,
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
                            meta={'parent_task_id': current_task_id},
                            retry=Retry(max=3)
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
                time.sleep(3)

            log_and_update_main_clustering("All clustering batch jobs completed. Finalizing best result...", 90,
                                           details_to_add_or_update={"best_score": best_diversity_score})
            if not best_clustering_results or best_diversity_score < 0:
                log_and_update_main_clustering("No valid clustering solution found after all runs.", 100, details_to_add_or_update={"error": "No suitable clustering found", "best_score": best_diversity_score}, task_state=TASK_STATUS_FAILURE)
                logger.error("No valid clustering solution found. Best score: %s", best_diversity_score)
                return {"status": "FAILURE", "message": "No valid clusters found after multiple runs."}
            current_progress = 92

            # Log final best parameters right before playlist creation
            best_params_log_string = (
                f"Final Best Parameters: "
                f"Method={best_clustering_results['parameters']['clustering_method_config']['method']}, "
                f"Runs={num_clustering_runs}, "
                f"MaxPerCluster={final_max_songs_per_cluster}, "
            )

            clustering_method_conf = best_clustering_results['parameters']['clustering_method_config']

            if clustering_method_conf['method'] == 'kmeans':
                best_params_log_string += f"K={clustering_method_conf['params']['n_clusters']}, "
            elif clustering_method_conf['method'] == 'dbscan':
                dbscan_params = clustering_method_conf['params']
                best_params_log_string += f"Eps={dbscan_params['eps']}, MinSamples={dbscan_params['min_samples']}, "
            elif clustering_method_conf['method'] == 'gmm':
                best_params_log_string += f"Components={clustering_method_conf['params']['n_components']}, "
            elif clustering_method_conf['method'] == 'spectral':
                best_params_log_string += f"N_Clusters={clustering_method_conf['params']['n_clusters']}, "


            pca_conf = best_clustering_results['parameters']['pca_config']
            best_params_log_string += f"PCA={pca_conf['enabled']}"
            if pca_conf['enabled']:
                best_params_log_string += f"({pca_conf['components']})"

            enable_embeddings_log_val = enable_clustering_embeddings_param if enable_clustering_embeddings_param is not None else False

            log_and_update_main_clustering(
                f"Best clustering parameters: {best_params_log_string}. Diversity score: {best_diversity_score:.2f}. Using Embeddings: {enable_embeddings_log_val}",
                91,  # Slightly before playlist creation
                details_to_add_or_update={"best_score": best_diversity_score, "best_params": best_clustering_results.get("parameters")},
                print_console=True
            )

            log_and_update_main_clustering(
            f"Score Weights: Diversity={score_weight_diversity_param:.2f}, Silhouette={score_weight_silhouette_param:.2f}, "
            f"DaviesBouldin={score_weight_davies_bouldin_param:.2f}, CalinskiHarabasz={score_weight_calinski_harabasz_param:.2f}, "
            f"Purity={score_weight_purity_param:.2f}, OtherDiv={score_weight_other_feature_diversity_param:.2f}, "
            f"OtherPur={score_weight_other_feature_purity_param:.2f}",
            91,
            details_to_add_or_update={"best_score": best_diversity_score, "best_params": best_clustering_results.get("parameters")},
            print_console=True
        )

            log_and_update_main_clustering(
                f"Configuration Parameters - Stratification: MinSongsPerGenre={min_songs_per_genre_for_stratification_param}, "
                f"TargetPercentile={stratified_sampling_target_percentile_param}, "
                f"AI Naming: Provider={ai_model_provider_param}, TopMoods={top_n_moods_for_clustering_param}, "
                f"MaxSongsPerPlaylist={max_songs_per_cluster_val}",
                91,  # Just before playlist creation
                details_to_add_or_update={"best_score": best_diversity_score, "best_params": best_clustering_results.get("parameters")},
                print_console=True
            )

            log_and_update_main_clustering(
            f"Ollama Settings: Server={ollama_server_url_param}, Model={ollama_model_name_param}. "
            f"Gemini Settings: Model={gemini_model_name_param}", # API Key not shown for security
            91,  # Just before playlist creation
            details_to_add_or_update={"best_score": best_diversity_score, "best_params": best_clustering_results.get("parameters")},
            print_console=True
        )

            log_and_update_main_clustering(f"Best clustering found with diversity score: {best_diversity_score:.2f}. Preparing to create playlists.", current_progress, details_to_add_or_update={"best_score": best_diversity_score, "best_params": best_clustering_results.get("parameters")})
            final_named_playlists = best_clustering_results["named_playlists"]
            final_playlist_centroids = best_clustering_results["playlist_centroids"]
            final_max_songs_per_cluster = best_clustering_results["parameters"]["max_songs_per_cluster"]
            final_pca_model_details = best_clustering_results["pca_model_details"]
            final_scaler_details = best_clustering_results["scaler_details"]
            log_prefix_main_task_ai = f"[MainClusteringTask-{current_task_id} AI Naming]"
            if ai_model_provider_for_run in ["OLLAMA", "GEMINI"]:
                logger.info("%s AI Naming block entered. Attempting to import 'ai' module.", log_prefix_main_task_ai)
                ai_naming_start_progress = 90
                ai_naming_end_progress = 95
                ai_naming_progress_range = ai_naming_end_progress - ai_naming_start_progress
                current_progress = ai_naming_start_progress
                log_and_update_main_clustering(f"Preparing for AI Naming...", current_progress, print_console=True)
                try:
                    ai_renamed_playlists_final = {}
                    ai_renamed_centroids_final = {}
                    ai_base_name_generation_count = defaultdict(int)
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
                            logger.info("%s Embeddings used for clustering. Using generic tags for AI: '%s', '%s', '%s'.", log_prefix_main_task_ai, feature1_for_ai, feature2_for_ai, feature3_for_ai)
                        else:
                            # If not using embeddings, derive tags from the rule-based original_name
                            name_parts = original_name.split('_')
                            if len(name_parts) >= 1: feature1_for_ai = name_parts[0]
                            if len(name_parts) >= 2: feature2_for_ai = name_parts[1]
                            if len(name_parts) >= 3 and name_parts[2].lower() not in ["slow", "medium", "fast"]:
                                feature3_for_ai = name_parts[2]
                            elif len(name_parts) >= 2 and name_parts[1].lower() not in ["slow", "medium", "fast"]:
                                feature3_for_ai = name_parts[1]
                        logger.info("%s Generating AI name for '%s' (%s songs) using provider '%s'. Tags for AI: F1: '%s', F2: '%s', F3: '%s'.", log_prefix_main_task_ai, original_name, len(song_list_for_ai), ai_model_provider_for_run, feature1_for_ai, feature2_for_ai, feature3_for_ai)
                        centroid_features_for_ai = final_playlist_centroids.get(original_name, {})
                        ai_generated_name_str = get_ai_playlist_name(
                            ai_model_provider_for_run,
                            ollama_server_url_for_run, ollama_model_name_for_run,
                            gemini_api_key_for_run, gemini_model_name_for_run,
                            creative_prompt_template,
                            feature1_for_ai, feature2_for_ai, feature3_for_ai,
                            song_list_for_ai,
                            centroid_features_for_ai)
                        logger.info("%s Raw AI output for '%s': '%s'", log_prefix_main_task_ai, original_name, ai_generated_name_str)
                        current_playlist_final_name = original_name
                        ai_generated_base_name = original_name

                        if ai_generated_name_str and not ai_generated_name_str.startswith("Error") and not ai_generated_name_str.startswith("AI Naming Skipped"):
                             clean_ai_name = ai_generated_name_str.strip().replace("\n", " ")
                             if clean_ai_name:
                                 if clean_ai_name.lower() != original_name.lower().strip().replace("_", " "):
                                      logger.info("%s AI: '%s' -> '%s'", log_prefix_main_task_ai, original_name, clean_ai_name)
                                      ai_generated_base_name = clean_ai_name
                                 else:
                                      ai_generated_base_name = original_name
                             else:
                                 logger.warning("%s AI for '%s' returned empty after cleaning. Raw: '%s'. Using original.", log_prefix_main_task_ai, original_name, ai_generated_name_str)
                                 ai_generated_base_name = original_name
                        else:
                            logger.warning("%s AI naming for '%s' failed or returned error/skip message: '%s'. Using original.", log_prefix_main_task_ai, original_name, ai_generated_name_str)
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

                        ai_renamed_playlists_final[current_playlist_final_name] = songs_in_playlist
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
                            logger.info("%s AI-generated base name '%s' was generated for %s input playlists. They are stored as distinct playlists with numerical suffixes (e.g., '%s (2)').", log_prefix_main_task_ai, ai_name, contribution_count, ai_name)

                    if ai_renamed_playlists_final:
                        final_named_playlists = ai_renamed_playlists_final
                        final_playlist_centroids = ai_renamed_centroids_final
                    log_and_update_main_clustering(f"{log_prefix_main_task_ai} AI Naming for best playlist set completed.", ai_naming_end_progress, print_console=True)
                except ImportError:
                    logger.error("%s Could not import 'ai' module. Skipping AI naming for final playlists.", log_prefix_main_task_ai, exc_info=True)
                    current_progress = ai_naming_end_progress
                    log_and_update_main_clustering(f"{log_prefix_main_task_ai} AI Naming skipped due to import error.", current_progress, print_console=True)
                except Exception as e_ai_final:
                    logger.error("%s Error during final AI playlist naming: %s. Using original names.", log_prefix_main_task_ai, e_ai_final, exc_info=True)
                    current_progress = ai_naming_end_progress
                    log_and_update_main_clustering(f"{log_prefix_main_task_ai} AI Naming skipped due to error.", current_progress, print_console=True)
            else:
                logger.info("%s AI Naming skipped: Provider is '%s'.", log_prefix_main_task_ai, ai_model_provider_for_run)
            playlists_to_create_on_media_server = {}
            centroids_for_media_server_playlists = {}
            for name, songs in final_named_playlists.items():
                 final_name_with_suffix = f"{name}_automatic"
                 playlists_to_create_on_media_server[final_name_with_suffix] = songs
                 if name in final_playlist_centroids:
                     centroids_for_media_server_playlists[final_name_with_suffix] = final_playlist_centroids[name]
            current_progress = 96
            log_and_update_main_clustering("Applying '_automatic' suffix to playlist names...", current_progress, print_console=True)
            
            log_and_update_main_clustering("Deleting existing '_automatic' playlists...", 97, print_console=True)
            try:
                delete_automatic_playlists()
                log_and_update_main_clustering("Deletion of existing playlists complete.", 98, print_console=True)
            except Exception as e_delete:
                logger.error(f"Failed to delete automatic playlists: {e_delete}", exc_info=True)
                log_and_update_main_clustering(f"Warning: Failed to delete existing playlists: {e_delete}", 98, print_console=True)

            current_progress = 98
            log_and_update_main_clustering("Creating/Updating playlists on media server...", current_progress, print_console=False)
            
            # Reformat playlists for the new function signature
            final_playlists_for_creation = {}
            for base_name, cluster in playlists_to_create_on_media_server.items():
                chunks = []
                if final_max_songs_per_cluster > 0:
                    chunks = [cluster[i:i+final_max_songs_per_cluster] for i in range(0, len(cluster), final_max_songs_per_cluster)]
                else:
                    if cluster: chunks = [cluster]
                for idx, chunk in enumerate(chunks, 1):
                    playlist_name_on_server = f"{base_name} ({idx})" if len(chunks) > 1 else base_name
                    item_ids = [item_id for item_id, _, _ in chunk]
                    if item_ids:
                        final_playlists_for_creation[playlist_name_on_server] = item_ids

            # MODIFIED: Loop through playlists and call the simplified create_playlist function
            for name, ids in final_playlists_for_creation.items():
                try:
                    create_playlist(name, ids) # Assuming create_playlist handles both creation and updates
                except Exception as e_create_playlist:
                    logger.error(f"Failed to create playlist '{name}' on media server: {e_create_playlist}", exc_info=True)

            final_db_summary = {
                "best_score": best_diversity_score,
                # "best_params": best_clustering_results.get("parameters"), # Removed
                "num_playlists_created": len(playlists_to_create_on_media_server)
            }
            current_progress = 100
            log_and_update_main_clustering("Updating playlist database with final suffixed names...", current_progress, print_console=True, task_state=TASK_STATUS_PROGRESS)
            update_playlist_table(playlists_to_create_on_media_server)
            log_and_update_main_clustering("Playlist database updated.", current_progress, print_console=True, task_state=TASK_STATUS_PROGRESS)
            log_and_update_main_clustering(f"Playlists generated and updated on the media server! Best diversity score: {best_diversity_score:.2f}.", current_progress, details_to_add_or_update=final_db_summary, task_state=TASK_STATUS_SUCCESS)
            return {"status": "SUCCESS", "message": f"Playlists generated and updated on the media server! Best run had diversity score of {best_diversity_score:.2f}."}
        except Exception as e:
            # Log the full traceback on the server for debugging.
            logger.critical("FATAL ERROR: Main clustering task failed: %s", e, exc_info=True)
            # The with app.app_context() is technically redundant here as we are already in one,
            # but we'll keep it to be safe and explicit.
            # The with app.app_context() is technically redundant here as we are already in one,
            # but we'll keep it to be safe and explicit.
            with app.app_context():
                # Save a generic error to the DB, without the stack trace.
                log_and_update_main_clustering(
                    "❌ Main clustering failed due to an unexpected error.",
                    current_progress,
                    details_to_add_or_update={"error_message": "An unexpected error occurred. Check server logs for details."},
                    task_state=TASK_STATUS_FAILURE,
                    print_console=False
                )

            # Re-raising the exception is crucial for RQ to handle retries if configured on the task itself
            raise
