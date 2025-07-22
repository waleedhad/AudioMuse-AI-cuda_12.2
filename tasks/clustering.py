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
from scipy.spatial.distance import cdist # *** NEW: For distance calculations ***

# RQ import
from rq import get_current_job, Retry
from rq.job import Job
from rq.exceptions import NoSuchJobError

# Import necessary components from the main app.py file
from app import (app, redis_conn, get_db, save_task_status, get_task_info_from_db,
                 update_playlist_table, JobStatus,
                 TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
                 TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED,
                 get_child_tasks_from_db)
from psycopg2.extras import DictCursor

# Import configuration
from config import (MAX_SONGS_PER_CLUSTER, MOOD_LABELS, STRATIFIED_GENRES,
                    MUTATION_KMEANS_COORD_FRACTION, MUTATION_INT_ABS_DELTA, MUTATION_FLOAT_ABS_DELTA,
                    TOP_N_ELITES, EXPLOITATION_START_FRACTION, EXPLOITATION_PROBABILITY_CONFIG,
                    SAMPLING_PERCENTAGE_CHANGE_PER_RUN, ITERATIONS_PER_BATCH_JOB, MAX_CONCURRENT_BATCH_JOBS)

# Import AI naming function and prompt template
from ai import get_ai_playlist_name, creative_prompt_template
# Import media server functions
from .mediaserver import create_playlist, delete_automatic_playlists
# Import refactored clustering helpers
from .clustering_helper import (
    _get_stratified_song_subset,
    get_job_result_safely,
    _perform_single_clustering_iteration
)

logger = logging.getLogger(__name__)

def _sanitize_for_json(obj):
    """
    Recursively converts numpy arrays and numpy numeric types to native Python types
    to ensure the object is JSON serializable.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle numpy numeric types which are not JSON serializable by default
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


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
    score_weights_dict,
    elite_solutions_params_list_json,
    exploitation_probability,
    mutation_config_json,
    initial_subset_track_ids_json,
    enable_clustering_embeddings_param
):
    """
    Executes a batch of clustering iterations. This task is enqueued by the main clustering task.
    """
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())
    logger.info(f"Starting clustering batch task {current_task_id} (Batch: {batch_id_str})")

    with app.app_context():
        # Helper for logging and updating task status
        def _log_and_update(message, progress, details=None, state=TASK_STATUS_PROGRESS):
            logger.info(f"[ClusteringBatchTask-{current_task_id}] {message}")
            db_details = {
                "batch_id": batch_id_str,
                "start_run_idx": start_run_idx,
                "num_iterations_in_batch": num_iterations_in_batch,
                "status_message": message,
                **(details or {})
            }
            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.save_meta()
            save_task_status(current_task_id, "clustering_batch", state, parent_task_id=parent_task_id,
                             sub_type_identifier=batch_id_str, progress=progress, details=db_details)

        try:
            _log_and_update("Batch started.", 0)
            genre_to_lightweight_track_data_map = json.loads(genre_to_lightweight_track_data_map_json)
            elite_solutions_params_list = json.loads(elite_solutions_params_list_json)
            mutation_config = json.loads(mutation_config_json)
            current_sampled_track_ids = json.loads(initial_subset_track_ids_json)

            best_result_in_batch = None
            best_score_in_batch = -1.0 # Use -1.0 as a safe initial value
            iterations_completed = 0

            for i in range(num_iterations_in_batch):
                current_run_global_idx = start_run_idx + i

                # Revocation Check
                if current_job:
                    task_info = get_task_info_from_db(current_task_id)
                    parent_task_info = get_task_info_from_db(parent_task_id)
                    if (task_info and task_info.get('status') == TASK_STATUS_REVOKED) or \
                       (parent_task_info and parent_task_info.get('status') in [TASK_STATUS_REVOKED, TASK_STATUS_FAILURE]):
                        _log_and_update("Stopping batch due to revocation.", i, state=TASK_STATUS_REVOKED)
                        return {"status": "REVOKED", "message": "Batch task revoked."}

                # Get a new subset of songs for this iteration, perturbing the previous one
                percentage_change = 0.0 if i == 0 else sampling_percentage_change_per_run
                current_subset_lightweight_data = _get_stratified_song_subset(
                    genre_to_lightweight_track_data_map,
                    target_songs_per_genre,
                    prev_ids=current_sampled_track_ids,
                    percent_change=percentage_change
                )
                item_ids_for_iteration = [t['item_id'] for t in current_subset_lightweight_data]
                current_sampled_track_ids = list(item_ids_for_iteration)

                if not item_ids_for_iteration:
                    logger.warning(f"No songs in subset for iteration {current_run_global_idx}. Skipping.")
                    continue

                iteration_result = _perform_single_clustering_iteration(
                    run_idx=current_run_global_idx,
                    item_ids_for_subset=item_ids_for_iteration,
                    clustering_method=clustering_method,
                    num_clusters_min_max=num_clusters_min_max_tuple,
                    dbscan_params_ranges=dbscan_params_ranges_dict,
                    gmm_params_ranges=gmm_params_ranges_dict,
                    spectral_params_ranges=spectral_params_ranges_dict,
                    pca_params_ranges=pca_params_ranges_dict,
                    active_mood_labels=active_mood_labels_for_batch,
                    max_songs_per_cluster=max_songs_per_cluster,
                    log_prefix=f"[Batch-{current_task_id}]",
                    elite_solutions_params_list=elite_solutions_params_list,
                    exploitation_probability=exploitation_probability,
                    mutation_config=mutation_config,
                    score_weights=score_weights_dict,
                    enable_clustering_embeddings=enable_clustering_embeddings_param
                )
                iterations_completed += 1

                if iteration_result and iteration_result.get("fitness_score", -1.0) > best_score_in_batch:
                    best_score_in_batch = iteration_result["fitness_score"]
                    best_result_in_batch = iteration_result

                progress = int(100 * (i + 1) / num_iterations_in_batch)
                _log_and_update(f"Iteration {current_run_global_idx} complete. Batch best score: {best_score_in_batch:.2f}", progress)

            # *** FIX: Sanitize the result to make it JSON-serializable before logging/returning ***
            if best_result_in_batch:
                best_result_in_batch = _sanitize_for_json(best_result_in_batch)

            final_details = {
                "best_score_in_batch": best_score_in_batch,
                "iterations_completed_in_batch": iterations_completed,
                "full_best_result_from_batch": best_result_in_batch,
                "final_subset_track_ids": current_sampled_track_ids
            }
            _log_and_update(f"Batch complete. Best score: {best_score_in_batch or -1:.2f}", 100, details=final_details, state=TASK_STATUS_SUCCESS)
            return {
                "status": "SUCCESS",
                "iterations_completed_in_batch": iterations_completed,
                "best_result_from_batch": best_result_in_batch,
                "final_subset_track_ids": current_sampled_track_ids
            }

        except Exception as e:
            logger.error(f"Clustering batch {batch_id_str} failed", exc_info=True)
            _log_and_update(f"Batch failed: {e}", 100, details={"error": str(e)}, state=TASK_STATUS_FAILURE)
            return {"status": "FAILURE", "message": str(e)}


def run_clustering_task(
    clustering_method, num_clusters_min, num_clusters_max,
    dbscan_eps_min, dbscan_eps_max, dbscan_min_samples_min, dbscan_min_samples_max,
    pca_components_min, pca_components_max, num_clustering_runs, max_songs_per_cluster_val,
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
    top_n_playlists_param, # *** NEW: Accept Top N parameter ***
    enable_clustering_embeddings_param):
    """
    Main entry point for the clustering process.
    Orchestrates data preparation, batch job creation, result aggregation, and playlist creation.
    """
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())
    logger.info(f"Starting main clustering task {current_task_id}")

    # Capture initial parameters for the final report
    initial_params = {
        "clustering_method": clustering_method,
        "pca_components_min": pca_components_min,
        "pca_components_max": pca_components_max,
        "use_embeddings": enable_clustering_embeddings_param,
        "top_n_playlists": top_n_playlists_param, # *** NEW: Log Top N parameter ***
        "stratification_percentile": stratified_sampling_target_percentile_param,
        "score_weights": {
            "mood_diversity": score_weight_diversity_param,
            "silhouette": score_weight_silhouette_param,
            "davies_bouldin": score_weight_davies_bouldin_param,
            "calinski_harabasz": score_weight_calinski_harabasz_param,
            "mood_purity": score_weight_purity_param,
            "other_feature_diversity": score_weight_other_feature_diversity_param,
            "other_feature_purity": score_weight_other_feature_purity_param
        }
    }
    if clustering_method == 'kmeans':
        initial_params["num_clusters_min"] = num_clusters_min
        initial_params["num_clusters_max"] = num_clusters_max
    elif clustering_method == 'gmm':
        initial_params["num_clusters_min"] = gmm_n_components_min
        initial_params["num_clusters_max"] = gmm_n_components_max
    elif clustering_method == 'spectral':
        initial_params["num_clusters_min"] = spectral_n_clusters_min
        initial_params["num_clusters_max"] = spectral_n_clusters_max

    with app.app_context():
        # IDEMPOTENCY CHECK
        task_info = get_task_info_from_db(current_task_id)
        if task_info and task_info.get('status') in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]:
            logger.info(f"Main clustering task {current_task_id} is already in a terminal state ('{task_info.get('status')}'). Skipping execution.")
            return {"status": task_info.get('status'), "message": f"Task already in terminal state '{task_info.get('status')}'.", "details": json.loads(task_info.get('details', '{}'))}

        # This dictionary will hold the state and be passed to the logging function.
        _main_task_accumulated_details = {
            "log": [],
            "total_runs": num_clustering_runs,
            "runs_completed": 0,
            "best_score": -1.0, # Use -1.0 as a safe initial value
            "best_result": None,
            "active_jobs": {},
            "elite_solutions": [],
            "last_subset_ids": [],
            "processed_job_ids": set() # *** FIX 1: Add set to track processed jobs ***
        }

        # Helper for logging and updating main task status, using a shared dictionary.
        def _log_and_update(message, progress, details_to_add_or_update=None, task_state=TASK_STATUS_PROGRESS):
            nonlocal _main_task_accumulated_details
            
            logger.info(f"[MainClusteringTask-{current_task_id}] {message}")
            if details_to_add_or_update:
                _main_task_accumulated_details.update(details_to_add_or_update)
            
            _main_task_accumulated_details["status_message"] = message
            
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            _main_task_accumulated_details["log"].append(log_entry)

            # Prepare details for saving (a copy to avoid modifying the original during iteration)
            details_for_db = _main_task_accumulated_details.copy()
            details_for_db.pop('active_jobs', None) # Don't save job objects to DB
            details_for_db.pop('best_result', None) # Don't save the full result object in every progress update
            details_for_db.pop('last_subset_ids', None) # Remove the large list of IDs
            details_for_db.pop('processed_job_ids', None) # Don't save the set of job IDs to DB

            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.save_meta()
            
            save_task_status(current_task_id, "main_clustering", task_state, progress=progress, details=details_for_db)

        try:
            _log_and_update("Initializing clustering process...", 0, task_state=TASK_STATUS_STARTED)

            # --- 1. Data Preparation and Stratification Setup ---
            _log_and_update("Fetching lightweight track data for stratification...", 1)
            db = get_db()
            cur = db.cursor(cursor_factory=DictCursor)
            cur.execute("SELECT item_id, author, mood_vector FROM score WHERE mood_vector IS NOT NULL AND mood_vector != ''")
            lightweight_rows = cur.fetchall()
            cur.close()

            if len(lightweight_rows) < (num_clusters_min or 2):
                raise ValueError(f"Not enough tracks in DB ({len(lightweight_rows)}) for clustering.")

            genre_map = _prepare_genre_map(lightweight_rows)
            target_songs_per_genre = _calculate_target_songs_per_genre(
                genre_map, stratified_sampling_target_percentile_param, min_songs_per_genre_for_stratification_param
            )
            _log_and_update(f"Target songs per genre for stratification: {target_songs_per_genre}", 5)

            # --- 2. Batch Job Orchestration ---
            num_total_batches = (num_clustering_runs + ITERATIONS_PER_BATCH_JOB - 1) // ITERATIONS_PER_BATCH_JOB if ITERATIONS_PER_BATCH_JOB > 0 else 0
            next_batch_to_launch = 0
            batches_completed_count = 0

            # STATE RECOVERY
            child_tasks_from_db = get_child_tasks_from_db(current_task_id)
            if child_tasks_from_db:
                logger.info(f"Found {len(child_tasks_from_db)} existing child tasks. Attempting state recovery.")
                _monitor_and_process_batches(_main_task_accumulated_details, current_task_id, initial_check=True)
                batches_completed_count = len([j for j in child_tasks_from_db if j['status'] in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE]])
                next_batch_to_launch = batches_completed_count
                logger.info(f"Recovery complete. Resuming. Batches completed: {batches_completed_count}, Next batch: {next_batch_to_launch}")

            if not _main_task_accumulated_details["last_subset_ids"]:
                initial_subset_data = _get_stratified_song_subset(genre_map, target_songs_per_genre)
                _main_task_accumulated_details["last_subset_ids"] = [t['item_id'] for t in initial_subset_data]

            while batches_completed_count < num_total_batches:
                if current_job and (current_job.is_stopped or get_task_info_from_db(current_task_id).get('status') == TASK_STATUS_REVOKED):
                    _log_and_update("Task revoked, stopping.", _main_task_accumulated_details['runs_completed'], task_state=TASK_STATUS_REVOKED)
                    return {"status": "REVOKED", "message": "Main clustering task revoked."}

                _monitor_and_process_batches(_main_task_accumulated_details, current_task_id)
                batches_completed_count = len([j for j in get_child_tasks_from_db(current_task_id) if j['status'] in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE]])

                while len(_main_task_accumulated_details["active_jobs"]) < MAX_CONCURRENT_BATCH_JOBS and next_batch_to_launch < num_total_batches:
                    _launch_batch_job(
                        _main_task_accumulated_details, current_task_id, next_batch_to_launch, num_clustering_runs,
                        genre_map, target_songs_per_genre, clustering_method,
                        num_clusters_min, num_clusters_max, dbscan_eps_min, dbscan_eps_max,
                        dbscan_min_samples_min, dbscan_min_samples_max, gmm_n_components_min,
                        gmm_n_components_max, spectral_n_clusters_min, spectral_n_clusters_max,
                        pca_components_min, pca_components_max, max_songs_per_cluster_val,
                        score_weight_diversity_param, score_weight_silhouette_param,
                        score_weight_davies_bouldin_param, score_weight_calinski_harabasz_param,
                        score_weight_purity_param, score_weight_other_feature_diversity_param,
                        score_weight_other_feature_purity_param, top_n_moods_for_clustering_param,
                        enable_clustering_embeddings_param
                    )
                    next_batch_to_launch += 1

                progress = 5 + int(85 * _main_task_accumulated_details["runs_completed"] / num_clustering_runs) if num_clustering_runs > 0 else 5
                _log_and_update(
                    f"Progress: {_main_task_accumulated_details['runs_completed']}/{num_clustering_runs} runs. Active batches: {len(_main_task_accumulated_details['active_jobs'])}. Best score: {_main_task_accumulated_details['best_score']:.2f}",
                    progress
                )
                
                # *** FIX 3: Starvation exit condition ***
                # If all runs are accounted for and no jobs are active, exit the loop.
                # This prevents getting stuck if a batch job fails to update its DB status.
                if _main_task_accumulated_details["runs_completed"] >= num_clustering_runs and len(_main_task_accumulated_details["active_jobs"]) == 0:
                    _log_and_update(f"All runs ({_main_task_accumulated_details['runs_completed']}) are processed and no active batches remain. Forcing loop exit to prevent starvation.", progress)
                    break
                
                time.sleep(3)

            _log_and_update("All batches completed. Finalizing...", 90)

            # --- 3. Finalization and Playlist Creation ---
            if not _main_task_accumulated_details["best_result"]:
                raise ValueError("No valid clustering solution found after all runs.")

            best_result = _main_task_accumulated_details["best_result"]

            # *** NEW STEP: Filter for Top N Most Diverse Playlists ***
            if top_n_playlists_param > 0 and len(best_result.get("named_playlists", {})) > top_n_playlists_param:
                _log_and_update(f"Filtering for Top {top_n_playlists_param} most diverse playlists...", 91)
                best_result = _select_top_n_diverse_playlists(best_result, top_n_playlists_param)
                _main_task_accumulated_details["best_result"] = best_result # Update main dict with filtered result

            _log_and_update(f"Best clustering found with score: {_main_task_accumulated_details['best_score']:.2f}. Creating playlists...", 92)

            final_playlists_with_details = _name_and_prepare_playlists(
                best_result, # Use the potentially filtered result
                ai_model_provider_param, ollama_server_url_param,
                ollama_model_name_param, gemini_api_key_param, gemini_model_name_param,
                enable_clustering_embeddings_param
            )

            _log_and_update("Deleting existing automatic playlists...", 97)
            delete_automatic_playlists()

            _log_and_update(f"Creating {len(final_playlists_with_details)} new playlists...", 98)
            for name, songs_with_details in final_playlists_with_details.items():
                item_ids = [item_id for item_id, _, _ in songs_with_details]
                create_playlist(name, item_ids)

            update_playlist_table(final_playlists_with_details)

            # --- Final Success Reporting ---
            final_message = "Clustering task completed successfully!"
            
            # Add final message to the log before preparing the summary
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {final_message}"
            _main_task_accumulated_details["log"].append(log_entry)
            logger.info(f"[MainClusteringTask-{current_task_id}] {final_message}")

            final_log = _main_task_accumulated_details.get('log', [])
            truncated_log = final_log[-10:]

            # This dictionary is the final, clean state for the DB.
            # It includes running parameters, excludes elite solutions, and has a truncated log.
            final_db_summary = {
                "status_message": final_message,
                "running_parameters": initial_params,
                "best_score": _main_task_accumulated_details["best_score"],
                "best_params": _main_task_accumulated_details["best_result"].get("parameters"),
                "num_playlists_created": len(final_playlists_with_details),
                "log": truncated_log,
                "log_storage_info": f"Log truncated to last {len(truncated_log)} entries. Original length: {len(final_log)}." if len(final_log) > 10 else "Full log."
            }
            
            if current_job:
                current_job.meta['progress'] = 100
                current_job.meta['status_message'] = final_message
                current_job.save_meta()

            # Direct call to save_task_status with the clean details object
            save_task_status(current_task_id, "main_clustering", TASK_STATUS_SUCCESS, progress=100, details=final_db_summary)

            return {"status": "SUCCESS", "message": f"Playlists created. Best score: {_main_task_accumulated_details['best_score']:.2f}"}

        except Exception as e:
            logger.critical("FATAL ERROR in main clustering task", exc_info=True)
            _log_and_update(f"Task failed: {e}", 100, details_to_add_or_update={"error": str(e)}, task_state=TASK_STATUS_FAILURE)
            raise

# --- Internal Helper Functions for run_clustering_task ---

def _prepare_genre_map(lightweight_rows):
    """Creates a map of genre -> list of tracks from raw DB rows."""
    genre_map = defaultdict(list)
    for row in lightweight_rows:
        if row.get('mood_vector'):
            mood_scores = {p.split(':')[0]: float(p.split(':')[1]) for p in row['mood_vector'].split(',') if ':' in p}
            top_genre = max((g for g in STRATIFIED_GENRES if g in mood_scores), key=mood_scores.get, default='__other__')
            genre_map[top_genre].append({'item_id': row['item_id'], 'mood_vector': row['mood_vector']})
    return genre_map

def _calculate_target_songs_per_genre(genre_map, percentile, min_songs):
    """Calculates the target number of songs per genre for stratification."""
    counts = [len(tracks) for g, tracks in genre_map.items() if g in STRATIFIED_GENRES]
    if not counts:
        return min_songs
    target = np.percentile(counts, np.clip(percentile, 0, 100))
    return max(min_songs, int(np.floor(target)))

def _monitor_and_process_batches(state_dict, parent_task_id, initial_check=False):
    """
    Checks status of active jobs, processes results of finished ones.
    This function is corrected to prevent race conditions and infinite loops.
    """
    # ALWAYS get the full list of child tasks from the database.
    # This is the single source of truth and prevents state inconsistencies.
    all_child_tasks = get_child_tasks_from_db(parent_task_id)
    
    # Identify jobs that are still supposedly running according to our database.
    job_ids_to_check = [
        task['task_id'] for task in all_child_tasks 
        if task['status'] not in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]
    ]

    # Also include jobs we know are active in-memory but might not be in the DB yet.
    # This covers the brief moment right after enqueueing.
    for job_id in state_dict["active_jobs"].keys():
        if job_id not in job_ids_to_check:
            job_ids_to_check.append(job_id)

    finished_job_ids = []
    for job_id in job_ids_to_check:
        try:
            job = Job.fetch(job_id, connection=redis_conn)
            if job.is_finished or job.is_failed or job.get_status() == JobStatus.CANCELED:
                finished_job_ids.append(job_id)
        except NoSuchJobError:
            # If the job is not in RQ, it's either finished and cleaned up, or something
            # went wrong. We check our DB again. If the DB says it's not finished,
            # we log a warning, but we treat it as 'finished' to avoid getting stuck.
            task_info = get_task_info_from_db(job_id)
            if not task_info or task_info.get('status') not in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]:
                 logger.warning(f"Job {job_id} not found in RQ, but its status in DB is not terminal. Treating as finished to prevent loop.")
            finished_job_ids.append(job_id)

    for job_id in finished_job_ids:
        # *** FIX 2: Prevent re-processing results ***
        if job_id in state_dict.get("processed_job_ids", set()):
            if job_id in state_dict["active_jobs"]:
                del state_dict["active_jobs"][job_id]
            continue
        
        result = get_job_result_safely(job_id, parent_task_id, "clustering_batch")
        if result:
            state_dict["runs_completed"] += result.get("iterations_completed_in_batch", 0)
            state_dict["last_subset_ids"] = result.get("final_subset_track_ids", state_dict["last_subset_ids"])
            best_from_batch = result.get("best_result_from_batch")
            if best_from_batch:
                current_best_score = best_from_batch.get("fitness_score", -1.0)
                state_dict["elite_solutions"].append({
                    "score": current_best_score,
                    "params": best_from_batch.get("parameters")
                })
                if current_best_score > state_dict["best_score"]:
                    state_dict["best_score"] = current_best_score
                    state_dict["best_result"] = best_from_batch
        
        # Mark as processed and remove from active jobs list
        state_dict.setdefault("processed_job_ids", set()).add(job_id)
        if job_id in state_dict["active_jobs"]:
            del state_dict["active_jobs"][job_id]

    # Prune elite solutions to keep only the best
    state_dict["elite_solutions"].sort(key=lambda x: x["score"], reverse=True)
    state_dict["elite_solutions"] = state_dict["elite_solutions"][:TOP_N_ELITES]


def _launch_batch_job(state_dict, parent_task_id, batch_idx, total_runs, genre_map, target_per_genre, *args):
    """Constructs and enqueues a single batch job."""
    from app import rq_queue_default # Local import to avoid circular dependency issues at top-level

    # Unpack all the parameters passed via *args
    (
        clustering_method,
        num_clusters_min, num_clusters_max, dbscan_eps_min, dbscan_eps_max,
        dbscan_min_samples_min, dbscan_min_samples_max, gmm_n_components_min,
        gmm_n_components_max, spectral_n_clusters_min, spectral_n_clusters_max,
        pca_components_min, pca_components_max, max_songs_per_cluster,
        score_weight_diversity, score_weight_silhouette, score_weight_davies_bouldin,
        score_weight_calinski_harabasz, score_weight_purity,
        score_weight_other_feature_diversity, score_weight_other_feature_purity,
        top_n_moods, enable_embeddings
    ) = args

    batch_job_id = f"{parent_task_id}_batch_{batch_idx}"
    start_run = batch_idx * ITERATIONS_PER_BATCH_JOB
    num_iterations = min(ITERATIONS_PER_BATCH_JOB, total_runs - start_run)

    exploitation_prob = EXPLOITATION_PROBABILITY_CONFIG if start_run >= (total_runs * EXPLOITATION_START_FRACTION) else 0.0

    # Package parameters for the batch task
    job_args = {
        "batch_id_str": f"Batch_{batch_idx}",
        "start_run_idx": start_run,
        "num_iterations_in_batch": num_iterations,
        "genre_to_lightweight_track_data_map_json": json.dumps(genre_map),
        "target_songs_per_genre": target_per_genre,
        "sampling_percentage_change_per_run": SAMPLING_PERCENTAGE_CHANGE_PER_RUN,
        "clustering_method": clustering_method,
        "active_mood_labels_for_batch": MOOD_LABELS[:top_n_moods] if top_n_moods > 0 else MOOD_LABELS,
        "num_clusters_min_max_tuple": (num_clusters_min, num_clusters_max),
        "dbscan_params_ranges_dict": {"eps_min": dbscan_eps_min, "eps_max": dbscan_eps_max, "samples_min": dbscan_min_samples_min, "samples_max": dbscan_min_samples_max},
        "gmm_params_ranges_dict": {"n_components_min": gmm_n_components_min, "n_components_max": gmm_n_components_max},
        "spectral_params_ranges_dict": {"n_clusters_min": spectral_n_clusters_min, "n_clusters_max": spectral_n_clusters_max},
        "pca_params_ranges_dict": {"components_min": pca_components_min, "components_max": pca_components_max},
        "max_songs_per_cluster": max_songs_per_cluster,
        "parent_task_id": parent_task_id,
        "score_weights_dict": {
            "mood_diversity": score_weight_diversity, 
            "silhouette": score_weight_silhouette,
            "davies_bouldin": score_weight_davies_bouldin, 
            "calinski_harabasz": score_weight_calinski_harabasz,
            "mood_purity": score_weight_purity, 
            "other_feature_diversity": score_weight_other_feature_diversity,
            "other_feature_purity": score_weight_other_feature_purity
        },
        "elite_solutions_params_list_json": json.dumps([e["params"] for e in state_dict["elite_solutions"]]),
        "exploitation_probability": exploitation_prob,
        "mutation_config_json": json.dumps({
            "int_abs_delta": MUTATION_INT_ABS_DELTA, "float_abs_delta": MUTATION_FLOAT_ABS_DELTA,
            "coord_mutation_fraction": MUTATION_KMEANS_COORD_FRACTION
        }),
        "initial_subset_track_ids_json": json.dumps(state_dict["last_subset_ids"]),
        "enable_clustering_embeddings_param": enable_embeddings
    }

    new_job = rq_queue_default.enqueue(
        run_clustering_batch_task,
        kwargs=job_args,
        job_id=batch_job_id,
        job_timeout=-1,
        retry=Retry(max=3)
    )
    state_dict["active_jobs"][new_job.id] = new_job
    logger.info(f"Enqueued batch job {new_job.id} for runs {start_run}-{start_run + num_iterations - 1}.")


def _name_and_prepare_playlists(best_result, ai_provider, ollama_url, ollama_model, gemini_key, gemini_model, embeddings_used):
    """
    Uses AI to name playlists and formats them for creation.
    Returns a dictionary mapping final playlist names to lists of song tuples (id, title, author).
    """
    final_playlists = {}
    centroids = best_result.get("playlist_centroids", {})
    named_playlists = best_result.get("named_playlists", {})
    max_songs = best_result.get("parameters", {}).get("max_songs_per_cluster", MAX_SONGS_PER_CLUSTER)

    for original_name, songs in named_playlists.items():
        if not songs:
            continue

        final_name = original_name
        if ai_provider in ["OLLAMA", "GEMINI"]:
            try:
                # Simplified feature extraction for AI prompt
                name_parts = original_name.split('_')
                feature1 = name_parts[0] if len(name_parts) > 0 else "Music"
                feature2 = name_parts[1] if len(name_parts) > 1 else "Vibes"
                feature3 = name_parts[2] if len(name_parts) > 2 else "Collection"
                if embeddings_used:
                    feature1, feature2, feature3 = "Vibe", "Focused", "Collection"

                ai_name = get_ai_playlist_name(
                    ai_provider, ollama_url, ollama_model, gemini_key, gemini_model,
                    creative_prompt_template, feature1, feature2, feature3,
                    [{'title': s_title, 'author': s_author} for _, s_title, s_author in songs],
                    centroids.get(original_name, {})
                )
                if ai_name and "Error" not in ai_name:
                    final_name = ai_name.strip().replace("\n", " ")
            except Exception as e:
                logger.warning(f"AI naming failed for '{original_name}': {e}. Using original name.")

        # Ensure unique names
        temp_name = final_name
        suffix = 1
        while temp_name in final_playlists:
            suffix += 1
            temp_name = f"{final_name} ({suffix})"
        final_name = temp_name

        # Add suffix and handle chunking
        base_name_with_suffix = f"{final_name}_automatic"
        
        # The 'songs' variable is already the list of tuples: [(item_id, title, author), ...]
        if max_songs > 0 and len(songs) > max_songs:
             chunks = [songs[i:i+max_songs] for i in range(0, len(songs), max_songs)]
             for idx, chunk in enumerate(chunks, 1):
                 final_playlists[f"{base_name_with_suffix} ({idx})"] = chunk # Store the chunk of tuples
        else:
            final_playlists[base_name_with_suffix] = songs # Store the list of tuples

    return final_playlists


def _select_top_n_diverse_playlists(best_result, n):
    """
    Selects the N most diverse playlists from a clustering result by weighting
    both distance (diversity) and size (usefulness).
    """
    playlist_to_vector = best_result.get("playlist_to_centroid_vector_map", {})
    original_playlists = best_result.get("named_playlists", {})
    original_centroids = best_result.get("playlist_centroids", {})

    if not playlist_to_vector or n <= 0 or n >= len(playlist_to_vector):
        logger.info(f"Skipping Top-N selection: N={n}, available playlists={len(playlist_to_vector)}. Returning original set.")
        return best_result

    logger.info(f"Starting selection of Top {n} diverse playlists from {len(playlist_to_vector)} candidates.")

    # Convert to lists for easier indexing
    available_names = list(playlist_to_vector.keys())
    available_vectors = np.array(list(playlist_to_vector.values()))

    if available_vectors.shape[0] <= n:
        return best_result

    selected_indices = []
    
    # 1. Start with the largest playlist to anchor the selection
    playlist_sizes = [len(original_playlists.get(name, [])) for name in available_names]
    first_idx = np.argmax(playlist_sizes)
    selected_indices.append(first_idx)

    # Create a boolean mask for available items
    is_available = np.ones(len(available_names), dtype=bool)
    is_available[first_idx] = False
    
    # 2. Iteratively select the playlist with the best combined score of distance and size
    for _ in range(n - 1):
        if not np.any(is_available):
            break # No more playlists to select

        selected_vectors = available_vectors[selected_indices]
        remaining_vectors = available_vectors[is_available]
        
        # --- Calculate Diversity Score (Distance) ---
        dist_matrix = cdist(remaining_vectors, selected_vectors, 'euclidean')
        min_distances = np.min(dist_matrix, axis=1)
        
        # --- Calculate Size Score ---
        original_indices_available = np.where(is_available)[0]
        sizes_available = np.array([len(original_playlists.get(available_names[i], [])) for i in original_indices_available])
        # Use log1p for a smooth curve with diminishing returns for size
        size_scores = np.log1p(sizes_available)

        # --- Normalize and Combine Scores ---
        # Normalize both scores to a 0-1 range to make them comparable
        max_dist = np.max(min_distances)
        normalized_dist_scores = min_distances / max_dist if max_dist > 0 else np.zeros_like(min_distances)

        max_size_score = np.max(size_scores)
        normalized_size_scores = size_scores / max_size_score if max_size_score > 0 else np.zeros_like(size_scores)
        
        # Combine the scores (equal weighting)
        # TEST USING * INSTEAD OF +
        combined_scores = normalized_dist_scores * normalized_size_scores
        
        # Find the playlist that has the maximum combined score
        best_candidate_local_idx = np.argmax(combined_scores)
        
        # Convert this local index back to the original full list index
        best_original_idx = original_indices_available[best_candidate_local_idx]
        
        # Add to selected and mark as unavailable
        selected_indices.append(best_original_idx)
        is_available[best_original_idx] = False

    # 3. Build the new, filtered result
    selected_names = [available_names[i] for i in selected_indices]
    
    filtered_playlists = {name: original_playlists[name] for name in selected_names if name in original_playlists}
    filtered_centroids = {name: original_centroids[name] for name in selected_names if name in original_centroids}
    filtered_vector_map = {name: playlist_to_vector[name] for name in selected_names if name in playlist_to_vector}

    new_result = best_result.copy()
    new_result["named_playlists"] = filtered_playlists
    new_result["playlist_centroids"] = filtered_centroids
    new_result["playlist_to_centroid_vector_map"] = filtered_vector_map
    
    logger.info(f"Selected {len(selected_names)} diverse playlists: {selected_names}")

    return new_result
