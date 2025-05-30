import os
import psycopg2
from psycopg2.extras import DictCursor
from flask import Flask, jsonify, request, render_template, g
from contextlib import closing
import json
import uuid # For generating job IDs if needed directly in API, though tasks handle their own

# RQ imports
from redis import Redis
from rq import Queue, Retry
from rq.job import Job, JobStatus
from rq.exceptions import NoSuchJobError
from rq.command import send_stop_job_command
JobStatus = JobStatus # Make JobStatus directly accessible within the app for tasks to import via `from app import JobStatus`

# Import configuration
from config import JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, HEADERS, TEMP_DIR, \
    REDIS_URL, DATABASE_URL, MAX_DISTANCE, MAX_SONGS_PER_CLUSTER, MAX_SONGS_PER_ARTIST, NUM_RECENT_ALBUMS, \
    CLUSTER_ALGORITHM, NUM_CLUSTERS_MIN, NUM_CLUSTERS_MAX, DBSCAN_EPS_MIN, DBSCAN_EPS_MAX, \
    DBSCAN_MIN_SAMPLES_MIN, DBSCAN_MIN_SAMPLES_MAX, GMM_N_COMPONENTS_MIN, GMM_N_COMPONENTS_MAX, \
    PCA_COMPONENTS_MIN, PCA_COMPONENTS_MAX, CLUSTERING_RUNS, MOOD_LABELS, TOP_N_MOODS

# --- Flask App Setup ---
app = Flask(__name__)

# --- RQ Setup ---
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
redis_conn = Redis.from_url(
    REDIS_URL,
    socket_connect_timeout=15,  # seconds to wait for connection
    socket_timeout=15           # seconds for read/write operations
)
rq_queue = Queue(connection=redis_conn)  # Default queue

# --- Database Setup (PostgreSQL) ---
# DATABASE_URL is now imported from config.py
MAX_LOG_ENTRIES_STORED = 10 # Max number of recent log entries to store in the database per task

# --- Status Constants ---
TASK_STATUS_PENDING = "PENDING"
TASK_STATUS_STARTED = "STARTED"
TASK_STATUS_PROGRESS = "PROGRESS" # For more granular updates within a task
TASK_STATUS_SUCCESS = "SUCCESS"
TASK_STATUS_FAILURE = "FAILURE"
TASK_STATUS_REVOKED = "REVOKED"
# RQ JobStatus (JobStatus.FINISHED, JobStatus.FAILED etc.) are used for RQ's direct state

def get_db():
    if 'db' not in g:
        try:
            g.db = psycopg2.connect(DATABASE_URL, connect_timeout=15) # 15-second connection timeout
        except psycopg2.OperationalError as e:
            app.logger.error(f"Failed to connect to database: {e}") # Use app.logger for Flask context
            raise # Re-raise to ensure the operation that needed the DB fails clearly
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
            parent_task_id TEXT,
            task_type TEXT NOT NULL,
            sub_type_identifier TEXT,
            status TEXT,
            progress INTEGER DEFAULT 0,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
    cur.close()

with app.app_context():
    init_db()

# --- DB Cleanup Utility ---
def clean_successful_task_details_on_new_start():
    """
    Cleans the 'details' field (specifically 'log' and 'log_storage_info')
    for all tasks in the database that are marked as SUCCESS.
    This is typically called when a new main task starts.
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    app.logger.info("Starting cleanup of details for previously successful tasks.")
    try:
        cur.execute("SELECT task_id, details FROM task_status WHERE status = %s", (TASK_STATUS_SUCCESS,))
        tasks_to_clean = cur.fetchall()
        
        cleaned_count = 0
        for task_row in tasks_to_clean:
            task_id = task_row['task_id']
            details_json = task_row['details']
            if details_json:
                try:
                    details_dict = json.loads(details_json)
                    needs_update = False
                    if 'log' in details_dict and (not isinstance(details_dict['log'], list) or len(details_dict['log']) > 1 or (len(details_dict['log']) == 1 and "cleaned" not in details_dict['log'][0].lower() and "success" not in details_dict['log'][0].lower() )):
                        # Replace log with a summary or remove it, preserving other info
                        summary_message = details_dict.get("status_message", "Task completed successfully.")
                        details_dict['log'] = [f"Log cleaned. Original status: {summary_message}"]
                        needs_update = True
                    if 'log_storage_info' in details_dict:
                        del details_dict['log_storage_info']
                        needs_update = True
                    
                    if needs_update:
                        updated_details_json = json.dumps(details_dict)
                        with db.cursor() as update_cur:
                            update_cur.execute("UPDATE task_status SET details = %s WHERE task_id = %s", (updated_details_json, task_id))
                        cleaned_count += 1
                except json.JSONDecodeError:
                    app.logger.warning(f"Could not parse details JSON for successful task {task_id} during cleanup.")
                except Exception as e_clean_item:
                    app.logger.error(f"Error cleaning details for successful task {task_id}: {e_clean_item}")
        
        if cleaned_count > 0:
            db.commit()
            app.logger.info(f"Cleaned details for {cleaned_count} previously successful tasks.")
        else:
            app.logger.info("No previously successful tasks required details cleaning.")
    except Exception as e_main_clean:
        db.rollback() # Rollback in case of error during the main query or commit
        app.logger.error(f"Error during the task details cleanup process: {e_main_clean}")
    finally:
        cur.close()

# --- DB Utility Functions (used by tasks.py and API) ---
def save_task_status(task_id, task_type, status=TASK_STATUS_PENDING, parent_task_id=None, sub_type_identifier=None, progress=0, details=None):
    db = get_db()
    cur = db.cursor()

    if details is not None and isinstance(details, dict):
        # Log truncation for non-SUCCESS states; 'log' should be a list from tasks.py helpers.
        if status != TASK_STATUS_SUCCESS and 'log' in details and isinstance(details['log'], list):
            log_list = details['log']
            if len(log_list) > MAX_LOG_ENTRIES_STORED:
                original_log_length = len(log_list)
                details['log'] = log_list[-MAX_LOG_ENTRIES_STORED:]
                details['log_storage_info'] = f"Log in DB truncated to last {MAX_LOG_ENTRIES_STORED} entries. Original length: {original_log_length}."
            else:
                # If log is not truncated, ensure no old log_storage_info persists
                details.pop('log_storage_info', None)
        elif status == TASK_STATUS_SUCCESS:
            # For successful tasks, 'log' should be minimal (set by task). Ensure no log_storage_info.
            details.pop('log_storage_info', None)
            if 'log' not in details or not isinstance(details.get('log'), list) or not details.get('log'):
                details['log'] = ["Task completed successfully."] # Default minimal log

    details_json = json.dumps(details) if details is not None else None
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

def track_exists(item_id): # Removed db_path
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT item_id FROM score WHERE item_id = %s", (item_id,))
    row = cur.fetchone()
    cur.close()
    return row is not None

def save_track_analysis(item_id, title, author, tempo, key, scale, moods): # Removed db_path
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

def get_all_tracks(): # Removed db_path
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT item_id, title, author, tempo, key, scale, mood_vector FROM score")
    rows = cur.fetchall() # Returns list of DictRow
    cur.close()
    return rows

def update_playlist_table(playlists): # Removed db_path
    conn = get_db()
    cur = conn.cursor()
    try:
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

# --- Import Task Functions ---
# Imports are moved into the respective endpoint functions to avoid circular dependencies
# from tasks import run_analysis_task, run_clustering_task
# analyze_album_task and run_single_clustering_iteration_task are called by other tasks, not directly by API

# --- API Endpoints ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analysis/start', methods=['POST'])
def start_analysis_endpoint():
    # Import task function here to break circular dependency
    from tasks import run_analysis_task

    data = request.json or {}
    jellyfin_url = data.get('jellyfin_url', JELLYFIN_URL)
    jellyfin_user_id = data.get('jellyfin_user_id', JELLYFIN_USER_ID)
    jellyfin_token = data.get('jellyfin_token', JELLYFIN_TOKEN)
    num_recent_albums = int(data.get('num_recent_albums', NUM_RECENT_ALBUMS))
    top_n_moods = int(data.get('top_n_moods', TOP_N_MOODS))

    job_id = str(uuid.uuid4())
    
    # Clean up details of previously successful tasks before starting a new one
    clean_successful_task_details_on_new_start()
    save_task_status(job_id, "main_analysis", TASK_STATUS_PENDING, details={"message": "Task enqueued."})

    job = rq_queue.enqueue(
        run_analysis_task,
        args=(jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods),
        job_id=job_id,
        description="Main Music Analysis", # No timeout
        retry=Retry(max=1), # Optional: retry once if fails
        job_timeout=-1 
    )
    return jsonify({"task_id": job.id, "task_type": "main_analysis", "status": job.get_status()}), 202

@app.route('/api/clustering/start', methods=['POST'])
def start_clustering_endpoint():
    # Import task function here to break circular dependency
    from tasks import run_clustering_task

    data = request.json
    clustering_method = data.get('clustering_method', CLUSTER_ALGORITHM)
    num_clusters_min_val = int(data.get('num_clusters_min', NUM_CLUSTERS_MIN))
    num_clusters_max_val = int(data.get('num_clusters_max', NUM_CLUSTERS_MAX))
    dbscan_eps_min_val = float(data.get('dbscan_eps_min', DBSCAN_EPS_MIN))
    dbscan_eps_max_val = float(data.get('dbscan_eps_max', DBSCAN_EPS_MAX))
    dbscan_min_samples_min_val = int(data.get('dbscan_min_samples_min', DBSCAN_MIN_SAMPLES_MIN))
    dbscan_min_samples_max_val = int(data.get('dbscan_min_samples_max', DBSCAN_MIN_SAMPLES_MAX))
    gmm_n_components_min_val = int(data.get('gmm_n_components_min', GMM_N_COMPONENTS_MIN))
    gmm_n_components_max_val = int(data.get('gmm_n_components_max', GMM_N_COMPONENTS_MAX))
    pca_components_min_val = int(data.get('pca_components_min', PCA_COMPONENTS_MIN))
    pca_components_max_val = int(data.get('pca_components_max', PCA_COMPONENTS_MAX))
    num_clustering_runs_val = int(data.get('clustering_runs', CLUSTERING_RUNS))
    max_songs_per_cluster_val = int(data.get('max_songs_per_cluster', MAX_SONGS_PER_CLUSTER))

    job_id = str(uuid.uuid4())

    # Clean up details of previously successful tasks before starting a new one
    clean_successful_task_details_on_new_start()
    save_task_status(job_id, "main_clustering", TASK_STATUS_PENDING, details={"message": "Task enqueued."})

    job = rq_queue.enqueue(
        run_clustering_task,
        args=(
            clustering_method, num_clusters_min_val, num_clusters_max_val,
            dbscan_eps_min_val, dbscan_eps_max_val, dbscan_min_samples_min_val, dbscan_min_samples_max_val,
            pca_components_min_val, pca_components_max_val, num_clustering_runs_val,
            max_songs_per_cluster_val, gmm_n_components_min_val, gmm_n_components_max_val
        ),
        job_id=job_id,
        description="Main Music Clustering", # No timeout
        retry=Retry(max=1), # Optional
        job_timeout=-1  
    )
    return jsonify({"task_id": job.id, "task_type": "main_clustering", "status": job.get_status()}), 202

@app.route('/api/status/<task_id>', methods=['GET'])
def get_task_status_endpoint(task_id):
    response = {'task_id': task_id, 'state': 'UNKNOWN', 'status_message': 'Task ID not found in RQ or DB.', 'progress': 0, 'details': {}, 'task_type_from_db': None}
    try:
        job = Job.fetch(task_id, connection=redis_conn)
        response['state'] = job.get_status() # e.g., queued, started, finished, failed
        response['status_message'] = job.meta.get('status_message', response['state'])
        response['progress'] = job.meta.get('progress', 0)
        response['details'] = job.meta.get('details', {})
        if job.is_failed:
            response['details']['error_message'] = job.exc_info if job.exc_info else "Job failed without error info."
            response['status_message'] = "FAILED"
        elif job.is_finished:
             response['status_message'] = "SUCCESS" # RQ uses 'finished' for success
             response['progress'] = 100
        elif job.is_canceled:
            response['status_message'] = "CANCELED"
            response['progress'] = 100

    except NoSuchJobError:
        # If not in RQ, it might have been cleared or never existed. Check DB.
        pass # Will fall through to DB check

    # Augment with DB data, DB is source of truth for persisted details
    db_task_info = get_task_info_from_db(task_id)
    if db_task_info:
        response['task_type_from_db'] = db_task_info.get('task_type')
        # If RQ state is more final (e.g. failed/finished), prefer that, else use DB
        if response['state'] not in [JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED]:
            response['state'] = db_task_info.get('status', response['state']) # Use DB status if RQ is still active
        
        response['progress'] = db_task_info.get('progress', response['progress'])
        db_details = json.loads(db_task_info.get('details')) if db_task_info.get('details') else {}
        # Merge details: RQ meta (live) can override DB details (persisted)
        response['details'] = {**db_details, **response['details']}
        
        # If task is marked REVOKED in DB, this is the most accurate status for cancellation
        if db_task_info.get('status') == TASK_STATUS_REVOKED:
            response['state'] = 'REVOKED'
            response['status_message'] = 'Task revoked.'
            response['progress'] = 100
    elif response['state'] == 'UNKNOWN': # Not in RQ and not in DB
        return jsonify(response), 404

    return jsonify(response)

def cancel_job_and_children_recursive(job_id, task_type_from_db=None):
    """Helper to cancel a job and its children based on DB records."""
    cancelled_count = 0
    
    # First, determine the task_type for the current job_id
    db_task_info = get_task_info_from_db(job_id)
    current_task_type = db_task_info.get('task_type') if db_task_info else task_type_from_db

    if not current_task_type:
        print(f"Warning: Could not determine task_type for job {job_id}. Cannot reliably mark as REVOKED in DB or cancel children.")
        # Try a best-effort RQ cancel if job_id is known, but DB update for this job is skipped.
        try:
            job_rq = Job.fetch(job_id, connection=redis_conn)
            if not job_rq.is_finished and not job_rq.is_failed and not job_rq.is_canceled:
                send_stop_job_command(redis_conn, job_id)
                cancelled_count += 1 # Count this as an action taken
                print(f"Job {job_id} (task_type unknown) stop command sent to RQ.")
            # else: Job already in a final state or not running
        except NoSuchJobError:
            pass # Job not in RQ, nothing to do there for this ID
        return cancelled_count

    # If current_task_type is known, proceed with RQ cancellation attempt
    action_taken_in_rq = False
    try:
        job_rq = Job.fetch(job_id, connection=redis_conn)
        current_rq_status = job_rq.get_status()
        print(f"Job {job_id} (type: {current_task_type}) found in RQ with status: {current_rq_status}")

        if job_rq.is_started:
            print(f"  Job {job_id} is STARTED. Attempting to send stop command.")
            try:
                send_stop_job_command(redis_conn, job_id) # This will likely move it to 'failed' in RQ
                action_taken_in_rq = True
                print(f"    Stop command sent for job {job_id}.")
            except InvalidJobOperation:
                print(f"    Job {job_id} was in 'started' state but became non-executable for stop command (InvalidJobOperation). Will mark as REVOKED in DB.")
            except Exception as e_stop_cmd: # Catch other potential errors from send_stop_job_command
                print(f"    Error sending stop command for job {job_id}: {e_stop_cmd}")
        elif not (job_rq.is_finished or job_rq.is_failed or job_rq.is_canceled):
            # If it's not started, and not in a terminal state (e.g., queued, deferred)
            print(f"  Job {job_id} is {current_rq_status}. Attempting to cancel via job.cancel().")
            job_rq.cancel() # This moves it to 'canceled' status in RQ
            action_taken_in_rq = True
        else:
            print(f"  Job {job_id} is already in a terminal RQ state: {current_rq_status}. No RQ action needed.")
            
    except NoSuchJobError:
        print(f"Job {job_id} (type: {current_task_type}) not found in RQ. Will mark as REVOKED in DB.")
    except Exception as e_rq_interaction:
        print(f"Warning: Error interacting with RQ for job {job_id} (type: {current_task_type}): {e_rq_interaction}")

    if action_taken_in_rq:
        cancelled_count += 1
        
    # Always mark as REVOKED in DB for the current job if its task_type is known
    save_task_status(job_id, current_task_type, TASK_STATUS_REVOKED, progress=100, details={"message": "Task cancellation processed by API."})

    # Attempt to cancel children based on DB parent_task_id
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    
    # Define terminal statuses for the query
    terminal_statuses_tuple = (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED, 
                               JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED)

    # Fetch children that are not already in a terminal state
    cur.execute("""
        SELECT task_id, task_type FROM task_status 
        WHERE parent_task_id = %s 
        AND status NOT IN %s
    """, (job_id, terminal_statuses_tuple))
    children_tasks = cur.fetchall()
    cur.close()

    for child_task_row in children_tasks:
        child_job_id = child_task_row['task_id']
        child_task_type = child_task_row['task_type'] # Child's own type
        print(f"Recursively cancelling child job: {child_job_id} of type {child_task_type}")
        # The count from recursive calls will be added
        cancelled_count += cancel_job_and_children_recursive(child_job_id, child_task_type) 
    
    return cancelled_count

@app.route('/api/cancel/<task_id>', methods=['POST'])
def cancel_task_endpoint(task_id):
    db_task_info = get_task_info_from_db(task_id)
    if not db_task_info:
        return jsonify({"message": f"Task {task_id} not found in database.", "task_id": task_id}), 404

    cancelled_count = cancel_job_and_children_recursive(task_id, db_task_info.get('task_type')) # Task type from DB

    if cancelled_count > 0:
        return jsonify({"message": f"Task {task_id} and its children cancellation initiated. {cancelled_count} total jobs affected.", "task_id": task_id, "cancelled_jobs_count": cancelled_count}), 200
    return jsonify({"message": "Task could not be cancelled (e.g., already completed or not found in active state).", "task_id": task_id}), 400


@app.route('/api/cancel_all/<task_type_prefix>', methods=['POST'])
def cancel_all_tasks_by_type_endpoint(task_type_prefix):
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    # Exclude terminal statuses
    terminal_statuses = (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED, 
                         JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED) # JobStatus for completeness if used directly
    cur.execute("SELECT task_id, task_type FROM task_status WHERE task_type = %s AND status NOT IN %s", (task_type_prefix, terminal_statuses))
    tasks_to_cancel = cur.fetchall()
    cur.close()

    total_cancelled_jobs = 0
    cancelled_main_task_ids = []
    for task_row in tasks_to_cancel:  # task_row has 'task_id' and 'task_type'
        cancelled_jobs_for_this_main_task = cancel_job_and_children_recursive(task_row['task_id'], task_row['task_type'])  # Use task type from DB for consistency
        if cancelled_jobs_for_this_main_task > 0:
           total_cancelled_jobs += cancelled_jobs_for_this_main_task
           cancelled_main_task_ids.append(task_row['task_id'])

    if total_cancelled_jobs > 0:
        return jsonify({"message": f"Cancellation initiated for {len(cancelled_main_task_ids)} main tasks of type '{task_type_prefix}' and their children. Total jobs affected: {total_cancelled_jobs}.", "cancelled_main_tasks": cancelled_main_task_ids}), 200
    return jsonify({"message": f"No active tasks of type '{task_type_prefix}' found to cancel."}), 404

@app.route('/api/last_task', methods=['GET'])
def get_last_overall_task_status_endpoint():
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT task_id, task_type, status, progress, details FROM task_status WHERE parent_task_id IS NULL ORDER BY timestamp DESC LIMIT 1")
    last_task_row = cur.fetchone()
    cur.close()

    if last_task_row:
        last_task_data = dict(last_task_row)
        if last_task_data.get('details'):
            try: last_task_data['details'] = json.loads(last_task_data['details'])
            except json.JSONDecodeError: pass # Keep as string if not valid JSON
        return jsonify(last_task_data), 200
    return jsonify({"task_id": None, "task_type": None, "status": "NO_PREVIOUS_MAIN_TASK", "details": {"log": ["No previous main task found."] }}), 200

@app.route('/api/active_tasks', methods=['GET'])
def get_active_tasks_endpoint():
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    non_terminal_statuses = (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED,
                             JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED) # JobStatus for completeness
    cur.execute("""
        SELECT task_id, parent_task_id, task_type, sub_type_identifier, status, progress, details, timestamp
        FROM task_status
        WHERE parent_task_id IS NULL AND status NOT IN ('SUCCESS', 'FAILURE', 'REVOKED', 'FINISHED', 'FAILED', 'CANCELED')
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    active_main_task_row = cur.fetchone() # This query was already good, just ensuring constants are used if it were checking status
    cur.close()

    if active_main_task_row:
        task_item = dict(active_main_task_row)
        if task_item.get('details'):
            try:
                task_item['details'] = json.loads(task_item['details'])
            except json.JSONDecodeError:
                task_item['details'] = {"raw_details": task_item['details'], "error": "Failed to parse details JSON."}
        return jsonify(task_item), 200
    return jsonify({}), 200 # Return empty object if no active main task

@app.route('/api/config', methods=['GET'])
def get_config_endpoint():
    return jsonify({
        "jellyfin_url": JELLYFIN_URL, "jellyfin_user_id": JELLYFIN_USER_ID, "jellyfin_token": JELLYFIN_TOKEN,
        "num_recent_albums": NUM_RECENT_ALBUMS, "max_distance": MAX_DISTANCE,
        "max_songs_per_cluster": MAX_SONGS_PER_CLUSTER, "max_songs_per_artist": MAX_SONGS_PER_ARTIST,
        "cluster_algorithm": CLUSTER_ALGORITHM, "num_clusters_min": NUM_CLUSTERS_MIN, "num_clusters_max": NUM_CLUSTERS_MAX,
        "dbscan_eps_min": DBSCAN_EPS_MIN, "dbscan_eps_max": DBSCAN_EPS_MAX,
        "dbscan_min_samples_min": DBSCAN_MIN_SAMPLES_MIN, "dbscan_min_samples_max": DBSCAN_MIN_SAMPLES_MAX,
        "gmm_n_components_min": GMM_N_COMPONENTS_MIN, "gmm_n_components_max": GMM_N_COMPONENTS_MAX,
        "pca_components_min": PCA_COMPONENTS_MIN, "pca_components_max": PCA_COMPONENTS_MAX,
        "top_n_moods": TOP_N_MOODS, "mood_labels": MOOD_LABELS, "clustering_runs": CLUSTERING_RUNS,
    })

@app.route('/api/playlists', methods=['GET'])
def get_playlists_endpoint():
    from collections import defaultdict # Local import if not used elsewhere globally
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT playlist_name, item_id, title, author FROM playlist ORDER BY playlist_name, title")
    rows = cur.fetchall()
    cur.close()
    playlists_data = defaultdict(list)
    for row in rows:
        playlists_data[row['playlist_name']].append({"item_id": row['item_id'], "title": row['title'], "author": row['author']})
    return jsonify(dict(playlists_data)), 200

if __name__ == '__main__':
    os.makedirs(TEMP_DIR, exist_ok=True)
    # The app context for init_db is handled at the top level of the script.
    app.run(debug=True, host='0.0.0.0', port=8000)
