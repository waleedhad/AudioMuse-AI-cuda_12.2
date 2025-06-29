import os
import psycopg2
from psycopg2.extras import DictCursor
from flask import Flask, jsonify, request, render_template, g
from contextlib import closing
import json
import logging
import numpy as np # Ensure numpy is imported
import uuid # For generating job IDs if needed directly in API, though tasks handle their own

# RQ imports

logger = logging.getLogger(__name__)

# Configure basic logging for the entire application
logging.basicConfig(
    level=logging.INFO, # Set the default logging level (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format='[%(levelname)s]-[%(asctime)s]-%(message)s', # Custom format string
    datefmt='%d-%m-%Y %H-%M-%S' # Custom date/time format
)

from redis import Redis
from rq import Queue, Retry
from rq.job import Job, JobStatus
from rq.exceptions import NoSuchJobError, InvalidJobOperation
from rq.command import send_stop_job_command
JobStatus = JobStatus # Make JobStatus directly accessible within the app for tasks to import via `from app import JobStatus`

# Swagger imports
from flasgger import Swagger, swag_from

# Import configuration
from config import JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, HEADERS, TEMP_DIR, \
    REDIS_URL, DATABASE_URL, MAX_DISTANCE, MAX_SONGS_PER_CLUSTER, MAX_SONGS_PER_ARTIST, NUM_RECENT_ALBUMS, \
    SCORE_WEIGHT_DIVERSITY, SCORE_WEIGHT_SILHOUETTE, SCORE_WEIGHT_DAVIES_BOULDIN, SCORE_WEIGHT_CALINSKI_HARABASZ, \
    SCORE_WEIGHT_PURITY, SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY, SCORE_WEIGHT_OTHER_FEATURE_PURITY, \
    MIN_SONGS_PER_GENRE_FOR_STRATIFICATION, STRATIFIED_SAMPLING_TARGET_PERCENTILE, \
    CLUSTER_ALGORITHM, NUM_CLUSTERS_MIN, NUM_CLUSTERS_MAX, DBSCAN_EPS_MIN, DBSCAN_EPS_MAX, GMM_COVARIANCE_TYPE, \
    DBSCAN_MIN_SAMPLES_MIN, DBSCAN_MIN_SAMPLES_MAX, GMM_N_COMPONENTS_MIN, GMM_N_COMPONENTS_MAX, ENABLE_CLUSTERING_EMBEDDINGS, \
    PCA_COMPONENTS_MIN, PCA_COMPONENTS_MAX, CLUSTERING_RUNS, MOOD_LABELS, TOP_N_MOODS, \
    AI_MODEL_PROVIDER, OLLAMA_SERVER_URL, OLLAMA_MODEL_NAME, GEMINI_API_KEY, GEMINI_MODEL_NAME

logger = logging.getLogger(__name__)

# --- Flask App Setup ---
app = Flask(__name__)

# --- Swagger Setup ---
app.config['SWAGGER'] = {
    'title': 'AudioMuse-AI API',
    'uiversion': 3,
    'openapi': '3.0.0'
}
swagger = Swagger(app)

# --- RQ Setup ---
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
redis_conn = Redis.from_url(
    REDIS_URL,
    socket_connect_timeout=15,  # seconds to wait for connection
    socket_timeout=15           # seconds for read/write operations
)
rq_queue_high = Queue('high', connection=redis_conn) # High priority for main tasks
rq_queue_default = Queue('default', connection=redis_conn) # Default queue for sub-tasks

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
    # Check if the 'energy' column exists and add it if not
    cur.execute("""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'score' AND column_name = 'energy'
        )
    """)
    column_exists_energy = cur.fetchone()[0]
    if not column_exists_energy:
        app.logger.info("Adding 'energy' column to the 'score' table.")
        cur.execute("ALTER TABLE score ADD COLUMN energy REAL")
    # Check if the 'other_features' column exists and add it if not
    cur.execute("""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'score' AND column_name = 'other_features'
        )
    """)
    column_exists = cur.fetchone()[0]
    if not column_exists:
        app.logger.info("Adding 'other_features' column to the 'score' table.")
        cur.execute("ALTER TABLE score ADD COLUMN other_features TEXT")
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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embedding (
            item_id TEXT PRIMARY KEY,
            embedding_vector JSONB,
            FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE
        )
    """)
    app.logger.info("Database tables checked/created successfully.")
    db.commit()
    cur.close()

with app.app_context():
    init_db()

# --- Import and Register Blueprints ---
# Import here to avoid circular dependencies during initial module loading
from app_chat import chat_bp

app.register_blueprint(chat_bp, url_prefix='/chat') # All routes in chat_bp will be prefixed with /chat


# --- DB Cleanup Utility ---
def clean_successful_task_details_on_new_start():
    """
    Cleans the 'details' field (specifically 'log' and 'log_storage_info')
    for all tasks in the database that are marked as SUCCESS.
    This is typically called when a new main task starts.
    This function will now change the status of these tasks to REVOKED
    and update their details to a minimal archival message.
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    app.logger.info("Starting archival of previously successful tasks (setting status to REVOKED and pruning details).")
    try:
        # Select tasks that are currently marked as SUCCESS
        cur.execute("SELECT task_id, details FROM task_status WHERE status = %s", (TASK_STATUS_SUCCESS,))
        tasks_to_archive = cur.fetchall()

        archived_count = 0
        for task_row in tasks_to_archive:
            task_id = task_row['task_id']
            original_details_json = task_row['details']
            original_status_message = "Task completed successfully." # Default

            if original_details_json:
                try:
                    original_details_dict = json.loads(original_details_json)
                    original_status_message = original_details_dict.get("status_message", original_status_message)
                except json.JSONDecodeError:
                    app.logger.warning(f"Could not parse original details JSON for task {task_id} during archival. Using default status message.")

            # New minimal details for the archived task
            archived_details = {
                "log": [f"[Archived] Task was previously successful. Original summary: {original_status_message}"],
                "original_status_before_archival": TASK_STATUS_SUCCESS, # Keep a record of its original success
                "archival_reason": "New main task started, old successful task archived."
            }
            archived_details_json = json.dumps(archived_details)

            # Update status to REVOKED, set new minimal details, progress to 100, and update timestamp
            with db.cursor() as update_cur:
                update_cur.execute(
                    "UPDATE task_status SET status = %s, details = %s, progress = 100, timestamp = NOW() WHERE task_id = %s AND status = %s",
                    (TASK_STATUS_REVOKED, archived_details_json, task_id, TASK_STATUS_SUCCESS) # Ensure we only update tasks that are still SUCCESS
                )
            archived_count += 1

        if archived_count > 0:
            db.commit()
            app.logger.info(f"Archived (set status to REVOKED and pruned details for) {archived_count} previously successful tasks.")
        else:
            app.logger.info("No previously successful tasks found to archive.")
    except Exception as e_main_clean:
        db.rollback() # Rollback in case of error during the main query or commit
        app.logger.error(f"Error during the task archival process: {e_main_clean}")
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

def get_child_tasks_from_db(parent_task_id):
    """Fetches all child tasks for a given parent_task_id from the database."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    # Select the task_id which is the job_id, and other necessary fields
    cur.execute("SELECT task_id, status, sub_type_identifier FROM task_status WHERE parent_task_id = %s", (parent_task_id,))
    tasks = cur.fetchall()
    cur.close()
    # DictCursor returns a list of dictionary-like objects, convert to plain dicts
    return [dict(row) for row in tasks]

def track_exists(item_id):
    """
    Checks if a track exists in the database AND has been analyzed for key features.
    in both the 'score' and 'embedding' tables.
    Returns True if:
    1. The track exists in 'score' table and 'other_features', 'energy', 'mood_vector', and 'tempo' are populated.
    2. The track exists in the 'embedding' table.
    Returns False otherwise, indicating a re-analysis is needed.
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.item_id
        FROM score s
        JOIN embedding e ON s.item_id = e.item_id
        WHERE s.item_id = %s
          AND s.other_features IS NOT NULL AND s.other_features != ''
          AND s.energy IS NOT NULL
          AND s.mood_vector IS NOT NULL AND s.mood_vector != ''
          AND s.tempo IS NOT NULL
    """, (item_id,))
    row = cur.fetchone()
    cur.close()
    return row is not None

def save_track_analysis(item_id, title, author, tempo, key, scale, moods, energy=None, other_features=None): # Added energy and other_features
    mood_str = ','.join(f"{k}:{v:.3f}" for k, v in moods.items())
    conn = get_db()
    cur = conn.cursor()
    try:
        # Use ON CONFLICT DO UPDATE to ensure existing records are updated
        cur.execute("""
            INSERT INTO score (item_id, title, author, tempo, key, scale, mood_vector, energy, other_features)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (item_id) DO UPDATE SET
                title = EXCLUDED.title,
                author = EXCLUDED.author,
                tempo = EXCLUDED.tempo,
                key = EXCLUDED.key,
                scale = EXCLUDED.scale,
                mood_vector = EXCLUDED.mood_vector,
                energy = EXCLUDED.energy,
                other_features = EXCLUDED.other_features
        """, (item_id, title, author, tempo, key, scale, mood_str, energy, other_features))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Error saving track analysis for %s: %s", item_id, e)
    finally:
        cur.close()

def save_track_embedding(item_id, embedding_vector):
    """Saves or updates the embedding vector for a track."""
    if embedding_vector is None: # Should not happen if analysis is successful
        logger.warning("Embedding vector for %s is None. Skipping save.", item_id)
        return

    conn = get_db()
    cur = conn.cursor()
    try:
        # Convert numpy array to list for JSON serialization
        embedding_list = embedding_vector.tolist() if isinstance(embedding_vector, np.ndarray) else embedding_vector
        embedding_json = json.dumps(embedding_list)
        cur.execute("""
            INSERT INTO embedding (item_id, embedding_vector) VALUES (%s, %s)
            ON CONFLICT (item_id) DO UPDATE SET embedding_vector = EXCLUDED.embedding_vector
        """, (item_id, embedding_json))
        conn.commit()
    except Exception as e:
        logger.error("Error saving track embedding for %s: %s", item_id, e)
    finally:
        cur.close()

def get_all_tracks(): # Removed db_path
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    # Join with embedding table to get embedding_vector
    cur.execute("""
        SELECT s.item_id, s.title, s.author, s.tempo, s.key, s.scale, s.mood_vector, s.energy, s.other_features, e.embedding_vector
        FROM score s
        LEFT JOIN embedding e ON s.item_id = e.item_id
    """)
    rows = cur.fetchall() # Returns list of DictRow
    cur.close()
    return rows

def get_tracks_by_ids(item_ids_list):
    """Fetches full track data (including embeddings) for a specific list of item_ids."""
    if not item_ids_list:
        return []
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    # Using a placeholder for a list of IDs
    # psycopg2 can convert a list to a tuple for the IN operator
    query = """
        SELECT s.item_id, s.title, s.author, s.tempo, s.key, s.scale, s.mood_vector, s.energy, s.other_features, e.embedding_vector
        FROM score s
        LEFT JOIN embedding e ON s.item_id = e.item_id
        WHERE s.item_id IN %s
    """
    cur.execute(query, (tuple(item_ids_list),)) # Pass list as a tuple for IN clause
    rows = cur.fetchall()
    cur.close()
    # app.logger.debug(f"Fetched {len(rows)} tracks for {len(item_ids_list)} IDs.")
    return rows

def get_score_data_by_ids(item_ids_list):
    """Fetches only score-related data (excluding embeddings) for a specific list of item_ids."""
    if not item_ids_list:
        return []
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    query = """
        SELECT s.item_id, s.title, s.author, s.tempo, s.key, s.scale, s.mood_vector, s.energy, s.other_features
        FROM score s
        WHERE s.item_id IN %s
    """
    try:
        cur.execute(query, (tuple(item_ids_list),))
        rows = cur.fetchall()
    except Exception as e:
        app.logger.error(f"Error fetching score data by IDs: {e}")
        rows = [] # Return empty list on error
    finally:
        cur.close()
    return rows


def update_playlist_table(playlists): # Removed db_path
    conn = get_db()
    cur = conn.cursor()
    try:
        # Clear all previous conceptual playlists to reflect only the current run.
        cur.execute("DELETE FROM playlist")
        for name, cluster in playlists.items():
            for item_id, title, author in cluster:
                cur.execute("INSERT INTO playlist (playlist_name, item_id, title, author) VALUES (%s, %s, %s, %s) ON CONFLICT (playlist_name, item_id) DO NOTHING", (name, item_id, title, author))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Error updating playlist table: %s", e)
    finally:
        cur.close()

# --- Import Task Functions ---
# Imports are moved into the respective endpoint functions to avoid circular dependencies
# from tasks import run_analysis_task, run_clustering_task
# analyze_album_task and run_single_clustering_iteration_task are called by other tasks, not directly by API

# --- API Endpoints ---

@app.route('/')
def index():
    """
    Serve the main HTML page.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the main page.
        content:
          text/html:
            schema:
              type: string
    """
    return render_template('index.html')

@app.route('/api/analysis/start', methods=['POST'])
def start_analysis_endpoint():
    """
    Start the music analysis process for recent albums.
    This endpoint enqueues a main analysis task.
    Note: Starting a new analysis task will archive previously successful tasks by setting their status to REVOKED.
    ---
    tags:
      - Analysis
    requestBody:
      description: Configuration for the analysis task.
      required: false
      content:
        application/json:
          schema:
            type: object
            properties:
              jellyfin_url:
                type: string
                description: URL of the Jellyfin server.
                default: "Configured JELLYFIN_URL"
              jellyfin_user_id:
                type: string
                description: Jellyfin User ID.
                default: "Configured JELLYFIN_USER_ID"
              jellyfin_token:
                type: string
                description: Jellyfin API Token.
                default: "Configured JELLYFIN_TOKEN"
              num_recent_albums:
                type: integer
                description: Number of recent albums to process.
                default: "Configured NUM_RECENT_ALBUMS"
              top_n_moods:
                type: integer
                description: Number of top moods to extract per track.
                default: "Configured TOP_N_MOODS"
    responses:
      202:
        description: Analysis task successfully enqueued.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                  description: The ID of the enqueued main analysis task.
                task_type:
                  type: string
                  description: Type of the task (e.g., main_analysis).
                  example: main_analysis
                status:
                  type: string
                  description: The initial status of the job in the queue (e.g., queued).
      400:
        description: Invalid input.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
      500:
        description: Server error during task enqueue.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                message:
                  type: string
    """
    # Import task function here to break circular dependency
    from tasks import run_analysis_task  # Import from the tasks package

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

    job = rq_queue_high.enqueue( # Enqueue main task on the HIGH priority queue
        run_analysis_task,
        args=(jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods),
        job_id=job_id,
        description="Main Music Analysis", # Already has retry=Retry(max=1)
        retry=Retry(max=3), # Optional: retry once if fails
        job_timeout=-1 # No timeout
    )
    return jsonify({"task_id": job.id, "task_type": "main_analysis", "status": job.get_status()}), 202

@app.route('/api/clustering/start', methods=['POST'])
def start_clustering_endpoint():
    """
    Start the music clustering and playlist generation process.
    This endpoint enqueues a main clustering task.
    Note: Starting a new clustering task will archive previously successful tasks by setting their status to REVOKED.
    ---
    tags:
      - Clustering
    requestBody:
      description: Configuration for the clustering task.
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              clustering_method:
                type: string
                description: Algorithm to use for clustering (e.g., kmeans, dbscan, gmm).
                default: "Configured CLUSTER_ALGORITHM"
              num_clusters_min:
                type: integer
                description: Minimum number of clusters (for kmeans/gmm).
                default: "Configured NUM_CLUSTERS_MIN"
              num_clusters_max:
                type: integer
                description: Maximum number of clusters (for kmeans/gmm).
                default: "Configured NUM_CLUSTERS_MAX"
              dbscan_eps_min:
                type: number
                format: float
                description: Minimum epsilon for DBSCAN.
                default: "Configured DBSCAN_EPS_MIN"
              dbscan_eps_max:
                type: number
                format: float
                description: Maximum epsilon for DBSCAN.
                default: "Configured DBSCAN_EPS_MAX"
              dbscan_min_samples_min:
                type: integer
                description: Minimum min_samples for DBSCAN.
                default: "Configured DBSCAN_MIN_SAMPLES_MIN"
              dbscan_min_samples_max:
                type: integer
                description: Maximum min_samples for DBSCAN.
                default: "Configured DBSCAN_MIN_SAMPLES_MAX"
              gmm_n_components_min:
                type: integer
                description: Minimum number of components for GMM.
                default: "Configured GMM_N_COMPONENTS_MIN"
              gmm_n_components_max:
                type: integer
                description: Maximum number of components for GMM.
                default: "Configured GMM_N_COMPONENTS_MAX"
              pca_components_min:
                type: integer
                description: Minimum number of PCA components.
                default: "Configured PCA_COMPONENTS_MIN"
              pca_components_max:
                type: integer
                description: Maximum number of PCA components.
                default: "Configured PCA_COMPONENTS_MAX"
              clustering_runs:
                type: integer
                description: Number of clustering iterations to perform.
                default: "Configured CLUSTERING_RUNS"
              score_weight_diversity:
                type: number
                format: float
                description: Weight for the inter-playlist mood diversity score component.
                default: "Configured SCORE_WEIGHT_DIVERSITY"
              score_weight_silhouette:
                type: number
                format: float
                description: Weight for the Silhouette score component.
                default: "Configured SCORE_WEIGHT_SILHOUETTE"
              score_weight_davies_bouldin:
                type: number
                format: float
                description: Weight for the Davies-Bouldin score component (higher is better for score calculation).
                default: "Configured SCORE_WEIGHT_DAVIES_BOULDIN"
              score_weight_calinski_harabasz:
                type: number
                format: float
                description: Weight for the Calinski-Harabasz score component (higher is better).
                default: "Configured SCORE_WEIGHT_CALINSKI_HARABASZ"
              score_weight_purity:
                type: number
                format: float
                description: Weight for playlist purity (intra-playlist mood consistency).
                default: "Configured SCORE_WEIGHT_PURITY"
              score_weight_other_feature_diversity:
                type: number
                format: float
                description: Weight for inter-playlist diversity of other features (e.g., danceability).
                default: "Configured SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY"
              score_weight_other_feature_purity:
                type: number
                format: float
                description: Weight for intra-playlist consistency of other features (e.g., danceability).
                default: "Configured SCORE_WEIGHT_OTHER_FEATURE_PURITY"
              min_songs_per_genre_for_stratification:
                type: integer
                description: Minimum number of songs to target per stratified genre.
                default: "Configured MIN_SONGS_PER_GENRE_FOR_STRATIFICATION"
              stratified_sampling_target_percentile:
                type: integer
                description: Percentile of genre song counts to use for target songs per genre.
                minimum: 0
                maximum: 100
                default: "Configured STRATIFIED_SAMPLING_TARGET_PERCENTILE"
              max_songs_per_cluster:
                type: integer
                description: Maximum number of songs per generated playlist/cluster.
                default: "Configured MAX_SONGS_PER_CLUSTER"
              ai_model_provider:
                type: string
                description: AI provider for playlist naming (OLLAMA, GEMINI, NONE).
                default: "Configured AI_MODEL_PROVIDER"
              ollama_server_url:
                type: string
                description: Override for the Ollama server URL for this run.
                nullable: true
                default: "Defaults to server-configured OLLAMA_SERVER_URL"
              ollama_model_name:
                type: string
                description: Override for the Ollama model name for this run.
                nullable: true
                default: "Defaults to server-configured OLLAMA_MODEL_NAME"
              gemini_api_key:
                type: string
                description: Override for the Gemini API key for this run.
                nullable: true
                default: "Defaults to server-configured GEMINI_API_KEY"
              gemini_model_name:
                type: string
                description: Override for the Gemini model name for this run.
                nullable: true
                default: "Defaults to server-configured GEMINI_MODEL_NAME"
              top_n_moods:
                type: integer
                description: Number of top moods to consider for clustering feature vectors (uses the first N from global MOOD_LABELS).
                default: "Configured TOP_N_MOODS"
              enable_clustering_embeddings:
                type: boolean
                description: Whether to use embeddings for clustering (True) or score_vector (False).
                default: false
    responses:
      202:
        description: Clustering task successfully enqueued.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                  description: The ID of the enqueued main clustering task.
                task_type:
                  type: string
                  description: Type of the task (e.g., main_clustering).
                  example: main_clustering
                status:
                  type: string
                  description: The initial status of the job in the queue (e.g., queued).
      409:
        description: An active clustering task is already in progress.
        content:
            application/json:
                schema:
                    type: object
                    properties:
                        error:
                            type: string
                        task_id:
                            type: string
                        status:
                            type: string
    """
    # Import task function here to break circular dependency
    from tasks import run_clustering_task # Import from the tasks package

    # Check if a main_clustering task is already active
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    # Define non-terminal statuses
    active_statuses = (TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS)
    cur.execute("""
        SELECT task_id, status FROM task_status
        WHERE task_type = 'main_clustering' AND status IN %s
        ORDER BY timestamp DESC
        LIMIT 1
    """, (active_statuses,))
    existing_active_task = cur.fetchone()
    cur.close()

    if existing_active_task:
        return jsonify({"error": "An active clustering task is already in progress.", "task_id": existing_active_task['task_id'], "status": existing_active_task['status']}), 409 # 409 Conflict

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
    min_songs_per_genre_for_stratification_val = int(data.get('min_songs_per_genre_for_stratification', MIN_SONGS_PER_GENRE_FOR_STRATIFICATION))
    stratified_sampling_target_percentile_val = int(data.get('stratified_sampling_target_percentile', STRATIFIED_SAMPLING_TARGET_PERCENTILE))

    # Retrieve score weights from request, falling back to config defaults
    score_weight_diversity_val = float(data.get('score_weight_diversity', SCORE_WEIGHT_DIVERSITY))
    score_weight_silhouette_val = float(data.get('score_weight_silhouette', SCORE_WEIGHT_SILHOUETTE))
    score_weight_davies_bouldin_val = float(data.get('score_weight_davies_bouldin', SCORE_WEIGHT_DAVIES_BOULDIN))
    score_weight_calinski_harabasz_val = float(data.get('score_weight_calinski_harabasz', SCORE_WEIGHT_CALINSKI_HARABASZ))
    score_weight_purity_val = float(data.get('score_weight_purity', SCORE_WEIGHT_PURITY))
    # *** NEW: Retrieve new 'other_feature' weights ***
    score_weight_other_feature_diversity_val = float(data.get('score_weight_other_feature_diversity', SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY))
    score_weight_other_feature_purity_val = float(data.get('score_weight_other_feature_purity', SCORE_WEIGHT_OTHER_FEATURE_PURITY))

    # Ensure percentile is within valid range
    stratified_sampling_target_percentile_val = max(0, min(100, stratified_sampling_target_percentile_val))

    # Get AI Naming parameters from request, fallback to global config
    ai_model_provider_param = data.get('ai_model_provider', AI_MODEL_PROVIDER).upper()
    ollama_url_param = data.get('ollama_server_url', OLLAMA_SERVER_URL)
    ollama_model_param = data.get('ollama_model_name', OLLAMA_MODEL_NAME)
    gemini_api_key_param = data.get('gemini_api_key', GEMINI_API_KEY)
    gemini_model_name_param = data.get('gemini_model_name', GEMINI_MODEL_NAME)
    top_n_moods_for_clustering = int(data.get('top_n_moods', TOP_N_MOODS))
    enable_clustering_embeddings_param = data.get('enable_clustering_embeddings', ENABLE_CLUSTERING_EMBEDDINGS) # Get new flag
    job_id = str(uuid.uuid4())

    # Clean up details of previously successful tasks before starting a new one
    clean_successful_task_details_on_new_start()
    save_task_status(job_id, "main_clustering", TASK_STATUS_PENDING, details={"message": "Task enqueued."})

    job = rq_queue_high.enqueue( # Enqueue main task on the HIGH priority queue
        run_clustering_task,
        args=(
            clustering_method, num_clusters_min_val, num_clusters_max_val,
            dbscan_eps_min_val, dbscan_eps_max_val, dbscan_min_samples_min_val, dbscan_min_samples_max_val,
            pca_components_min_val, pca_components_max_val,
            num_clustering_runs_val,
            max_songs_per_cluster_val, gmm_n_components_min_val, gmm_n_components_max_val,
            min_songs_per_genre_for_stratification_val, # Added
            stratified_sampling_target_percentile_val,  # Added
            score_weight_diversity_val, score_weight_silhouette_val,
            score_weight_davies_bouldin_val, score_weight_calinski_harabasz_val,
            score_weight_purity_val,
            # *** NEW: Pass the new 'other_feature' weights to the task ***
            score_weight_other_feature_diversity_val,
            score_weight_other_feature_purity_val,
            # Pass AI params and new top_n_moods for clustering
            ai_model_provider_param, ollama_url_param, ollama_model_param, gemini_api_key_param, gemini_model_name_param,
            top_n_moods_for_clustering, # Pass to the task
            enable_clustering_embeddings_param # Pass new flag
        ),
        job_id=job_id,
        description="Main Music Clustering",
        retry=Retry(max=3),
        job_timeout=-1 # No timeout
    )
    return jsonify({"task_id": job.id, "task_type": "main_clustering", "status": job.get_status()}), 202


@app.route('/api/status/<task_id>', methods=['GET'])
def get_task_status_endpoint(task_id):
    """
    Get the status of a specific task.
    Retrieves status information from both RQ and the database.
    ---
    tags:
      - Status
    parameters:
      - name: task_id
        in: path
        required: true
        description: The ID of the task.
        schema:
          type: string
    responses:
      200:
        description: Status information for the task.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                state:
                  type: string
                  description: Current state of the task (e.g., PENDING, STARTED, PROGRESS, SUCCESS, FAILURE, REVOKED, queued, finished, failed, canceled).
                status_message:
                  type: string
                  description: A human-readable status message.
                progress:
                  type: integer
                  description: Task progress percentage (0-100).
                details:
                  type: object
                  description: Detailed information about the task. Structure varies by task type and state.
                  additionalProperties: true
                  example: {"log": ["Log message 1"], "current_album": "Album X"}
                task_type_from_db:
                  type: string
                  nullable: true
                  description: The type of the task as recorded in the database (e.g., main_analysis, album_analysis, main_clustering, clustering_batch).
      404:
        description: Task ID not found in RQ or database.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                state:
                  type: string
                  example: UNKNOWN
                status_message:
                  type: string
                  example: Task ID not found in RQ or DB.
    """
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

    # First, determine the task_type for the current job_id # type: ignore
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
        job_rq = Job.fetch(job_id, connection=redis_conn) # type: ignore
        current_rq_status = job_rq.get_status()
        logger.info("Job %s (type: %s) found in RQ with status: %s", job_id, current_task_type, current_rq_status)

        if job_rq.is_started:
            logger.info("  Job %s is STARTED. Attempting to send stop command.", job_id)
            try:
                send_stop_job_command(redis_conn, job_id) # This will likely move it to 'failed' in RQ
                action_taken_in_rq = True
                logger.info("    Stop command sent for job %s.", job_id)
            except InvalidJobOperation:
                logger.warning("    Job %s was in 'started' state but became non-executable for stop command (InvalidJobOperation). Will mark as REVOKED in DB.", job_id)
            except Exception as e_stop_cmd: # Catch other potential errors from send_stop_job_command
                logger.error("    Error sending stop command for job %s: %s", job_id, e_stop_cmd)
        elif not (job_rq.is_finished or job_rq.is_failed or job_rq.is_canceled):
            # If it's not started, and not in a terminal state (e.g., queued, deferred)
            logger.info("  Job %s is %s. Attempting to cancel via job.cancel().", job_id, current_rq_status)
            job_rq.cancel() # This moves it to 'canceled' status in RQ
            action_taken_in_rq = True
        else:
            logger.info("  Job %s is already in a terminal RQ state: %s. No RQ action needed.", job_id, current_rq_status)

    except NoSuchJobError:
        logger.warning("Job %s (type: %s) not found in RQ. Will mark as REVOKED in DB.", job_id, current_task_type)
    except Exception as e_rq_interaction:
        logger.error("Error interacting with RQ for job %s (type: %s): %s", job_id, current_task_type, e_rq_interaction)

    if action_taken_in_rq:
        cancelled_count += 1

    # Always mark as REVOKED in DB for the current job if its task_type is known
    save_task_status(job_id, current_task_type, TASK_STATUS_REVOKED, progress=100, details={"message": "Task cancellation processed by API."})

    # Attempt to cancel children based on DB parent_task_id
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)

    # Define terminal statuses for the query
    terminal_statuses_tuple = (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED,
                               JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED) # JobStatus for completeness if used directly

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
        logger.info("Recursively cancelling child job: %s of type %s", child_job_id, child_task_type)
        # The count from recursive calls will be added
        cancelled_count += cancel_job_and_children_recursive(child_job_id, child_task_type)

    return cancelled_count

@app.route('/api/cancel/<task_id>', methods=['POST'])
def cancel_task_endpoint(task_id):
    """
    Cancel a specific task and its children.
    Marks the task and its descendants as REVOKED in the database and attempts to stop/cancel them in RQ.
    ---
    tags:
      - Control
    parameters:
      - name: task_id
        in: path
        required: true
        description: The ID of the task.
        schema:
          type: string
    responses:
      200:
        description: Cancellation initiated for the task and its children.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                task_id:
                  type: string
                cancelled_jobs_count:
                  type: integer
      400:
        description: Task could not be cancelled (e.g., already completed or not in an active state).
      404:
        description: Task ID not found in the database.
    """
    db_task_info = get_task_info_from_db(task_id)
    if not db_task_info:
        return jsonify({"message": f"Task {task_id} not found in database.", "task_id": task_id}), 404

    cancelled_count = cancel_job_and_children_recursive(task_id, db_task_info.get('task_type')) # Task type from DB

    if cancelled_count > 0:
        return jsonify({"message": f"Task {task_id} and its children cancellation initiated. {cancelled_count} total jobs affected.", "task_id": task_id, "cancelled_jobs_count": cancelled_count}), 200
    return jsonify({"message": "Task could not be cancelled (e.g., already completed or not found in active state).", "task_id": task_id}), 400


@app.route('/api/cancel_all/<task_type_prefix>', methods=['POST'])
def cancel_all_tasks_by_type_endpoint(task_type_prefix):
    """
    Cancel all active tasks of a specific type (e.g., main_analysis, main_clustering) and their children.
    ---
    tags:
      - Control
    parameters:
      - name: task_type_prefix
        in: path
        required: true
        description: The type of main tasks to cancel (e.g., "main_analysis", "main_clustering").
        schema:
          type: string
    responses:
      200:
        description: Cancellation initiated for all matching active tasks and their children.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                cancelled_main_tasks:
                  type: array
                  items:
                    type: string
      404:
        description: No active tasks of the specified type found to cancel.
    """
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
    """
    Get the status of the most recent overall main task (analysis or clustering).
    ---
    tags:
      - Status
    responses:
      200:
        description: Status information for the last main task.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                  nullable: true
                task_type:
                  type: string
                  nullable: true
                status:
                  type: string
                progress:
                  type: integer
                  nullable: true
                details:
                  type: object
                  additionalProperties: true
                  nullable: true
      404: # Although current code returns 200 with NO_PREVIOUS_MAIN_TASK
        description: No previous main task found (current implementation returns 200).

    """
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
    """
    Get the status of the currently active main task, if any.
    An active main task is one that is not in a SUCCESS, FAILURE, or REVOKED state.
    ---
    tags:
      - Status
    responses:
      200:
        description: Status information for the active main task, or an empty object if none.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                parent_task_id:
                  type: string
                  nullable: true
                task_type:
                  type: string
                sub_type_identifier:
                  type: string
                  nullable: true
                status:
                  type: string
                progress:
                  type: integer
                details:
                  type: object
                  additionalProperties: true
    """
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
    active_main_task_row = cur.fetchone()
    cur.close()

    if active_main_task_row:
        task_item = dict(active_main_task_row)
        if task_item.get('details'):
            try:
                task_item['details'] = json.loads(task_item['details'])
                # Prune specific large or internal keys from details
                if isinstance(task_item['details'], dict):
                    task_item['details'].pop('clustering_run_job_ids', None)
                    if 'best_params' in task_item['details'] and \
                       isinstance(task_item['details']['best_params'], dict) and \
                       'clustering_method_config' in task_item['details']['best_params'] and \
                       isinstance(task_item['details']['best_params']['clustering_method_config'], dict) and \
                       'params' in task_item['details']['best_params']['clustering_method_config']['params'] and \
                       isinstance(task_item['details']['best_params']['clustering_method_config']['params'], dict):
                        task_item['details']['best_params']['clustering_method_config']['params'].pop('initial_centroids', None)

            except json.JSONDecodeError:
                task_item['details'] = {"raw_details": task_item['details'], "error": "Failed to parse details JSON."}
        return jsonify(task_item), 200
    return jsonify({}), 200 # Return empty object if no active main task

@app.route('/api/config', methods=['GET'])
def get_config_endpoint():
    """
    Get the current server configuration values.
    ---
    tags:
      - Configuration
    responses:
      200:
        description: A JSON object containing various configuration parameters.
        content:
          application/json:
            schema:
              type: object
              properties:
                jellyfin_url:
                  type: string
                jellyfin_user_id:
                  type: string
                jellyfin_token:
                  type: string
                num_recent_albums:
                  type: integer
                max_distance:
                  type: number
                max_songs_per_cluster:
                  type: integer
                max_songs_per_artist:
                  type: integer
                cluster_algorithm:
                  type: string
                num_clusters_min:
                  type: integer
                num_clusters_max:
                  type: integer
                dbscan_eps_min:
                  type: number
                dbscan_eps_max:
                  type: number
                dbscan_min_samples_min:
                  type: integer
                dbscan_min_samples_max:
                  type: integer
                gmm_n_components_min:
                  type: integer
                gmm_n_components_max:
                  type: integer
                pca_components_min:
                  type: integer
                pca_components_max:
                  type: integer
                min_songs_per_genre_for_stratification:
                  type: integer
                stratified_sampling_target_percentile:
                  type: integer
                top_n_moods:
                  type: integer
                mood_labels:
                  type: array
                  items:
                    type: string
                ai_model_provider:
                  type: string
                  description: Configured AI provider for playlist naming (OLLAMA, GEMINI, NONE).
                ollama_server_url:
                  type: string
                  description: URL of the Ollama server for AI naming.
                  nullable: true
                ollama_model_name:
                  type: string
                  description: Name of the Ollama model to use for AI naming.
                  nullable: true
                gemini_api_key:
                  type: string
                  description: Configured Gemini API key (may be a default placeholder).
                  nullable: true
                gemini_model_name:
                  type: string
                  description: Configured Gemini model name.
                  nullable: true
                clustering_runs:
                  type: integer
                score_weight_diversity:
                  type: number
                  format: float
                score_weight_silhouette:
                  type: number
                  format: float
                score_weight_davies_bouldin:
                  type: number
                  format: float
                score_weight_calinski_harabasz:
                  type: number
                  format: float
                score_weight_purity:
                  type: number
                  format: float
                score_weight_other_feature_diversity:
                  type: number
                  format: float
                score_weight_other_feature_purity:
                  type: number
                  format: float
                gmm_covariance_type:
                  type: string
                  description: Default GMM covariance type.
                enable_clustering_embeddings:
                  type: boolean
                  description: Default state for using embeddings in clustering.
    """
    return jsonify({
        "jellyfin_url": JELLYFIN_URL, "jellyfin_user_id": JELLYFIN_USER_ID, "jellyfin_token": JELLYFIN_TOKEN,
        "num_recent_albums": NUM_RECENT_ALBUMS, "max_distance": MAX_DISTANCE,
        "max_songs_per_cluster": MAX_SONGS_PER_CLUSTER, "max_songs_per_artist": MAX_SONGS_PER_ARTIST,
        "cluster_algorithm": CLUSTER_ALGORITHM, "num_clusters_min": NUM_CLUSTERS_MIN, "num_clusters_max": NUM_CLUSTERS_MAX,
        "dbscan_eps_min": DBSCAN_EPS_MIN, "dbscan_eps_max": DBSCAN_EPS_MAX, "gmm_covariance_type": GMM_COVARIANCE_TYPE,
        "dbscan_min_samples_min": DBSCAN_MIN_SAMPLES_MIN, "dbscan_min_samples_max": DBSCAN_MIN_SAMPLES_MAX,
        "gmm_n_components_min": GMM_N_COMPONENTS_MIN, "gmm_n_components_max": GMM_N_COMPONENTS_MAX,
        "pca_components_min": PCA_COMPONENTS_MIN, "pca_components_max": PCA_COMPONENTS_MAX,
        "min_songs_per_genre_for_stratification": MIN_SONGS_PER_GENRE_FOR_STRATIFICATION,
        "stratified_sampling_target_percentile": STRATIFIED_SAMPLING_TARGET_PERCENTILE,
        "ai_model_provider": AI_MODEL_PROVIDER,
        "ollama_server_url": OLLAMA_SERVER_URL, "ollama_model_name": OLLAMA_MODEL_NAME,
        "gemini_api_key": GEMINI_API_KEY, "gemini_model_name": GEMINI_MODEL_NAME,
        "top_n_moods": TOP_N_MOODS, "mood_labels": MOOD_LABELS, "clustering_runs": CLUSTERING_RUNS,
        "enable_clustering_embeddings": ENABLE_CLUSTERING_EMBEDDINGS, # Expose new flag
        "score_weight_diversity": SCORE_WEIGHT_DIVERSITY,
        "score_weight_silhouette": SCORE_WEIGHT_SILHOUETTE,
        "score_weight_davies_bouldin": SCORE_WEIGHT_DAVIES_BOULDIN,
        "score_weight_calinski_harabasz": SCORE_WEIGHT_CALINSKI_HARABASZ,
        "score_weight_purity": SCORE_WEIGHT_PURITY,
        # *** NEW: Add new 'other_feature' weights to config response ***
        "score_weight_other_feature_diversity": SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY,
        "score_weight_other_feature_purity": SCORE_WEIGHT_OTHER_FEATURE_PURITY
    })

@app.route('/api/playlists', methods=['GET'])
def get_playlists_endpoint():
    """
    Get all generated playlists and their tracks from the database.
    ---
    tags:
      - Playlists
    responses:
      200:
        description: A dictionary of playlists.
        content:
          application/json:
            schema:
              type: object
              additionalProperties:
                type: array
                items:
                  type: object
                  properties:
                    item_id:
                      type: string
                      description: The Jellyfin Item ID of the track.
                    title:
                      type: string
                      description: The title of the track.
                    author:
                      type: string
                      description: The artist of the track.
                example:
                  "Energetic_Fast_1": [{"item_id": "xyz", "title": "Song A", "author": "Artist X"}]

    """
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
    app.run(debug=False, host='0.0.0.0', port=8000)
