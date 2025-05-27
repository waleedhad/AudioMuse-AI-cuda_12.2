# config.py
import os

# --- Jellyfin and DB Constants (Read from Environment Variables first) ---

# JELLYFIN_USER_ID and JELLYFIN_TOKEN come from a Kubernetes Secret
JELLYFIN_USER_ID = os.getenv("JELLYFIN_USER_ID", "0e45c44b3e2e4da7a2be11a72a1c8575")
JELLYFIN_TOKEN = os.getenv("JELLYFIN_TOKEN", "e0b8c325bc1b426c81922b90c0aa2ff1")

# Other variables come from the audiomuse-ai-config ConfigMap
JELLYFIN_URL = os.getenv("JELLYFIN_URL", "http://jellyfin.192.168.3.131.nip.io:8087")
TEMP_DIR = os.getenv("TEMP_DIR", "/app/temp_audio") # Now explicitly using /app/temp_audio as default for emptyDir
DB_PATH = os.getenv("DB_PATH", "/app/db.sqlite") # Default to /app if env var not found (local dev fallback)
STATUS_DB_PATH = os.getenv("STATUS_DB_PATH", "/app/status_db.sqlite") # Ensure status DB path is also from env var

HEADERS = {"X-Emby-Token": JELLYFIN_TOKEN}

# --- General Constants (Read from Environment Variables where applicable) ---
MAX_DISTANCE = 0.5
MAX_SONGS_PER_CLUSTER = 40 # This is a hard limit for chunking playlists for Jellyfin, not the desired playlist size
MAX_SONGS_PER_ARTIST = 3

# Changed NUM_RECENT_ALBUMS to MIN_RECENT_ALBUMS and added MAX_RECENT_ALBUMS
MIN_RECENT_ALBUMS = int(os.getenv("MIN_RECENT_ALBUMS", "40")) # Min recent albums to fetch
MAX_RECENT_ALBUMS = int(os.getenv("MAX_RECENT_ALBUMS", "60")) # Max recent albums to fetch


# --- Algorithm Choose Constant (Read from Environment Variables) ---
CLUSTER_ALGORITHM = os.getenv("CLUSTER_ALGORITHM", "kmeans") # accepted dbscan or kmeans

# --- DBSCAN Only Constants (Ranges for Evolutionary Approach) ---
# Default ranges for DBSCAN parameters
DBSCAN_EPS_MIN = float(os.getenv("DBSCAN_EPS_MIN", "0.1"))
DBSCAN_EPS_MAX = float(os.getenv("DBSCAN_EPS_MAX", "0.5"))
DBSCAN_MIN_SAMPLES_MIN = int(os.getenv("DBSCAN_MIN_SAMPLES_MIN", "5")) # Changed default
DBSCAN_MIN_SAMPLES_MAX = int(os.getenv("DBSCAN_MIN_SAMPLES_MAX", "20")) # Changed default

# --- KMEANS and DBSCAN (PCA Components) ---
PCA_COMPONENTS_MIN = int(os.getenv("PCA_COMPONENTS_MIN", "0")) # 0 to disable PCA initially
PCA_COMPONENTS_MAX = int(os.getenv("PCA_COMPONENTS_MAX", "10")) # Max components for PCA

# --- KMEANS Only Constants (Ranges for Evolutionary Approach) ---
NUM_CLUSTERS_MIN = int(os.getenv("NUM_CLUSTERS_MIN", "10")) # Changed default
NUM_CLUSTERS_MAX = int(os.getenv("NUM_CLUSTERS_MAX", "50")) # Changed default

# --- Evolutionary Clustering Runs ---
CLUSTERING_RUNS = int(os.getenv("CLUSTERING_RUNS", "1000")) # Changed default

# --- Playlist Song Number Constraints (User Configurable) ---
MIN_SONGS_PER_PLAYLIST = int(os.getenv("MIN_SONGS_PER_PLAYLIST", "20"))
MAX_SONGS_PER_PLAYLIST = int(os.getenv("MAX_SONGS_PER_PLAYLIST", "40"))


# --- Celery Constants ---
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# --- Essentia Model Paths ---
# These are the paths to the models within the Docker container / app environment
# Ensure these are correct based on where your models are mounted or stored
EMBEDDING_MODEL_PATH = "/app/msd-musicnn-1.pb"
PREDICTION_MODEL_PATH = "/app/msd-discogs-effnet.pb" # Corrected path for consistency, assuming discogs for mood prediction

# --- Mood Labels ---
# Updated list of mood labels based on a common set, ensures consistency if model changes
MOOD_LABELS = [
    'danceable', 'energetic', 'calm', 'romantic', 'sad', 'happy',
    'aggressive', 'relaxed', 'upbeat', 'dark', 'exciting', 'peaceful',
    'dreamy', 'driving', 'groovy', 'laid-back', 'melancholy', 'pensive',
    'rowdy', 'soothing', 'spiritual', 'stirring', 'thought-provoking',
    'uplifting', 'mellow', 'intense', 'epic', 'light', 'mysterious',
    'passion', 'resolute', 'serene', 'smooth', 'somber', 'sparkling',
    'spooky', 'unsettling', 'vibrant', 'whimsical', 'gritty'
]

TOP_N_MOODS = 5

# --- Debugging (Optional, remove in production if not needed) ---
print(f"DEBUG: JELLYFIN_USER_ID: {JELLYFIN_USER_ID}")
print(f"DEBUG: JELLYFIN_URL: {JELLYFIN_URL}")
print(f"DEBUG: JELLYFIN_TOKEN: {'***hidden***' if JELLYFIN_TOKEN else 'None'}")
print(f"DEBUG: TEMP_DIR: {TEMP_DIR}")
print(f"DEBUG: DB_PATH: {DB_PATH}")
print(f"DEBUG: STATUS_DB_PATH: {STATUS_DB_PATH}")
print(f"DEBUG: MIN_RECENT_ALBUMS: {MIN_RECENT_ALBUMS}") # Updated debug print
print(f"DEBUG: MAX_RECENT_ALBUMS: {MAX_RECENT_ALBUMS}") # New debug print
print(f"DEBUG: CLUSTER_ALGORITHM: {CLUSTER_ALGORITHM}")
print(f"DEBUG: NUM_CLUSTERS_MIN: {NUM_CLUSTERS_MIN}")
print(f"DEBUG: NUM_CLUSTERS_MAX: {NUM_CLUSTERS_MAX}")
print(f"DEBUG: DBSCAN_EPS_MIN: {DBSCAN_EPS_MIN}")
print(f"DEBUG: DBSCAN_EPS_MAX: {DBSCAN_EPS_MAX}")
print(f"DEBUG: DBSCAN_MIN_SAMPLES_MIN: {DBSCAN_MIN_SAMPLES_MIN}")
print(f"DEBUG: DBSCAN_MIN_SAMPLES_MAX: {DBSCAN_MIN_SAMPLES_MAX}")
print(f"DEBUG: PCA_COMPONENTS_MIN: {PCA_COMPONENTS_MIN}")
print(f"DEBUG: PCA_COMPONENTS_MAX: {PCA_COMPONENTS_MAX}")
print(f"DEBUG: CLUSTERING_RUNS: {CLUSTERING_RUNS}")
print(f"DEBUG: MIN_SONGS_PER_PLAYLIST: {MIN_SONGS_PER_PLAYLIST}")
print(f"DEBUG: MAX_SONGS_PER_PLAYLIST: {MAX_SONGS_PER_PLAYLIST}")
