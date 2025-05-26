import os

# --- Jellyfin and DB Constants (Read from Environment Variables first) ---

# JELLYFIN_USER_ID and JELLYFIN_TOKEN come from a Kubernetes Secret
JELLYFIN_USER_ID = os.getenv("JELLYFIN_USER_ID", "0e45c44b3e2e4da7a2be11a72a1c8575")
JELLYFIN_TOKEN = os.getenv("JELLYFIN_TOKEN", "e0b8c325bc1b426c81922b90c0aa2ff1")

# Other variables come from the audiomuse-ai-config ConfigMap
JELLYFIN_URL = os.getenv("JELLYFIN_URL", "http://jellyfin.192.168.3.131.nip.io:8087")
TEMP_DIR = os.getenv("TEMP_DIR", "temp_audio") # Now explicitly using /app/temp_audio as default for emptyDir
DB_PATH = os.getenv("DB_PATH", "db.sqlite") # Default to /app if env var not found (local dev fallback)
STATUS_DB_PATH = os.getenv("STATUS_DB_PATH", "status_db.sqlite") # Ensure status DB path is also from env var

HEADERS = {"X-Emby-Token": JELLYFIN_TOKEN}

# --- General Constants (Read from Environment Variables where applicable) ---
MAX_DISTANCE = 0.5
MAX_SONGS_PER_CLUSTER = 40
MAX_SONGS_PER_ARTIST = 3
NUM_RECENT_ALBUMS = int(os.getenv("NUM_RECENT_ALBUMS", "0")) # Convert to int

# --- Algorithm Chose Constant (Read from Environment Variables) ---
CLUSTER_ALGORITHM = os.getenv("CLUSTER_ALGORITHM", "kmeans") # accepted dbscan or kmeans
PCA_ENABLED = os.getenv("PCA_ENABLED", "False").lower() == 'true' # Convert string "False" to boolean False

# --- DBSCAN Only Constant ---
DBSCAN_EPS = 0.15
DBSCAN_MIN_SAMPLES = 10

# --- KMEANS Only Constant (Read from Environment Variables) ---
NUM_CLUSTERS = int(os.getenv("NUM_CLUSTERS", "40")) # if 0 it automatically defined

# --- Clustering Runs for Diversity (New Constant) ---
CLUSTERING_RUNS = int(os.getenv("CLUSTERING_RUNS", "10")) # Default to 10 runs

# --- Celery Broker/Backend URLs (from ConfigMap in your deployment) ---
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")


# --- Classifier Constant ---
MOOD_LABELS = [
    'rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists', 'dance', '00s', 'alternative rock', 'jazz',
    'beautiful', 'metal', 'chillout', 'male vocalists', 'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica', '80s',
    'folk', '90s', 'chill', 'instrumental', 'punk', 'oldies', 'blues', 'hard rock', 'ambient', 'acoustic', 'experimental',
    'female vocalist', 'guitar', 'Hip-Hop', '70s', 'party', 'country', 'easy listening', 'sexy', 'catchy', 'funk', 'electro',
    'heavy metal', 'Progressive rock', '60s', 'rnb', 'indie pop', 'sad', 'House', 'happy'
]

TOP_N_MOODS = 5
EMBEDDING_MODEL_PATH = "/app/msd-musicnn-1.pb"
PREDICTION_MODEL_PATH = "/app/msd-msd-musicnn-1.pb"

# --- Debugging (Optional, remove in production if not needed) ---
print(f"DEBUG: JELLYFIN_USER_ID: {JELLYFIN_USER_ID}")
print(f"DEBUG: JELLYFIN_URL: {JELLYFIN_URL}")
print(f"DEBUG: JELLYFIN_TOKEN: {'***hidden***' if JELLYFIN_TOKEN else 'None'}")
print(f"DEBUG: TEMP_DIR: {TEMP_DIR}")
print(f"DEBUG: DB_PATH: {DB_PATH}")
print(f"DEBUG: STATUS_DB_PATH: {STATUS_DB_PATH}")
print(f"DEBUG: NUM_RECENT_ALBUMS: {NUM_RECENT_ALBUMS}")
print(f"DEBUG: CLUSTER_ALGORITHM: {CLUSTER_ALGORITHM}")
print(f"DEBUG: PCA_ENABLED: {PCA_ENABLED}")
print(f"DEBUG: NUM_CLUSTERS: {NUM_CLUSTERS}")
print(f"DEBUG: CLUSTERING_RUNS: {CLUSTERING_RUNS}") # New debug print
print(f"DEBUG: CELERY_BROKER_URL: {CELERY_BROKER_URL}")
print(f"DEBUG: CELERY_RESULT_BACKEND: {CELERY_RESULT_BACKEND}")
