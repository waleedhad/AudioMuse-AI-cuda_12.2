import os

# --- Jellyfin and DB Constants (Read from Environment Variables first) ---

# JELLYFIN_USER_ID and JELLYFIN_TOKEN come from a Kubernetes Secret
JELLYFIN_USER_ID = os.getenv("JELLYFIN_USER_ID", "0e45c44b3e2e4da7a2be11a72a1c8575")
JELLYFIN_TOKEN = os.getenv("JELLYFIN_TOKEN", "e0b8c325bc1b426c81922b90c0aa2ff1")

# Other variables come from the audiomuse-ai-config ConfigMap
JELLYFIN_URL = os.getenv("JELLYFIN_URL", "http://jellyfin.192.168.3.131.nip.io:8087")
TEMP_DIR = os.getenv("TEMP_DIR", "/workspace/temp_audio") # Now explicitly using /app/temp_audio as default for emptyDir

HEADERS = {"X-Emby-Token": JELLYFIN_TOKEN}

# --- PostgreSQL Database Constants ---
# These will be set as environment variables in the Kubernetes deployment
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "audiomuse_db")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_PORT = os.getenv("DB_PORT", "5432")

# Construct the PostgreSQL connection URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Define table names as constants
STATUS_DB_TABLE_NAME = os.getenv("STATUS_DB_TABLE_NAME", "task_status")
SCORE_DB_TABLE_NAME = os.getenv("SCORE_DB_TABLE_NAME", "score")
PLAYLIST_DB_TABLE_NAME = os.getenv("PLAYLIST_DB_TABLE_NAME", "playlist")


# --- General Constants (Read from Environment Variables where applicable) ---
MAX_DISTANCE = 0.5
MAX_SONGS_PER_CLUSTER = 40
MAX_SONGS_PER_ARTIST = 3
NUM_RECENT_ALBUMS = int(os.getenv("NUM_RECENT_ALBUMS", "2000")) # Convert to int

# --- Algorithm Choose Constant (Read from Environment Variables) ---
CLUSTER_ALGORITHM = os.getenv("CLUSTER_ALGORITHM", "kmeans") # accepted dbscan, kmeans, or gmm

# --- DBSCAN Only Constants (Ranges for Evolutionary Approach) ---
# Default ranges for DBSCAN parameters
DBSCAN_EPS_MIN = float(os.getenv("DBSCAN_EPS_MIN", "0.1"))
DBSCAN_EPS_MAX = float(os.getenv("DBSCAN_EPS_MAX", "0.5"))
DBSCAN_MIN_SAMPLES_MIN = int(os.getenv("DBSCAN_MIN_SAMPLES_MIN", "5"))
DBSCAN_MIN_SAMPLES_MAX = int(os.getenv("DBSCAN_MIN_SAMPLES_MAX", "20"))


# --- KMEANS Only Constants (Ranges for Evolutionary Approach) ---
# Default ranges for KMeans parameters
NUM_CLUSTERS_MIN = int(os.getenv("NUM_CLUSTERS_MIN", "20"))
NUM_CLUSTERS_MAX = int(os.getenv("NUM_CLUSTERS_MAX", "60"))

# --- GMM Only Constants (Ranges for Evolutionary Approach) ---
# Default ranges for GMM parameters
GMM_N_COMPONENTS_MIN = int(os.getenv("GMM_N_COMPONENTS_MIN", "20"))
GMM_N_COMPONENTS_MAX = int(os.getenv("GMM_N_COMPONENTS_MAX", "60"))
GMM_COVARIANCE_TYPE = os.getenv("GMM_COVARIANCE_TYPE", "full") # 'full', 'tied', 'diag', 'spherical'

# --- PCA Constants (Ranges for Evolutionary Approach) ---
# Default ranges for PCA components
PCA_COMPONENTS_MIN = int(os.getenv("PCA_COMPONENTS_MIN", "0")) # 0 to disable PCA
PCA_COMPONENTS_MAX = int(os.getenv("PCA_COMPONENTS_MAX", "5")) # Max components for PCA

# --- Clustering Runs for Diversity (New Constant) ---
CLUSTERING_RUNS = int(os.getenv("CLUSTERING_RUNS", "1000")) # Default to 100 runs for evolutionary search

# --- Celery Broker/Backend URLs (from ConfigMap in your deployment) ---
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis-service.playlist:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis-service.playlist:6379/0")


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
print(f"DEBUG: DB_HOST: {DB_HOST}")
print(f"DEBUG: DB_NAME: {DB_NAME}")
print(f"DEBUG: DB_USER: {DB_USER}")
print(f"DEBUG: DB_PORT: {DB_PORT}")
print(f"DEBUG: DATABASE_URL: {'***hidden***' if DB_PASSWORD else DATABASE_URL}") # Hide password
print(f"DEBUG: STATUS_DB_TABLE_NAME: {STATUS_DB_TABLE_NAME}")
print(f"DEBUG: SCORE_DB_TABLE_NAME: {SCORE_DB_TABLE_NAME}")
print(f"DEBUG: PLAYLIST_DB_TABLE_NAME: {PLAYLIST_DB_TABLE_NAME}")
print(f"DEBUG: NUM_RECENT_ALBUMS: {NUM_RECENT_ALBUMS}")
print(f"DEBUG: CLUSTER_ALGORITHM: {CLUSTER_ALGORITHM}")
print(f"DEBUG: NUM_CLUSTERS_MIN: {NUM_CLUSTERS_MIN}")
print(f"DEBUG: NUM_CLUSTERS_MAX: {NUM_CLUSTERS_MAX}")
print(f"DEBUG: DBSCAN_EPS_MIN: {DBSCAN_EPS_MIN}")
print(f"DEBUG: DBSCAN_EPS_MAX: {DBSCAN_EPS_MAX}")
print(f"DEBUG: DBSCAN_MIN_SAMPLES_MIN: {DBSCAN_MIN_SAMPLES_MIN}")
print(f"DEBUG: DBSCAN_MIN_SAMPLES_MAX: {DBSCAN_MIN_SAMPLES_MAX}")
print(f"DEBUG: GMM_N_COMPONENTS_MIN: {GMM_N_COMPONENTS_MIN}")
print(f"DEBUG: GMM_N_COMPONENTS_MAX: {GMM_N_COMPONENTS_MAX}")
print(f"DEBUG: GMM_COVARIANCE_TYPE: {GMM_COVARIANCE_TYPE}")
print(f"DEBUG: PCA_COMPONENTS_MIN: {PCA_COMPONENTS_MIN}")
print(f"DEBUG: PCA_COMPONENTS_MAX: {PCA_COMPONENTS_MAX}")
print(f"DEBUG: CLUSTERING_RUNS: {CLUSTERING_RUNS}")
print(f"DEBUG: CELERY_BROKER_URL: {CELERY_BROKER_URL}")
print(f"DEBUG: CELERY_RESULT_BACKEND: {CELERY_RESULT_BACKEND}")
