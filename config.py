import os

# --- Jellyfin and DB Constants (Read from Environment Variables first) ---

# JELLYFIN_USER_ID and JELLYFIN_TOKEN come from a Kubernetes Secret
JELLYFIN_USER_ID = os.environ.get("JELLYFIN_USER_ID", "your_default_user_id")  # Replace with a suitable default or handle missing case
JELLYFIN_TOKEN = os.environ.get("JELLYFIN_TOKEN", "your_default_token")  # Replace with a suitable default or handle missing case

# Other variables come from the audiomuse-ai-config ConfigMap
JELLYFIN_URL = os.environ.get("JELLYFIN_URL", "http://your_jellyfin_url:8096") # Replace with your default URL
TEMP_DIR = "/app/temp_audio"  # Always use /app/temp_audio



HEADERS = {"X-Emby-Token": JELLYFIN_TOKEN}

# --- General Constants (Read from Environment Variables where applicable) ---
MAX_DISTANCE = 0.5
MAX_SONGS_PER_CLUSTER = 40
MAX_SONGS_PER_ARTIST = 3
NUM_RECENT_ALBUMS = int(os.getenv("NUM_RECENT_ALBUMS", "2000")) # Convert to int

# --- Algorithm Choose Constant (Read from Environment Variables) ---
CLUSTER_ALGORITHM = os.environ.get("CLUSTER_ALGORITHM", "kmeans") # accepted dbscan, kmeans, or gmm

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
GMM_COVARIANCE_TYPE = os.environ.get("GMM_COVARIANCE_TYPE", "full") # 'full', 'tied', 'diag', 'spherical'

# --- PCA Constants (Ranges for Evolutionary Approach) ---
# Default ranges for PCA components
PCA_COMPONENTS_MIN = int(os.getenv("PCA_COMPONENTS_MIN", "0")) # 0 to disable PCA
PCA_COMPONENTS_MAX = int(os.getenv("PCA_COMPONENTS_MAX", "5")) # Max components for PCA

# --- Clustering Runs for Diversity (New Constant) ---
CLUSTERING_RUNS = int(os.environ.get("CLUSTERING_RUNS", "100")) # Default to 100 runs for evolutionary search

# --- Celery Broker/Backend URLs (No longer used with RQ) ---
# CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis-service.playlist:6379/0")
# CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis-service.playlist:6379/0")

# --- RQ / Redis / Database Configuration ---
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://audiomuse:audiomusepassword@postgres-service.playlist:5432/audiomusedb")

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
