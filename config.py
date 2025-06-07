# /home/guido/Music/AudioMuse-AI/config.py
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

# --- Algorithm Choose Constants (Read from Environment Variables) ---
CLUSTER_ALGORITHM = os.environ.get("CLUSTER_ALGORITHM", "kmeans") # accepted dbscan, kmeans, or gmm
AI_MODEL_PROVIDER = os.environ.get("AI_MODEL_PROVIDER", "NONE").upper() # Accepted: OLLAMA, GEMINI, NONE

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
CLUSTERING_RUNS = int(os.environ.get("CLUSTERING_RUNS", "1000")) # Default to 100 runs for evolutionary search

# --- Guided Evolutionary Clustering Constants ---
TOP_N_ELITES = int(os.environ.get("CLUSTERING_TOP_N_ELITES", "10")) # Number of best solutions to keep as elites
EXPLOITATION_START_FRACTION = float(os.environ.get("CLUSTERING_EXPLOITATION_START_FRACTION", "0.2")) # Fraction of runs before starting to use elites (e.g., 0.2 means after 20% of runs)
EXPLOITATION_PROBABILITY_CONFIG = float(os.environ.get("CLUSTERING_EXPLOITATION_PROBABILITY", "0.7")) # Probability of mutating an elite vs. random generation, once exploitation starts
MUTATION_INT_ABS_DELTA = int(os.environ.get("CLUSTERING_MUTATION_INT_ABS_DELTA", "3")) # Max absolute change for integer parameter mutation
MUTATION_FLOAT_ABS_DELTA = float(os.environ.get("CLUSTERING_MUTATION_FLOAT_ABS_DELTA", "0.05")) # Max absolute change for float parameter mutation (e.g., for DBSCAN eps)
MUTATION_KMEANS_COORD_FRACTION = float(os.environ.get("CLUSTERING_MUTATION_KMEANS_COORD_FRACTION", "0.05")) # Fractional change for KMeans centroid coordinates based on data range

# --- Scoring Weights for Enhanced Diversity Score ---
SCORE_WEIGHT_DIVERSITY = float(os.environ.get("SCORE_WEIGHT_DIVERSITY", "0.4")) # Weight for the base diversity (inter-playlist mood diversity)
SCORE_WEIGHT_PURITY = float(os.environ.get("SCORE_WEIGHT_PURITY", "0.3"))    # Weight for playlist purity (intra-playlist mood consistency)
SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY = float(os.environ.get("SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY", "0.2")) # New: Weight for inter-playlist other feature diversity
SCORE_WEIGHT_OTHER_FEATURE_PURITY = float(os.environ.get("SCORE_WEIGHT_OTHER_FEATURE_PURITY", "0.1"))       # New: Weight for intra-playlist other feature consistency

# --- AI Playlist Naming ---
# USE_AI_PLAYLIST_NAMING is replaced by AI_MODEL_PROVIDER
OLLAMA_SERVER_URL = os.environ.get("OLLAMA_SERVER_URL", "http://192.168.3.15:11434/api/generate") # URL for your Ollama instance
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "mistral:7b") # Ollama model to use

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR-GEMINI-API-KEY-HERE") # Default API key
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest") # Default Gemini model
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# Construct DATABASE_URL from individual components for better security in K8s
POSTGRES_USER = os.environ.get("POSTGRES_USER", "audiomuse")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "audiomusepassword")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres-service.playlist") # Default for K8s
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "audiomusedb")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

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

# --- Other Essentia Model Paths ---
# Paths for models used in predict_other_models (VGGish-based)
VGGISH_EMBEDDING_MODEL_PATH = os.environ.get("VGGISH_EMBEDDING_MODEL_PATH", "/app/audioset-vggish-3.pb")
DANCEABILITY_MODEL_PATH = os.environ.get("DANCEABILITY_MODEL_PATH", "/app/danceability-audioset-vggish-1.pb")
AGGRESSIVE_MODEL_PATH = os.environ.get("AGGRESSIVE_MODEL_PATH", "/app/mood_aggressive-audioset-vggish-1.pb")
HAPPY_MODEL_PATH = os.environ.get("HAPPY_MODEL_PATH", "/app/mood_happy-audioset-vggish-1.pb")
PARTY_MODEL_PATH = os.environ.get("PARTY_MODEL_PATH", "/app/mood_party-audioset-vggish-1.pb")
RELAXED_MODEL_PATH = os.environ.get("RELAXED_MODEL_PATH", "/app/mood_relaxed-audioset-vggish-1.pb")
SAD_MODEL_PATH = os.environ.get("SAD_MODEL_PATH", "/app/mood_sad-audioset-vggish-1.pb")

# --- Energy Normalization Range ---
ENERGY_MIN = float(os.getenv("ENERGY_MIN", "0.01"))
ENERGY_MAX = float(os.getenv("ENERGY_MAX", "0.15"))
OTHER_FEATURE_LABELS = ['danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad']
