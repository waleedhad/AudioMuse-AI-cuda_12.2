#Jellyfin and db constant
JELLYFIN_USER_ID = "0e45c44b3e2e4da7a2be11a72a1c8575"
JELLYFIN_URL = "http://jellyfin.192.168.3.131.nip.io:8087"
JELLYFIN_TOKEN = "e0b8c325bc1b426c81922b90c0aa2ff1"
TEMP_DIR = "temp_audio"
DB_PATH = "db.sqlite"

HEADERS = {"X-Emby-Token": JELLYFIN_TOKEN}

#General constant
MAX_DISTANCE = 0.5
MAX_SONGS_PER_CLUSTER = 40
MAX_SONGS_PER_ARTIST = 3
NUM_RECENT_ALBUMS = 0

#Algorithm chose constant
CLUSTER_ALGORITHM = "kmeans" # accepted dbscan or kmeans
PCA_ENABLED = False  # Set False to disable PCA usage.

#dbscan only constant
DBSCAN_EPS = 0.15
DBSCAN_MIN_SAMPLES = 10

#kmeans only constant
NUM_CLUSTERS = 40 #if 0 it automatically defined

#Classifier constant
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
