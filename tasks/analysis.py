# tasks/analysis.py

import os
import shutil
import requests
from collections import defaultdict
import numpy as np
import json
import time
import random
import logging
import uuid
import traceback

import librosa
import tensorflow.compat.v1 as tf
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import tensor_util
import tensorflow.keras.backend as K # Import Keras backend

tf.disable_v2_behavior() # Necessary for loading frozen graphs

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# RQ import
from rq import get_current_job, Retry
from rq.job import Job
from rq.exceptions import NoSuchJobError

# Import configuration from the user's provided config file
from config import (
    TEMP_DIR, MAX_DISTANCE, MAX_SONGS_PER_CLUSTER, MAX_SONGS_PER_ARTIST,
    GMM_COVARIANCE_TYPE, MOOD_LABELS, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, ENERGY_MIN, ENERGY_MAX,
    TEMPO_MIN_BPM, TEMPO_MAX_BPM, JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, OTHER_FEATURE_LABELS, REDIS_URL, DATABASE_URL,
    OLLAMA_SERVER_URL, OLLAMA_MODEL_NAME, AI_MODEL_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL_NAME,
    DANCEABILITY_MODEL_PATH, AGGRESSIVE_MODEL_PATH, HAPPY_MODEL_PATH, PARTY_MODEL_PATH, RELAXED_MODEL_PATH, SAD_MODEL_PATH,
    SCORE_WEIGHT_SILHOUETTE, SCORE_WEIGHT_DAVIES_BOULDIN, SCORE_WEIGHT_CALINSKI_HARABASZ,
    SCORE_WEIGHT_DIVERSITY, SCORE_WEIGHT_PURITY, SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY, SCORE_WEIGHT_OTHER_FEATURE_PURITY,
    MUTATION_KMEANS_COORD_FRACTION, MUTATION_INT_ABS_DELTA, MUTATION_FLOAT_ABS_DELTA,
    TOP_N_ELITES, EXPLOITATION_START_FRACTION, EXPLOITATION_PROBABILITY_CONFIG, TOP_N_MOODS, TOP_N_OTHER_FEATURES,
    STRATIFIED_GENRES, MIN_SONGS_PER_GENRE_FOR_STRATIFICATION, SAMPLING_PERCENTAGE_CHANGE_PER_RUN, ITERATIONS_PER_BATCH_JOB, MAX_CONCURRENT_BATCH_JOBS, REBUILD_INDEX_BATCH_SIZE,
    MAX_QUEUED_ANALYSIS_JOBS,
    TOP_K_MOODS_FOR_PURITY_CALCULATION, LN_MOOD_DIVERSITY_STATS, LN_MOOD_PURITY_STATS,
    LN_OTHER_FEATURES_DIVERSITY_STATS, LN_OTHER_FEATURES_PURITY_STATS,
    STRATIFIED_SAMPLING_TARGET_PERCENTILE,
    OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY as CONFIG_OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY
)


# Import other project modules
from ai import get_ai_playlist_name, creative_prompt_template
from .commons import score_vector
from .annoy_manager import build_and_store_annoy_index

from psycopg2 import OperationalError
logger = logging.getLogger(__name__)

# --- Tensor Name Definitions ---
# Based on a full review of all error logs and the Essentia examples,
# this is the definitive mapping.
DEFINED_TENSOR_NAMES = {
    # Takes spectrograms, outputs embeddings
    'embedding': {
        'input': 'model/Placeholder:0',
        'output': 'model/dense/BiasAdd:0'
    },
    # Takes embeddings, outputs mood predictions
    'prediction': {
        'input': 'serving_default_model_Placeholder:0',
        'output': 'PartitionedCall:0'
    },
    # Takes a single aggregated embedding, outputs a binary classification
    'danceable': {
        'input': 'model/Placeholder:0',
        'output': 'model/Softmax:0'
    },
    'aggressive': {
        'input': 'model/Placeholder:0',
        'output': 'model/Softmax:0'
    },
    'happy': {
        'input': 'model/Placeholder:0',
        'output': 'model/Softmax:0'
    },
    'party': {
        'input': 'model/Placeholder:0',
        'output': 'model/Softmax:0'
    },
    'relaxed': {
        'input': 'model/Placeholder:0',
        'output': 'model/Softmax:0'
    },
    'sad': {
        'input': 'model/Placeholder:0',
        'output': 'model/Softmax:0'
    }
}

# --- Class Index Mapping ---
# Based on confirmed metadata from the user.
CLASS_INDEX_MAP = {
    "aggressive": 0,
    "happy": 0,
    "relaxed": 1,
    "sad": 1,
    "danceable": 0,
    "party": 1,
}


# --- Utility Functions ---
def clean_temp(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.warning(f"Could not remove {file_path} from {temp_dir}: {e}")

def get_recent_albums(jellyfin_url, jellyfin_user_id, headers, limit):
    url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {"IncludeItemTypes": "MusicAlbum", "SortBy": "DateCreated", "SortOrder": "Descending", "Recursive": True}
    if limit != 0: # A limit of 0 means get all
        params["Limit"] = limit
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"get_recent_albums: {e}", exc_info=True)
        return []

def get_tracks_from_album(jellyfin_url, jellyfin_user_id, headers, album_id):
    url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {"ParentId": album_id, "IncludeItemTypes": "Audio"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("Items", []) if r.ok else []
    except Exception as e:
        logger.error(f"get_tracks_from_album {album_id}: {e}", exc_info=True)
        return []

def download_track(jellyfin_url, headers, temp_dir, item):
    sanitized_track_name = item['Name'].replace('/', '_').replace('\\', '_')
    sanitized_artist_name = item.get('AlbumArtist', 'Unknown').replace('/', '_').replace('\\', '_')
    filename = f"{sanitized_track_name}-{sanitized_artist_name}.mp3"
    path = os.path.join(temp_dir, filename)
    try:
        r = requests.get(f"{jellyfin_url}/Items/{item['Id']}/Download", headers=headers, timeout=120)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        return path
    except Exception as e:
        logger.error(f"download_track {item['Name']}: {e}", exc_info=True)
        return None

# --- Core Analysis Functions ---

def run_inference(session, feed_dict, output_tensor_name):
    """
    Runs inference using the provided session and tensor names.
    Handles multiple input tensors via the feed_dict.
    """
    graph = session.graph
    logger.debug(f"Running inference for output '{output_tensor_name}' with feed_dict keys: {list(feed_dict.keys())}")
    
    final_feed_dict = {}
    for tensor_name, value in feed_dict.items():
        try:
            tensor = graph.get_tensor_by_name(tensor_name)
            final_feed_dict[tensor] = value
        except KeyError:
            logger.error(f"Could not find tensor '{tensor_name}' in the current graph. Skipping.")
            return None

    output_tensor = graph.get_tensor_by_name(output_tensor_name)
    return session.run(output_tensor, feed_dict=final_feed_dict)

def sigmoid(x):
    """Numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-x))

def analyze_track(file_path, mood_labels_list, model_paths):
    """
    Analyzes a single track. This function is now completely self-contained to ensure
    that no TensorFlow state bleeds over between different track analyses.
    """
    K.clear_session()
    logger.info(f"Starting analysis for: {os.path.basename(file_path)}")

    # --- 1. Load Audio and Compute Basic Features ---
    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
    except Exception as e:
        logger.warning(f"Librosa loading error for {os.path.basename(file_path)}: {e}")
        return None, None

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    average_energy = np.mean(librosa.feature.rms(y=audio))
    
    # Improved key/scale detection
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key_vals = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    
    major_correlations = np.array([np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1] for i in range(12)])
    minor_correlations = np.array([np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1] for i in range(12)])

    major_key_idx = np.argmax(major_correlations)
    minor_key_idx = np.argmax(minor_correlations)

    if major_correlations[major_key_idx] > minor_correlations[minor_key_idx]:
        musical_key = key_vals[major_key_idx]
        scale = 'major'
    else:
        musical_key = key_vals[minor_key_idx]
        scale = 'minor'


    # --- 2. Prepare Spectrograms --- 
    try:
        # Using the spectrogram settings confirmed to work for the main model
        n_mels, hop_length, n_fft, frame_size = 96, 256, 512, 187
        #mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=512, hop_length=256, n_mels=96, window='hann', center=True, power=1.0)
        #mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        #mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window='hann')   
        #mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window='hann', center=False, power=1.0, norm='slaney', htk=False)
        #mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window='hann', center=True, power=2.0)
        #mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window='hann', center=False, power=1.0, norm=None, htk=False)

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window='hann', center=False, power=2.0, norm='slaney', htk=False)


        #log_mel_spec = np.log(1 + 10000 * mel_spec)
        log_mel_spec = np.log10(1 + 10000 * mel_spec)

        spec_patches = [log_mel_spec[:, i:i+frame_size] for i in range(0, log_mel_spec.shape[1] - frame_size + 1, frame_size)]
        if not spec_patches:
            logger.warning(f"Track too short to create spectrogram patches: {os.path.basename(file_path)}")
            return None, None
        
        transposed_patches = np.array(spec_patches).transpose(0, 2, 1)
    except Exception as e:
        logger.error(f"Spectrogram creation failed for {os.path.basename(file_path)}: {e}", exc_info=True)
        return None, None

    # --- 3. Run Main Models (Embedding and Prediction) ---
    try:
        # Load and run embedding model
        embedding_graph = tf.Graph()
        with embedding_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_paths['embedding'], "rb") as f:
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        
        with tf.Session(graph=embedding_graph) as sess:
            embedding_feed_dict = {DEFINED_TENSOR_NAMES['embedding']['input']: transposed_patches}
            embeddings_per_patch = run_inference(sess, embedding_feed_dict, DEFINED_TENSOR_NAMES['embedding']['output'])

        # Load and run prediction model
        prediction_graph = tf.Graph()
        with prediction_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_paths['prediction'], "rb") as f:
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        with tf.Session(graph=prediction_graph) as sess:
            prediction_feed_dict = {
                DEFINED_TENSOR_NAMES['prediction']['input']: embeddings_per_patch,
                'saver_filename:0': ''
            }
            mood_predictions_raw = run_inference(sess, prediction_feed_dict, DEFINED_TENSOR_NAMES['prediction']['output'])
            
            if isinstance(mood_predictions_raw, bytes):
                proto = tensor_pb2.TensorProto()
                proto.ParseFromString(mood_predictions_raw)
                mood_logits = tensor_util.MakeNdarray(proto)
            else:
                mood_logits = mood_predictions_raw
            
            averaged_logits = np.mean(mood_logits, axis=0)
            final_mood_predictions = averaged_logits

            moods = {label: float(score) for label, score in zip(mood_labels_list, final_mood_predictions)}

    except Exception as e:
        logger.error(f"Main model inference failed for {os.path.basename(file_path)}: {e}", exc_info=True)
        return None, None
        
    # --- 4. Run Secondary Models ---
    other_predictions = {}

    for key in ["danceable", "aggressive", "happy", "party", "relaxed", "sad"]:
        try:
            other_model_graph = tf.Graph()
            model_path = model_paths[key]
            
            with other_model_graph.as_default():
                graph_def = tf.GraphDef()
                with tf.gfile.GFile(model_path, "rb") as f:
                    graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
            
            with tf.Session(graph=other_model_graph) as sess:
                feed_dict = {DEFINED_TENSOR_NAMES[key]['input']: embeddings_per_patch}
                
                probabilities_raw = run_inference(sess, feed_dict, DEFINED_TENSOR_NAMES[key]['output'])

                if isinstance(probabilities_raw, bytes):
                    proto = tensor_pb2.TensorProto()
                    proto.ParseFromString(probabilities_raw)
                    probabilities_per_patch = tensor_util.MakeNdarray(proto)
                else:
                    probabilities_per_patch = probabilities_raw
                
                if probabilities_per_patch.ndim == 2 and probabilities_per_patch.shape[1] == 2:
                    # Using the CLASS_INDEX_MAP to select the correct probability
                    positive_class_index = CLASS_INDEX_MAP.get(key, 0)
                    class_probs = probabilities_per_patch[:, positive_class_index]
                    other_predictions[key] = float(np.mean(class_probs))
                else:
                    other_predictions[key] = 0.0

        except Exception as e:
            logger.error(f"Error predicting '{key}' for {os.path.basename(file_path)}: {e}", exc_info=True)
            other_predictions[key] = 0.0

    # --- 5. Final Aggregation for Storage ---
    processed_embeddings = np.mean(embeddings_per_patch, axis=0)

    return {
        "tempo": float(tempo), "key": musical_key, "scale": scale,
        "moods": moods, "energy": float(average_energy), **other_predictions
    }, processed_embeddings



# --- RQ Task Definitions ---
def analyze_album_task(album_id, album_name, jellyfin_url, jellyfin_user_id, jellyfin_token, top_n_moods, parent_task_id):
    from app import (app, redis_conn, get_db, save_task_status, get_task_info_from_db,
                     save_track_analysis, save_track_embedding, JobStatus,
                     TASK_STATUS_STARTED, TASK_STATUS_PROGRESS, TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)
    
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        initial_details = {"album_name": album_name, "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Album analysis task started."]}
        save_task_status(current_task_id, "album_analysis", TASK_STATUS_STARTED, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=0, details=initial_details)
        headers = {"X-Emby-Token": jellyfin_token}
        tracks_analyzed_count, tracks_skipped_count, current_progress_val = 0, 0, 0
        current_task_logs = initial_details["log"]
        
        model_paths = {
            'embedding': EMBEDDING_MODEL_PATH,
            'prediction': PREDICTION_MODEL_PATH,
            'danceable': DANCEABILITY_MODEL_PATH,
            'aggressive': AGGRESSIVE_MODEL_PATH,
            'happy': HAPPY_MODEL_PATH,
            'party': PARTY_MODEL_PATH,
            'relaxed': RELAXED_MODEL_PATH,
            'sad': SAD_MODEL_PATH
        }

        def log_and_update_album_task(message, progress, **kwargs):
            nonlocal current_progress_val, current_task_logs
            current_progress_val = progress
            logger.info(f"[AlbumTask-{current_task_id}-{album_name}] {message}")
            db_details = {"album_name": album_name, **kwargs}
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            task_state = kwargs.get('task_state', TASK_STATUS_PROGRESS)

            if task_state in [TASK_STATUS_FAILURE, TASK_STATUS_REVOKED] or task_state != TASK_STATUS_SUCCESS:
                current_task_logs.append(log_entry)
                db_details["log"] = current_task_logs
            else:
                db_details["log"] = [f"Task completed successfully. Final status: {message}"]
            
            if current_job:
                current_job.meta.update({'progress': progress, 'status_message': message})
                current_job.save_meta()
            save_task_status(current_task_id, "album_analysis", task_state, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=progress, details=db_details)

        try:
            log_and_update_album_task(f"Fetching tracks for album: {album_name}", 5)
            tracks = get_tracks_from_album(jellyfin_url, jellyfin_user_id, headers, album_id)
            if not tracks:
                log_and_update_album_task(f"No tracks found for album: {album_name}", 100, task_state=TASK_STATUS_SUCCESS)
                return {"status": "SUCCESS", "message": f"No tracks in album {album_name}", "tracks_analyzed": 0}

            def get_existing_track_ids(track_ids):
                if not track_ids: return set()
                with get_db() as conn, conn.cursor() as cur:
                    cur.execute("SELECT s.item_id FROM score s JOIN embedding e ON s.item_id = e.item_id WHERE s.item_id IN %s AND s.other_features IS NOT NULL AND s.energy IS NOT NULL AND s.mood_vector IS NOT NULL AND s.tempo IS NOT NULL", (tuple(track_ids),))
                    return {row[0] for row in cur.fetchall()}

            existing_track_ids_set = get_existing_track_ids( [t['Id'] for t in tracks])
            total_tracks_in_album = len(tracks)

            for idx, item in enumerate(tracks, 1):
                if current_job:
                    task_info = get_task_info_from_db(current_task_id)
                    parent_info = get_task_info_from_db(parent_task_id) if parent_task_id else None
                    if (task_info and task_info.get('status') == 'REVOKED') or (parent_info and parent_info.get('status') in ['REVOKED', 'FAILURE']):
                        log_and_update_album_task(f"Stopping album analysis for '{album_name}' due to parent/self revocation.", current_progress_val, task_state=TASK_STATUS_REVOKED)
                        return {"status": "REVOKED"}

                track_name_full = f"{item['Name']} by {item.get('AlbumArtist', 'Unknown')}"
                progress = 10 + int(85 * (idx / float(total_tracks_in_album)))
                log_and_update_album_task(f"Analyzing track: {track_name_full} ({idx}/{total_tracks_in_album})", progress, current_track_name=track_name_full)

                if item['Id'] in existing_track_ids_set:
                    tracks_skipped_count += 1
                    continue

                path = download_track(jellyfin_url, headers, TEMP_DIR, item)
                if not path:
                    continue

                try:
                    analysis, embedding = analyze_track(path, MOOD_LABELS, model_paths)
                    if analysis is None:
                        tracks_skipped_count += 1
                        continue
                    
                    top_moods = dict(sorted(analysis['moods'].items(), key=lambda i: i[1], reverse=True)[:top_n_moods])
                    other_features = ",".join([f"{k}:{analysis.get(k, 0.0):.2f}" for k in OTHER_FEATURE_LABELS])
                    
                    logger.info(f"SUCCESSFULLY ANALYZED '{track_name_full}' (ID: {item['Id']}):")
                    logger.info(f"  - Tempo: {analysis['tempo']:.2f}, Energy: {analysis['energy']:.4f}, Key: {analysis['key']} {analysis['scale']}")
                    logger.info(f"  - Top Moods: {top_moods}")
                    logger.info(f"  - Other Features: {other_features}")
                    
                    save_track_analysis(item['Id'], item['Name'], item.get('AlbumArtist', 'Unknown'), analysis['tempo'], analysis['key'], analysis['scale'], top_moods, energy=analysis['energy'], other_features=other_features)
                    save_track_embedding(item['Id'], embedding)
                    
                    logger.info(f"Saved analysis and embedding for track ID {item['Id']} to database.")

                    tracks_analyzed_count += 1
                finally:
                    if path and os.path.exists(path):
                        os.remove(path)

            summary = {"tracks_analyzed": tracks_analyzed_count, "tracks_skipped": tracks_skipped_count, "total_tracks_in_album": total_tracks_in_album}
            log_and_update_album_task(f"Album '{album_name}' analysis complete.", 100, task_state=TASK_STATUS_SUCCESS, final_summary_details=summary)
            return {"status": "SUCCESS", **summary}

        except OperationalError as e:
            logger.error(f"Database connection error during album analysis {album_id}: {e}. This job will be retried.", exc_info=True)
            log_and_update_album_task(f"Database connection failed for album '{album_name}'. Retrying...", current_progress_val, task_state=TASK_STATUS_FAILURE, final_summary_details={"error": str(e), "traceback": traceback.format_exc()})
            # Re-raising the exception is crucial for RQ to trigger the retry mechanism
            raise
        except Exception as e:
            logger.critical(f"Album analysis {album_id} failed: {e}", exc_info=True)
            log_and_update_album_task(f"Failed to analyze album '{album_name}': {e}", current_progress_val, task_state=TASK_STATUS_FAILURE, final_summary_details={"error": str(e), "traceback": traceback.format_exc()})
            raise

def run_analysis_task(jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods):
    from app import (app, redis_conn, get_db, save_task_status, get_task_info_from_db, rq_queue_default,
                     TASK_STATUS_STARTED, TASK_STATUS_PROGRESS, TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())    

    with app.app_context():
        if num_recent_albums < 0:
             logger.warning("num_recent_albums is negative, treating as 0 (all albums).")
             num_recent_albums = 0

        task_info = get_task_info_from_db(current_task_id)
        if task_info and task_info.get('status') in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]:
            return {"status": task_info.get('status'), "message": "Task already in terminal state."}
        
        checked_album_ids = set(json.loads(task_info.get('details', '{}')).get('checked_album_ids', [])) if task_info else set()
        
        initial_details = {"message": "Fetching albums...", "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Main analysis task started."]}

        save_task_status(current_task_id, "main_analysis", TASK_STATUS_STARTED, progress=0, details=initial_details)
        headers = {"X-Emby-Token": jellyfin_token}
        current_progress = 0
        current_task_logs = initial_details["log"]

        def log_and_update_main(message, progress, **kwargs):
            nonlocal current_progress, current_task_logs
            current_progress = progress
            logger.info(f"[MainAnalysisTask-{current_task_id}] {message}")
            details = {**kwargs, "status_message": message}
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            task_state = kwargs.get('task_state', TASK_STATUS_PROGRESS)
            
            if task_state != TASK_STATUS_SUCCESS:
                current_task_logs.append(log_entry)
                details["log"] = current_task_logs
            else:
                details["log"] = [f"Task completed successfully. Final status: {message}"]

            if current_job:
                current_job.meta.update({'progress': progress, 'status_message': message, 'details':details})
                current_job.save_meta()
            save_task_status(current_task_id, "main_analysis", task_state, progress=progress, details=details)

        try:
            log_and_update_main("🚀 Starting main analysis process...", 0)
            clean_temp(TEMP_DIR)
            all_albums = get_recent_albums(jellyfin_url, jellyfin_user_id, headers, num_recent_albums)
            if not all_albums:
                log_and_update_main("⚠️ No new albums to analyze.", 100, albums_found=0, task_state=TASK_STATUS_SUCCESS)
                return {"status": "SUCCESS", "message": "No new albums to analyze."}

            total_albums_to_check = len(all_albums)
            active_jobs, launched_jobs = {}, []
            albums_skipped, albums_launched, albums_completed, last_rebuild_count = 0, 0, 0, 0

            def get_existing_track_ids(track_ids):
                if not track_ids: return set()
                with get_db() as conn, conn.cursor() as cur:
                    cur.execute("SELECT s.item_id FROM score s JOIN embedding e ON s.item_id = e.item_id WHERE s.item_id IN %s AND s.other_features IS NOT NULL AND s.energy IS NOT NULL AND s.mood_vector IS NOT NULL AND s.tempo IS NOT NULL", (tuple(track_ids),))
                    return {row[0] for row in cur.fetchall()}

            def monitor_and_clear_jobs():
                nonlocal albums_completed, last_rebuild_count
                for job_id in list(active_jobs.keys()):
                    try:
                        job = Job.fetch(job_id, connection=redis_conn)
                        if job.is_finished or job.is_failed or job.is_canceled:
                            del active_jobs[job_id]
                            albums_completed += 1
                    except NoSuchJobError:
                        del active_jobs[job_id]
                        albums_completed += 1
                
                if albums_completed > last_rebuild_count and (albums_completed - last_rebuild_count) >= REBUILD_INDEX_BATCH_SIZE:
                    log_and_update_main(f"Batch of {albums_completed - last_rebuild_count} albums complete. Rebuilding index...", current_progress)
                    build_and_store_annoy_index(get_db())
                    redis_conn.publish('index-updates', 'reload')
                    last_rebuild_count = albums_completed

            for idx, album in enumerate(all_albums):
                # Periodically check for completed jobs to update progress
                monitor_and_clear_jobs()

                if album['Id'] in checked_album_ids:
                    albums_skipped += 1
                    continue
                
                while len(active_jobs) >= MAX_QUEUED_ANALYSIS_JOBS:
                    monitor_and_clear_jobs()
                    time.sleep(5)

                tracks = get_tracks_from_album(jellyfin_url, jellyfin_user_id, headers, album['Id'])
                if not tracks or len(get_existing_track_ids([t['Id'] for t in tracks])) >= len(tracks):
                    albums_skipped += 1
                    checked_album_ids.add(album['Id'])
                    continue

                job = rq_queue_default.enqueue('tasks.analysis.analyze_album_task', args=(album['Id'], album['Name'], jellyfin_url, jellyfin_user_id, jellyfin_token, top_n_moods, current_task_id), job_id=str(uuid.uuid4()), job_timeout=-1, retry=Retry(max=3))
                active_jobs[job.id] = job
                launched_jobs.append(job)
                albums_launched += 1
                checked_album_ids.add(album['Id'])
                
                progress = 5 + int(85 * (idx / float(total_albums_to_check)))
                status_message = f"Launched: {albums_launched}. Completed: {albums_completed}/{albums_launched}. Active: {len(active_jobs)}. Skipped: {albums_skipped}/{total_albums_to_check}."
                log_and_update_main(
                    status_message,
                    progress,
                    albums_to_process=albums_launched,
                    albums_skipped=albums_skipped,
                    checked_album_ids=list(checked_album_ids)
                )
                
            while active_jobs:
                monitor_and_clear_jobs()
                progress = 5 + int(85 * ((albums_skipped + albums_completed) / float(total_albums_to_check)))
                status_message = f"Launched: {albums_launched}. Completed: {albums_completed}/{albums_launched}. Active: {len(active_jobs)}. Skipped: {albums_skipped}/{total_albums_to_check}. (Finalizing)"
                log_and_update_main(status_message, progress, checked_album_ids=list(checked_album_ids))
                time.sleep(5)

            log_and_update_main("Performing final index rebuild...", 95)
            build_and_store_annoy_index(get_db())
            redis_conn.publish('index-updates', 'reload')

            final_message = f"Main analysis complete. Launched {albums_launched}, Skipped {albums_skipped}."
            log_and_update_main(final_message, 100, task_state=TASK_STATUS_SUCCESS)
            clean_temp(TEMP_DIR)
            return {"status": "SUCCESS", "message": final_message}

        except OperationalError as e:
            logger.critical(f"FATAL ERROR: Main analysis task failed due to DB connection issue: {e}", exc_info=True)
            log_and_update_main(f"❌ Main analysis failed due to a database connection error. The task may be retried.", current_progress, task_state=TASK_STATUS_FAILURE, error_message=str(e), traceback=traceback.format_exc())
            # Re-raise to allow RQ to handle retries if configured on the task itself
            raise
        except Exception as e:
            logger.critical(f"FATAL ERROR: Analysis failed: {e}", exc_info=True)
            log_and_update_main(f"❌ Main analysis failed: {e}", current_progress, task_state=TASK_STATUS_FAILURE, error_message=str(e), traceback=traceback.format_exc())
            raise
