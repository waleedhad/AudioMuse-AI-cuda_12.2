import os
import json
import time
import logging
import sqlite3
import requests # Re-add requests, it was implicitly removed in my mind but needed for JellyfinClient
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from celery import Celery
from celery.result import AsyncResult
from celery.signals import task_postrun
from datetime import datetime

# Imports for Essentia Analysis (now internal)
import essentia.standard as es
import numpy as np
# Removed librosa and soundfile as per user's request
import tempfile
import shutil # For cleaning up temporary directories
import pathlib # For handling paths

# Imports for Clustering (now internal)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import uuid
import math

# Import your configuration
import config


# --- Flask App Setup ---
app = Flask(__name__, static_folder='.')
CORS(app) # Enable CORS for all origins

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Celery Configuration ---
celery_app = Celery(
    'essentia_tasks',
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND
)
celery_app.conf.update(app.config)


# --- JellyfinClient Class (Integrated) ---
class JellyfinClient:
    def __init__(self, jellyfin_url, user_id, api_token):
        self.jellyfin_url = jellyfin_url
        self.headers = {
            "X-Emby-Token": api_token,
            "X-Emby-User-ID": user_id,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.user_id = user_id
        logger.info(f"JellyfinClient initialized for URL: {self.jellyfin_url} and User ID: {self.user_id}")

    def _make_request(self, endpoint, method="GET", params=None, data=None):
        url = f"{self.jellyfin_url}/{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, params=params, json=data)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e.response.status_code} - {e.response.text}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {url}: {e}")
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error for {url}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {url}: {e} - Response: {response.text}")
            raise

    def get_user_views(self):
        return self._make_request(f"Users/{self.user_id}/Views")

    def get_latest_albums(self, limit=config.NUM_RECENT_ALBUMS):
        # Adjust limit based on config, 0 means no limit (get all)
        if limit == 0:
            params = {"Recursive": "true", "IncludeItemTypes": "MusicAlbum", "SortBy": "DateCreated", "SortOrder": "Descending"}
        else:
            params = {"Recursive": "true", "IncludeItemTypes": "MusicAlbum", "SortBy": "DateCreated", "SortOrder": "Descending", "Limit": limit}
        
        # Discover the library ID for Music
        views_data = self.get_user_views()
        music_library_id = None
        for item in views_data.get('Items', []):
            if item.get('CollectionType') == 'music':
                music_library_id = item['Id']
                logger.info(f"Found Music Library ID: {music_library_id}")
                break

        if not music_library_id:
            logger.warning("No music library found for the user.")
            return [] # Return empty if no music library found

        return self._make_request(f"Users/{self.user_id}/Items", params={"ParentId": music_library_id, **params})

    def get_album_tracks(self, album_id):
        return self._make_request(f"Albums/{album_id}/Items")

    def get_track_stream_url(self, track_id):
        # Assumes direct access for simplicity, adjust for transcoding if needed
        return f"{self.jellyfin_url}/Audio/{track_id}/stream.mp3?static=true"


# --- EssentiaAnalyzer Class (Integrated) ---
class EssentiaAnalyzer:
    def __init__(self, jellyfin_url, jellyfin_user_id, jellyfin_token, db_manager, temp_dir, model_path, prediction_model_path, mood_labels=None):
        self.jellyfin_client = JellyfinClient(jellyfin_url, jellyfin_user_id, jellyfin_token)
        self.db_manager = db_manager
        self.temp_dir = pathlib.Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True) # Ensure temp directory exists

        # Load models
        self.musicnn_embeddings_model = es.TensorflowPredictMusiCNN(graph=model_path, output="activations")
        self.musicnn_prediction_model = es.TensorflowPredictMusiCNN(graph=prediction_model_path, output="predictions")
        logger.info(f"Essentia models loaded: {model_path}, {prediction_model_path}")

        self.num_recent_albums = config.NUM_RECENT_ALBUMS # Default from config
        self.top_n_moods = config.TOP_N_MOODS # Default from config

        # Essentia configuration (these can be tuned)
        self.frame_size = 2048
        self.hop_size = 1024
        self.sample_rate = 44100 # Standard sample rate for audio analysis

        # Placeholder for mood labels; ideally loaded from a mapping file or passed
        if mood_labels is None:
            # Example mood labels based on a common MusicNN output
            self.mood_labels = [
                'acoustic', 'aggresive', 'ambient', 'chilled', 'dark', 'epic', 'funky', 'groovy',
                'happy', 'intense', 'light', 'melancholic', 'party', 'relaxing', 'romantic',
                'sad', 'sexy', 'uplifting', 'energetic', 'calm'
            ]
        else:
            self.mood_labels = mood_labels


    def _download_audio(self, track_id, output_path):
        stream_url = self.jellyfin_client.get_track_stream_url(track_id)
        logger.info(f"Attempting to download audio from: {stream_url} to {output_path}")
        try:
            response = requests.get(stream_url, stream=True)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Successfully downloaded {track_id} to {output_path}")
            return output_path
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {track_id}: {e}")
            raise

    def _analyze_audio(self, audio_path):
        # *** ONLY USING ESSENTIA'S MONOLOADER AS REQUESTED ***
        try:
            audio = es.MonoLoader(filename=audio_path, sampleRate=self.sample_rate)()
            if audio.size == 0:
                raise ValueError(f"Essentia MonoLoader loaded empty audio from {audio_path}")
        except Exception as e:
            logger.error(f"Failed to load audio with Essentia MonoLoader: {e}", exc_info=True)
            raise ValueError(f"Could not load audio file {audio_path} using Essentia: {e}")


        # Feature Extraction
        # Danceability, Energy, Valence (from Low-Level features)
        rhythm_extractor = es.RhythmExtractor2013(frameSize=self.frame_size, hopSize=self.hop_size)
        rhythm = rhythm_extractor(audio)
        danceability = rhythm[0] # BPM
        # For energy and valence, we might need a more direct psychoacoustic model or use aggregated spectral features
        # For simplicity, let's use aggregated spectral centroid/energy for a proxy
        frame_cutter = es.FrameCutter(frameSize=self.frame_size, hopSize=self.hop_size)
        window = es.Windowing(type="blackmanharris62")
        fft = es.FFT()
        spectrum = es.Spectrum()
        spectral_centroid = es.SpectralCentroid()
        energy_feature = es.Energy()

        all_energies = []
        all_spectral_centroids = []
        for frame in es.FrameGenerator(audio, frameSize=self.frame_size, hopSize=self.hop_size):
            windowed_frame = window(frame)
            mag_spectrum = spectrum(fft(windowed_frame))
            all_energies.append(energy_feature(windowed_frame))
            all_spectral_centroids.append(spectral_centroid(mag_spectrum))

        # Simple average of energy and centroid as proxies for energy/valence
        avg_energy = np.mean(all_energies) if all_energies else 0.0
        avg_spectral_centroid = np.mean(all_spectral_centroids) if all_spectral_centroids else 0.0

        # Normalize energy and centroid to a 0-1 range for a rough proxy (not true perceptual values)
        # These normalization ranges are heuristic; proper ranges would come from a larger dataset
        normalized_energy = np.clip(avg_energy / 0.1, 0, 1) # Assuming max energy around 0.1
        normalized_valence = np.clip((avg_spectral_centroid - 1000) / 4000, 0, 1) # Assuming centroid 1000-5000

        # Essentia Features: Chroma, MFCC, Spectral Contrast, Tonnetz
        # For consistency and simplicity, let's use the aggregated features from Essentia's pre-trained models or common pipelines
        # Using a fixed-size descriptor approach for overall features suitable for clustering.

        # For overall_features, let's create a combined feature vector:
        # MusicNN embeddings provide a good high-level representation
        embeddings = self.musicnn_embeddings_model(audio)
        # Average embeddings across time to get a single vector per track
        overall_features = np.mean(embeddings, axis=0) if embeddings.shape[0] > 0 else np.zeros(50) # MusicNN outputs 50-dim features

        # Mood Prediction using MusicNN's prediction model
        predictions = self.musicnn_prediction_model(audio) # Raw predictions from the prediction model
        # Average predictions over time
        avg_predictions = np.mean(predictions, axis=0) if predictions.shape[0] > 0 else np.zeros(len(self.mood_labels))
        
        # Get top N moods
        top_mood_indices = avg_predictions.argsort()[-self.top_n_moods:][::-1]
        top_moods_with_scores = [
            {'mood': self.mood_labels[i], 'score': float(avg_predictions[i])}
            for i in top_mood_indices
        ]
        
        # Standard Essentia features (for more granular control if needed)
        # Using aggregated values for simplicity.
        try:
            mfcc = es.MFCC(numberCoefficients=13)(es.Spectrum(es.FFT()(es.Windowing(type="blackmanharris62")(audio))))
            avg_mfcc = np.mean(mfcc, axis=0) if mfcc.shape[0] > 0 else np.zeros(13)
        except Exception:
            avg_mfcc = np.zeros(13) # Fallback if audio too short or problem

        try:
            spectral_contrast = es.SpectralContrast()(es.Spectrum(es.FFT()(es.Windowing(type="blackmanharris62")(audio))))
            avg_spectral_contrast = np.mean(spectral_contrast, axis=0) if spectral_contrast.shape[0] > 0 else np.zeros(7) # 7 bands
        except Exception:
            avg_spectral_contrast = np.zeros(7)

        try:
            tonnetz = es.TonalExtractor(frameSize=self.frame_size, hopSize=self.hop_size)(audio)
            avg_tonnetz = np.mean(tonnetz[0], axis=0) if tonnetz[0].shape[0] > 0 else np.zeros(6) # 6 coefficients
        except Exception:
            avg_tonnetz = np.zeros(6)

        try:
            chroma = es.ChromaExtractor(frameSize=self.frame_size, hopSize=self.hop_size)(audio)
            avg_chroma = np.mean(chroma[0], axis=0) if chroma[0].shape[0] > 0 else np.zeros(12) # 12 pitches
        except Exception:
            avg_chroma = np.zeros(12)

        # Concatenate all features into a single, comprehensive vector for clustering
        # Ensure all components are 1-D arrays
        combined_features = np.concatenate([
            overall_features,
            np.array([danceability, normalized_energy, normalized_valence]),
            avg_mfcc,
            avg_spectral_contrast,
            avg_tonnetz,
            avg_chroma
        ]).tolist() # Convert to list for JSON serialization


        return {
            "danceability": float(danceability),
            "energy": float(normalized_energy),
            "valence": float(normalized_valence),
            "moods": top_moods_with_scores,
            "chroma_features": avg_chroma.tolist(),
            "mfcc_features": avg_mfcc.tolist(),
            "spectral_contrast_features": avg_spectral_contrast.tolist(),
            "tonnetz_features": avg_tonnetz.tolist(),
            "overall_features": combined_features # This will be the primary vector for clustering
        }

    def run_analysis(self, progress_callback=None):
        logger.info(f"Starting analysis for {self.num_recent_albums if self.num_recent_albums > 0 else 'all'} recent albums...")
        try:
            albums_data = self.jellyfin_client.get_latest_albums(limit=self.num_recent_albums)
            if not albums_data or not albums_data.get('Items'):
                logger.warning("No albums found for analysis.")
                if progress_callback:
                    progress_callback("No albums found for analysis.", 0, 0, 100)
                return

            albums = albums_data['Items']
            total_albums = len(albums)
            total_tracks_processed = 0
            all_tracks_data = []

            for i, album in enumerate(albums):
                album_title = album.get('Name', 'Unknown Album')
                album_id = album.get('Id')
                
                # Update progress per album
                progress_percent = (i / total_albums) * 100
                if progress_callback:
                    progress_callback(f"Analyzing album: {album_title}", i + 1, total_albums, progress_percent)
                logger.info(f"Processing album {i+1}/{total_albums}: {album_title} (ID: {album_id})")

                tracks_in_album = self.jellyfin_client.get_album_tracks(album_id)
                if not tracks_in_album or not tracks_in_album.get('Items'):
                    logger.info(f"No tracks found for album {album_title}.")
                    continue

                for track in tracks_in_album['Items']:
                    track_id = track.get('Id')
                    track_title = track.get('Name', 'Unknown Track')
                    track_artist = track.get('ArtistItems', [{}])[0].get('Name', 'Unknown Artist')

                    logger.info(f"  Processing track: {track_title} by {track_artist} (ID: {track_id})")

                    temp_audio_file = self.temp_dir / f"{track_id}.mp3"
                    try:
                        self._download_audio(track_id, temp_audio_file)
                        features = self._analyze_audio(str(temp_audio_file))
                        track_data = {
                            "id": track_id,
                            "title": track_title,
                            "author": track_artist,
                            "album_id": album_id,
                            "album_name": album_title,
                            **features
                        }
                        self.db_manager.insert_track(track_data)
                        all_tracks_data.append(track_data)
                        total_tracks_processed += 1
                        logger.info(f"  Analyzed and saved features for track: {track_title}")
                    except Exception as e:
                        logger.error(f"Failed to analyze track {track_title} (ID: {track_id}): {e}")
                    finally:
                        if temp_audio_file.exists():
                            os.remove(temp_audio_file)
                            logger.debug(f"Removed temporary file: {temp_audio_file}")

            if progress_callback:
                progress_callback(f"Analysis complete. Processed {total_tracks_processed} tracks.", total_albums, total_albums, 100)
            logger.info(f"Analysis finished. Total tracks processed: {total_tracks_processed}")

        except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)
            if progress_callback:
                progress_callback(f"Analysis failed: {e}", 0, 0, 0)
            raise
        finally:
            # Clean up the entire temp directory after analysis
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")


# --- SQLite Database Management for Main Data ---
# This remains a class as it handles complex data structures
class DbManager:
    def __init__(self, db_path="db.sqlite"):
        self.db_path = db_path
        self.conn = None

    def _get_conn(self):
        if self.conn is None or not self._is_conn_valid():
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row # Allows accessing columns by name
        return self.conn

    def _is_conn_valid(self):
        try:
            # Try to execute a simple query to check connection validity
            self.conn.execute("SELECT 1")
            return True
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            return False

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def create_tables(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
                id TEXT PRIMARY KEY,
                title TEXT,
                author TEXT,
                album_id TEXT,
                album_name TEXT,
                danceability REAL,
                energy REAL,
                valence REAL,
                moods TEXT,
                chroma_features TEXT,
                mfcc_features TEXT,
                spectral_contrast_features TEXT,
                tonnetz_features TEXT,
                overall_features TEXT,
                playlist_id TEXT,
                FOREIGN KEY (playlist_id) REFERENCES playlists(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS playlists (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE,
                description TEXT,
                created_at TEXT
            )
        ''')
        conn.commit()
        self.close()

    def insert_track(self, track_data):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO tracks (
                id, title, author, album_id, album_name, danceability, energy, valence, moods,
                chroma_features, mfcc_features, spectral_contrast_features, tonnetz_features, overall_features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            track_data['id'], track_data['title'], track_data['author'], track_data['album_id'], track_data['album_name'],
            track_data['danceability'], track_data['energy'], track_data['valence'], json.dumps(track_data['moods']),
            json.dumps(track_data['chroma_features']), json.dumps(track_data['mfcc_features']),
            json.dumps(track_data['spectral_contrast_features']), json.dumps(track_data['tonnetz_features']),
            json.dumps(track_data['overall_features'])
        ))
        conn.commit()
        self.close()

    def get_all_tracks_with_features(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, title, author, album_name, overall_features FROM tracks
        ''')
        rows = cursor.fetchall()
        self.close()
        tracks = []
        for row in rows:
            track = dict(row)
            track['overall_features'] = json.loads(track['overall_features'])
            tracks.append(track)
        return tracks

    def insert_playlist(self, playlist_id, name, description):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO playlists (id, name, description, created_at)
            VALUES (?, ?, ?, ?)
        ''', (playlist_id, name, description, datetime.now().isoformat()))
        conn.commit()
        self.close()

    def update_track_playlist_id(self, track_id, playlist_id):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE tracks SET playlist_id = ? WHERE id = ?
        ''', (playlist_id, track_id))
        conn.commit()
        self.close()

    def get_all_playlists(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT id, name FROM playlists ORDER BY created_at DESC')
        playlists_data = cursor.fetchall()

        playlists = {}
        for p_data in playlists_data:
            playlist_id = p_data['id']
            playlist_name = p_data['name']
            cursor.execute('SELECT title, author FROM tracks WHERE playlist_id = ?', (playlist_id,))
            tracks_data = cursor.fetchall()
            playlists[playlist_name] = [dict(t) for t in tracks_data]
        self.close()
        return playlists

    def clear_all_data(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM tracks')
        cursor.execute('DELETE FROM playlists')
        conn.commit()
        self.close()

# --- Initialize Main DB Manager ---
db_manager = DbManager(config.DB_PATH)

# --- SQLite Database Management for Task Status (Directly in app.py) ---
def _get_status_db_conn():
    conn = sqlite3.connect(config.STATUS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_status_table():
    conn = _get_status_db_conn()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS task_status (
            task_id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            status TEXT NOT NULL,
            progress REAL,
            current_album TEXT,       -- Re-used for current step message in clustering
            current_album_idx INTEGER, -- Re-used for current step index in clustering
            total_albums INTEGER,      -- Re-used for total steps in clustering
            log_output TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_task_status(task_id, task_type, status, progress, current_album, current_album_idx=0, total_albums=0, log_output='', timestamp=''):
    conn = _get_status_db_conn()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO task_status
        (task_id, task_type, status, progress, current_album, current_album_idx, total_albums, log_output, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (task_id, task_type, status, progress, current_album, current_album_idx, total_albums, log_output, timestamp or datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_task_status_from_db(task_id):
    conn = _get_status_db_conn()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM task_status WHERE task_id = ?', (task_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_last_task_status_from_db(task_type):
    conn = _get_status_db_conn()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM task_status WHERE task_type = ? ORDER BY timestamp DESC LIMIT 1', (task_type,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# --- ClusterGenerator Class (Integrated) ---
class ClusterGenerator:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        # Default values from config
        self.cluster_algorithm = config.CLUSTER_ALGORITHM
        self.num_clusters = config.NUM_CLUSTERS
        self.dbscan_eps = config.DBSCAN_EPS
        self.dbscan_min_samples = config.DBSCAN_MIN_SAMPLES
        self.pca_enabled = config.PCA_ENABLED
        self.pca_components = config.PCA_COMPONENTS
        self.clustering_runs = config.CLUSTERING_RUNS

    def generate_and_save_playlists(self, progress_callback=None):
        logger.info("Starting playlist generation...")
        tracks = self.db_manager.get_all_tracks_with_features()

        if not tracks:
            raise ValueError("No tracks with features found in the database. Please run analysis first.")

        # Extract features and track info
        track_ids = [t['id'] for t in tracks]
        track_info = {t['id']: {'title': t['title'], 'author': t['author'], 'album_name': t['album_name']} for t in tracks}
        features = np.array([t['overall_features'] for t in tracks])

        if features.size == 0 or features.shape[1] == 0:
            raise ValueError("Extracted features are empty or malformed. Cannot perform clustering.")

        # Determine total steps for progress bar
        # 5% for Standardizing, 85% for Clustering Runs, 10% for Saving Playlists
        total_clustering_runs_for_progress = self.clustering_runs if self.cluster_algorithm == 'kmeans' else 1
        # Steps: 1. Standardizing/PCA, 2. Clustering runs, 3. Saving Playlists
        total_steps_overall = 2 + total_clustering_runs_for_progress


        # Standardize features (Step 1 of overall)
        current_step_idx = 1
        current_progress_percent = (current_step_idx / total_steps_overall) * 100
        if progress_callback:
            progress_callback("Standardizing features...", current_step_idx, total_steps_overall, current_progress_percent)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        logger.info(f"Scaled features shape: {scaled_features.shape}")

        # Apply PCA if enabled (Part of Step 1 or a sub-step of it)
        if self.pca_enabled:
            n_components_actual = min(self.pca_components, scaled_features.shape[1])
            if n_components_actual <= 0:
                logger.warning("PCA components set to 0 or less, or insufficient features. PCA will be skipped.")
                pca_features = scaled_features
                if progress_callback:
                    progress_callback("PCA skipped.", current_step_idx, total_steps_overall, current_progress_percent)
            else:
                pca = PCA(n_components=n_components_actual)
                pca_features = pca.fit_transform(scaled_features)
                logger.info(f"PCA applied. Features reduced to {n_components_actual} components. Shape: {pca_features.shape}")
                if progress_callback:
                    progress_callback(f"PCA applied: features reduced to {n_components_actual} components.", current_step_idx, total_steps_overall, current_progress_percent)
        else:
            pca_features = scaled_features
            if progress_callback:
                progress_callback("PCA is disabled.", current_step_idx, total_steps_overall, current_progress_percent)


        best_labels = None
        best_score = -float('inf') # For KMeans, higher score is better (e.g., silhouette)

        if self.cluster_algorithm == 'kmeans':
            logger.info(f"Using KMeans clustering with {self.num_clusters} clusters and {self.clustering_runs} runs.")
            if self.num_clusters == 0:
                logger.info("num_clusters is 0. Using a default of 20 for KMeans.")
                self.num_clusters = 20

            if self.num_clusters > len(pca_features) and len(pca_features) > 0:
                logger.warning(f"Number of clusters ({self.num_clusters}) is greater than the number of tracks ({len(pca_features)}). Adjusting to number of tracks.")
                self.num_clusters = len(pca_features)
            elif len(pca_features) == 0:
                raise ValueError("No features available to cluster.")

            for run_idx in range(self.clustering_runs):
                # Calculate progress for each KMeans run.
                # Base progress (from standardization/PCA) + (proportion of clustering runs) * (percentage for clustering)
                clustering_start_percentage = (1 / total_steps_overall)
                clustering_end_percentage = ( (total_steps_overall - 1) / total_steps_overall) # Roughly, before save step
                progress_range_for_clustering = clustering_end_percentage - clustering_start_percentage

                # Current step progress within the clustering phase
                progress_within_clustering_phase = (run_idx + 1) / total_clustering_runs_for_progress
                current_progress_percent = (clustering_start_percentage + progress_within_clustering_phase * progress_range_for_clustering) * 100

                if progress_callback:
                    progress_callback(f"KMeans run {run_idx + 1}/{self.clustering_runs}...", run_idx + 1, total_clustering_runs_for_progress, current_progress_percent)

                # n_init='auto' or n_init=10 is good default for robustness against random centroid initialization.
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=run_idx, n_init='auto')
                labels = kmeans.fit_predict(pca_features)

                # TODO: Implement silhouette score comparison if you want to truly pick the 'best' run
                # from sklearn.metrics import silhouette_score
                # if len(set(labels)) > 1: # Silhouette requires at least 2 clusters
                #     score = silhouette_score(pca_features, labels)
                #     if score > best_score:
                #         best_score = score
                #         best_labels = labels
                # else:
                #     # If only one cluster, no score, or assign if it's the first run
                #     if best_labels is None:
                #         best_labels = labels

                # For now, if not tracking best score, just assign the last run's labels
                best_labels = labels


        elif self.cluster_algorithm == 'dbscan':
            logger.info(f"Using DBSCAN clustering with eps={self.dbscan_eps}, min_samples={self.dbscan_min_samples}.")
            # DBSCAN is a single run, so its progress is fixed within the clustering phase.
            clustering_start_percentage = (1 / total_steps_overall)
            clustering_end_percentage = ( (total_steps_overall - 1) / total_steps_overall)
            current_progress_percent = (clustering_start_percentage + (1/total_clustering_runs_for_progress) * (clustering_end_percentage - clustering_start_percentage)) * 100

            if progress_callback:
                progress_callback("Running DBSCAN...", 1, 1, current_progress_percent)

            dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
            best_labels = dbscan.fit_predict(pca_features)
            logger.info(f"DBSCAN found {len(set(best_labels)) - (1 if -1 in best_labels else 0)} clusters.")

        else:
            raise ValueError(f"Unknown clustering algorithm: {self.cluster_algorithm}")

        if best_labels is None:
             # This should ideally not happen if tracks are present, but as a safeguard
             best_labels = np.array([-1] * len(tracks)) # All noise or unclustered

        # Save playlists in DB (Final step of overall)
        self.db_manager.clear_all_data() # Clear existing playlists and tracks (optional, depends on desired behavior)

        clusters = {}
        for i, label in enumerate(best_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(track_ids[i])

        if -1 in clusters:
            noise_tracks = clusters.pop(-1)
            logger.info(f"DBSCAN identified {len(noise_tracks)} noise tracks.")
            if progress_callback:
                # This message is part of the final saving phase
                progress_callback(f"Identified {len(noise_tracks)} noise tracks.", total_steps_overall, total_steps_overall, 95) # Just before 100%


        for cluster_id, track_ids_in_cluster in clusters.items():
            playlist_name = f"Cluster {cluster_id}"
            playlist_description = f"Auto-generated playlist for cluster {cluster_id}"
            playlist_uuid = str(uuid.uuid4())
            self.db_manager.insert_playlist(playlist_uuid, playlist_name, playlist_description)
            for track_id in track_ids_in_cluster:
                self.db_manager.update_track_playlist_id(track_id, playlist_uuid)
            logger.info(f"Created playlist '{playlist_name}' with {len(track_ids_in_cluster)} tracks.")

        final_progress_step_idx = total_steps_overall
        final_progress_percent = 100
        if progress_callback:
            progress_callback("Playlists saved successfully!", final_progress_step_idx, total_steps_overall, final_progress_percent)

        logger.info("Playlist generation complete.")
        return True


# --- Initialize Essentia Analyzer and Cluster Generator ---
# Initialize JellyfinClient first, then pass an instance to EssentiaAnalyzer
jellyfin_client_instance = JellyfinClient(
    jellyfin_url=config.JELLYFIN_URL,
    user_id=config.JELLYFIN_USER_ID,
    api_token=config.JELLYFIN_TOKEN
)

essentia_analyzer = EssentiaAnalyzer(
    jellyfin_url=config.JELLYFIN_URL, # These are just passed to the internal JellyfinClient constructor
    jellyfin_user_id=config.JELLYFIN_USER_ID, # The class now contains the JellyfinClient logic
    jellyfin_token=config.JELLYFIN_TOKEN,
    db_manager=db_manager,
    temp_dir=config.TEMP_DIR,
    model_path=config.EMBEDDING_MODEL_PATH,
    prediction_model_path=config.PREDICTION_MODEL_PATH,
    # config.MOOD_LABELS was not defined, assume it's from essentia_analyzer or remove if not needed
    # mood_labels=config.MOOD_LABELS
)
cluster_generator = ClusterGenerator(db_manager=db_manager)


# --- Celery Task Status Storage ---
@task_postrun.connect
def save_task_status_postrun(sender=None, task_id=None, state=None, retval=None, **kwargs):
    """
    Signal handler to save task status after a task has run.
    This ensures that even if the app process restarts, we have the last known state.
    """
    logger.info(f"Task {task_id} finished with state {state}")
    meta = {}
    if isinstance(retval, dict) and 'meta' in retval:
        meta = retval['meta'] # If task returns structured result, capture its meta
    elif isinstance(retval, Exception):
        meta['error'] = str(retval)

    task_type = 'analysis' if sender.name == 'app.start_analysis_task' else 'clustering'

    save_task_status(
        task_id=task_id,
        task_type=task_type,
        status=state,
        progress=meta.get('progress', 0),
        current_album=meta.get('current_album', meta.get('current_step', '')), # Prioritize current_album, fallback to current_step
        current_album_idx=meta.get('current_album_idx', meta.get('current_step_idx', 0)),
        total_albums=meta.get('total_albums', meta.get('total_steps', 0)),
        log_output='\n'.join(meta.get('log_output', [])),
        timestamp=datetime.now().isoformat()
    )


# --- Celery Tasks ---

@celery_app.task(bind=True, name='app.start_analysis_task')
def start_analysis_task(self, jellyfin_config):
    logger.info(f"Analysis task {self.request.id} started with config: {jellyfin_config}")
    log_messages = []

    def update_frontend_progress(msg, current_album="", current_album_idx=0, total_albums=0, progress_percent=0):
        log_messages.append(msg)
        self.update_state(
            state='PROGRESS',
            meta={
                'progress': progress_percent,
                'current_album': current_album,
                'current_album_idx': current_album_idx,
                'total_albums': total_albums,
                'log_output': log_messages,
                'status': msg # User-friendly status
            }
        )
        logger.info(f"Task {self.request.id} progress: {progress_percent}% - {msg}")

    try:
        # Update parameters on the analyzer instance
        essentia_analyzer.num_recent_albums = jellyfin_config.get('num_recent_albums', config.NUM_RECENT_ALBUMS)
        essentia_analyzer.top_n_moods = jellyfin_config.get('top_n_moods', config.TOP_N_MOODS)
        # Update the jellyfin_client within the essentia_analyzer if its config changed
        essentia_analyzer.jellyfin_client.jellyfin_url = jellyfin_config['jellyfin_url']
        essentia_analyzer.jellyfin_client.user_id = jellyfin_config['jellyfin_user_id']
        essentia_analyzer.jellyfin_client.headers["X-Emby-User-ID"] = jellyfin_config['jellyfin_user_id']
        essentia_analyzer.jellyfin_client.headers["X-Emby-Token"] = jellyfin_config['jellyfin_token']


        essentia_analyzer.run_analysis(update_frontend_progress)

        update_frontend_progress("Analysis complete!", progress_percent=100)
        return {'status': 'Analysis completed successfully!', 'progress': 100, 'state': 'SUCCESS', 'meta': {'log_output': log_messages}}

    except Exception as e:
        logger.error(f"Analysis task {self.request.id} failed: {e}", exc_info=True)
        update_frontend_progress(f"Analysis failed: {e}", progress_percent=0)
        self.update_state(
            state='FAILURE',
            meta={'status': f"Analysis failed: {e}", 'progress': 0, 'log_output': log_messages}
        )
        return {'status': f"Analysis failed: {e}", 'progress': 0, 'state': 'FAILURE', 'meta': {'log_output': log_messages}}


@celery_app.task(bind=True, name='app.start_clustering_task')
def start_clustering_task(self, clustering_params):
    logger.info(f"Clustering task {self.request.id} started with params: {clustering_params}")
    log_messages = []

    def update_clustering_progress(msg, current_step_idx=0, total_steps=0, progress_percent=0):
        log_messages.append(msg)
        self.update_state(
            state='PROGRESS',
            meta={
                'progress': progress_percent,
                'current_step': msg,
                'current_step_idx': current_step_idx,
                'total_steps': total_steps,
                'log_output': log_messages,
                'status': msg # User-friendly status
            }
        )
        logger.info(f"Task {self.request.id} progress: {progress_percent}% - {msg}")

    try:
        # Update clustering parameters on the instance from the request
        cluster_generator.cluster_algorithm = clustering_params.get('clustering_method', config.CLUSTER_ALGORITHM)
        cluster_generator.num_clusters = clustering_params.get('num_clusters', config.NUM_CLUSTERS)
        cluster_generator.dbscan_eps = clustering_params.get('dbscan_eps', config.DBSCAN_EPS)
        cluster_generator.dbscan_min_samples = clustering_params.get('dbscan_min_samples', config.DBSCAN_MIN_SAMPLES)
        cluster_generator.pca_enabled = clustering_params.get('pca_enabled', config.PCA_ENABLED)
        cluster_generator.pca_components = clustering_params.get('pca_components', config.PCA_COMPONENTS)
        cluster_generator.clustering_runs = clustering_params.get('clustering_runs', config.CLUSTERING_RUNS)

        # Pass the progress callback to the clustering logic
        cluster_generator.generate_and_save_playlists(update_clustering_progress)

        update_clustering_progress("Clustering complete!", current_step_idx=100, total_steps=100, progress_percent=100)
        return {'status': 'Clustering completed successfully!', 'progress': 100, 'state': 'SUCCESS', 'meta': {'log_output': log_messages}}

    except Exception as e:
        logger.error(f"Clustering task {self.request.id} failed: {e}", exc_info=True)
        update_clustering_progress(f"Clustering failed: {e}", progress_percent=0)
        self.update_state(
            state='FAILURE',
            meta={'status': f"Clustering failed: {e}", 'progress': 0, 'log_output': log_messages}
        )
        return {'status': f"Clustering failed: {e}", 'progress': 0, 'state': 'FAILURE', 'meta': {'log_output': log_messages}}


# --- Flask Routes ---

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    # Return relevant config values to the frontend
    return jsonify({
        "jellyfin_url": config.JELLYFIN_URL,
        "jellyfin_user_id": config.JELLYFIN_USER_ID,
        "jellyfin_token": config.JELLYFIN_TOKEN,
        "num_recent_albums": config.NUM_RECENT_ALBUMS,
        "top_n_moods": config.TOP_N_MOODS,
        "cluster_algorithm": config.CLUSTER_ALGORITHM,
        "num_clusters": config.NUM_CLUSTERS,
        "dbscan_eps": config.DBSCAN_EPS,
        "dbscan_min_samples": config.DBSCAN_MIN_SAMPLES,
        "pca_enabled": config.PCA_ENABLED,
        "pca_components": config.PCA_COMPONENTS, # Include PCA components
        "clustering_runs": config.CLUSTERING_RUNS
    })

@app.route('/api/analysis/start', methods=['POST'])
def start_analysis():
    data = request.json
    task = start_analysis_task.delay(data)
    # Save initial task status
    save_task_status(
        task_id=task.id,
        task_type='analysis',
        status='PENDING',
        progress=0,
        current_album='Starting...',
        timestamp=datetime.now().isoformat()
    )
    return jsonify({"task_id": task.id, "message": "Analysis started."})

@app.route('/api/analysis/status/<task_id>', methods=['GET'])
def get_analysis_status(task_id):
    task = AsyncResult(task_id, app=celery_app)
    response = {}
    if task.state == 'PENDING':
        # Try to get from DB first, then default
        db_status = get_task_status_from_db(task_id)
        response = {
            'task_id': task.id,
            'state': db_status['status'] if db_status else task.state,
            'status': db_status['current_album'] if db_status else 'Pending...',
            'progress': db_status['progress'] if db_status else 0,
            'current_album': db_status['current_album'] if db_status else 'Waiting for worker',
            'current_album_idx': db_status['current_album_idx'] if db_status else 0,
            'total_albums': db_status['total_albums'] if db_status else 0,
            'log_output': db_status['log_output'].split('\n') if db_status and db_status['log_output'] else []
        }
    elif task.state == 'PROGRESS':
        response = {
            'task_id': task.id,
            'state': task.state,
            'status': task.info.get('status', 'In Progress'),
            'progress': task.info.get('progress', 0),
            'current_album': task.info.get('current_album', 'N/A'),
            'current_album_idx': task.info.get('current_album_idx', 0),
            'total_albums': task.info.get('total_albums', 0),
            'log_output': task.info.get('log_output', [])
        }
    elif task.state in ['SUCCESS', 'FAILURE', 'REVOKED']:
        # For completed tasks, retrieve final status from DB for robustness
        db_status = get_task_status_from_db(task_id)
        if db_status:
            response = {
                'task_id': db_status['task_id'],
                'state': db_status['status'],
                'status': db_status['current_album'] or db_status['status'],
                'progress': db_status['progress'],
                'current_album': db_status['current_album'],
                'current_album_idx': db_status['current_album_idx'],
                'total_albums': db_status['total_albums'],
                'log_output': db_status['log_output'].split('\n') if db_status['log_output'] else []
            }
        else:
            # Fallback if not in DB (shouldn't happen with task_postrun)
            response = {
                'task_id': task.id,
                'state': task.state,
                'status': f"Task {task.state}",
                'progress': 100 if task.state == 'SUCCESS' else 0,
                'current_album': 'Finished',
                'current_album_idx': 0,
                'total_albums': 0,
                'log_output': [str(task.info)]
            }
    else:
        # For other states like STARTED (before first update), or if result isn't available
        db_status = get_task_status_from_db(task_id)
        if db_status:
             response = {
                'task_id': db_status['task_id'],
                'state': db_status['status'],
                'status': db_status['current_album'] or db_status['status'],
                'progress': db_status['progress'],
                'current_album': db_status['current_album'],
                'current_album_idx': db_status['current_album_idx'],
                'total_albums': db_status['total_albums'],
                'log_output': db_status['log_output'].split('\n') if db_status['log_output'] else []
            }
        else:
            response = {
                'task_id': task.id,
                'state': task.state,
                'status': 'Unknown status',
                'progress': 0,
                'current_album': 'N/A',
                'current_album_idx': 0,
                'total_albums': 0,
                'log_output': []
            }
    return jsonify(response)


@app.route('/api/analysis/cancel/<task_id>', methods=['POST'])
def cancel_analysis(task_id):
    task = AsyncResult(task_id, app=celery_app)
    if task.state not in ['PENDING', 'STARTED', 'PROGRESS']:
        return jsonify({"message": f"Task {task_id} is already in state {task.state}. Cannot cancel."}), 400
    task.revoke(terminate=True, signal='SIGKILL')
    # Update DB status directly as revoke might not trigger postrun immediately
    save_task_status(
        task_id=task_id,
        task_type='analysis',
        status='REVOKED',
        progress=0,
        current_album='Cancelled by user',
        log_output='Task was cancelled by the user.',
        timestamp=datetime.now().isoformat()
    )
    return jsonify({"message": f"Task {task_id} cancellation requested."})

@app.route('/api/analysis/last_task', methods=['GET'])
def get_last_analysis_task_status():
    last_task = get_last_task_status_from_db(task_type='analysis')
    if last_task:
        return jsonify({
            'task_id': last_task['task_id'],
            'status': last_task['status'],
            'progress': last_task['progress'],
            'current_album': last_task['current_album'],
            'current_album_idx': last_task['current_album_idx'],
            'total_albums': last_task['total_albums'],
            'log_output': last_task['log_output'].split('\n') if last_task['log_output'] else []
        })
    return jsonify({"task_id": None, "status": "No analysis task found."})


# --- NEW CLUSTERING ENDPOINTS ---

@app.route('/api/clustering', methods=['POST'])
def start_clustering():
    data = request.json
    logger.info(f"Received clustering request: {data}")
    task = start_clustering_task.delay(data)
    # Save initial task status
    save_task_status(
        task_id=task.id,
        task_type='clustering',
        status='PENDING',
        progress=0,
        current_album='Starting clustering...', # Re-using current_album for the first message
        timestamp=datetime.now().isoformat()
    )
    return jsonify({"task_id": task.id, "message": "Clustering started."})

@app.route('/api/clustering/status/<task_id>', methods=['GET'])
def get_clustering_status(task_id):
    task = AsyncResult(task_id, app=celery_app)
    response = {}
    if task.state == 'PENDING':
        db_status = get_task_status_from_db(task_id)
        response = {
            'task_id': task.id,
            'state': db_status['status'] if db_status else task.state,
            'status': db_status['current_album'] if db_status else 'Pending...',
            'progress': db_status['progress'] if db_status else 0,
            'current_step': db_status['current_album'] if db_status else 'Waiting for worker',
            'current_step_idx': db_status['current_album_idx'] if db_status else 0,
            'total_steps': db_status['total_albums'] if db_status else 0,
            'log_output': db_status['log_output'].split('\n') if db_status and db_status['log_output'] else []
        }
    elif task.state == 'PROGRESS':
        response = {
            'task_id': task.id,
            'state': task.state,
            'status': task.info.get('status', 'In Progress'),
            'progress': task.info.get('progress', 0),
            'current_step': task.info.get('current_step', 'N/A'),
            'current_step_idx': task.info.get('current_step_idx', 0),
            'total_steps': task.info.get('total_steps', 0),
            'log_output': task.info.get('log_output', [])
        }
    elif task.state in ['SUCCESS', 'FAILURE', 'REVOKED']:
        db_status = get_task_status_from_db(task_id)
        if db_status:
            response = {
                'task_id': db_status['task_id'],
                'state': db_status['status'],
                'status': db_status['current_album'] or db_status['status'],
                'progress': db_status['progress'],
                'current_step': db_status['current_album'],
                'current_step_idx': db_status['current_album_idx'],
                'total_steps': db_status['total_albums'],
                'log_output': db_status['log_output'].split('\n') if db_status['log_output'] else []
            }
        else:
            response = {
                'task_id': task.id,
                'state': task.state,
                'status': f"Task {task.state}",
                'progress': 100 if task.state == 'SUCCESS' else 0,
                'current_step': 'Finished',
                'current_step_idx': 0,
                'total_steps': 0,
                'log_output': [str(task.info)]
            }
    else:
        db_status = get_task_status_from_db(task_id)
        if db_status:
             response = {
                'task_id': db_status['task_id'],
                'state': db_status['status'],
                'status': db_status['current_album'] or db_status['status'],
                'progress': db_status['progress'],
                'current_step': db_status['current_album'],
                'current_step_idx': db_status['current_album_idx'],
                'total_steps': db_status['total_albums'],
                'log_output': db_status['log_output'].split('\n') if db_status['log_output'] else []
            }
        else:
            response = {
                'task_id': task.id,
                'state': task.state,
                'status': 'Unknown status',
                'progress': 0,
                'current_step': 'N/A',
                'current_step_idx': 0,
                'total_steps': 0,
                'log_output': []
            }
    return jsonify(response)


@app.route('/api/clustering/cancel/<task_id>', methods=['POST'])
def cancel_clustering(task_id):
    task = AsyncResult(task_id, app=celery_app)
    if task.state not in ['PENDING', 'STARTED', 'PROGRESS']:
        return jsonify({"message": f"Task {task_id} is already in state {task.state}. Cannot cancel."}), 400
    task.revoke(terminate=True, signal='SIGKILL')
    save_task_status(
        task_id=task_id,
        task_type='clustering',
        status='REVOKED',
        progress=0,
        current_album='Cancelled by user', # Re-using for message
        log_output='Clustering task was cancelled by the user.',
        timestamp=datetime.now().isoformat()
    )
    return jsonify({"message": f"Clustering task {task_id} cancellation requested."})

@app.route('/api/clustering/last_task', methods=['GET'])
def get_last_clustering_task_status():
    last_task = get_last_task_status_from_db(task_type='clustering')
    if last_task:
        return jsonify({
            'task_id': last_task['task_id'],
            'status': last_task['status'],
            'progress': last_task['progress'],
            'current_step': last_task['current_album'],
            'current_step_idx': last_task['current_album_idx'],
            'total_steps': last_task['total_albums'],
            'log_output': last_task['log_output'].split('\n') if last_task['log_output'] else []
        })
    return jsonify({"task_id": None, "status": "No clustering task found."})


@app.route('/api/playlists', methods=['GET'])
def get_playlists():
    playlists = db_manager.get_all_playlists()
    return jsonify(playlists)

# --- Main entry point for Flask ---
if __name__ == '__main__':
    # Ensure temporary directory exists
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    # Initialize DB schema if it doesn't exist
    db_manager.create_tables()
    create_status_table() # Ensure status table is created
    app.run(host='0.0.0.0', port=8000)
