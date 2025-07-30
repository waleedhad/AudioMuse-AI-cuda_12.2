import logging
import numpy as np
from datetime import datetime, timezone

from config import SONIC_FINGERPRINT_TOP_N_SONGS, SONIC_FINGERPRINT_NEIGHBORS, EMBEDDING_DIMENSION
from .mediaserver import get_top_played_songs, get_last_played_time
from .voyager_manager import find_nearest_neighbors_by_vector
from app import get_tracks_by_ids

logger = logging.getLogger(__name__)

def generate_sonic_fingerprint():
    """
    Generates a 'sonic fingerprint' by averaging the embeddings of the most played songs,
    weighted by recency, and then finds similar songs to this fingerprint.
    """
    logger.info("Generating sonic fingerprint...")

    # 1. Get top N played songs from the media server
    top_songs = get_top_played_songs(limit=SONIC_FINGERPRINT_TOP_N_SONGS)
    if not top_songs:
        logger.warning("No top played songs found. Cannot generate sonic fingerprint.")
        return []

    top_song_ids = [song['Id'] for song in top_songs]
    logger.info(f"Found {len(top_song_ids)} top played songs to create fingerprint from.")

    # 2. Get embeddings for these songs from our DB
    track_details = get_tracks_by_ids(top_song_ids)
    if not track_details:
        logger.warning("Could not retrieve embeddings for any of the top songs.")
        return []

    embeddings_map = {track['item_id']: track['embedding_vector'] for track in track_details if 'embedding_vector' in track and track['embedding_vector'].size > 0}
    
    weighted_vectors = []
    total_weight = 0

    # 3. Calculate weighted average of embeddings
    for song_id in top_song_ids:
        if song_id not in embeddings_map:
            logger.debug(f"Skipping song {song_id} as it has no embedding in the database.")
            continue

        embedding_vector = embeddings_map[song_id]
        
        # Get last played time for weighting
        last_played_str = get_last_played_time(song_id)
        
        weight = 1.0  # Default weight
        days_since_played = "N/A"
        if last_played_str:
            try:
                # Navidrome format: "2023-10-27T10:00:00Z"
                # Jellyfin format:  "2024-01-15T18:30:00.0000000Z"
                # Python's fromisoformat can only handle up to 6 microsecond digits.
                # We need to truncate longer fractional seconds from Jellyfin before parsing.
                if '.' in last_played_str and last_played_str.endswith('Z'):
                    dot_index = last_played_str.rfind('.')
                    z_index = last_played_str.rfind('Z')
                    
                    # If the fractional part has more than 6 digits, truncate it
                    if z_index > dot_index and (z_index - dot_index - 1) > 6:
                        last_played_str = last_played_str[:dot_index+7] + 'Z'

                # The fromisoformat method handles both formats if we normalize the 'Z'
                last_played_dt = datetime.fromisoformat(last_played_str.replace('Z', '+00:00'))
                days_since_played = (datetime.now(timezone.utc) - last_played_dt).days
                
                # Apply exponential decay weight.
                # A shorter half-life gives more importance to very recent songs.
                half_life = 30.0 # days
                decay_rate = -np.log(0.5) / half_life
                weight = np.exp(-decay_rate * max(0, days_since_played))

            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse date '{last_played_str}' for song {song_id}. Using lower weight. Error: {e}")
                weight = 0.5 # Lower weight if date is unparsable
        else:
            # Song has a play count but no last played date, give it a lower weight
            weight = 0.25

        weighted_vectors.append(embedding_vector * weight)
        total_weight += weight
        logger.info(f"Song {song_id}, days since played: {days_since_played}, weight: {weight:.4f}")

    if not weighted_vectors:
        logger.error("No valid embeddings with weights could be calculated. Cannot generate fingerprint.")
        return []

    # Calculate the average vector (the "sonic fingerprint")
    average_vector = np.sum(weighted_vectors, axis=0) / total_weight
    
    logger.info(f"Calculated average vector (sonic fingerprint) from {len(weighted_vectors)} songs.")

    # 4. Use Voyager to find similar songs to the fingerprint
    try:
        similar_songs = find_nearest_neighbors_by_vector(
            query_vector=average_vector,
            n=SONIC_FINGERPRINT_NEIGHBORS,
            eliminate_duplicates=True # Good default to get a diverse playlist
        )
        logger.info(f"Found {len(similar_songs)} similar songs for the sonic fingerprint.")
        return similar_songs
    except Exception as e:
        logger.error(f"Error finding neighbors for sonic fingerprint: {e}", exc_info=True)
        return []
