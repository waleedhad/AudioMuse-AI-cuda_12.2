import os
import json
import logging
import tempfile
import numpy as np
import voyager # type: ignore
import psycopg2 # type: ignore
from psycopg2.extras import DictCursor
import io # Import the io module
import re
from rapidfuzz import fuzz

from config import (
    EMBEDDING_DIMENSION, INDEX_NAME, VOYAGER_METRIC, VOYAGER_EF_CONSTRUCTION,
    VOYAGER_M, VOYAGER_QUERY_EF, MAX_SONGS_PER_ARTIST
)

# Import from other project modules
from .mediaserver import create_instant_playlist

logger = logging.getLogger(__name__)

# --- Global cache for the loaded Voyager index ---
voyager_index = None
id_map = None # {voyager_int_id: item_id_str}
reverse_id_map = None # {item_id_str: voyager_int_id}


def build_and_store_voyager_index(db_conn):
    """
    Fetches all song embeddings, builds a new Voyager index, and stores it
    atomically in the 'voyager_index_data' table in PostgreSQL.
    """
    logger.info("Starting to build and store Voyager index...")

    # Map the string metric from config to the voyager.Space enum
    metric_str = VOYAGER_METRIC.lower()
    if metric_str == 'angular':
        space = voyager.Space.Cosine
    elif metric_str == 'euclidean':
        space = voyager.Space.Euclidean
    elif metric_str == 'dot':
        space = voyager.Space.InnerProduct
    else:
        logger.warning(f"Unknown Voyager metric '{VOYAGER_METRIC}'. Defaulting to Cosine.")
        space = voyager.Space.Cosine

    cur = db_conn.cursor()
    try:
        logger.info("Fetching all embeddings from the database...")
        cur.execute("SELECT item_id, embedding FROM embedding")
        all_embeddings = cur.fetchall()

        if not all_embeddings:
            logger.warning("No embeddings found in DB. Voyager index will not be built.")
            return

        logger.info(f"Found {len(all_embeddings)} embeddings to index.")

        voyager_index_builder = voyager.Index(
            space=space, # Use the mapped enum value
            num_dimensions=EMBEDDING_DIMENSION,
            M=VOYAGER_M,
            ef_construction=VOYAGER_EF_CONSTRUCTION
        )
        
        local_id_map = {}
        voyager_item_index = 0
        vectors_to_add = []
        ids_to_add = []

        for item_id, embedding_blob in all_embeddings:
            if embedding_blob is None:
                logger.warning(f"Skipping item_id {item_id}: embedding data is NULL.")
                continue
            
            embedding_vector = np.frombuffer(embedding_blob, dtype=np.float32)
            
            if embedding_vector.shape[0] != EMBEDDING_DIMENSION:
                logger.warning(f"Skipping item_id {item_id}: embedding dimension mismatch. "
                               f"Expected {EMBEDDING_DIMENSION}, got {embedding_vector.shape[0]}.")
                continue
            
            vectors_to_add.append(embedding_vector)
            ids_to_add.append(voyager_item_index)
            local_id_map[voyager_item_index] = item_id
            voyager_item_index += 1

        if not vectors_to_add:
            logger.warning("No valid embeddings were found to add to the Voyager index. Aborting build process.")
            return

        logger.info(f"Adding {len(vectors_to_add)} items to the index...")
        voyager_index_builder.add_items(np.array(vectors_to_add), ids=np.array(ids_to_add))

        logger.info(f"Building index with {len(vectors_to_add)} items...")
        
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".voyager") as tmp:
                temp_file_path = tmp.name
            
            voyager_index_builder.save(temp_file_path)

            with open(temp_file_path, 'rb') as f:
                index_binary_data = f.read()
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        logger.info(f"Voyager index binary data size to be stored: {len(index_binary_data)} bytes.")

        if not index_binary_data:
            logger.error("CRITICAL: Generated Voyager index file is empty. Aborting database storage.")
            return

        id_map_json = json.dumps(local_id_map)

        logger.info(f"Storing Voyager index '{INDEX_NAME}' in the database...")
        upsert_query = """
            INSERT INTO voyager_index_data (index_name, index_data, id_map_json, embedding_dimension, created_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (index_name) DO UPDATE SET
                index_data = EXCLUDED.index_data,
                id_map_json = EXCLUDED.id_map_json,
                embedding_dimension = EXCLUDED.embedding_dimension,
                created_at = CURRENT_TIMESTAMP;
        """
        cur.execute(upsert_query, (INDEX_NAME, psycopg2.Binary(index_binary_data), id_map_json, EMBEDDING_DIMENSION))
        db_conn.commit()
        logger.info("Voyager index build and database storage complete.")

    except Exception as e:
        logger.error("An error occurred during Voyager index build: %s", e, exc_info=True)
        db_conn.rollback()
    finally:
        cur.close()


def load_voyager_index_for_querying(force_reload=False):
    """
    Loads the Voyager index from the database into the global in-memory cache.
    """
    global voyager_index, id_map, reverse_id_map

    if voyager_index is not None and not force_reload:
        logger.info("Voyager index is already loaded in memory. Skipping reload.")
        return

    from app import get_db

    logger.info("Attempting to load Voyager index from database into memory...")
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT index_data, id_map_json, embedding_dimension FROM voyager_index_data WHERE index_name = %s", (INDEX_NAME,))
        record = cur.fetchone()

        if not record:
            logger.warning(f"Voyager index '{INDEX_NAME}' not found in the database. Cache will be empty.")
            voyager_index, id_map, reverse_id_map = None, None, None
            return
        
        index_binary_data, id_map_json, db_embedding_dim = record

        if not index_binary_data:
            logger.error(f"Voyager index '{INDEX_NAME}' data in database is empty.")
            voyager_index, id_map, reverse_id_map = None, None, None
            return

        if db_embedding_dim != EMBEDDING_DIMENSION:
            logger.error(f"FATAL: Voyager index dimension mismatch! DB has {db_embedding_dim}, config expects {EMBEDDING_DIMENSION}.")
            voyager_index, id_map, reverse_id_map = None, None, None
            return

        index_stream = io.BytesIO(index_binary_data)
        loaded_index = voyager.Index.load(index_stream)
        loaded_index.ef = VOYAGER_QUERY_EF 
        voyager_index = loaded_index
        id_map = {int(k): v for k, v in json.loads(id_map_json).items()}
        reverse_id_map = {v: k for k, v in id_map.items()}

        logger.info(f"Voyager index with {len(id_map)} items loaded successfully into memory.")

    except Exception as e:
        logger.error("Failed to load Voyager index from database: %s", e, exc_info=True)
        voyager_index, id_map, reverse_id_map = None, None, None
    finally:
        cur.close()

def _normalize_title(title: str) -> str:
    """Lowercase, remove content in brackets, and normalize separators."""
    if not title:
        return ""
    title = re.sub(r'[\[\(].*?[\]\)]', '', title)
    title = re.sub(r'[_\-]', ' ', title)
    title = re.sub(r'\s+', ' ', title)
    return title.strip().lower()

def _normalize_artist(artist: str) -> str:
    """Lowercase and strip whitespace."""
    if not artist:
        return ""
    return artist.strip().lower()

def _is_likely_same_song(title1, artist1, title2, artist2, threshold=90):
    """Determines if two songs are likely the same based on title and artist similarity."""
    norm_title1 = _normalize_title(title1)
    norm_title2 = _normalize_title(title2)
    norm_artist1 = _normalize_artist(artist1)
    norm_artist2 = _normalize_artist(artist2)
    
    artist_score = fuzz.ratio(norm_artist1, norm_artist2)
    if artist_score < 90:
        return False

    title_score = fuzz.ratio(norm_title1, norm_title2)
    return title_score >= threshold

def _deduplicate_and_filter_neighbors(song_results: list, db_conn):
    """
    Filters a list of songs to remove duplicates based on title/artist similarity.
    """
    if not song_results:
        return []

    item_ids = [r['item_id'] for r in song_results]
    item_details = {}
    with db_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT item_id, title, author FROM score WHERE item_id = ANY(%s)", (item_ids,))
        rows = cur.fetchall()
        for row in rows:
            item_details[row['item_id']] = {'title': row['title'], 'author': row['author']}

    unique_songs = []
    added_songs_details = [] 

    for song in song_results:
        current_details = item_details.get(song['item_id'])
        if not current_details:
            logger.warning(f"Could not find details for item_id {song['item_id']} during deduplication. Skipping.")
            continue

        is_duplicate = False
        for added_detail in added_songs_details:
            if _is_likely_same_song(
                current_details['title'], current_details['author'],
                added_detail['title'], added_detail['author']
            ):
                is_duplicate = True
                logger.info(f"Found duplicate: '{current_details['title']}' is similar to '{added_detail['title']}'.")
                break
        
        if not is_duplicate:
            unique_songs.append(song)
            added_songs_details.append(current_details)

    return unique_songs

def find_nearest_neighbors_by_id(target_item_id: str, n: int = 10, eliminate_duplicates: bool = False):
    """
    Finds the N nearest neighbors for a given item_id using the globally cached Voyager index.
    The results will NOT include the original item and will be deduplicated based on title/artist similarity.
    If eliminate_duplicates is True, it will also limit the number of songs from a single artist.
    """
    if voyager_index is None or id_map is None or reverse_id_map is None:
        raise RuntimeError("Voyager index is not loaded in memory. It may be missing, empty, or the server failed to load it on startup.")

    target_voyager_id = reverse_id_map.get(target_item_id)
    if target_voyager_id is None:
        logger.warning(f"Target item_id '{target_item_id}' not found in the loaded Voyager index map.")
        return []

    try:
        query_vector = voyager_index.get_vector(target_voyager_id)
    except Exception as e:
        logger.error(f"Could not retrieve vector for Voyager ID {target_voyager_id} (item_id: {target_item_id}): {e}")
        return []

    # If eliminating duplicates by artist, search for a much larger number of neighbors first.
    if eliminate_duplicates:
        # Query more to have a pool for filtering. +1 for the original song.
        k_increase = max(5, int(n * 3))
        num_to_query = n + k_increase + 1 # +1 to account for the query song itself
    else:
        # Standard query size increase for basic song-level deduplication
        k_increase = max(5, int(n * 0.20))
        num_to_query = n + k_increase + 1 # +1 to account for the query song itself

    # Query for neighbors.
    neighbor_voyager_ids, distances = voyager_index.query(query_vector, k=num_to_query)

    initial_results = []
    # Create a list of results, excluding the original song itself.
    for voyager_id, dist in zip(neighbor_voyager_ids, distances):
        item_id = id_map.get(voyager_id)
        if item_id and item_id != target_item_id:
            initial_results.append({"item_id": item_id, "distance": float(dist)})

    from app import get_db, get_score_data_by_ids
    db_conn = get_db()

    # First, perform the song-level deduplication (e.g., remove remixes)
    unique_results_by_song = _deduplicate_and_filter_neighbors(initial_results, db_conn)
    
    # Now, if requested, filter by artist count
    if eliminate_duplicates:
        # We need author information. We can fetch it once for all unique songs.
        item_ids_to_check = [r['item_id'] for r in unique_results_by_song]
        
        track_details_list = get_score_data_by_ids(item_ids_to_check)
        details_map = {d['item_id']: {'author': d['author']} for d in track_details_list}

        artist_counts = {}
        final_results = []
        for song in unique_results_by_song:
            song_id = song['item_id']
            author = details_map.get(song_id, {}).get('author')

            if not author:
                logger.warning(f"Could not find author for item_id {song_id} during artist deduplication. Skipping.")
                continue

            current_count = artist_counts.get(author, 0)
            if current_count < MAX_SONGS_PER_ARTIST:
                final_results.append(song)
                artist_counts[author] = current_count + 1
    else:
        final_results = unique_results_by_song

    # Finally, return the top N results from the processed list.
    return final_results[:n]


def get_item_id_by_title_and_artist(title: str, artist: str):
    """
    Finds the item_id for an exact title and artist match.
    Returns the item_id string or None if not found.
    """
    from app import get_db
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    try:
        query = "SELECT item_id FROM score WHERE title = %s AND author = %s LIMIT 1"
        cur.execute(query, (title, artist))
        result = cur.fetchone()
        if result:
            return result['item_id']
        return None
    except Exception as e:
        logger.error(f"Error fetching item_id for '{title}' by '{artist}': {e}", exc_info=True)
        return None
    finally:
        cur.close()

def search_tracks_by_title_and_artist(title_query: str, artist_query: str, limit: int = 15):
    """
    Searches for tracks using partial title and artist names for autocomplete.
    """
    from app import get_db
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    results = []
    try:
        query_parts = []
        params = []
        
        if artist_query:
            query_parts.append("author ILIKE %s")
            params.append(f"%{artist_query}%")
            
        if title_query:
            query_parts.append("title ILIKE %s")
            params.append(f"%{title_query}%")

        if not query_parts:
            return []

        where_clause = " AND ".join(query_parts)
        
        query = f"""
            SELECT item_id, title, author 
            FROM score 
            WHERE {where_clause}
            ORDER BY author, title 
            LIMIT %s
        """
        params.append(limit)
        
        cur.execute(query, tuple(params))
        results = [dict(row) for row in cur.fetchall()]

    except Exception as e:
        logger.error(f"Error searching tracks with query '{title_query}', '{artist_query}': {e}", exc_info=True)
    finally:
        cur.close()
    
    return results


def create_playlist_from_ids(playlist_name: str, track_ids: list):
    """
    Creates a new playlist on the configured media server with the provided name and track IDs.
    """
    try:
        created_playlist = create_instant_playlist(playlist_name, track_ids)
        
        if not created_playlist:
            raise Exception("Playlist creation failed. The media server did not return a playlist object.")

        playlist_id = created_playlist.get('Id')

        if not playlist_id:
            raise Exception("Media server API response did not include a playlist ID.")

        return playlist_id

    except Exception as e:
        raise e
