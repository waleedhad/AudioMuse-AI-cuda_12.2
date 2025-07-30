# tasks/mediaserver.py

import requests
import logging
import os
import config  # Import the config module to access server type and settings

logger = logging.getLogger(__name__)

# Define a global timeout for all requests
REQUESTS_TIMEOUT = 300

# ##############################################################################
# JELLYFIN IMPLEMENTATION
# ##############################################################################

def _jellyfin_get_recent_albums(limit):
    """
    Fetches a list of the most recently added albums from Jellyfin using pagination.
    If limit is 0, it fetches all albums.
    If limit > 0, it fetches up to the specified limit.
    """
    all_albums = []
    start_index = 0
    page_size = 500  # The number of items to request per page
    fetch_all = (limit == 0)

    while fetch_all or len(all_albums) < limit:
        # Determine how many items to fetch in the current request
        size_to_fetch = page_size
        if not fetch_all:
            size_to_fetch = min(page_size, limit - len(all_albums))

        if size_to_fetch <= 0:
            break

        url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
        params = {
            "IncludeItemTypes": "MusicAlbum",
            "SortBy": "DateCreated",
            "SortOrder": "Descending",
            "Recursive": True,
            "Limit": size_to_fetch,
            "StartIndex": start_index
        }
        
        try:
            r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
            response_data = r.json()
            albums = response_data.get("Items", [])
            
            if not albums:
                # No more albums to fetch, break the loop
                break

            all_albums.extend(albums)
            start_index += len(albums)

            # If the number of returned albums is less than requested, it's the last page
            if len(albums) < size_to_fetch:
                break
            
            # If fetching all, check if we have reached the total record count
            if fetch_all and start_index >= response_data.get("TotalRecordCount", float('inf')):
                 break

        except Exception as e:
            logger.error(f"Jellyfin get_recent_albums failed during pagination: {e}", exc_info=True)
            # Exit loop on any request error
            break

    return all_albums

def _jellyfin_get_tracks_from_album(album_id):
    """Fetches all audio tracks for a given album ID from Jellyfin."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"ParentId": album_id, "IncludeItemTypes": "Audio"}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"Jellyfin get_tracks_from_album failed for album {album_id}: {e}", exc_info=True)
        return []

def _jellyfin_download_track(temp_dir, item):
    """Downloads a single track from Jellyfin."""
    try:
        track_id = item['Id']
        file_extension = os.path.splitext(item.get('Path', ''))[1] or '.tmp'
        download_url = f"{config.JELLYFIN_URL}/Items/{track_id}/Download"
        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")

        with requests.get(download_url, headers=config.HEADERS, stream=True, timeout=REQUESTS_TIMEOUT) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Downloaded '{item['Name']}' to '{local_filename}'")
        return local_filename
    except Exception as e:
        logger.error(f"Failed to download track {item.get('Name', 'Unknown')}: {e}", exc_info=True)
        return None


def _jellyfin_get_all_songs():
    """Fetches all songs from Jellyfin."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Audio", "Recursive": True}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"Jellyfin get_all_songs failed: {e}", exc_info=True)
        return []

def _jellyfin_get_playlist_by_name(playlist_name):
    """Finds a Jellyfin playlist by its exact name."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True, "Name": playlist_name}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        playlists = r.json().get("Items", [])
        return playlists[0] if playlists else None
    except Exception as e:
        logger.error(f"Jellyfin get_playlist_by_name failed for '{playlist_name}': {e}", exc_info=True)
        return None

def _jellyfin_create_playlist(base_name, item_ids):
    """Creates a new playlist on Jellyfin."""
    url = f"{config.JELLYFIN_URL}/Playlists"
    body = {"Name": base_name, "Ids": item_ids, "UserId": config.JELLYFIN_USER_ID}
    try:
        r = requests.post(url, headers=config.HEADERS, json=body, timeout=REQUESTS_TIMEOUT)
        if r.ok:
            logger.info("✅ Created Jellyfin playlist '%s' with %s tracks", base_name, len(item_ids))
    except Exception as e:
        logger.error("Exception creating Jellyfin playlist '%s': %s", base_name, e, exc_info=True)

def _jellyfin_create_instant_playlist(playlist_name, item_ids):
    """Creates a new instant playlist on Jellyfin and returns the response object."""
    final_playlist_name = f"{playlist_name.strip()}_instant"
    url = f"{config.JELLYFIN_URL}/Playlists"
    body = {"Name": final_playlist_name, "Ids": item_ids, "UserId": config.JELLYFIN_USER_ID}
    try:
        r = requests.post(url, headers=config.HEADERS, json=body, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        created_playlist = r.json()
        logger.info("✅ Created Jellyfin instant playlist '%s' with ID: %s", final_playlist_name, created_playlist.get('Id'))
        return created_playlist
    except Exception as e:
        logger.error("Exception creating Jellyfin instant playlist '%s': %s", playlist_name, e, exc_info=True)
        return None

def _jellyfin_get_all_playlists():
    """Fetches all playlists from Jellyfin."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"Jellyfin get_all_playlists failed: {e}", exc_info=True)
        return []

def _jellyfin_delete_playlist(playlist_id):
    """Deletes a playlist on Jellyfin."""
    url = f"{config.JELLYFIN_URL}/Items/{playlist_id}"
    try:
        r = requests.delete(url, headers=config.HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        logger.info(f"🗑️ Deleted Jellyfin playlist ID: {playlist_id}")
        return True
    except Exception as e:
        logger.error(f"Exception deleting Jellyfin playlist ID {playlist_id}: {e}", exc_info=True)
        return False

def _jellyfin_get_top_played_songs(limit):
    """Fetches the top N most played songs from Jellyfin."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {
        "IncludeItemTypes": "Audio",
        "SortBy": "PlayCount",
        "SortOrder": "Descending",
        "Recursive": True,
        "Limit": limit,
        "Fields": "UserData,Path"  # Request UserData to get PlayCount and LastPlayedDate
    }
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("Items", [])
        return items
    except Exception as e:
        logger.error(f"Jellyfin get_top_played_songs failed: {e}", exc_info=True)
        return []

def _jellyfin_get_last_played_time(item_id):
    """Fetches the last played time for a specific track from Jellyfin."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items/{item_id}"
    params = {"Fields": "UserData"}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        user_data = r.json().get("UserData", {})
        return user_data.get("LastPlayedDate") # Returns a string like "2024-01-15T18:30:00.0000000Z" or None
    except Exception as e:
        logger.error(f"Jellyfin get_last_played_time failed for item {item_id}: {e}", exc_info=True)
        return None

# ##############################################################################
# NAVIDROME (SUBSONIC API) IMPLEMENTATION
# ##############################################################################

def get_navidrome_auth_params():
    """
    Generates authentication parameters for the Navidrome (Subsonic) API.
    Navidrome does not support client-side password hashing (like md5 or pbkdf2)
    due to its use of secure Argon2 password hashing on the server.
    The correct method for password-based auth is to send the password hex-encoded.
    Returns a dictionary of parameters required for every Navidrome API call.
    """
    if not config.NAVIDROME_PASSWORD or not config.NAVIDROME_USER:
        logger.warning("Navidrome User or Password is not configured in environment variables.")
        return {}
    
    # Hex-encode the password. This is the method Navidrome supports for password auth.
    hex_encoded_password = config.NAVIDROME_PASSWORD.encode('utf-8').hex()
    
    return {
        "u": config.NAVIDROME_USER,
        "p": f"enc:{hex_encoded_password}", # The 'enc:' prefix is required by the Subsonic API
        "v": "1.16.1",
        "c": config.APP_VERSION,
        "f": "json"
    }

def _navidrome_request(endpoint, params=None, method='get', stream=False):
    """Helper function to make requests to the Navidrome API."""
    if params is None:
        params = {}
    
    auth_params = get_navidrome_auth_params()
    if not auth_params:
        logger.error("Navidrome credentials are not configured. Cannot make API call.")
        return None

    url = f"{config.NAVIDROME_URL}/rest/{endpoint}.view"
    all_params = {**auth_params, **params}

    try:
        if method.lower() == 'get':
            r = requests.get(url, params=all_params, timeout=REQUESTS_TIMEOUT, stream=stream)
        elif method.lower() == 'post':
            r = requests.post(url, params=all_params, timeout=REQUESTS_TIMEOUT)
        else:
            logger.error(f"Unsupported HTTP method: {method}")
            return None
            
        r.raise_for_status()
        if stream:
            return r
        
        response_data = r.json()
        
        subsonic_response = response_data.get("subsonic-response", {})
        if subsonic_response.get("status") == "failed":
            error = subsonic_response.get("error", {})
            logger.error(f"Navidrome API Error on endpoint '{endpoint}': {error.get('message')} (Code: {error.get('code')})")
            return None
            
        return subsonic_response
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Navidrome API endpoint '{endpoint}': {e}", exc_info=True)
        return None

def _navidrome_download_track(temp_dir, item):
    """Downloads a single track from Navidrome."""
    try:
        # Use 'id' (original key) for the API call, not the normalized 'Id'
        track_id = item['id'] 
        file_extension = os.path.splitext(item.get('path', ''))[1] or '.tmp'
        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")
        
        # Navidrome uses the 'stream' endpoint for downloads
        response = _navidrome_request("stream", params={"id": track_id}, stream=True)
        if response:
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded '{item.get('title', 'Unknown')}' to '{local_filename}'")
            return local_filename
    except Exception as e:
        logger.error(f"Failed to download Navidrome track {item.get('title', 'Unknown')}: {e}", exc_info=True)
    return None

def _navidrome_get_recent_albums(limit):
    """
    Fetches a list of the most recently added albums from Navidrome.
    If limit is 0, it fetches all albums by paginating.
    If limit > 0, it fetches up to the specified limit, handling pagination if necessary.
    """
    all_albums = []
    offset = 0
    page_size = 500  # Max size per Subsonic API request
    fetch_all = (limit == 0)

    while fetch_all or len(all_albums) < limit:
        # Determine the size for the current page request
        size_to_fetch = page_size
        if not fetch_all:
            size_to_fetch = min(page_size, limit - len(all_albums))

        if size_to_fetch <= 0:
            break

        params = {"type": "newest", "size": size_to_fetch, "offset": offset}
        response = _navidrome_request("getAlbumList2", params)

        if response and "albumList2" in response and "album" in response["albumList2"]:
            albums = response["albumList2"]["album"]
            if not albums:
                break  # No more albums to fetch

            all_albums.extend([{**a, 'Id': a.get('id'), 'Name': a.get('name')} for a in albums])
            offset += len(albums)

            if len(albums) < size_to_fetch:
                break # This was the last page
        else:
            logger.error("Failed to fetch recent albums page from Navidrome.")
            break
            
    return all_albums

def _navidrome_get_tracks_from_album(album_id):
    """
    Fetches all audio tracks for a given album ID from Navidrome and
    normalizes the keys to match Jellyfin's format for compatibility.
    """
    params = {"id": album_id}
    response = _navidrome_request("getAlbum", params)
    if response and "album" in response and "song" in response["album"]:
        songs = response["album"]["song"]
        # Normalize keys to match Jellyfin ('Id', 'Name', etc.)
        return [{
            **s,
            'Id': s.get('id'),
            'Name': s.get('title'),
            'AlbumArtist': s.get('artist'),
            'Path': s.get('path')
        } for s in songs]
    return []

def _navidrome_get_all_songs():
    """Fetches all songs from Navidrome using the search3 endpoint."""
    logger.warning("Fetching all songs from Navidrome. This may be slow for large libraries.")
    all_songs = []
    offset = 0
    limit = 500
    while True:
        params = {"query": '', "songCount": limit, "songOffset": offset}
        response = _navidrome_request("search3", params)
        if response and "searchResult3" in response and "song" in response["searchResult3"]:
            songs = response["searchResult3"]["song"]
            if not songs:
                break
            # Normalize keys to match Jellyfin's output
            all_songs.extend([{
                **s,
                'Id': s.get('id'),
                'Name': s.get('title'),
                'AlbumArtist': s.get('artist'),
                'Path': s.get('path')
            } for s in songs])
            offset += len(songs)
            if len(songs) < limit:
                break
        else:
            logger.error("Failed to fetch songs using search3. The Navidrome server may not support this method for fetching all songs.")
            break
    return all_songs

def _navidrome_get_playlist_by_name(playlist_name):
    """Finds a Navidrome playlist by its exact name."""
    response = _navidrome_request("getPlaylists")
    if response and "playlists" in response and "playlist" in response["playlists"]:
        for playlist in response["playlists"]["playlist"]:
            if playlist.get("name") == playlist_name:
                playlist_details_response = _navidrome_request("getPlaylist", {"id": playlist["id"]})
                if playlist_details_response and "playlist" in playlist_details_response:
                    # Normalize the playlist object
                    p = playlist_details_response["playlist"]
                    return {**p, 'Id': p.get('id'), 'Name': p.get('name')}
    return None

def _navidrome_create_playlist(base_name, item_ids):
    """Creates a new playlist on Navidrome."""
    params = {"name": base_name, "songId": item_ids}
    response = _navidrome_request("createPlaylist", params, method='post')
    if response and response.get("status") == "ok":
        logger.info("✅ Created Navidrome playlist '%s' with %s tracks", base_name, len(item_ids))
    else:
        logger.error("Failed to create playlist '%s' on Navidrome", base_name)

def _navidrome_create_instant_playlist(playlist_name, item_ids):
    """Creates a new instant playlist on Navidrome and returns it."""
    final_playlist_name = f"{playlist_name.strip()}_instant"
    params = {"name": final_playlist_name, "songId": item_ids}
    response = _navidrome_request("createPlaylist", params, method='post')
    
    if response and response.get("status") == "ok":
        logger.info("✅ Created playlist '%s' on Navidrome. Now fetching its details.", final_playlist_name)
        return _navidrome_get_playlist_by_name(final_playlist_name)
    else:
        logger.error("Failed to create instant playlist '%s' on Navidrome", final_playlist_name)
        return None

def _navidrome_get_all_playlists():
    """Fetches all playlists from Navidrome."""
    response = _navidrome_request("getPlaylists")
    if response and "playlists" in response and "playlist" in response["playlists"]:
        playlists = response["playlists"]["playlist"]
        # Normalize keys
        return [{**p, 'Id': p.get('id'), 'Name': p.get('name')} for p in playlists]
    return []

def _navidrome_delete_playlist(playlist_id):
    """Deletes a playlist on Navidrome."""
    params = {"id": playlist_id}
    response = _navidrome_request("deletePlaylist", params, method='post')
    if response and response.get("status") == "ok":
        logger.info(f"🗑️ Deleted Navidrome playlist ID: {playlist_id}")
        return True
    else:
        logger.error(f"Failed to delete playlist ID '{playlist_id}' on Navidrome")
        return False

def _navidrome_get_top_played_songs(limit):
    """Fetches the top N most played songs from Navidrome using the getTopSongs endpoint."""
    params = {"count": limit}
    response = _navidrome_request("getTopSongs", params)
    if response and "topSongs" in response and "song" in response["topSongs"]:
        songs = response["topSongs"]["song"]
        # Normalize keys to match Jellyfin's output for compatibility
        return [{
            'Id': s.get('id'),
            'Name': s.get('title'),
            'AlbumArtist': s.get('artist'),
            'Path': s.get('path'),
            'UserData': { # Emulate Jellyfin's UserData structure
                'PlayCount': s.get('playCount', 0),
                # lastPlayed is not in getTopSongs, must be fetched separately
                'LastPlayedDate': None 
            }
        } for s in songs]
    return []

def _navidrome_get_last_played_time(item_id):
    """Fetches the last played time for a specific track from Navidrome."""
    params = {"id": item_id}
    response = _navidrome_request("getSong", params)
    if response and "song" in response:
        return response["song"].get("lastPlayed") # Returns a timestamp string like "2023-10-27T10:00:00Z" or None
    return None

# ##############################################################################
# PUBLIC API (Dispatcher functions)
# ##############################################################################

def delete_automatic_playlists():
    """
    Finds and deletes all playlists ending with '_automatic' on the configured media server.
    """
    logger.info("Starting deletion of all '_automatic' playlists.")
    deleted_count = 0
    if config.MEDIASERVER_TYPE == 'jellyfin':
        playlists = _jellyfin_get_all_playlists()
        for playlist in playlists:
            if playlist.get('Name', '').endswith('_automatic'):
                #logger.info(f"Found Jellyfin playlist to delete: {playlist.get('Name')} (ID: {playlist.get('Id')})")
                if _jellyfin_delete_playlist(playlist.get('Id')):
                    deleted_count += 1
    elif config.MEDIASERVER_TYPE == 'navidrome':
        playlists = _navidrome_get_all_playlists()
        for playlist in playlists:
            # Use the normalized 'Name' and 'Id' keys
            if playlist.get('Name', '').endswith('_automatic'):
                logger.info(f"Found Navidrome playlist to delete: {playlist.get('Name')} (ID: {playlist.get('Id')})")
                if _navidrome_delete_playlist(playlist.get('id')): # API still needs original 'id'
                    deleted_count += 1
    else:
        logger.error(f"Unsupported media server type for automatic playlist deletion: {config.MEDIASERVER_TYPE}")
    
    logger.info(f"Finished deletion of '_automatic' playlists. Deleted {deleted_count} playlists.")

def get_recent_albums(limit):
    """
    Fetches a list of the most recently added albums from the configured media server.
    """
    if config.MEDIASERVER_TYPE == 'jellyfin':
        return _jellyfin_get_recent_albums(limit)
    elif config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_get_recent_albums(limit)
    else:
        logger.error(f"Unsupported media server type: {config.MEDIASERVER_TYPE}")
        return []

def get_tracks_from_album(album_id):
    """
    Fetches all audio tracks for a given album ID from the configured media server.
    """
    if config.MEDIASERVER_TYPE == 'jellyfin':
        return _jellyfin_get_tracks_from_album(album_id)
    elif config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_get_tracks_from_album(album_id)
    else:
        logger.error(f"Unsupported media server type: {config.MEDIASERVER_TYPE}")
        return []

def download_track(temp_dir, item):
    """
    Downloads a track from the configured media server.
    """
    if config.MEDIASERVER_TYPE == 'jellyfin':
        return _jellyfin_download_track(temp_dir, item)
    elif config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_download_track(temp_dir, item)
    else:
        logger.error(f"Unsupported media server type for download: {config.MEDIASERVER_TYPE}")
        return None



def get_all_songs():
    """
    Fetches all songs from the configured media server.
    """
    if config.MEDIASERVER_TYPE == 'jellyfin':
        return _jellyfin_get_all_songs()
    elif config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_get_all_songs()
    else:
        logger.error(f"Unsupported media server type: {config.MEDIASERVER_TYPE}")
        return []

def get_playlist_by_name(playlist_name):
    """
    Finds a playlist by its exact name on the configured media server.
    """
    if not playlist_name or not playlist_name.strip():
        raise ValueError("Playlist name cannot be empty.")
        
    if config.MEDIASERVER_TYPE == 'jellyfin':
        return _jellyfin_get_playlist_by_name(playlist_name)
    elif config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_get_playlist_by_name(playlist_name)
    else:
        logger.error(f"Unsupported media server type: {config.MEDIASERVER_TYPE}")
        return None

def create_playlist(base_name, item_ids):
    """
    Creates a new playlist on the configured media server.
    """
    if not base_name or not base_name.strip():
        raise ValueError("Playlist name cannot be empty.")
    if not item_ids:
        raise ValueError("No item IDs provided for playlist creation.")
        
    if config.MEDIASERVER_TYPE == 'jellyfin':
        _jellyfin_create_playlist(base_name, item_ids)
    elif config.MEDIASERVER_TYPE == 'navidrome':
        _navidrome_create_playlist(base_name, item_ids)
    else:
        logger.error(f"Unsupported media server type: {config.MEDIASERVER_TYPE}")

def create_instant_playlist(playlist_name, item_ids):
    """
    Creates a new playlist with an '_instant' suffix on the configured media server.
    Returns a JSON object for the created playlist on success.
    """
    if not playlist_name or not playlist_name.strip():
        raise ValueError("Playlist name cannot be empty.")
    if not item_ids:
        raise ValueError("No item IDs provided for playlist creation.")

    if config.MEDIASERVER_TYPE == 'jellyfin':
        return _jellyfin_create_instant_playlist(playlist_name, item_ids)
    elif config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_create_instant_playlist(playlist_name, item_ids)
    else:
        logger.error(f"Unsupported media server type: {config.MEDIASERVER_TYPE}")
        return None

def get_top_played_songs(limit):
    """
    Fetches the top N most played songs from the configured media server.
    """
    if config.MEDIASERVER_TYPE == 'jellyfin':
        return _jellyfin_get_top_played_songs(limit)
    elif config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_get_top_played_songs(limit)
    else:
        logger.error(f"Unsupported media server type: {config.MEDIASERVER_TYPE}")
        return []

def get_last_played_time(item_id):
    """
    Fetches the last played time for a specific track from the configured media server.
    Returns a UTC timestamp string or None.
    """
    if config.MEDIASERVER_TYPE == 'jellyfin':
        return _jellyfin_get_last_played_time(item_id)
    elif config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_get_last_played_time(item_id)
    else:
        logger.error(f"Unsupported media server type: {config.MEDIASERVER_TYPE}")
        return None
