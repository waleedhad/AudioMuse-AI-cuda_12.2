# tasks/mediaserver.py

import requests
import logging
import os
import hashlib
import string
import random
import config  # Import the config module to access server type and settings

logger = logging.getLogger(__name__)

# ##############################################################################
# JELLYFIN IMPLEMENTATION
# ##############################################################################

def _jellyfin_get_recent_albums(limit):
    """Fetches a list of the most recently added albums from Jellyfin."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "MusicAlbum", "SortBy": "DateCreated", "SortOrder": "Descending", "Recursive": True}
    if limit != 0:
        params["Limit"] = limit
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"Jellyfin get_recent_albums failed: {e}", exc_info=True)
        return []

def _jellyfin_get_tracks_from_album(album_id):
    """Fetches all audio tracks for a given album ID from Jellyfin."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"ParentId": album_id, "IncludeItemTypes": "Audio"}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=30)
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

        with requests.get(download_url, headers=config.HEADERS, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Downloaded '{item['Name']}' to '{local_filename}'")
        return local_filename
    except Exception as e:
        logger.error(f"Failed to download track {item.get('Name', 'Unknown')}: {e}", exc_info=True)
        return None

def _jellyfin_get_all_albums():
    """Fetches all music albums from Jellyfin."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "MusicAlbum", "Recursive": True}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=120)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"Jellyfin get_all_albums failed: {e}", exc_info=True)
        return []

def _jellyfin_get_all_songs():
    """Fetches all songs from Jellyfin."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Audio", "Recursive": True}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=300)
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
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=30)
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
        r = requests.post(url, headers=config.HEADERS, json=body, timeout=60)
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
        r = requests.post(url, headers=config.HEADERS, json=body, timeout=60)
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
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=60)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"Jellyfin get_all_playlists failed: {e}", exc_info=True)
        return []

def _jellyfin_delete_playlist(playlist_id):
    """Deletes a playlist on Jellyfin."""
    url = f"{config.JELLYFIN_URL}/Items/{playlist_id}"
    try:
        r = requests.delete(url, headers=config.HEADERS, timeout=30)
        r.raise_for_status()
        logger.info(f"🗑️ Deleted Jellyfin playlist ID: {playlist_id}")
        return True
    except Exception as e:
        logger.error(f"Exception deleting Jellyfin playlist ID {playlist_id}: {e}", exc_info=True)
        return False


# ##############################################################################
# NAVIDROME (SUBSONIC API) IMPLEMENTATION
# ##############################################################################

def get_navidrome_auth_params():
    """
    Generates the salt and token for Subsonic API authentication using the password.
    Returns a dictionary of parameters required for every Navidrome API call.
    """
    if not config.NAVIDROME_PASSWORD or not config.NAVIDROME_USER:
        logger.warning("Navidrome User or Password is not configured in environment variables.")
        return {}
    
    salt = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    token = hashlib.md5((config.NAVIDROME_PASSWORD + salt).encode('utf-8')).hexdigest()
    
    return {
        "u": config.NAVIDROME_USER,
        "t": token,
        "s": salt,
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
            r = requests.get(url, params=all_params, timeout=30, stream=stream)
        elif method.lower() == 'post':
            r = requests.post(url, params=all_params, timeout=60)
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
    """Fetches a list of the most recently added albums from Navidrome."""
    params = {"type": "newest", "size": limit if limit != 0 else 500}
    response = _navidrome_request("getAlbumList2", params)
    if response and "albumList2" in response and "album" in response["albumList2"]:
        return response["albumList2"]["album"]
    return []

def _navidrome_get_tracks_from_album(album_id):
    """Fetches all audio tracks for a given album ID from Navidrome."""
    params = {"id": album_id}
    response = _navidrome_request("getAlbum", params)
    if response and "album" in response and "song" in response["album"]:
        return response["album"]["song"]
    return []

def _navidrome_get_all_albums():
    """Fetches all albums from Navidrome, paginating through the results."""
    all_albums = []
    offset = 0
    limit = 500
    while True:
        params = {"type": "alphabeticalByName", "size": limit, "offset": offset}
        response = _navidrome_request("getAlbumList2", params)
        if response and "albumList2" in response and "album" in response["albumList2"]:
            albums = response["albumList2"]["album"]
            if not albums:
                break
            all_albums.extend(albums)
            offset += len(albums)
            if len(albums) < limit:
                break
        else:
            logger.error("Failed to fetch albums page from Navidrome.")
            break
    return all_albums

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
            all_songs.extend(songs)
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
                    return playlist_details_response["playlist"]
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
        return response["playlists"]["playlist"]
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
            if playlist.get('name', '').endswith('_automatic'):
                logger.info(f"Found Navidrome playlist to delete: {playlist.get('name')} (ID: {playlist.get('id')})")
                if _navidrome_delete_playlist(playlist.get('id')):
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

def get_all_albums():
    """
    Fetches all music albums from the configured media server.
    """
    if config.MEDIASERVER_TYPE == 'jellyfin':
        return _jellyfin_get_all_albums()
    elif config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_get_all_albums()
    else:
        logger.error(f"Unsupported media server type: {config.MEDIASERVER_TYPE}")
        return []

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
