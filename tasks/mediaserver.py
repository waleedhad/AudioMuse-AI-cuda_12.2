# tasks/mediaserver.py

import requests
import logging
import os
import random
import config  # Import the config module to access server type and settings

logger = logging.getLogger(__name__)

# Define a global timeout for all requests
REQUESTS_TIMEOUT = 300

# ##############################################################################
# JELLYFIN IMPLEMENTATION
# ##############################################################################

def _jellyfin_get_users(token):
    """Fetches a list of all users from Jellyfin using a provided token."""
    url = f"{config.JELLYFIN_URL}/Users"
    headers = {"X-Emby-Token": token}
    try:
        r = requests.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Jellyfin get_users failed: {e}", exc_info=True)
        return None

def _jellyfin_resolve_user(identifier, token):
    """
    Resolves a Jellyfin username to a User ID.
    If the identifier doesn't match any username, it's returned as is, assuming it's already an ID.
    """
    users = _jellyfin_get_users(token)
    if users:
        for user in users:
            if user.get('Name', '').lower() == identifier.lower():
                logger.info(f"Matched username '{identifier}' to User ID '{user['Id']}'.")
                return user['Id']
    
    logger.info(f"No username match for '{identifier}'. Assuming it is a User ID.")
    return identifier # Return original identifier if no match is found

# --- ADMIN/GLOBAL JELLYFIN FUNCTIONS ---
def _jellyfin_get_recent_albums(limit):
    """
    Fetches a list of the most recently added albums from Jellyfin using pagination.
    Uses global admin credentials.
    """
    all_albums = []
    start_index = 0
    page_size = 500
    fetch_all = (limit == 0)
    while fetch_all or len(all_albums) < limit:
        size_to_fetch = page_size if fetch_all else min(page_size, limit - len(all_albums))
        if size_to_fetch <= 0: break
        url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
        params = {"IncludeItemTypes": "MusicAlbum", "SortBy": "DateCreated", "SortOrder": "Descending", "Recursive": True, "Limit": size_to_fetch, "StartIndex": start_index}
        try:
            r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
            response_data = r.json()
            albums = response_data.get("Items", [])
            if not albums: break
            all_albums.extend(albums)
            start_index += len(albums)
            if len(albums) < size_to_fetch: break
            if fetch_all and start_index >= response_data.get("TotalRecordCount", float('inf')): break
        except Exception as e:
            logger.error(f"Jellyfin get_recent_albums failed: {e}", exc_info=True)
            break
    return all_albums

def _jellyfin_get_tracks_from_album(album_id):
    """Fetches all audio tracks for a given album ID from Jellyfin using admin credentials."""
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
    """Downloads a single track from Jellyfin using admin credentials."""
    try:
        track_id = item['Id']
        file_extension = os.path.splitext(item.get('Path', ''))[1] or '.tmp'
        download_url = f"{config.JELLYFIN_URL}/Items/{track_id}/Download"
        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")
        with requests.get(download_url, headers=config.HEADERS, stream=True, timeout=REQUESTS_TIMEOUT) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        logger.info(f"Downloaded '{item['Name']}' to '{local_filename}'")
        return local_filename
    except Exception as e:
        logger.error(f"Failed to download track {item.get('Name', 'Unknown')}: {e}", exc_info=True)
        return None

def _jellyfin_get_all_songs():
    """Fetches all songs from Jellyfin using admin credentials."""
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
    """Finds a Jellyfin playlist by its exact name using admin credentials."""
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
    """Creates a new playlist on Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Playlists"
    body = {"Name": base_name, "Ids": item_ids, "UserId": config.JELLYFIN_USER_ID}
    try:
        r = requests.post(url, headers=config.HEADERS, json=body, timeout=REQUESTS_TIMEOUT)
        if r.ok: logger.info("✅ Created Jellyfin playlist '%s'", base_name)
    except Exception as e:
        logger.error("Exception creating Jellyfin playlist '%s': %s", base_name, e, exc_info=True)

def _jellyfin_get_all_playlists():
    """Fetches all playlists from Jellyfin using admin credentials."""
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
    """Deletes a playlist on Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Items/{playlist_id}"
    try:
        r = requests.delete(url, headers=config.HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Exception deleting Jellyfin playlist ID {playlist_id}: {e}", exc_info=True)
        return False

# --- USER-SPECIFIC JELLYFIN FUNCTIONS ---
def _jellyfin_get_top_played_songs(limit, user_id, token):
    """Fetches the top N most played songs from Jellyfin for a specific user."""
    url = f"{config.JELLYFIN_URL}/Users/{user_id}/Items"
    headers = {"X-Emby-Token": token}
    params = {"IncludeItemTypes": "Audio", "SortBy": "PlayCount", "SortOrder": "Descending", "Recursive": True, "Limit": limit, "Fields": "UserData,Path"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"Jellyfin get_top_played_songs failed for user {user_id}: {e}", exc_info=True)
        return []

def _jellyfin_get_last_played_time(item_id, user_id, token):
    """Fetches the last played time for a specific track from Jellyfin for a specific user."""
    url = f"{config.JELLYFIN_URL}/Users/{user_id}/Items/{item_id}"
    headers = {"X-Emby-Token": token}
    params = {"Fields": "UserData"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("UserData", {}).get("LastPlayedDate")
    except Exception as e:
        logger.error(f"Jellyfin get_last_played_time failed for item {item_id}, user {user_id}: {e}", exc_info=True)
        return None

def _jellyfin_create_instant_playlist(playlist_name, item_ids, user_id, token):
    """Creates a new instant playlist on Jellyfin for a specific user."""
    final_playlist_name = f"{playlist_name.strip()}_instant"
    url = f"{config.JELLYFIN_URL}/Playlists"
    headers = {"X-Emby-Token": token}
    body = {"Name": final_playlist_name, "Ids": item_ids, "UserId": user_id}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Exception creating Jellyfin instant playlist '%s' for user %s: %s", playlist_name, user_id, e, exc_info=True)
        return None

# ##############################################################################
# NAVIDROME (SUBSONIC API) IMPLEMENTATION
# ##############################################################################
def get_navidrome_auth_params(username=None, password=None):
    """Generates Navidrome auth params, using provided creds or falling back to global config."""
    auth_user = username or config.NAVIDROME_USER
    auth_pass = password or config.NAVIDROME_PASSWORD
    if not auth_user or not auth_pass: 
        logger.warning("Navidrome User or Password is not configured.")
        return {}
    hex_encoded_password = auth_pass.encode('utf-8').hex()
    return {"u": auth_user, "p": f"enc:{hex_encoded_password}", "v": "1.16.1", "c": config.APP_VERSION, "f": "json"}

def _navidrome_request(endpoint, params=None, method='get', stream=False, user_creds=None):
    """Helper to make Navidrome API requests using specific or global user credentials."""
    params = params or {}
    auth_params = get_navidrome_auth_params(username=user_creds.get('user') if user_creds else None, password=user_creds.get('password') if user_creds else None)
    if not auth_params: 
        logger.error("Navidrome credentials not configured. Cannot make API call.")
        return None
    url = f"{config.NAVIDROME_URL}/rest/{endpoint}.view"
    all_params = {**auth_params, **params}
    try:
        r = requests.request(method, url, params=all_params, timeout=REQUESTS_TIMEOUT, stream=stream)
        r.raise_for_status()
        if stream: return r
        subsonic_response = r.json().get("subsonic-response", {})
        if subsonic_response.get("status") == "failed":
            error = subsonic_response.get("error", {})
            logger.error(f"Navidrome API Error on '{endpoint}': {error.get('message')}")
            return None
        return subsonic_response
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Navidrome API endpoint '{endpoint}': {e}", exc_info=True)
        return None

def _navidrome_download_track(temp_dir, item):
    """Downloads a single track from Navidrome using admin credentials."""
    try:
        track_id = item['id'] 
        file_extension = os.path.splitext(item.get('path', ''))[1] or '.tmp'
        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")
        
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
    """Fetches a list of the most recently added albums from Navidrome using admin credentials."""
    all_albums = []
    offset = 0
    page_size = 500
    fetch_all = (limit == 0)

    while fetch_all or len(all_albums) < limit:
        size_to_fetch = page_size if fetch_all else min(page_size, limit - len(all_albums))
        if size_to_fetch <= 0: break

        params = {"type": "newest", "size": size_to_fetch, "offset": offset}
        response = _navidrome_request("getAlbumList2", params)

        if response and "albumList2" in response and "album" in response["albumList2"]:
            albums = response["albumList2"]["album"]
            if not albums: break 

            all_albums.extend([{**a, 'Id': a.get('id'), 'Name': a.get('name')} for a in albums])
            offset += len(albums)

            if len(albums) < size_to_fetch: break
        else:
            logger.error("Failed to fetch recent albums page from Navidrome.")
            break
            
    return all_albums

def _navidrome_get_all_songs():
    """Fetches all songs from Navidrome using admin credentials."""
    all_songs = []
    offset = 0
    limit = 500
    while True:
        params = {"query": '', "songCount": limit, "songOffset": offset}
        response = _navidrome_request("search3", params)
        if response and "searchResult3" in response and "song" in response["searchResult3"]:
            songs = response["searchResult3"]["song"]
            if not songs: break
            all_songs.extend([{'Id': s.get('id'), 'Name': s.get('title'), 'AlbumArtist': s.get('artist'), 'Path': s.get('path')} for s in songs])
            offset += len(songs)
            if len(songs) < limit: break
        else:
            logger.error("Failed to fetch all songs from Navidrome.")
            break
    return all_songs

def _navidrome_create_playlist(base_name, item_ids):
    """Creates a new playlist on Navidrome using admin credentials."""
    params = {"name": base_name, "songId": item_ids}
    response = _navidrome_request("createPlaylist", params, method='post')
    if response and response.get("status") == "ok":
        logger.info("✅ Created Navidrome playlist '%s'", base_name)
    else:
        logger.error("Failed to create playlist '%s' on Navidrome", base_name)

def _navidrome_get_all_playlists():
    """Fetches all playlists from Navidrome using admin credentials."""
    response = _navidrome_request("getPlaylists")
    if response and "playlists" in response and "playlist" in response["playlists"]:
        return [{**p, 'Id': p.get('id'), 'Name': p.get('name')} for p in response["playlists"]["playlist"]]
    return []

def _navidrome_delete_playlist(playlist_id):
    """Deletes a playlist on Navidrome using admin credentials."""
    response = _navidrome_request("deletePlaylist", {"id": playlist_id}, method='post')
    if response and response.get("status") == "ok":
        logger.info(f"🗑️ Deleted Navidrome playlist ID: {playlist_id}")
        return True
    logger.error(f"Failed to delete playlist ID '{playlist_id}' on Navidrome")
    return False

# --- USER-SPECIFIC NAVIDROME FUNCTIONS ---
def _navidrome_get_tracks_from_album(album_id, user_creds=None):
    """Fetches all audio tracks for an album. Uses specific user_creds if provided."""
    params = {"id": album_id}
    response = _navidrome_request("getAlbum", params, user_creds=user_creds)
    if response and "album" in response and "song" in response["album"]:
        songs = response["album"]["song"]
        return [{**s, 'Id': s.get('id'), 'Name': s.get('title'), 'AlbumArtist': s.get('artist'), 'Path': s.get('path')} for s in songs]
    return []

def _navidrome_get_playlist_by_name(playlist_name, user_creds=None):
    """Finds a Navidrome playlist by name. Uses specific user_creds if provided."""
    response = _navidrome_request("getPlaylists", user_creds=user_creds)
    if response and "playlists" in response and "playlist" in response["playlists"]:
        for playlist in response["playlists"]["playlist"]:
            if playlist.get("name") == playlist_name:
                details_resp = _navidrome_request("getPlaylist", {"id": playlist["id"]}, user_creds=user_creds)
                if details_resp and "playlist" in details_resp:
                    p = details_resp["playlist"]
                    return {**p, 'Id': p.get('id'), 'Name': p.get('name')}
    return None

def _navidrome_get_top_played_songs(limit, user_creds):
    """Fetches the top N most played songs from Navidrome for a specific user."""
    all_top_songs = []
    num_albums_to_fetch = (limit // 10) + 10
    params = {"type": "frequent", "size": num_albums_to_fetch}
    response = _navidrome_request("getAlbumList2", params, user_creds=user_creds)
    if response and "albumList2" in response and "album" in response["albumList2"]:
        for album in response["albumList2"]["album"]:
            tracks = _navidrome_get_tracks_from_album(album.get("id"), user_creds=user_creds)
            if tracks: all_top_songs.extend(tracks)
    return random.sample(all_top_songs, limit) if len(all_top_songs) > limit else all_top_songs

def _navidrome_get_last_played_time(item_id, user_creds):
    """Fetches the last played time for a track for a specific user."""
    response = _navidrome_request("getSong", {"id": item_id}, user_creds=user_creds)
    if response and "song" in response: return response["song"].get("lastPlayed")
    return None

def _navidrome_create_instant_playlist(playlist_name, item_ids, user_creds):
    """Creates a new instant playlist on Navidrome for a specific user."""
    final_playlist_name = f"{playlist_name.strip()}_instant"
    params = {"name": final_playlist_name, "songId": item_ids}
    response = _navidrome_request("createPlaylist", params, method='post', user_creds=user_creds)
    if response and response.get("status") == "ok":
        return _navidrome_get_playlist_by_name(final_playlist_name, user_creds=user_creds)
    return None

# ##############################################################################
# PUBLIC API (Dispatcher functions)
# ##############################################################################

def resolve_jellyfin_user(identifier, token):
    """Public dispatcher for resolving a Jellyfin user identifier."""
    return _jellyfin_resolve_user(identifier, token)

def delete_automatic_playlists():
    """Deletes all playlists ending with '_automatic' using admin credentials."""
    logger.info("Starting deletion of all '_automatic' playlists.")
    deleted_count = 0
    if config.MEDIASERVER_TYPE == 'jellyfin':
        for p in _jellyfin_get_all_playlists():
            if p.get('Name', '').endswith('_automatic') and _jellyfin_delete_playlist(p.get('Id')):
                deleted_count += 1
    elif config.MEDIASERVER_TYPE == 'navidrome':
        for p in _navidrome_get_all_playlists():
            if p.get('Name', '').endswith('_automatic') and _navidrome_delete_playlist(p.get('id')):
                deleted_count += 1
    logger.info(f"Finished deletion. Deleted {deleted_count} playlists.")

def get_recent_albums(limit):
    """Fetches recently added albums using admin credentials."""
    if config.MEDIASERVER_TYPE == 'jellyfin': return _jellyfin_get_recent_albums(limit)
    if config.MEDIASERVER_TYPE == 'navidrome': return _navidrome_get_recent_albums(limit)
    return []

def get_tracks_from_album(album_id):
    """Fetches tracks for an album using admin credentials."""
    if config.MEDIASERVER_TYPE == 'jellyfin': return _jellyfin_get_tracks_from_album(album_id)
    if config.MEDIASERVER_TYPE == 'navidrome': return _navidrome_get_tracks_from_album(album_id)
    return []

def download_track(temp_dir, item):
    """Downloads a track using admin credentials."""
    if config.MEDIASERVER_TYPE == 'jellyfin': return _jellyfin_download_track(temp_dir, item)
    if config.MEDIASERVER_TYPE == 'navidrome': return _navidrome_download_track(temp_dir, item)
    return None

def get_all_songs():
    """Fetches all songs using admin credentials."""
    if config.MEDIASERVER_TYPE == 'jellyfin': return _jellyfin_get_all_songs()
    if config.MEDIASERVER_TYPE == 'navidrome': return _navidrome_get_all_songs()
    return []

def get_playlist_by_name(playlist_name):
    """Finds a playlist by name using admin credentials."""
    if not playlist_name: raise ValueError("Playlist name is required.")
    if config.MEDIASERVER_TYPE == 'jellyfin': return _jellyfin_get_playlist_by_name(playlist_name)
    if config.MEDIASERVER_TYPE == 'navidrome': return _navidrome_get_playlist_by_name(playlist_name)
    return None

def create_playlist(base_name, item_ids):
    """Creates a playlist using admin credentials."""
    if not base_name: raise ValueError("Playlist name is required.")
    if not item_ids: raise ValueError("Track IDs are required.")
    if config.MEDIASERVER_TYPE == 'jellyfin': _jellyfin_create_playlist(base_name, item_ids)
    elif config.MEDIASERVER_TYPE == 'navidrome': _navidrome_create_playlist(base_name, item_ids)

def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    """Creates an instant playlist. Uses user_creds if provided, otherwise admin."""
    if not playlist_name: raise ValueError("Playlist name is required.")
    if not item_ids: raise ValueError("Track IDs are required.")
    
    if config.MEDIASERVER_TYPE == 'jellyfin':
        token = user_creds.get('token') if user_creds else config.JELLYFIN_TOKEN
        if not token: raise ValueError("Jellyfin Token is required.")
        
        identifier = user_creds.get('user_identifier') if user_creds else config.JELLYFIN_USER_ID
        if not identifier: raise ValueError("Jellyfin User Identifier is required.")

        user_id = _jellyfin_resolve_user(identifier, token)
        return _jellyfin_create_instant_playlist(playlist_name, item_ids, user_id, token)

    if config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_create_instant_playlist(playlist_name, item_ids, user_creds)
    return None

def get_top_played_songs(limit, user_creds=None):
    """Fetches top played songs. Uses user_creds if provided, otherwise admin."""
    if config.MEDIASERVER_TYPE == 'jellyfin':
        user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
        token = user_creds.get('token') if user_creds else config.JELLYFIN_TOKEN
        if not user_id or not token: raise ValueError("Jellyfin User ID and Token are required.")
        return _jellyfin_get_top_played_songs(limit, user_id, token)
    if config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_get_top_played_songs(limit, user_creds)
    return []

def get_last_played_time(item_id, user_creds=None):
    """Fetches last played time for a track. Uses user_creds if provided, otherwise admin."""
    if config.MEDIASERVER_TYPE == 'jellyfin':
        user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
        token = user_creds.get('token') if user_creds else config.JELLYFIN_TOKEN
        if not user_id or not token: raise ValueError("Jellyfin User ID and Token are required.")
        return _jellyfin_get_last_played_time(item_id, user_id, token)
    if config.MEDIASERVER_TYPE == 'navidrome':
        return _navidrome_get_last_played_time(item_id, user_creds)
    return None
