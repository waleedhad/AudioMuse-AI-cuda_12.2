# tasks/mediaserver.py

import requests
import logging
import os

logger = logging.getLogger(__name__)

def get_recent_albums(jellyfin_url, jellyfin_user_id, headers, limit):
    """
    Fetches a list of the most recently added albums from Jellyfin.
    """
    url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {"IncludeItemTypes": "MusicAlbum", "SortBy": "DateCreated", "SortOrder": "Descending", "Recursive": True}
    if limit != 0:  # A limit of 0 means get all
        params["Limit"] = limit
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"get_recent_albums: {e}", exc_info=True)
        return []

def get_tracks_from_album(jellyfin_url, jellyfin_user_id, headers, album_id):
    """
    Fetches all audio tracks for a given album ID from Jellyfin.
    """
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
    """
    Downloads a single track from Jellyfin to a temporary directory.
    """
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

def delete_old_automatic_playlists(jellyfin_url, jellyfin_user_id, headers):
    """
    Deletes all playlists from Jellyfin that have '_automatic' in their name.
    """
    url = f"{jellyfin_url}/Users/{jellyfin_user_id}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        for item in r.json().get("Items", []):
            if "_automatic" in item.get("Name", ""):
                del_url = f"{jellyfin_url}/Items/{item['Id']}"
                del_resp = requests.delete(del_url, headers=headers, timeout=10)
                if del_resp.ok:
                    logger.info("🗑️ Deleted old playlist: %s", item['Name'])
    except Exception as e:
        logger.error("Failed to clean old playlists: %s", e, exc_info=True)

def create_or_update_playlists_on_jellyfin(jellyfin_url, jellyfin_user_id, headers, playlists, cluster_centers, mood_labels_list, max_songs_per_cluster):
    """
    Creates new playlists on Jellyfin after deleting old ones.
    """
    delete_old_automatic_playlists(jellyfin_url, jellyfin_user_id, headers)
    for base_name, item_ids in playlists.items():
        if not item_ids:
            continue
        body = {"Name": base_name, "Ids": item_ids, "UserId": jellyfin_user_id}
        try:
            r = requests.post(f"{jellyfin_url}/Playlists", headers=headers, json=body, timeout=60)
            if r.ok:
                logger.info("✅ Created playlist '%s' with %s tracks", base_name, len(item_ids))
        except Exception as e:
            logger.error("Exception creating playlist '%s': %s", base_name, e, exc_info=True)

def create_instant_playlist(jellyfin_url, jellyfin_user_id, headers, playlist_name, item_ids):
    """
    Creates a new playlist on Jellyfin with an '_instant' suffix.
    Returns the full JSON response object for the created playlist on success.
    """
    if not playlist_name or not playlist_name.strip():
        raise ValueError("Playlist name cannot be empty.")
    if not item_ids:
        raise ValueError("No item IDs provided for playlist creation.")

    final_playlist_name = f"{playlist_name.strip()}_instant"
    url = f"{jellyfin_url}/Playlists"
    body = {
        "Name": final_playlist_name,
        "Ids": item_ids,
        "UserId": jellyfin_user_id
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        created_playlist = r.json()
        logger.info("✅ Created instant playlist '%s' with ID: %s", final_playlist_name, created_playlist.get('Id'))
        return created_playlist
    except requests.exceptions.RequestException as e:
        logger.error("Failed to create instant playlist '%s': %s", final_playlist_name, e, exc_info=True)
        raise Exception(f"Failed to communicate with Jellyfin: {e}") from e
