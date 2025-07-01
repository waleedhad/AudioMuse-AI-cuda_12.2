# app_annoy.py
from flask import Blueprint, jsonify, request, render_template
import logging

# Import the new playlist creation function
from tasks.annoy_manager import find_nearest_neighbors_by_id, create_jellyfin_playlist_from_ids

logger = logging.getLogger(__name__)

# Create a Blueprint for Annoy (similarity) related routes
annoy_bp = Blueprint('annoy_bp', __name__)

@annoy_bp.route('/similarity', methods=['GET'])
def similarity_page():
    """
    Serves the frontend page for finding similar tracks.
    """
    return render_template('similarity.html')

@annoy_bp.route('/api/similar_tracks/<item_id>', methods=['GET'])
def get_similar_tracks_endpoint(item_id):
    """
    Find similar tracks for a given track ID.
    This endpoint uses the pre-loaded Annoy index for fast lookups.
    ---
    tags:
      - Similarity
    parameters:
      - name: item_id
        in: path
        required: true
        description: The Jellyfin Item ID of the track to find neighbors for.
        schema:
          type: string
      - name: n
        in: query
        description: The number of similar tracks to return.
        schema:
          type: integer
          default: 10
    responses:
      200:
        description: A list of similar tracks with their details.
        content:
          application/json:
            schema:
              type: array
              items:
                $ref: '#/components/schemas/SimilarTrack'
      404:
        description: Target track not found in the index or no similar tracks found.
      500:
        description: Server error, e.g., Annoy index not loaded.
    components:
      schemas:
        SimilarTrack:
          type: object
          properties:
            item_id:
              type: string
            title:
              type: string
            author:
              type: string
            distance:
              type: number
              format: float
              description: The similarity distance (lower is more similar).
    """
    num_neighbors = request.args.get('n', 10, type=int)
    try:
        # This function now returns a list of dicts with {'item_id': id, 'distance': dist}
        neighbor_results = find_nearest_neighbors_by_id(item_id, n=num_neighbors)
        if not neighbor_results:
            return jsonify({"error": "Target track not found in index or no similar tracks found."}), 404

        # Import the DB utility function locally to avoid circular import with app.py
        from app import get_score_data_by_ids

        # Get the full track details for the neighbors
        neighbor_ids = [n['item_id'] for n in neighbor_results]
        neighbor_details = get_score_data_by_ids(neighbor_ids)

        # Create a map for easy lookup
        details_map = {d['item_id']: d for d in neighbor_details}
        distance_map = {n['item_id']: n['distance'] for n in neighbor_results}

        # Combine details with distance, preserving the order from Annoy
        final_results = []
        for neighbor_id in neighbor_ids:
            if neighbor_id in details_map:
                track_info = details_map[neighbor_id]
                final_results.append({
                    "item_id": track_info['item_id'],
                    "title": track_info['title'],
                    "author": track_info['author'],
                    "distance": distance_map[neighbor_id]
                })

        return jsonify(final_results)
    except RuntimeError as e:
        logger.error(f"Runtime error finding neighbors for {item_id}: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error finding neighbors for {item_id}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500

@annoy_bp.route('/api/create_playlist', methods=['POST'])
def create_jellyfin_playlist():
    """
    Creates a new playlist in Jellyfin with the provided tracks.
    ---
    tags:
      - Similarity
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              playlist_name:
                type: string
                description: The name for the new playlist.
              track_ids:
                type: array
                items:
                  type: string
                description: A list of Jellyfin Item IDs to add to the playlist, including the original and similar tracks.
    responses:
      201:
        description: Playlist created successfully.
      400:
        description: Bad request, missing playlist name or track IDs.
      500:
        description: Failed to create playlist in Jellyfin.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    playlist_name = data.get('playlist_name')
    # The incoming track_ids list is the source of truth, containing the seed and similar tracks.
    track_ids_raw = data.get('track_ids', []) 

    if not playlist_name:
        return jsonify({"error": "Missing 'playlist_name'"}), 400

    # Process the incoming list to handle different formats and remove duplicates.
    final_track_ids = []
    if isinstance(track_ids_raw, list):
        for item in track_ids_raw:
            item_id = None
            if isinstance(item, str):
                item_id = item
            elif isinstance(item, dict) and 'item_id' in item:
                item_id = item['item_id']
            
            # Add the track ID if it's valid and not already in the list
            if item_id and item_id not in final_track_ids:
                final_track_ids.append(item_id)

    if not final_track_ids:
        return jsonify({"error": "No valid track IDs were provided to create the playlist"}), 400

    try:
        # === REAL JELLYFIN CLIENT LOGIC ===
        # Call the helper function from annoy_manager with the final, ordered list of track IDs
        new_playlist_id = create_jellyfin_playlist_from_ids(playlist_name, final_track_ids)
        
        logger.info(f"Successfully created playlist '{playlist_name}' with ID {new_playlist_id}.")
        
        return jsonify({
            "message": f"Playlist '{playlist_name}' created successfully!",
            "playlist_id": new_playlist_id
        }), 201

    except Exception as e:
        logger.error(f"Failed to create Jellyfin playlist '{playlist_name}': {e}", exc_info=True)
        return jsonify({"error": f"Failed to create playlist: {str(e)}"}), 500
