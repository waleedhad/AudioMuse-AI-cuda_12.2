# app_annoy.py
from flask import Blueprint, jsonify, request
import logging

logger = logging.getLogger(__name__)

# Create a Blueprint for Annoy (similarity) related routes
annoy_bp = Blueprint('annoy_bp', __name__)

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
    # Import locally to prevent circular dependency at module load time.
    from tasks.annoy_manager import find_nearest_neighbors_by_id
    num_neighbors = request.args.get('n', 10, type=int)
    try:
        # This function now returns a list of dicts with {'item_id': id, 'distance': dist}
        neighbor_results = find_nearest_neighbors_by_id(item_id, n=num_neighbors)
        if not neighbor_results:
            return jsonify({"error": "Target track not found in index or no neighbors found."}), 404

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
