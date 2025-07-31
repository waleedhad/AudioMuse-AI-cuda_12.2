# app_sonic_fingerprint.py
from flask import Blueprint, jsonify, request, render_template
import logging

from tasks.sonic_fingerprint_manager import generate_sonic_fingerprint
from app import get_score_data_by_ids
from config import MEDIASERVER_TYPE # Import MEDIASERVER_TYPE

logger = logging.getLogger(__name__)

# Create a blueprint for the new feature
sonic_fingerprint_bp = Blueprint('sonic_fingerprint_bp', __name__, template_folder='../templates')

@sonic_fingerprint_bp.route('/sonic_fingerprint', methods=['GET'])
def sonic_fingerprint_page():
    """
    Serves the frontend page for the Sonic Fingerprint feature.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the Sonic Fingerprint page.
        content:
          text/html:
            schema:
              type: string
    """
    try:
        # Pass the media server type to the template
        return render_template('sonic_fingerprint.html', mediaserver_type=MEDIASERVER_TYPE)
    except Exception as e:
         logger.error(f"Error rendering sonic_fingerprint.html: {e}", exc_info=True)
         return "Sonic Fingerprint page not implemented yet. Use the API at /api/sonic_fingerprint/generate"


@sonic_fingerprint_bp.route('/api/sonic_fingerprint/generate', methods=['GET'])
def generate_sonic_fingerprint_endpoint():
    """
    Generates a sonic fingerprint based on user's listening habits
    and returns a list of recommended tracks.
    ---
    tags:
      - Sonic Fingerprint
    parameters:
      - name: n
        in: query
        type: integer
        required: false
        description: The number of results to return. Overrides the server default.
      - name: jellyfin_user_id
        in: query
        type: string
        required: false
        description: The Jellyfin User ID (required if media server is Jellyfin).
      - name: jellyfin_token
        in: query
        type: string
        required: false
        description: The Jellyfin API Token (required if media server is Jellyfin).
      - name: navidrome_user
        in: query
        type: string
        required: false
        description: The Navidrome username (required if media server is Navidrome).
      - name: navidrome_password
        in: query
        type: string
        required: false
        description: The Navidrome password (required if media server is Navidrome).
    responses:
      200:
        description: A list of recommended tracks based on the sonic fingerprint.
      400:
        description: Bad Request - Missing necessary credentials for the configured media server.
      500:
        description: Server error during generation.
    """
    try:
        num_results = request.args.get('n', type=int)
        
        # --- Extract user credentials from request ---
        user_creds = {}
        if MEDIASERVER_TYPE == 'jellyfin':
            user_creds['user_id'] = request.args.get('jellyfin_user_id')
            user_creds['token'] = request.args.get('jellyfin_token')
            if not user_creds['user_id'] or not user_creds['token']:
                return jsonify({"error": "Jellyfin User ID and API Token are required."}), 400
        elif MEDIASERVER_TYPE == 'navidrome':
            user_creds['user'] = request.args.get('navidrome_user')
            user_creds['password'] = request.args.get('navidrome_password')
            if not user_creds['user'] or not user_creds['password']:
                return jsonify({"error": "Navidrome username and password are required."}), 400
        
        # The manager function now accepts user credentials
        fingerprint_results = generate_sonic_fingerprint(
            num_neighbors=num_results,
            user_creds=user_creds
        )

        if not fingerprint_results:
            return jsonify([])

        # Enrich results with title and author from the database for the response
        result_ids = [r['item_id'] for r in fingerprint_results]
        details_list = get_score_data_by_ids(result_ids)
        
        details_map = {d['item_id']: d for d in details_list}
        distance_map = {r['item_id']: r['distance'] for r in fingerprint_results}

        # Reconstruct the final list, ensuring order is preserved
        final_results = []
        for res_id in result_ids:
            if res_id in details_map:
                track_info = details_map[res_id]
                final_results.append({
                    "item_id": track_info['item_id'],
                    "title": track_info['title'],
                    "author": track_info['author'],
                    "distance": distance_map[res_id]
                })

        return jsonify(final_results)
    except Exception as e:
        logger.error(f"Error in sonic_fingerprint endpoint: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred while generating the sonic fingerprint."}), 500
