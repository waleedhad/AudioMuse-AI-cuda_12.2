from app import celery, init_db, clean_temp, JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, HEADERS, NUM_RECENT_ALBUMS, TOP_N_MOODS # Import everything needed by the task
import os

# You might need to adjust the path if essentia models are not directly accessible
# in the worker environment. Ensure these paths are correct relative to where
# the worker is started or are absolute paths.
os.environ["EMBEDDING_MODEL_PATH"] = "/app/msd-musicnn-1.pb"
os.environ["PREDICTION_MODEL_PATH"] = "/app/msd-msd-musicnn-1.pb"

# If your essentia models need to be loaded at worker startup or are defined
# as global constants, make sure they are accessible. For example, if they're
# in config.py, ensure config is imported and those paths are correctly set.
# The `app.py` script imports `config.py` which defines these.
# When the worker runs, it uses the `celery` object defined in `app.py`,
# which means it will have access to the global variables from `config.py`.

# This file only needs to define the celery app instance, and the tasks are
# automatically discovered from `app.py` because `app` is imported.

# To run this worker:
# celery -A celery_worker worker --loglevel=info
