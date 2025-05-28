import os
import redis
from rq import Connection, Worker, Queue

# Ensure model paths are set for the worker environment
# These should match the paths accessible by your worker instances.
os.environ["EMBEDDING_MODEL_PATH"] = os.environ.get("EMBEDDING_MODEL_PATH", "/app/msd-musicnn-1.pb")
os.environ["PREDICTION_MODEL_PATH"] = os.environ.get("PREDICTION_MODEL_PATH", "/app/msd-msd-musicnn-1.pb")

# Import the Flask app instance (for app context in tasks)
# This is crucial because tasks in tasks.py use `with app.app_context():`
from app import app

# Import task functions from the tasks.py file
from tasks import analyze_album_task, run_analysis_task, run_single_clustering_iteration_task, run_clustering_task

listen_queues = ['default']  # Specify the queues this worker will listen to

# Get Redis URL from environment or use a default
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

if __name__ == '__main__':
    redis_conn = redis.from_url(redis_url)
    with Connection(redis_conn):
        # The tasks are imported from tasks.py, RQ will find them by their names
        # when they are enqueued.
        # The 'app' instance is available in the global scope of this worker
        # process, so tasks.py can import and use it for app_context.
        worker = Worker(
            queues=[Queue(name, connection=redis_conn) for name in listen_queues],
            connection=redis_conn
        )
        # worker.work(with_scheduler=True) # Uncomment if you plan to use RQ Scheduler
        worker.work()
