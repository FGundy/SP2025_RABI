"""
SAM 2 Celery Worker
"""
# sam2_worker/worker.py
import os
from celery import Celery

# Configure Celery app
celery = Celery('sam2_worker')

# Configure from environment
celery.conf.update(
    broker_url=os.getenv('CELERY_BROKER', 'redis://redis:6379/0'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'),
    accept_content=['json'],
    task_serializer='json',
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task routing
    task_routes={
        'sam2_service.tasks.*': {'queue': 'sam2_queue'},
    },
    
    # Worker configuration for GPU tasks
    worker_prefetch_multiplier=1,  # Important for GPU tasks
    task_acks_late=True,
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks to prevent memory leaks
)

# Make sure celery app is available for autodiscovery
app = celery

if __name__ == '__main__':
    celery.start()