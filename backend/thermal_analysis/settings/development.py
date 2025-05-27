# backend/thermal_analysis/settings/development.py
from .base import *
import os

DEBUG = os.getenv('DEBUG', '1') == '1'

ALLOWED_HOSTS = ['localhost', '127.0.0.1', 'backend', '*']

# Additional development settings
CORS_ALLOW_ALL_ORIGINS = True

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}

# Environment-specific database override if needed
if os.getenv('DATABASE_URL'):
    import dj_database_url
    DATABASES['default'] = dj_database_url.parse(
        os.getenv('DATABASE_URL'),
        conn_max_age=600,
        conn_health_checks=True,
    )
    DATABASES['default']['ENGINE'] = 'django.contrib.gis.db.backends.postgis'