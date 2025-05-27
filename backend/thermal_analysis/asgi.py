"""
ASGI config for thermal_analysis project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""
# backend/thermal_analysis/asgi.py
import os

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

# Import your routing
try:
    import apps.analysis.routing
    websocket_urlpatterns = apps.analysis.routing.websocket_urlpatterns
except ImportError:
    websocket_urlpatterns = []

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'thermal_analysis.settings.development')

django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
})