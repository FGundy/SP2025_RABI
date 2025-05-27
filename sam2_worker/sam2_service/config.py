
# sam2_worker/sam2_service/config.py
import os
from pathlib import Path
from decouple import config

# SAM 2 Configuration
APP_ROOT = Path(os.getenv("APP_ROOT", "/opt/sam2_worker"))
MODEL_SIZE = config("MODEL_SIZE", default="base_plus")
SAM2_BUILD_CUDA = config("SAM2_BUILD_CUDA", default="0")

# Device configuration
FORCE_CPU_DEVICE = config("SAM2_DEMO_FORCE_CPU_DEVICE", default="0") == "1"

# Model paths
CHECKPOINTS_DIR = APP_ROOT / "sam2" / "checkpoints"

MODEL_CONFIGS = {
    "tiny": {
        "checkpoint": CHECKPOINTS_DIR / "sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml"
    },
    "small": {
        "checkpoint": CHECKPOINTS_DIR / "sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml"
    },
    "base_plus": {
        "checkpoint": CHECKPOINTS_DIR / "sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml"
    },
    "large": {
        "checkpoint": CHECKPOINTS_DIR / "sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml"
    }
}

# MinIO Configuration
MINIO_ENDPOINT = config('MINIO_ENDPOINT', default='minio:9000')
MINIO_ACCESS_KEY = config('MINIO_ACCESS_KEY', default='minioadmin')
MINIO_SECRET_KEY = config('MINIO_SECRET_KEY', default='minioadmin123')
MINIO_SECURE = config('MINIO_SECURE', default=False, cast=bool)