# Based on SAM 2's backend.Dockerfile
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
ARG MODEL_SIZE=base_plus

FROM ${BASE_IMAGE}

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SAM2_BUILD_CUDA=0
ENV MODEL_SIZE=${MODEL_SIZE}
ENV APP_ROOT=/opt/sam2_worker

# Install system requirements (from SAM 2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    build-essential \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
RUN mkdir -p ${APP_ROOT}
WORKDIR ${APP_ROOT}

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools
RUN pip install -r requirements.txt

# Clone and install SAM 2 (or copy if you have it locally)
RUN git clone https://github.com/facebookresearch/sam2.git
WORKDIR ${APP_ROOT}/sam2
RUN pip install -e ".[interactive-demo]"

# Fix FFmpeg path issue (make it conditional)
RUN if [ -f /opt/conda/bin/ffmpeg ]; then \
        rm /opt/conda/bin/ffmpeg && ln -s /bin/ffmpeg /opt/conda/bin/ffmpeg; \
    else \
        echo "FFmpeg already properly configured"; \
    fi

# Download SAM 2.1 checkpoints
RUN mkdir -p checkpoints
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt checkpoints/sam2.1_hiera_tiny.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt checkpoints/sam2.1_hiera_small.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt checkpoints/sam2.1_hiera_base_plus.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt checkpoints/sam2.1_hiera_large.pt

# Copy our worker code
WORKDIR ${APP_ROOT}
COPY . .

# Set Python path to include SAM 2
ENV PYTHONPATH="${APP_ROOT}/sam2:${PYTHONPATH}"

CMD ["celery", "-A", "worker", "worker", "--loglevel=info", "--queues=sam2_queue"]