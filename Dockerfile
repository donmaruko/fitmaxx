# Use official Python 3.11.12 base image
FROM python:3.11.12-slim

# Avoid prompts during installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages needed by OpenCV, matplotlib, MediaPipe, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    build-essential \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy all project files into container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r full_requirements.txt

# Default command – runs the silhouette + drip score pipeline
# You can override this in `docker run` if needed
CMD ["python", "models/fusionrunner.py", "athleisure"]
