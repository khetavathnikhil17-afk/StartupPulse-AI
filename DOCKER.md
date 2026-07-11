# StartupPulse AI

Explainable Aspect-Based Sentiment Analysis for Employee Feedback using DeBERTa-v3 and SHAP.

## Quick Start with Docker

### Prerequisites
- Docker installed on your system
- Docker Compose (optional, for easier management)

### Build and Run

```bash
# Build the Docker image
docker build -t startuppulse-ai .

# Run the container
docker run -p 8501:8501 startuppulse-ai
```

### Using Docker Compose

```bash
# Build and run with docker-compose
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop the container
docker-compose down
```

### Access the Dashboard

Open your browser and navigate to:
```
http://localhost:8501
```

### Environment Variables

You can customize the behavior using environment variables:

```bash
docker run -p 8501:8501 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TOKENIZERS_PARALLELISM=false \
  startuppulse-ai
```

### Volume Mounts

To persist data or use custom models:

```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  startuppulse-ai
```

### Development Mode

For development with live reload:

```bash
docker run -p 8501:8501 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/dashboard:/app/dashboard \
  startuppulse-ai
```

## Troubleshooting

### Container fails to start
```bash
# Check container logs
docker logs <container_id>

# Run with interactive shell
docker run -it startuppulse-ai /bin/bash
```

### GPU not available
```bash
# Ensure nvidia-docker is installed
docker run --gpus all startuppulse-ai
```

### Port already in use
```bash
# Use a different port
docker run -p 8502:8501 startuppulse-ai
```
