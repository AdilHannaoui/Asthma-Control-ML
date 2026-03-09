# =============================================================================
# Dockerfile — Asthma Poor-Control Prediction
# Author: Adil Hannaoui Anaaoui
#
# Build:
#   docker build -t asthma-control-ml .
#
# Run:
#   docker run -v $(pwd)/data:/app/data asthma-control-ml \
#       --input data/patients.csv --output data/predictions.csv --proba
# =============================================================================

# Base image — slim variant for smaller image size
FROM python:3.11-slim

# Metadata
LABEL author="Adil Hannaoui Anaaoui"
LABEL description="Asthma poor-control prediction pipeline"

# Working directory inside the container
WORKDIR /app

# Install system dependencies required by XGBoost and scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artefacts and prediction script
COPY models/ models/
COPY predict.py .

# Default entrypoint — runs predict.py with any arguments passed to docker run
ENTRYPOINT ["python", "predict.py"]
