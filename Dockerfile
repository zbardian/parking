FROM docker.io/library/python:3.11-slim

# Instalăm doar strictul necesar de sistem
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    "numpy==1.26.4" \
    "flask==3.0.3" \
    "opencv-python-headless==4.9.0.80"

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.3.1" \
    "torchvision==0.18.1"

RUN pip install --no-cache-dir "ultralytics==8.2.103"

COPY parking_flask.py .
COPY parking_rois.json .

ENV PYTHONUNBUFFERED=1

CMD ["python3", "parking_flask.py"]
