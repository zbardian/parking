FROM docker.io/library/python:3.11-slim

# Instalăm doar strictul necesar de sistem
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Adăugăm indexul piwheels.org - acesta are binare optimizate pt Pi 4
# 2. Instalăm NumPy < 2.0 (Bariera anti-Illegal Instruction)
# 3. Instalăm Ultralytics care va trage singur Torch-ul corect
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://www.piwheels.org "numpy<2.0" flask opencv-python-headless ultralytics

COPY parking_flask.py .
COPY parking_rois.json .

ENV PYTHONUNBUFFERED=1

CMD ["python3", "parking_flask.py"]
