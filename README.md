# Parking Detector (Raspberry Pi 4)

Detector de ocupare locuri de parcare cu:
- stream RTSP (MediaMTX / cameră IP)
- inferență YOLO (`ultralytics`)
- overlay ROI din `parking_rois.json`
- stream HTTP MJPEG prin Flask (`/video_feed`)

## Conținut proiect

- `parking_flask.py` – aplicația Flask + YOLO + procesare ROI
- `parking_rois.json` – poligoane ROI pentru locurile de parcare
- `Dockerfile` – imagine Python + OpenCV + Torch CPU + Ultralytics
- `docker-compose.yml` – configurare serviciu (`parking_detector`)
- `.env` – credențiale RTSP (local, neversionat)

## Cerințe

- Raspberry Pi 4 cu OS 64-bit (`aarch64`)
- Podman + podman-compose **sau** Docker + docker compose
- Sursă RTSP funcțională (ex: `rtsp://127.0.0.1:8554/stream1`)

Verificare arhitectură:

```bash
uname -m
getconf LONG_BIT
```

Rezultat recomandat: `aarch64` și `64`.

## Configurare `.env`

Creează fișierul `.env` în directorul proiectului:

```env
RTSP_USER=sd6csc3
RTSP_PASSWORD=12345678
```

Notă: `.env` este ignorat de Git prin `.gitignore`.

## Config RTSP

În `docker-compose.yml` sunt folosite:

- `RTSP_URL=rtsp://127.0.0.1:8554/stream1`
- `RTSP_FALLBACK_URL=rtsp://192.168.50.50:8554/stream1`
- `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`

Aplicația încearcă fallback automat dacă URL-ul principal nu răspunde.

## Pornire serviciu

### Cu Podman (recomandat pe setup-ul curent)

```bash
podman compose up -d --build
podman ps -a | grep parking_detector
podman logs -f parking_detector
```

### Cu Docker

```bash
docker compose up -d --build
docker ps -a | grep parking_detector
docker logs -f parking_detector
```

## Endpoint-uri

- `http://<IP_PI>:5010/` – pagină web simplă cu stream
- `http://<IP_PI>:5010/video_feed` – stream MJPEG
- `http://<IP_PI>:5010/status` – status simplu JSON

## Troubleshooting rapid

### 1) `Illegal instruction (core dumped)` la build/run

- verifică OS 64-bit (`aarch64`)
- rebuild complet:

```bash
podman compose down
podman compose build --no-cache
podman compose up -d
```

### 2) `RTSP indisponibil - retry...`

- verifică stream-ul local:

```bash
ffprobe -rtsp_transport tcp rtsp://127.0.0.1:8554/stream1
```

- verifică loguri aplicație:

```bash
podman logs --tail 200 parking_detector
```

- dacă primești `401 Unauthorized`, verifică `RTSP_USER` / `RTSP_PASSWORD` din `.env`.

### 3) Nu se vede nimic pe root

- verifică endpoint direct: `http://<IP_PI>:5010/video_feed`
- dacă reverse proxy (Nginx), folosește `proxy_pass http://<IP_PI>:5010/;` pe prefix (nu direct pe `/video_feed`).

## GitHub (first push)

```bash
git init
git add .
git commit -m "Initial parking detector setup"
git branch -M main
git remote add origin git@github.com:USER/REPO.git
git push -u origin main
```

Dacă folosești SSH și apare eroare de cheie:

```bash
ssh-keygen -t ed25519 -C "email@example.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Adaugă cheia publică în GitHub → Settings → SSH and GPG keys.

## Observații

- Model implicit: `yolov8n.pt` (rapid pentru Pi 4)
- ROI-urile se citesc din `parking_rois.json`
- Pentru performanță mai bună poți reduce rezoluția stream-ului sursă RTSP.
