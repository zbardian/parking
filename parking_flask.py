import cv2
import json
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO
import os
import time
from urllib.parse import urlparse, urlunparse, quote

# ===== Configurație =====
# Folosim MediaMTX de pe localhost (mai stabil)
RTSP_URL = os.environ.get("RTSP_URL", "rtsp://localhost:8554/stream1")
RTSP_FALLBACK_URL = os.environ.get("RTSP_FALLBACK_URL", "rtsp://192.168.50.50:8554/stream1")
RTSP_USER = os.environ.get("RTSP_USER", "")
RTSP_PASSWORD = os.environ.get("RTSP_PASSWORD", "")
MODEL_PATH = "yolov8n.pt"  # Modelul Nano este cel mai rapid pe Pi 4
ROIS_FILE = "parking_rois.json"

# Inițializare model YOLO
model = YOLO(MODEL_PATH)

# Încărcare ROI-uri (cu protecție dacă fișierul nu există încă)
if os.path.exists(ROIS_FILE):
    with open(ROIS_FILE, 'r') as f:
        rois = json.load(f)
else:
    print(f"ATENȚIE: Fișierul {ROIS_FILE} nu a fost găsit! Creează-l pentru a vedea locurile.")
    rois = {}

# ===== Funcție Verificare Ocupare =====
def roi_occupied(roi, bboxes):
    """
    Verifică dacă un poligon (ROI) este ocupat de o mașină.
    Calculăm intersecția dintre poligon și bounding box-ul detectat.
    """
    poly = np.array(roi, dtype=np.int32)
    for box_data in bboxes:
        # x1, y1, x2, y2, conf, cls
        x1, y1, x2, y2, conf, cls = box_data
        
        if int(cls) != 2:  # 2 = 'car' în COCO dataset
            continue
            
        # Creăm poligon pentru box-ul mașinii
        box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        
        # Calculăm aria intersecției
        inter, _ = cv2.intersectConvexConvex(poly, box)
        if inter > 500:  # Prag de pixeli (ajustează dacă e nevoie)
            return True
    return False

# ===== Generator Stream Video =====
def with_rtsp_auth(url):
    parsed = urlparse(url)
    if parsed.scheme.lower() != "rtsp":
        return url
    if parsed.username is not None:
        return url
    if not RTSP_USER:
        return url

    username = quote(RTSP_USER, safe='')
    password = quote(RTSP_PASSWORD, safe='') if RTSP_PASSWORD else ""
    auth = f"{username}:{password}@" if password else f"{username}@"
    netloc = f"{auth}{parsed.hostname or ''}"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    return urlunparse((
        parsed.scheme,
        netloc,
        parsed.path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))


def rtsp_candidates():
    urls = [RTSP_URL]
    if RTSP_FALLBACK_URL and RTSP_FALLBACK_URL not in urls:
        urls.append(RTSP_FALLBACK_URL)
    if "127.0.0.1" in RTSP_URL:
        alt = RTSP_URL.replace("127.0.0.1", "192.168.50.50")
        if alt not in urls:
            urls.append(alt)
    if "localhost" in RTSP_URL:
        alt = RTSP_URL.replace("localhost", "192.168.50.50")
        if alt not in urls:
            urls.append(alt)
    return [with_rtsp_auth(url) for url in urls]


def open_rtsp_capture():
    for url in rtsp_candidates():
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            print(f"RTSP conectat: {url}")
            return cap, url
        cap.release()
    return None, None


def gen_frames():
    while True:
        cap, active_url = open_rtsp_capture()

        if cap is None:
            print(f"Eroare: Nu pot deschide RTSP. Candidate: {rtsp_candidates()}")
            frame = np.zeros((480, 854, 3), dtype=np.uint8)
            cv2.putText(frame, "RTSP indisponibil - retry...", (20, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)
            continue

        while True:
            success, frame = cap.read()
            if not success:
                print(f"Eroare: Nu pot citi fluxul RTSP de la MediaMTX ({active_url}).")
                frame = np.zeros((480, 854, 3), dtype=np.uint8)
                cv2.putText(frame, "Nu se primesc frame-uri RTSP - reconnect...", (20, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                break

            # Rulăm YOLO doar pe acest cadru (imgsz 640 pentru viteză)
            results = model(frame, imgsz=640, verbose=False)[0]
            bboxes = results.boxes.data.cpu().numpy()

            # Desenăm locurile de parcare
            for spot, roi in rois.items():
                occupied = roi_occupied(roi, bboxes)

                # Culori BGR: Roșu dacă e ocupat, Verde dacă e liber
                color = (0, 0, 255) if occupied else (0, 255, 0)
                pts = np.array(roi, np.int32).reshape((-1, 1, 2))

                # Desenăm conturul și textul
                cv2.polylines(frame, [pts], True, color, 3)

                # Overlay semi-transparent pentru locurile ocupate
                if occupied:
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 0, 150))
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                cv2.putText(frame, f"{spot}: {'OCUPAT' if occupied else 'LIBER'}",
                            tuple(roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Codare JPEG pentru stream HTTP
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()
        time.sleep(0.5)

# ===== Flask App =====
app = Flask(__name__)

@app.route('/video_feed')
@app.route('/video_feed/')
def video_feed():
    """Ruta apelată de Nginx proxy_pass."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return (
        "<html><head><title>Parking Detector</title></head>"
        "<body style='margin:0;background:#111;color:#fff;font-family:Arial,sans-serif;'>"
        "<h2 style='padding:12px 16px;margin:0;'>Parking Detector Live</h2>"
        "<div style='padding:12px 16px;'><img src='video_feed' style='max-width:100%;height:auto;border:1px solid #333;'/></div>"
        "<div style='padding:0 16px 16px;'><a href='status' style='color:#7fb3ff;'>Status JSON</a></div>"
        "</body></html>"
    )

@app.route('/status')
def status():
    """Ruta opțională pentru a primi doar JSON cu starea locurilor."""
    # Aici ar trebui o logică similară cu gen_frames dar fără video
    return {"msg": "Folosește /video_feed pentru stream vizual"}

if __name__ == '__main__':
    # Rulăm pe portul 5010, vizibil din exterior (0.0.0.0)
    app.run(host='0.0.0.0', port=5010, threaded=True)
