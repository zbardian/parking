import cv2
import json
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO
import os

# ===== Configurație =====
# Folosim MediaMTX de pe localhost (mai stabil)
RTSP_URL = os.environ.get("RTSP_URL", "rtsp://localhost:8554/stream1")
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
def gen_frames():
    cap = cv2.VideoCapture(RTSP_URL)
    # Setăm buffer mic pentru latență scăzută
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        success, frame = cap.read()
        if not success:
            print("Eroare: Nu pot citi fluxul RTSP de la MediaMTX.")
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
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ===== Flask App =====
app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    """Ruta apelată de Nginx proxy_pass."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Ruta opțională pentru a primi doar JSON cu starea locurilor."""
    # Aici ar trebui o logică similară cu gen_frames dar fără video
    return {"msg": "Folosește /video_feed pentru stream vizual"}

if __name__ == '__main__':
    # Rulăm pe portul 5010, vizibil din exterior (0.0.0.0)
    app.run(host='0.0.0.0', port=5010, threaded=True)
