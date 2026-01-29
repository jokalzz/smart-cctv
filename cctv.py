import cv2
import torch
import time
import os
import requests
import numpy as np
from ultralytics import YOLO
from playsound import playsound
from datetime import datetime

# ======================================
# LOAD MODELS
# ======================================
device = "cuda" if torch.cuda.is_available() else "cpu"
detector = YOLO("yolov8s-world.pt").to(device)
pose_model = YOLO("yolov8n-pose.pt").to(device)
custom_model = YOLO("best.pt").to(device) 


# Detector YOLO - hanya kucing dan burung
detector.set_classes([
    "kucing",
    "burung"
])

# Custom model configuration - hanya ular dan biawak
CUSTOM_MODEL_CLASSES = ["ular", "biawak"]
CUSTOM_MODEL_THRESHOLDS = {"ular": 25, "biawak": 25}

# ======================================
# CONFIG
# ======================================
FRAME_W, FRAME_H = 640, 360
SKIP_FRAMES = 8  # Jalankan deteksi setiap 5 frame untuk mengurangi beban
POSE_INTERVAL = 8
MIN_OBJ_AREA_RATIO = 0.05
MIN_KEYPOINTS = 8
MIN_ASPECT_RATIO = 1.2
MIN_ANIMAL_CONFIDENCE = 30  # persen

# Per-class confidence thresholds
ALARM_THRESHOLDS = {
    "kucing": 30,      # Kucing: 30%
    "burung": 30,     # Burung: 30%
    "ular": 25,     # Ular: 25%
    "biawak": 25    # Biawak: 25%
}

STRONG_ANIMALS = ["kucing", "burung", "ular", "biawak"]

ALARM_DELAY = 5
SNAPSHOT_DELAY = 5

last_alarm = 0
last_snapshot = 0
animal_counter = 0

TELEGRAM_BOT_TOKEN = "8554115860:AAEBbgxKhM_nea38VkNDImzaHG3Z9cM9hvc"
TELEGRAM_CHAT_ID = "1368725503"

# ======================================
# SNAPSHOT CONFIG
# ======================================
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def save_snapshot(frame, animals_dict):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"{SNAPSHOT_DIR}/animal_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

    # Format dengan persentase
    animals_detail = "\n".join([f"  • {k}: <b>{v:.1f}%</b>" for k, v in animals_dict.items()])

    caption = (
        f"{animals_detail}\n"
        f"<b>Waktu:</b> {timestamp}"
    )

    log_file = "riwayat_deteksi.txt"
    print(f"[ALARM] Snapshot + Log → Telegram | Hewan: {list(animals_dict.keys())}")
    send_telegram_snapshot(filename, caption, log_file)

def send_telegram_snapshot(image_path, caption, log_file=None):
    """Send photo with caption and optionally attach log file"""
    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    url_doc = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"

    try:
        with open(image_path, "rb") as img:
            r = requests.post(
                url_photo,
                files={"photo": img},
                data={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": caption,
                    "parse_mode": "HTML"
                },
                timeout=10
            )
        print(f"[TELEGRAM Photo] Status: {r.status_code}")
    except Exception as e:
        print(f"[ERROR] Telegram Photo: {e}")

    # Send log file if exists
    if log_file and os.path.exists(log_file):
        try:
            with open(log_file, "rb") as doc:
                r2 = requests.post(
                    url_doc,
                    files={"document": doc},
                    data={"chat_id": TELEGRAM_CHAT_ID},
                    timeout=10
                )
            print(f"[TELEGRAM Document] Status: {r2.status_code}")
        except Exception as e:
            print(f"[ERROR] Telegram Document: {e}")

# local camera:
cap = cv2.VideoCapture(0)
# ======================================
# CCTV
# ======================================
# RTSP_URL = "rtsp://admin:hikvision12@192.168.1.68:554/Streaming/Channels/102"
# cap = cv2.VideoCapture(RTSP_URL)

prev_time = 0
frame_count = 0

detected_animals = {}  # { "KUCING": 85.5 }

# ======================================
# DETECTION HANDLER
# ======================================
def process_detections(results, names, frame, frame_area, allowed_classes=None):
    """Process model results dan cek threshold per kelas
    
    Args:
        allowed_classes: List of class names to process. If None, process all.
    """
    found = False
    allowed_lc = [c.lower() for c in allowed_classes] if allowed_classes else None
    
    for r in results:
        for box in r.boxes:
            cls_name = names[int(box.cls[0])]
            cls_lc = cls_name.lower()
            confidence = float(box.conf[0]) * 100

            # Filter classes jika ada allowed_classes
            if allowed_lc and cls_lc not in allowed_lc:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            obj_area = w * h
            aspect_ratio = h / w if w != 0 else 0

            if obj_area / frame_area < 0.05:
                continue

            # Anti-false human (untuk kelas yang bukan STRONG_ANIMALS)
            is_human = False
            if cls_lc not in STRONG_ANIMALS:
                if (
                    frame_count % POSE_INTERVAL == 0 and
                    obj_area / frame_area >= MIN_OBJ_AREA_RATIO and
                    aspect_ratio >= MIN_ASPECT_RATIO
                ):
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        pose_results = pose_model(crop, conf=0.3)
                        for pr in pose_results:
                            if pr.keypoints is not None:
                                kpts = pr.keypoints.xy[0]
                                visible = sum(
                                    1 for x, y in kpts if x > 0 and y > 0
                                )
                                if visible >= MIN_KEYPOINTS:
                                    is_human = True

            if is_human:
                continue

            # Cek threshold per kelas dari ALARM_THRESHOLDS
            threshold = ALARM_THRESHOLDS.get(cls_lc, MIN_ANIMAL_CONFIDENCE)

            if confidence >= threshold:
                key = cls_name.upper()
                if key not in detected_animals or confidence > detected_animals[key]:
                    detected_animals[key] = confidence
                found = True

            # DRAW
            label = f"{cls_name.upper()} {confidence:.1f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return found

# ======================================
# MAIN LOOP
# ======================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    frame_area = FRAME_W * FRAME_H
    frame_count += 1

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Run detections (setiap SKIP_FRAMES untuk mengurangi beban GPU)
    if frame_count % SKIP_FRAMES == 0:
        det_results = detector(frame, conf=0.25)
        custom_results = custom_model(frame, conf=0.25)
    else:
        det_results = []
        custom_results = []

    animal_detected_this_frame = False

    # Process YOLO results (kucing, burung HANYA dari YOLO)
    if det_results:
        animal_detected_this_frame |= process_detections(det_results, detector.names, frame, frame_area)

    # Process custom model results (ular, biawak HANYA dari best.pt)
    if custom_results:
        animal_detected_this_frame |= process_detections(custom_results, custom_model.names, frame, frame_area, 
                                                         allowed_classes=CUSTOM_MODEL_CLASSES)

    # ======================================
    # ALARM + SNAPSHOT
    # ======================================
    if animal_detected_this_frame:
        animal_counter += 1
    else:
        animal_counter = 0

    if animal_counter >= 1 and time.time() - last_alarm > ALARM_DELAY:
        playsound("alarm.wav", block=False)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open("riwayat_deteksi.txt", "a") as f:
            f.write(f"\n[{now}]\n")
            for animal, conf in detected_animals.items():
                f.write(f"  {animal}: {conf:.1f}%\n")

        if time.time() - last_snapshot > SNAPSHOT_DELAY:
            save_snapshot(frame, detected_animals)
            last_snapshot = time.time()

        detected_animals.clear()
        last_alarm = time.time()

    # ======================================
    # INFO
    # ======================================
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("CCTV 1", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
