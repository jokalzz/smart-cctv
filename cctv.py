import cv2
import time
import os
import requests
from ultralytics import YOLO
from playsound import playsound
from datetime import datetime

# ======================================
# LOAD MODELS
# ======================================
detector = YOLO("yolov8s-world.pt")
pose_model = YOLO("yolov8n-pose.pt")

detector.set_classes([
    "person",
    "cat",
    "dog",
    "snake",
    "chicken",
    "bird",
    "cow",
    "goat"
])

# ======================================
# CONFIG
# ======================================
FRAME_W, FRAME_H = 640, 480
POSE_INTERVAL = 5
MIN_OBJ_AREA_RATIO = 0.15
MIN_KEYPOINTS = 8
MIN_ASPECT_RATIO = 1.2
MIN_ANIMAL_CONFIDENCE = 65  # persen

STRONG_ANIMALS = ["dog", "cat", "snake", "bird", "chicken"]

ALARM_DELAY = 10
SNAPSHOT_DELAY = 10

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

def save_snapshot(frame, detected_animals):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{SNAPSHOT_DIR}/animal_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

    animals_text = ", ".join(detected_animals)
    caption = f"🚨 HEWAN TERDETEKSI 🚨\n{animals_text}\n🕒 {timestamp}"

    send_telegram_snapshot(filename, caption)


def send_telegram_snapshot(image_path, caption):
    url = f"https://api.telegram.org/bot8554115860:AAEBbgxKhM_nea38VkNDImzaHG3Z9cM9hvc/sendPhoto"

    with open(image_path, "rb") as img:
        files = {
            "photo": img
        }

        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption,  
            "parse_mode": "HTML"  
        }

        response = requests.post(url, files=files, data=data)

        if response.status_code != 200:
            print("TELEGRAM ERROR:", response.text)

# ======================================
# CAMERA
# ======================================
cap = cv2.VideoCapture(0)
prev_time = 0
frame_count = 0

detected_animals = set()
detected_animals_frame = []


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

    results = detector(frame, conf=0.25)
    animal_detected_this_frame = False

    for r in results:
        for box in r.boxes:
            cls_name = detector.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w = x2 - x1
            h = y2 - y1
            obj_area = w * h
            aspect_ratio = h / w if w != 0 else 0

            # Skip objek terlalu kecil

            if cls_name != "person" and obj_area / frame_area < 0.05:
                continue
            
            is_human = False

            confidence = float(box.conf[0]) * 100  # jadi persen
            
            if cls_name == "person" and confidence < 30:
                continue
            # ======================================
            # PERSON → MANUSIA
            # ======================================
            if cls_name == "person":
                is_human = True

            # ======================================
            # HEWAN → VERIFIKASI POSE
            # ======================================
            else:
                if cls_name in STRONG_ANIMALS:
                    is_human = False
                else:
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

            # ======================================
            # DRAW
            # ======================================
            if is_human:
                color = (0, 255, 0)
                label = f"Manusia {confidence:.1f}%"
            else:
                color = (0, 0, 255)
                label = f"{cls_name.upper()} {confidence:.1f}%"
                
                if confidence >= MIN_ANIMAL_CONFIDENCE:
                    animal_detected_this_frame = True
                    detected_animals.add(label)
                    detected_animals_frame.append(cls_name)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ======================================
    # ALARM + SNAPSHOT (ANTI SPAM)
    # ======================================
    if animal_detected_this_frame:
        animal_counter += 1
    else:
        animal_counter = 0

    if animal_counter >= 3 and time.time() - last_alarm > ALARM_DELAY:
        playsound("alarm.wav", block=False)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("riwayat_deteksi.txt", "a") as f:
            for animal in detected_animals:
                f.write(f"[{now}] {animal}\n")
        detected_animals.clear()

        # SIMPAN SNAPSHOT
        if time.time() - last_snapshot > SNAPSHOT_DELAY:
            save_snapshot(frame, detected_animals)
            last_snapshot = time.time()

        last_alarm = time.time()

    # ======================================
    # INFO
    # ======================================
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("SMART CCTV YOLO + POSE", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()