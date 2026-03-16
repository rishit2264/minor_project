import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import json

# ── Init ─────────────────────────────────────────────────────
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

# ── Load existing database ────────────────────────────────────
DB_FILE = "face_db.json"
EMB_DIR = "embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

if os.path.exists(DB_FILE):
    with open(DB_FILE, "r") as f:
        face_db = json.load(f)
else:
    face_db = {}

print(f"Existing registered users: {list(face_db.keys())}")

# ── Get name ──────────────────────────────────────────────────
name = input("Enter name to register: ").strip()
if not name:
    print("Name cannot be empty.")
    exit()

if name in face_db:
    print(f"WARNING: {name} already registered. Will overwrite.")

# ── Camera ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"\nRegistering: {name}")
print("Press 's' to capture | ESC to quit")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ── Skip frames to reduce load ────────────────────────────
    if frame_count % 3 != 0:
        cv2.imshow("Register", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # ── Resize for faster inference ───────────────────────────
    small = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
    faces = app.get(small)

    # ── Scale boxes back to original size ─────────────────────
    scale = 1 / 0.75
    for face in faces:
        box = (face.bbox * scale).astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    count_text = f"Faces: {len(faces)}"
    cv2.putText(frame, count_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Registering: {name}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, "Press S to Save", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("Register", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        if len(faces) == 0:
            print("No face detected. Reposition yourself.")
        else:
            # ── Normalize embedding ───────────────────────────
            emb = faces[0].embedding
            emb = emb / np.linalg.norm(emb)

            # ── Save embedding file ───────────────────────────
            emb_path = os.path.join(EMB_DIR, f"{name}.npy")
            np.save(emb_path, emb)

            # ── Update database ───────────────────────────────
            face_db[name] = emb_path
            with open(DB_FILE, "w") as f:
                json.dump(face_db, f, indent=2)

            print(f"\n✅ Registered: {name}")
            print(f"   Embedding saved → {emb_path}")
            print(f"   Database → {DB_FILE}")
            print(f"\nAll registered users: {list(face_db.keys())}")
            break

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()