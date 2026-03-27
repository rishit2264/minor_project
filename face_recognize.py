import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import json
import subprocess

# ── Init ArcFace ─────────────────────────────────────────────
print("Loading ArcFace model...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)
print("Model loaded.")

# ── Load Database ────────────────────────────────────────────
DB_FILE = "face_db.json"

if not os.path.exists(DB_FILE):
    print("No database found. Run register first.")
    exit()

with open(DB_FILE, "r") as f:
    face_db = json.load(f)

if len(face_db) == 0:
    print("No registered users.")
    exit()

print(f"Registered users: {list(face_db.keys())}")

cpp_binary = "../security_layer/security_layer"

# ── Camera Setup ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nRecognition running... Press ESC to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)

        # ── Send to C++ Security Layer ───────────────────
        embedding_str = ",".join(map(str, emb.tolist()))

        result = subprocess.run(
            [cpp_binary, embedding_str],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("C++ Error:", result.stderr)
            continue

        response = json.loads(result.stdout.strip())
        live_hash = response["public_hash"]

        # ── Compare hashes ────────────────────────────────
        name = "Unknown"

        for user, stored_hash in face_db.items():
            if live_hash == stored_hash:
                name = user
                break

        box = face.bbox.astype(int)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (box[0], box[1]),
                      (box[2], box[3]), color, 2)

        cv2.putText(frame, name,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

    cv2.imshow("SecureAttend", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()