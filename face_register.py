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

# ── Database ─────────────────────────────────────────────────
DB_FILE = "face_db.json"

if os.path.exists(DB_FILE):
    with open(DB_FILE, "r") as f:
        face_db = json.load(f)
else:
    face_db = {}

print(f"Existing registered users: {list(face_db.keys())}")

# ── User Input ───────────────────────────────────────────────
name = input("Enter name to register: ").strip()
if not name:
    print("Name cannot be empty.")
    exit()

# ── Camera Setup ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"\nRegistering: {name}")
print("Press 's' to capture | ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]),
                      (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("Register", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        if len(faces) == 0:
            print("No face detected.")
        else:
            emb = faces[0].embedding
            emb = emb / np.linalg.norm(emb)

            # ── Send to C++ Security Layer ───────────────────
            embedding_str = ",".join(map(str, emb.tolist()))
            cpp_binary = "../security_layer/security_layer"

            result = subprocess.run(
                [cpp_binary, embedding_str],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print("C++ Error:", result.stderr)
                break

            response = json.loads(result.stdout.strip())
            public_hash = response["public_hash"]

            # ── Store ONLY hash ──────────────────────────────
            face_db[name] = public_hash

            with open(DB_FILE, "w") as f:
                json.dump(face_db, f, indent=2)

            print(f"\n✅ Registered {name}")
            print(f"Stored hash: {public_hash}")
            break

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()