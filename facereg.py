import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import sys
import json
import subprocess

# ── Init ArcFace ──────────────────────────────────────────────
print("Loading ArcFace model...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)
print("Model loaded.")

# ── Database ──────────────────────────────────────────────────
DB_FILE = "face_db.json"

if os.path.exists(DB_FILE):
    with open(DB_FILE, "r") as f:
        face_db = json.load(f)
else:
    face_db = {}

print(f"Existing registered users: {list(face_db.keys())}")

# ── C++ binary path ───────────────────────────────────────────
binary_name = "security_layer.exe" if sys.platform == "win32" else "security_layer"
CPP_BINARY  = os.path.join("..", "security_layer", binary_name)

if not os.path.isfile(CPP_BINARY):
    print(f"ERROR: C++ binary not found at '{CPP_BINARY}'")
    print("Compile: cd ../security_layer && g++ main.cpp embedding_processor.cpp "
          "hash_utils.cpp nonce.cpp proof.cpp -o security_layer.exe")
    exit(1)

# ── User Input ────────────────────────────────────────────────
name = input("Enter name to register: ").strip()
if not name:
    print("Name cannot be empty.")
    exit()

if name in face_db:
    print(f"WARNING: '{name}' already registered. Will overwrite.")

# ── Camera Setup ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"\nRegistering: {name}")
print("Press 's' 5 times to capture samples | ESC to quit")

# ── Multi-capture averaging ────────────────────────────────────
# Average 5 frames to get a stable embedding for registration.
# The normalized average embedding is stored for cosine similarity
# at recognition time. The hash is stored for proof generation.
CAPTURE_COUNT       = 5
captured_embeddings = []
frame_count         = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        cv2.imshow("Register", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    small = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
    faces = app.get(small)

    scale = 1 / 0.75
    for face in faces:
        box = (face.bbox * scale).astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    status = (f"Captures: {len(captured_embeddings)}/{CAPTURE_COUNT} — keep pressing S"
              if captured_embeddings else "Press S to start capturing")
    cv2.putText(frame, f"Registering: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, status,                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    cv2.imshow("Register", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        if len(faces) == 0:
            print("No face detected. Reposition yourself.")
            continue

        emb = faces[0].embedding
        captured_embeddings.append(emb)
        print(f"  Captured {len(captured_embeddings)}/{CAPTURE_COUNT}")

        if len(captured_embeddings) < CAPTURE_COUNT:
            continue

        # Average and normalize — this is what we compare against
        # at recognition time using cosine similarity
        avg_emb  = np.mean(captured_embeddings, axis=0)
        avg_emb  = avg_emb / np.linalg.norm(avg_emb)   # normalize AFTER averaging

        # Send to C++ for hashing and proof generation
        embedding_str = ",".join(map(str, avg_emb.tolist()))

        try:
            result = subprocess.run(
                [CPP_BINARY, embedding_str],
                capture_output=True,
                text=True,
                timeout=10
            )
        except subprocess.TimeoutExpired:
            print("ERROR: C++ security layer timed out.")
            break
        except FileNotFoundError:
            print(f"ERROR: Cannot execute '{CPP_BINARY}'.")
            break

        if result.returncode != 0:
            print("C++ Error:", result.stderr.strip())
            break

        try:
            response    = json.loads(result.stdout.strip())
            public_hash = response["public_hash"]
            nonce       = response["nonce"]
            proof       = response["proof"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"ERROR: Bad C++ output — {e}")
            print(f"Raw: {result.stdout}")
            break

        # Store:
        # - embedding (normalized) → used for cosine similarity at recognition
        # - public_hash            → used for proof verification
        # - registration_proof     → verifiable record of registration event
        # - registration_nonce     → needed to verify the proof later
        face_db[name] = {
            "embedding":          avg_emb.tolist(),
            "public_hash":        public_hash,
            "registration_proof": proof,
            "registration_nonce": nonce
        }

        with open(DB_FILE, "w") as f:
            json.dump(face_db, f, indent=2)

        print(f"\n✅ Registered: {name}")
        print(f"   Public hash → {public_hash}")
        print(f"   DB updated  → {DB_FILE}")
        print(f"\nAll registered users: {list(face_db.keys())}")
        break

    if key == 27:
        print("Registration cancelled.")
        break

cap.release()
cv2.destroyAllWindows()