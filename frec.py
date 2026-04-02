import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import sys
import json
import subprocess
import threading
import time

# ── Init ArcFace ──────────────────────────────────────────────
print("Loading ArcFace model...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)
print("Model loaded.")

# ── C++ binary path ───────────────────────────────────────────
binary_name = "security_layer.exe" if sys.platform == "win32" else "security_layer"
CPP_BINARY  = os.path.join("..", "security_layer", binary_name)

if not os.path.isfile(CPP_BINARY):
    print(f"ERROR: C++ binary not found at '{CPP_BINARY}'")
    print("Compile: cd ../security_layer && g++ main.cpp embedding_processor.cpp "
          "hash_utils.cpp nonce.cpp proof.cpp -o security_layer.exe")
    exit(1)

# ── Load database ─────────────────────────────────────────────
DB_FILE = "face_db.json"

if not os.path.exists(DB_FILE):
    print("ERROR: face_db.json not found. Run face_register.py first.")
    exit(1)

with open(DB_FILE, "r") as f:
    face_db = json.load(f)

if len(face_db) == 0:
    print("No registered users. Run face_register.py first.")
    exit(1)

# ── Load embeddings and hashes into memory ────────────────────
registered = {}   # name → { "embedding": np.array, "public_hash": str }

for user, entry in face_db.items():
    if isinstance(entry, dict) and "embedding" in entry and "public_hash" in entry:
        registered[user] = {
            "embedding":   np.array(entry["embedding"]),
            "public_hash": entry["public_hash"]
        }
    else:
        print(f"  WARNING: '{user}' has old-format entry. Re-register them.")

if len(registered) == 0:
    print("No valid entries found. Re-register all users.")
    exit(1)

print(f"Loaded {len(registered)} user(s): {list(registered.keys())}")

# Cosine similarity threshold.
# 0.4 is a good starting point for ArcFace buffalo_l.
# Raise it (e.g. 0.5) to be stricter, lower it (e.g. 0.35) if
# you get false unknowns in different lighting conditions.
THRESHOLD = 0.4


# ─────────────────────────────────────────────────────────────
# Helper — call C++ security layer to generate a fresh proof
# for a confirmed identity. Returns (public_hash, proof) or
# (None, None) on failure.
# ─────────────────────────────────────────────────────────────
def generate_proof_from_cpp(normalized_embedding: np.ndarray):
    embedding_str = ",".join(map(str, normalized_embedding.tolist()))
    try:
        result = subprocess.run(
            [CPP_BINARY, embedding_str],
            capture_output=True,
            text=True,
            timeout=5
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None, None

    if result.returncode != 0:
        return None, None

    try:
        response = json.loads(result.stdout.strip())
        return response["public_hash"], response["proof"]
    except (json.JSONDecodeError, KeyError):
        return None, None


# ── Shared state for threaded inference ───────────────────────
latest_frame   = None
latest_results = []       # list of (box, name, score, matched)
frame_lock     = threading.Lock()
result_lock    = threading.Lock()
running        = True


# ─────────────────────────────────────────────────────────────
# INFERENCE THREAD
#
# Step 1 — ArcFace extracts embedding
# Step 2 — Cosine similarity finds best match (noise-tolerant)
# Step 3 — If matched, C++ generates a fresh proof for the event
#
# This separation is the correct architecture:
# Python handles biometric matching (needs float tolerance)
# C++ handles cryptographic proof generation (needs exactness)
# ─────────────────────────────────────────────────────────────
def inference_thread():
    global latest_results, running

    while running:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()

        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        faces = app.get(small)
        scale = 2.0

        results = []

        for face in faces:
            # Normalize live embedding
            live_emb = face.embedding
            live_emb = live_emb / np.linalg.norm(live_emb)

            # ── Step 2: Cosine similarity match ───────────────
            best_name  = "Unknown"
            best_score = -1.0

            for user, data in registered.items():
                score = float(np.dot(data["embedding"], live_emb))
                if score > best_score:
                    best_score = score
                    best_name  = user

            matched = best_score >= THRESHOLD
            if not matched:
                best_name = "Unknown"

            # ── Step 3: Generate proof if matched ─────────────
            # C++ generates a fresh cryptographic proof for this
            # recognition event. The proof links the live embedding
            # hash to a nonce, creating a verifiable attendance record.
            if matched:
                pub_hash, proof = generate_proof_from_cpp(live_emb)
                if pub_hash:
                    # Optional: log proof for audit trail
                    pass  # extend here to write to attendance log

            box = (face.bbox * scale).astype(int)
            results.append((box, best_name, best_score, matched))

        with result_lock:
            latest_results = results

        time.sleep(0.05)


# ── Start inference thread ────────────────────────────────────
t = threading.Thread(target=inference_thread, daemon=True)
t.start()

# ── Camera setup ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

print("\nRecognition running... Press ESC to quit.\n")

fps         = 0
fps_counter = 0
fps_timer   = time.time()


# ─────────────────────────────────────────────────────────────
# DISPLAY LOOP — runs at full camera FPS
# ─────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    with frame_lock:
        latest_frame = frame.copy()

    with result_lock:
        results = list(latest_results)

    for (box, name, score, matched) in results:
        color      = (0, 255, 0) if matched else (0, 0, 255)
        label_text = f"{name} ({score:.2f})" if matched else "Unknown"

        h, w = frame.shape[:2]
        x1 = max(0, box[0]);  y1 = max(0, box[1])
        x2 = min(w, box[2]);  y2 = min(h, box[3])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label_text,
                    (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

    if len(results) == 0:
        cv2.putText(frame, "No face detected",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (100, 100, 100), 2)

    fps_counter += 1
    if time.time() - fps_timer >= 1.0:
        fps         = fps_counter
        fps_counter = 0
        fps_timer   = time.time()

    cv2.putText(frame, f"FPS: {fps}",
                (10, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Users: {len(registered)}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("SecureAttend — ArcFace + C++ Security", frame)

    if cv2.waitKey(1) == 27:
        break

# ── Cleanup ───────────────────────────────────────────────────
running = False
cap.release()
cv2.destroyAllWindows()
print("Stopped.")