import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import json
import threading
import time

# ── Init model ────────────────────────────────────────────────
print("Loading ArcFace model...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)
print("Model loaded.")

# ── Load database ─────────────────────────────────────────────
DB_FILE = "face_db.json"

if not os.path.exists(DB_FILE):
    print("ERROR: face_db.json not found. Run register.py first.")
    exit()

with open(DB_FILE, "r") as f:
    face_db = json.load(f)

if len(face_db) == 0:
    print("No registered users. Run register.py first.")
    exit()

# ── Load all embeddings into memory ───────────────────────────
registered = {}
for name, emb_path in face_db.items():
    if os.path.exists(emb_path):
        registered[name] = np.load(emb_path)
        print(f"  Loaded: {name}")
    else:
        print(f"  WARNING: Missing embedding for {name}")

print(f"\nTotal users: {len(registered)}")

THRESHOLD = 0.5

# ── Shared state between threads ──────────────────────────────
latest_frame   = None          # latest raw frame from camera
latest_results = []            # latest detection results [(box, name, score)]
frame_lock     = threading.Lock()
result_lock    = threading.Lock()
running        = True

# ─────────────────────────────────────────────────────────────
# INFERENCE THREAD — runs recognition in background
# never blocks the display
# ─────────────────────────────────────────────────────────────
def inference_thread():
    global latest_results, running

    while running:
        # ── Grab latest frame ─────────────────────────────────
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()

        # ── Resize to 50% for speed ───────────────────────────
        small  = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        faces  = app.get(small)
        scale  = 2.0   # scale boxes back up

        results = []

        for face in faces:
            live_emb = face.embedding
            live_emb = live_emb / np.linalg.norm(live_emb)

            best_name  = "Unknown"
            best_score = -1.0

            for name, reg_emb in registered.items():
                score = float(np.dot(reg_emb, live_emb))
                if score > best_score:
                    best_score = score
                    best_name  = name

            if best_score < THRESHOLD:
                best_name = "Unknown"

            box = (face.bbox * scale).astype(int)
            results.append((box, best_name, best_score))

        with result_lock:
            latest_results = results

        # ── Small sleep to avoid maxing CPU ───────────────────
        time.sleep(0.03)


# ── Start inference thread ────────────────────────────────────
t = threading.Thread(target=inference_thread, daemon=True)
t.start()

# ── Camera setup ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # reduce camera buffer lag

print("\nRecognition running... Press ESC to quit.\n")

# ── FPS tracking ──────────────────────────────────────────────
fps        = 0
fps_counter = 0
fps_timer   = time.time()

# ─────────────────────────────────────────────────────────────
# DISPLAY LOOP — always runs at full camera FPS
# draws last known results on every frame
# ─────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── Push frame to inference thread ────────────────────────
    with frame_lock:
        latest_frame = frame.copy()

    # ── Read latest results ───────────────────────────────────
    with result_lock:
        results = list(latest_results)

    # ── Draw results ──────────────────────────────────────────
    for (box, name, score) in results:

        if name == "Unknown":
            color      = (0, 0, 255)    # red   — unknown
            label_text = "Unknown"
        else:
            color      = (0, 255, 0)    # green — matched
            label_text = f"{name}  {score:.2f}"

        # ── Clamp box to frame boundaries ─────────────────────
        h, w = frame.shape[:2]
        x1 = max(0, box[0]);  y1 = max(0, box[1])
        x2 = min(w, box[2]);  y2 = min(h, box[3])

        # ── Bounding box ──────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ── Label background for readability ──────────────────
        (tw, th), _ = cv2.getTextSize(label_text,
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.65, 2)
        cv2.rectangle(frame,
                      (x1, y1 - th - 14),
                      (x1 + tw + 8, y1),
                      color, -1)   # filled rectangle

        cv2.putText(frame, label_text,
                    (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)  # white text on colored bg

    # ── No face message ───────────────────────────────────────
    if len(results) == 0:
        cv2.putText(frame, "No face detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (100, 100, 100), 2)

    # ── FPS counter ───────────────────────────────────────────
    fps_counter += 1
    if time.time() - fps_timer >= 1.0:
        fps       = fps_counter
        fps_counter = 0
        fps_timer   = time.time()

    cv2.putText(frame, f"FPS: {fps}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)

    cv2.putText(frame, f"Users: {len(registered)}",
                (10, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)

    cv2.imshow("SecureAttend — ArcFace", frame)

    if cv2.waitKey(1) == 27:   # ESC
        break

# ── Cleanup ───────────────────────────────────────────────────
running = False
cap.release()
cv2.destroyAllWindows()
print("Stopped.")