# sec_cam.py — Real-time face recognition security camera
import argparse, os, time, csv
from collections import deque
import numpy as np
import cv2
import face_recognition
import json

def ensure_dirs(*ds):
    for d in ds:
        os.makedirs(d, exist_ok=True)

def load_encodings(enc_dir):
    enc_path = os.path.join(enc_dir, "people.npy")
    lab_path = os.path.join(enc_dir, "people.json")
    if not (os.path.exists(enc_path) and os.path.exists(lab_path)):
        raise FileNotFoundError("Encodings not found. Run enroll_faces.py first.")
    encs = np.load(enc_path)
    with open(lab_path) as f:
        names = json.load(f)["names"]
    return encs, names

def draw_hud(frame, fps, ident, dist):
    cv2.rectangle(frame, (10,10), (360,120), (0,0,0), -1)
    cv2.putText(frame, f"FPS: {int(fps)}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"ID: {ident}", (20,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    if dist is not None:
        cv2.putText(frame, f"Dist: {dist:.3f}", (20,105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc_dir", default="encodings")
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--snap_dir", default="snapshots")
    ap.add_argument("--threshold", type=float, default=0.50, help="Distance threshold (~0.4-0.6 typical)")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    ensure_dirs(args.log_dir, args.snap_dir)
    encs, names = load_encodings(args.enc_dir)

    cap = cv2.VideoCapture(args.camera)
    pTime = 0
    recent = deque(maxlen=5)  # smooth identity over last few frames

    # prepare logger
    log_path = os.path.join(args.log_dir, time.strftime("events_%Y%m%d_%H%M%S.csv"))
    f = open(log_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["ts", "frame_idx", "identity", "distance", "x", "y", "w", "h"])

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)

            ident = "no_face"
            best_dist = None
            best_box = None

            for encoding, (top,right,bottom,left) in zip(encodings, boxes):
                if len(encs) > 0:
                    dists = face_recognition.face_distance(encs, encoding)
                    idx = int(np.argmin(dists))
                    dist = float(dists[idx])
                    name = names[idx] if dist < args.threshold else "unknown"
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        ident = name
                        best_box = (left, top, right-left, bottom-top)

            # smoothing identity
            recent.append(ident)
            smoothed = max(set(recent), key=recent.count)

            # draw boxes
            if best_box:
                x,y,w,h = best_box
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0) if smoothed!="unknown" else (0,0,255), 2)
                cv2.putText(frame, f"{smoothed}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            else:
                x=y=w=h=None

            # fps
            cTime = time.time()
            fps = 1/(cTime - pTime) if pTime else 0.0
            pTime = cTime
            draw_hud(frame, fps, smoothed, best_dist)

            # logging + snapshots
            ts = time.time()
            writer.writerow([ts, frame_idx, smoothed, best_dist if best_dist is not None else "", x or "", y or "", w or "", h or ""])
            if smoothed == "unknown" and best_box:
                x,y,w,h = best_box
                face_crop = frame[y:y+h, x:x+w]
                snap_path = os.path.join(args.snap_dir, f"unknown_{int(ts)}_{frame_idx}.jpg")
                try:
                    cv2.imwrite(snap_path, face_crop)
                except Exception:
                    pass

            cv2.imshow("SentinelFaceCam", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            frame_idx += 1
    finally:
        f.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"[LOG] Saved events to {log_path}")

if __name__ == "__main__":
    main()
