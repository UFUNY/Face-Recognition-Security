# Face Recognition Security Camera

**One‑sentence:** Real‑time camera that recognizes enrolled faces, logs events to CSV, and captures snapshots of unknown visitors — no heavy training required.

## Features
- **Enroll once:** Generate face encodings from labeled images
- **Real‑time recognition:** 25–30 FPS webcam loop with HUD overlay
- **Event logging:** CSV log with timestamps, identity, confidence proxy, and frame index
- **Unknown snapshots:** Save cropped face images of unknowns for review
- **Reporting:** Summary notebook/script to compute counts, dwell time, and recognition stats

## Quickstart
```bash
# 0) (Recommended) Use Conda for best compatibility:
conda create -n facerec python=3.11 dlib
conda activate facerec
pip install -r requirements.txt

# 1) Add some reference images (see structure below), then:
python enroll_faces.py --ref_dir references --enc_dir encodings

# 2) Run the security camera (try different camera indices if you see a black screen):
python sec_cam.py --enc_dir encodings --snap_dir snapshots --log_dir logs --camera 0
# If you see a black screen, try:
python sec_cam.py --enc_dir encodings --snap_dir snapshots --log_dir logs --camera 1
# or
python sec_cam.py --enc_dir encodings --snap_dir snapshots --log_dir logs --camera 2

# 3) Generate a quick report:
python summarize_logs.py --log_dir logs
```

## Reference Image Structure
Place 1–5 clear face images per person (frontal if possible):
```
references/
  Mbappe/
    alice1.jpg
    alice2.jpg
  Ronaldo/
    bob1.jpg
```

## CLI Overview
- `enroll_faces.py`: Builds encodings from `references/` and writes to `encodings/people.json` + `encodings/people.npy`  
- `sec_cam.py`: Runs webcam loop; logs events and saves snapshots of unknown faces  
- `summarize_logs.py`: Aggregates CSV log(s) into simple metrics and charts
