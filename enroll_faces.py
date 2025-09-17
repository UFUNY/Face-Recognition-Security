# enroll_faces.py — build face encodings from labeled reference images
import argparse, os, json, glob
import numpy as np
import face_recognition

def load_images_and_encode(ref_dir):
    names = []
    encs = []
    for person_dir in sorted(glob.glob(os.path.join(ref_dir, "*"))):
        if not os.path.isdir(person_dir): 
            continue
        label = os.path.basename(person_dir)
        for img_path in glob.glob(os.path.join(person_dir, "*")):
            try:
                img = face_recognition.load_image_file(img_path)
                boxes = face_recognition.face_locations(img, model="hog")
                if not boxes:
                    continue
                encoding = face_recognition.face_encodings(img, known_face_locations=boxes)[0]
                names.append(label)
                encs.append(encoding)
            except Exception as e:
                print(f"[WARN] Skipping {img_path}: {e}")
    return names, np.array(encs) if encs else np.zeros((0,128))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", default="references", help="Folder with labeled subfolders of images")
    ap.add_argument("--enc_dir", default="encodings", help="Output folder for encodings")
    args = ap.parse_args()
    os.makedirs(args.enc_dir, exist_ok=True)

    names, encs = load_images_and_encode(args.ref_dir)
    if len(names) == 0:
        print("[INFO] No encodings created — add images under references/<name>/image.jpg")
        return

    npy_path = os.path.join(args.enc_dir, "people.npy")
    json_path = os.path.join(args.enc_dir, "people.json")
    np.save(npy_path, encs)
    with open(json_path, "w") as f:
        json.dump({"names": names}, f, indent=2)
    print(f"[OK] Saved {len(names)} encodings -> {npy_path} and labels -> {json_path}")

if __name__ == "__main__":
    main()
