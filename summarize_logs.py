# summarize_logs.py — aggregate CSV logs into simple metrics
import argparse, os, glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.log_dir, "events_*.csv")))
    if not files:
        print("No logs found.")
        return
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    counts = df["identity"].value_counts()
    print("Identity counts:\n", counts)

    # Bar chart
    plt.figure(figsize=(6,4))
    counts.plot(kind="bar")
    plt.title("Detections per Identity")
    plt.ylabel("Frames detected")
    plt.tight_layout()
    out = os.path.join(args.log_dir, "summary.png")
    plt.savefig(out, dpi=150)
    print(f"[OK] Saved summary chart -> {out}")

if __name__ == "__main__":
    main()
