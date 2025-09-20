# summarize_logs.py — comprehensive log analysis script
import argparse, os, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def analyze_sessions(df):
    """Analyze session duration and frames per user"""
    user_stats = {}
    
    for identity in df['identity'].unique():
        if identity in ['no_face', '']:
            continue
            
        user_df = df[df['identity'] == identity].copy()
        if len(user_df) == 0:
            continue
            
        # Convert timestamp to datetime
        user_df['datetime'] = pd.to_datetime(user_df['ts'], unit='s')
        
        # Calculate session duration (first to last detection)
        session_duration = (user_df['datetime'].max() - user_df['datetime'].min()).total_seconds()
        
        user_stats[identity] = {
            'total_frames': len(user_df),
            'session_duration_seconds': session_duration,
            'avg_confidence': user_df['distance'].apply(lambda x: float(x) if str(x).replace('.','').isdigit() else np.nan).mean()
        }
    
    return user_stats

def create_time_series_chart(df, output_dir):
    """Create line chart of detections over time"""
    if len(df) == 0:
        return
        
    df['datetime'] = pd.to_datetime(df['ts'], unit='s')
    df['minute'] = df['datetime'].dt.floor('T')  # Round to nearest minute
    
    # Count detections per minute for each identity
    time_counts = df.groupby(['minute', 'identity']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(12, 6))
    for identity in time_counts.columns:
        if identity not in ['no_face', '']:
            plt.plot(time_counts.index, time_counts[identity], label=identity, marker='o', linewidth=2)
    
    plt.title('Detections Over Time')
    plt.xlabel('Time')
    plt.ylabel('Detections per Minute')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "detections_timeline.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved timeline chart -> {out_path}")

def create_pie_chart(df, output_dir):
    """Create pie chart of known vs unknown detections"""
    # Filter out no_face entries
    face_df = df[df['identity'] != 'no_face'].copy()
    
    if len(face_df) == 0:
        print("[WARNING] No face detections found for pie chart")
        return
    
    # Categorize as known vs unknown
    face_df['category'] = face_df['identity'].apply(lambda x: 'Known' if x != 'unknown' else 'Unknown')
    category_counts = face_df['category'].value_counts()
    
    plt.figure(figsize=(8, 8))
    colors = ['#2ecc71', '#e74c3c']  # Green for known, red for unknown
    wedges, texts, autotexts = plt.pie(category_counts.values, 
                                      labels=category_counts.index,
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90,
                                      explode=(0.05, 0.05))
    
    plt.title('Known vs Unknown Face Detections', fontsize=16, fontweight='bold')
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    out_path = os.path.join(output_dir, "known_vs_unknown_pie.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved pie chart -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.log_dir, "events_*.csv")))
    if not files:
        print("No logs found.")
        return
    
    print(f"Found {len(files)} log file(s): {[os.path.basename(f) for f in files]}")
    
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    
    print(f"\n=== LOG ANALYSIS REPORT ===")
    print(f"Total log entries: {len(df)}")
    
    # 1. Number of unique users detected
    unique_users = df['identity'].unique()
    known_users = [u for u in unique_users if u not in ['no_face', 'unknown', '']]
    print(f"Unique known users detected: {len(known_users)}")
    if known_users:
        print(f"Known users: {', '.join(known_users)}")
    
    # 2. Total unknown detections
    unknown_count = len(df[df['identity'] == 'unknown'])
    print(f"Total unknown detections: {unknown_count}")
    
    # 3. Session analysis
    user_stats = analyze_sessions(df)
    if user_stats:
        print(f"\n=== USER SESSION ANALYSIS ===")
        for user, stats in user_stats.items():
            print(f"\n{user}:")
            print(f"  Total frames detected: {stats['total_frames']}")
            print(f"  Session duration: {stats['session_duration_seconds']:.1f} seconds")
            if stats['session_duration_seconds'] > 0:
                print(f"  Avg frames per second: {stats['total_frames']/stats['session_duration_seconds']:.2f}")
            if not np.isnan(stats['avg_confidence']):
                print(f"  Average confidence (distance): {stats['avg_confidence']:.3f}")
    
    # 4. Overall identity counts
    print(f"\n=== DETECTION SUMMARY ===")
    counts = df["identity"].value_counts()
    print("All identity counts:")
    for identity, count in counts.items():
        print(f"  {identity}: {count}")
    
    # 5. Generate visualizations
    print(f"\n=== GENERATING CHARTS ===")
    
    # Bar chart (existing)
    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar", color=['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6'][:len(counts)])
    plt.title("Detections per Identity", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Detections")
    plt.xlabel("Identity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    bar_out = os.path.join(args.log_dir, "detections_bar_chart.png")
    plt.savefig(bar_out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved bar chart -> {bar_out}")
    
    # Time series chart
    create_time_series_chart(df, args.log_dir)
    
    # Pie chart
    create_pie_chart(df, args.log_dir)
    
    # Summary statistics to CSV
    summary_path = os.path.join(args.log_dir, "analysis_summary.csv")
    summary_data = []
    
    for user, stats in user_stats.items():
        summary_data.append({
            'user': user,
            'total_frames': stats['total_frames'],
            'session_duration_seconds': stats['session_duration_seconds'],
            'avg_confidence': stats['avg_confidence']
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)
        print(f"[OK] Saved detailed summary -> {summary_path}")
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Charts and summaries saved to: {args.log_dir}")

if __name__ == "__main__":
    main()
