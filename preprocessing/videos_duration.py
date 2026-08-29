#claude script to see what is the average duration of the videos in the GSL dataset

import os
import cv2

video_folder = "gsl_isolated_videos/"  # ο φάκελος με τα isolated βίντεο

frame_counts = []

for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counts.append(num_frames)
        cap.release()

frame_counts = np.array(frame_counts)

print(f"Πλήθος βίντεο: {len(frame_counts)}")
print(f"Μέσος όρος frames: {frame_counts.mean():.1f}")
print(f"Διάμεσος: {np.median(frame_counts):.1f}")
print(f"Ελάχιστο: {frame_counts.min()}")
print(f"Μέγιστο: {frame_counts.max()}")
print(f"95ο percentile: {np.percentile(frame_counts, 95):.1f}")

# Σε δευτερόλεπτα (30fps)
print(f"\nΣε δευτερόλεπτα (30fps):")
print(f"Μέσος όρος: {frame_counts.mean()/30:.2f} sec")
print(f"95ο percentile: {np.percentile(frame_counts, 95)/30:.2f} sec")