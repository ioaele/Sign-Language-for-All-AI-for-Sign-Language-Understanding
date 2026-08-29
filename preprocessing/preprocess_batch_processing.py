# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

import os
import cv2
import numpy as np
import mediapipe as mp
from visualization import visualization
from normalization import normalization

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

DATA_ROOT = "data/GSL_isol"
OUTPUT_ROOT = "processed_landmarks"

#epeksergasia for a single gloss folder p periexei ta frames (.jpg picture)
def process_gloss(gloss_folder_path, holistic):
    
    frame_files = sorted(f for f in os.listdir(gloss_folder_path) if f.endswith(".jpg")) # sort frames 000-...

    if not frame_files:
        return None

    gloss_landmarks = []

    for frame_file in frame_files:
        frame_path = os.path.join(gloss_folder_path, frame_file) # enono to path me to actual frame gia na mporw na to diavasw
        frame = cv2.imread(frame_path)

        if frame is None:
            continue

         # BGR -> RGB MEDIAPIPE EXPECTS THE FRAME IN RGB 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

        # MediaPipe detection
        results = holistic.process(image) 

        # # RGB -> BGR SO WE CAN THEN SHOW THEM IN CV2
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks = normalization(results, image)

        if pose_landmarks is None:
            continue  # skip frame pou den "entopistike swma"

        all_landmarks = np.concatenate([
            pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks
        ], axis=0)

        gloss_landmarks.append(all_landmarks)

    if not gloss_landmarks:
        return None

    return np.array(gloss_landmarks)  # (num_frames, num_points, 2)


def main():

    with mp_holistic.Holistic( # mediaPipe holistic detects:
                            # left hand 21 landmarks
                            # right hand 21 landmarks
                            # pose 33 landmarks
                            # face up to 468 facial landmarks

        min_detection_confidence=0.5, # if the confidence that it detect sth is 50% and above then show it 
        min_tracking_confidence=0.5
    ) as holistic:

        folders = [
            f for f in os.listdir(DATA_ROOT)
            if os.path.isdir(os.path.join(DATA_ROOT, f))
        ]

        total_processed = 0
        total_skipped = 0

        for folder_name in folders:

            path = os.path.join(DATA_ROOT, folder_name)
            output_path = os.path.join(OUTPUT_ROOT, folder_name)
            os.makedirs(output_path, exist_ok=True)

            gloss_folders = sorted(
                f for f in os.listdir(path)
                if os.path.isdir(os.path.join(path, f))
            )

            for gloss_name in gloss_folders:

                gloss_folder_path = os.path.join(path, gloss_name)
                output_file = os.path.join(output_path, f"{gloss_name}.npy")

                # skip exei idi epeksergastei
                if os.path.exists(output_file):
                    continue

                landmarks = process_gloss(gloss_folder_path, holistic)

                if landmarks is None:
                    print(f"Skip {folder_name}/{gloss_name}. Frames not found")
                    total_skipped += 1
                    continue

                np.save(output_file, landmarks)
                total_processed += 1

            print(f"Folder {folder_name} done ({len(gloss_folders)} glosses)")

        print(f"\nDone! Processed {total_processed} glosses, skipped {total_skipped}.")


if __name__ == "__main__":
    main()