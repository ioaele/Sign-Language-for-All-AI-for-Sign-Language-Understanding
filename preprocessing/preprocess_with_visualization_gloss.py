# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

import os
import cv2
import mediapipe as mp
from visualization import visualization
from normalization import normalization

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

gloss_folder = "data/GSL_isol/health1_signer1_rep1_glosses/glosses0000" 
frame_files = sorted(os.listdir(gloss_folder))  

if not frame_files:
    print("No frames found in folder.")
    exit()


print(f"Βρέθηκαν {len(frame_files)} frames στο {gloss_folder}")


with mp_holistic.Holistic( # mediaPipe holistic detects:
                          # left hand 21 landmarks
                          # right hand 21 landmarks
                          # pose 33 landmarks
                          # face up to 468 facial landmarks

    min_detection_confidence=0.5, # if the confidence that it detect sth is 50% and above then show it 
    min_tracking_confidence=0.5
) as holistic:

    for frame_file in frame_files:

        frame_path = os.path.join(gloss_folder, frame_file)
        frame = cv2.imread(frame_path)  

        if frame is None:
            print(f"Could not read {frame_file}")
            continue
        
        # BGR -> RGB MEDIAPIPE EXPECTS THE FRAME IN RGB 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

        # MediaPipe detection
        results = holistic.process(image) 

        # RGB -> BGR SO WE CAN THEN SHOW THEM IN CV2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        visualization(image,results,frame, holistic, mp_drawing, mp_holistic)
        
        pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks=normalization(results,image)
        
        # print (pose_landmarks)
        # display
        cv2.imshow(
            "Greek Sign Language Video",
            image
        )

        if cv2.waitKey(25) & 0xFF == ord("q"): # if q key is pressed quit/stop video
            break

cv2.destroyAllWindows()
