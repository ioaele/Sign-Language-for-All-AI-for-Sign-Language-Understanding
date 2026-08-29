import numpy as np
import math 

#related paper: https://ieeexplore.ieee.org/document/9289551
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


# keep the most important facial landmarks because according to the paper if we keep them all 468 the performance is much lower
LEFT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]
RIGHT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]
LEFT_EYEBROW = [
    70, 63, 105, 66, 107
]
RIGHT_EYEBROW = [
    336, 296, 334, 293, 300
]
MOUTH = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 409, 270, 269, 267,
    0, 37, 39, 40, 185, 191, 80, 81,
    82, 13, 312, 311, 310, 415
]

FACE= LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW + MOUTH 

def normalization(results,image): 

#function gia na metaferthoun afto to mediapipe data type se array (numpy) for better number handling
# In MediaPipe Holistic, results.face_landmarks.landmark is a list-like protobuf container of NormalizedLandmark objects.
# More specifically results.face_landmarks.landmark is a RepeatedCompositeContainer
# containing 478 NormalizedLandmark objects (for the standard MediaPipe Face Mesh with refined landmarks).
# Each landmark has fields such as: landmark.x,landmark.y,landmark.z,landmark.visibility  
    pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks,x_left, y_left, d=transfer_results(results,image)
    
    if pose_landmarks is None:
        return None, None, None, None
    
    # normalize landmarks
    normalize_formula(pose_landmarks,d,x_left,y_left)
    normalize_formula(face_landmarks,d,x_left,y_left)
    normalize_formula(left_hand_landmarks,d,x_left,y_left)
    normalize_formula(right_hand_landmarks,d,x_left,y_left)
    
    return pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks
    
def normalize_formula(landmarks,d,x_left,y_left):
    #h formula apo to paper me vasi to starting point p einai to left shoulder kai to stathero distance metaksi twn dio shoulders
    landmarks[:, 0] = (landmarks[:, 0] - x_left) / d
    landmarks[:, 1] = (landmarks[:, 1] - y_left) / d

def transfer_results(results,image):
    # get the height and width of the video frame
    h, w, _ = image.shape
    
    #metatrepo se numpy afou prota ta kano pixels 
    #an sto sigkekrimeno frame de entopistoun landmarks gia to sigkekrimeno body part, mpenoun analoga posa landmarks exoume gia to kathe ena 0.
    #ref:https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker
    #The following models are packaged together into a downloadable model bundle:
        # Pose detection and landmark model: tracks 33 body pose coordinates.
        # ace detection and mesh model: detects and tracks 468 3D face mesh landmarks, with optional 52 blendshape coefficients.
        # Palm detection and hand landmark model: tracks 21 knuckle coordinates per hand.
    
    if results.face_landmarks: 
        face_landmarks = np.array([[results.face_landmarks.landmark[l].x * w, results.face_landmarks.landmark[l].y * h] for l in FACE])
    else:
        face_landmarks = np.zeros((len(FACE), 2)) 
        
    if results.left_hand_landmarks:
        left_hand_landmarks = np.array([[l.x * w, l.y * h]for l in results.left_hand_landmarks.landmark])
    else:
        left_hand_landmarks = np.zeros((21, 2))

    if results.right_hand_landmarks:
        right_hand_landmarks = np.array([[l.x * w, l.y * h] for l in results.right_hand_landmarks.landmark])
    else:
        right_hand_landmarks = np.zeros((21, 2))

    if results.pose_landmarks:
        pose_landmarks = np.array([[l.x * w, l.y * h] for l in results.pose_landmarks.landmark])
        
    if not results.pose_landmarks:
        return None, None, None, None, None, None, None #MOST IMPORTANT valloume to NONE giati etsi den tha exoume oute left oute right shoulder for calculations

    x_left, y_left, d = calculate_shoulder_distance(pose_landmarks)
    
    return  pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks,x_left, y_left, d

def calculate_shoulder_distance(pose_landmarks):
    left_shoulder = pose_landmarks[LEFT_SHOULDER]
    right_shoulder = pose_landmarks[RIGHT_SHOULDER]

    d=math.sqrt((left_shoulder[0] - right_shoulder[0])**2 + (left_shoulder[1] - right_shoulder[1])**2) #basic distance formula between 2 coordinates
        
    return left_shoulder[0], left_shoulder[1], d



