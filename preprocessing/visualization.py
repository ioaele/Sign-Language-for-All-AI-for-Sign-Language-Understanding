#reference: 
#https://github.com/nicknochnack/Full-Body-Estimation-using-Media-Pipe-Holistic/blob/main/Media%20Pipe%20Holistic%20Tutorial.ipynb
#https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker/python


# python3 -m pip install "mediapipe==0.10.14"

import cv2
#facial landmarks i want to visualize
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


def visualization(image,results,frame, holistic, mp_drawing, mp_holistic):
        if results.face_landmarks:

            face = results.face_landmarks.landmark # all facial landmarks but i only choose to display the eyes,brows and mouth 

            # goint through the facial selected landmarks  
            for i in (LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW+ RIGHT_EYEBROW+ MOUTH):

                # get the specific facial landmark using its index
                landmark = face[i]

                # get the height and width of the video frame
                h, w, _ = image.shape

                # convert MediaPipe's coordinates (0 - 1) into actual pixel coordinates
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                # Draw a dot (green) at the landmark position
                cv2.circle(image,(x, y),2,(0, 255, 0), -1) # 2:radius 2 pixels, -1:fill the circle


        if results.left_hand_landmarks:

            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks, # all left hand landmarks 
                mp_holistic.HAND_CONNECTIONS
            )


        if results.right_hand_landmarks:

            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks, # all right hand landmarks 
                mp_holistic.HAND_CONNECTIONS
            )


        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks, # all pose landmarks
                mp_holistic.POSE_CONNECTIONS
            )
      