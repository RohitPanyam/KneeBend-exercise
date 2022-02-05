import mediapipe as mp
import cv2
import numpy as np

# importing dependencies
mp_drw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# input
# vid = cv2.VideoCapture(0)
vid = cv2.VideoCapture("reso/trial1.mp4")
with mp_pose.Pose(min_tracking_confidence=0.5,min_detection_confidence=0.5) as pose:
    while vid.isOpened():
        _,frame = vid.read()

        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result = pose.process(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        mp_drw.draw_landmarks(img,result.pose_landmarks,mp_pose.POSE_CONNECTIONS)

        cv2.imshow('img',img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

