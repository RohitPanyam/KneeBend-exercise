import mediapipe as mp
import cv2
import numpy as np
import time
import threading

lock = 0


def countdown(time_sec):
    start = time.time()
    end = time.time()
    sec = end - start
    if sec >= 8:
        validity = True
    else:
        validity = False
    return validity


# def countdown(time_sec):
#     while time_sec:
#         mins, secs = divmod(time_sec, 60)
#         timeformat = '{:02d}:{:02d}'.format(mins, secs)
#         print(timeformat, end='\r')
#         time.sleep(1)
#         time_sec -= 1
#         print(time_sec)
#
#     global lock
#     lock = 0
#     print("Done")


# countdown(5)


# importing dependencies
mp_drw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# input
# vid = cv2.VideoCapture(0)
vid = cv2.VideoCapture("reso/trial1.mp4")
# vid = cv2.VideoCapture("reso/KneeBend.mp4")
i = 1
with mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5) as pose:
    count = 0
    while vid.isOpened():
        _, frame = vid.read()

        # print(i)
        # i += 1

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # getting the points
        try:
            landmarks = result.pose_landmarks.landmark
            # print(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
            # print(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y)
            # print(landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y)

            # check the leg
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            l_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            r_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]

            # safe_h =

            # viz
            cv2.putText(img, str(count), count, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            if ((l_knee.y < l_hip.y) and (l_knee.y < l_heel.y)) or ((r_knee.y < r_hip.y) and (r_knee.y < r_heel.y)):
                if lock == 0:
                    lock = 1
                    t1 = threading.Thread(countdown(8))
                    t1.start()
                    # use it directly
                    count += 1
                    print('gg')

            if lock == 1:
                if ((l_knee.y > l_hip.y) and (l_knee.y > l_heel.y)) or ((r_knee.y > r_hip.y) and (r_knee.y > r_heel.y)):
                    status = "hold the leg up"
                    color = "Red"

        except:
            pass

        mp_drw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
