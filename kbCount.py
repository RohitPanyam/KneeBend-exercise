import mediapipe as mp
import cv2
import numpy as np
import time
import threading


def countdown(time_sec):
    while time_sec:
        mins, secs = divmod(time_sec, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        time_sec -= 1
        print(time_sec)

    global lock
    lock = 0
    print("Done")


def calc_angle(a, b, c):  # a-first(hip), b-center(knee), c-last(heel)
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


# importing dependencies
mp_drw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# input
vid = cv2.VideoCapture("reso/KneeBend.mp4")
# vid = cv2.VideoCapture("reso/trial1.mp4")
# vid = cv2.VideoCapture("reso/trial2.mp4")
# vid = cv2.VideoCapture("reso/trial3.mp4")
# vid = cv2.VideoCapture("reso/trial4.mp4")
i = 1
reps = 0
status = None
lock = 0

with mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5) as pose:
    sec = 0
    start = None
    end = None
    while vid.isOpened():
        r, frame = vid.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # getting the points
        try:
            landmarks = result.pose_landmarks.landmark

            # check the leg
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            l_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            r_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y]

            # angle
            angle = calc_angle(l_hip, l_knee, l_heel)

            # reps
            if angle < 120:
                status = "yes"
                # sec = round(time.time()) - round(start)
                if lock == 0:
                    # t1 = threading.Thread(countdown(8))
                    # t1.start()
                    # t1.join()
                    start = time.time()
                    lock = 1
                sec = round(time.time()) - round(start)

            if angle > 160 and status == "yes":
                status = "no"
                print(reps)

                if lock == 1:
                    end = time.time()
                    sec = round(end) - round(start)
                    print(sec)
                    if sec >=8:
                        start = end - start
                        lock = 0
                        reps += 1
                        status = "Yes"
                    else:
                        status = "No"

            # viz
            cv2.putText(img, str(angle), tuple(np.multiply(l_knee, [640, 480]).astype(int)),
                        cv2.FONT_ITALIC, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.rectangle(img, (0, 0), (225, 73), (222, 111, 0), -1)
            cv2.putText(img, "Reps:", (12, 50), cv2.FONT_ITALIC, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, str(reps), (120, 60), cv2.FONT_ITALIC, 2, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.rectangle(img, (415, 0), (640, 73), (222, 111, 0), -1)
            # cv2.putText(img, "Timer per rep:", (427, 50), cv2.FONT_ITALIC, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(img, str(reps), (535, 60), cv2.FONT_ITALIC, 2, (0, 0, 0), 1, cv2.LINE_AA)
            if status == "No":
                cv2.putText(img, "Keep your knee bent", (427, 50), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "Rep Timing:", (427, 50), cv2.FONT_ITALIC, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, str(sec), (540, 60), cv2.FONT_ITALIC, 2, (0, 0, 0), 1, cv2.LINE_AA)

        except:
            pass

        mp_drw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
