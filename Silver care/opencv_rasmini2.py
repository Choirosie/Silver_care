from gpiozero import TonalBuzzer
from gpiozero.tones import Tone
from time import sleep
import cv2
import numpy as np
from datetime import datetime
import os
from gpiozero import Servo

buzzer = TonalBuzzer(18)
servo = Servo(15)
servo.detach()

is_time = datetime.now()
target_hour = 14
target_minute = 36

def CCTV():
    thresh = 80
    max_diff = 30

    a, b, c = None, None, None
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    if cap.isOpened():
        ret, a = cap.read()
        ret, b = cap.read()

        while ret:
            ret, c = cap.read()
            draw = c.copy()
            if not ret:
                break
            a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            c_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

            diff1 = cv2.absdiff(a_gray, b_gray)
            diff2 = cv2.absdiff(b_gray, c_gray)

            ret, diff1_t = cv2.threshold(diff1, thresh, 255, cv2.THRESH_BINARY)
            ret, diff2_t = cv2.threshold(diff2, thresh, 255, cv2.THRESH_BINARY)

            diff = cv2.bitwise_and(diff1_t, diff2_t)

            k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k)
            diff_cnt = cv2.countNonZero(diff)

            if diff_cnt > max_diff:
                nzero = np.nonzero(diff)
                cv2.rectangle(
                    draw,
                    (min(nzero[1]), min(nzero[0])),
                    (max(nzero[1]), max(nzero[0])),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    draw,
                    "Motion Detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (0, 0, 255),
                )
                buzzer.stop()  
                break
            cv2.imshow("motion sensor", draw)
            a = b
            b = c
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()


def harr():
    if not os.path.exists('./image'):
            os.makedirs('./image')

    xml = "/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml" 
    face_cascade = cv2.CascadeClassifier(xml)
        
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

        # 얼굴 감지된 경우
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                "Face Detected",
                (10, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 0, 255),
            )
            filename = f"./image/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print("얼굴 감지: 이미지 저장 완료")
            break
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def set_angle():
    servo.value = -1.0
    sleep(5)
    servo.value = 1.0 
    print("서보모터 각도: -90도")
    sleep(1)
    servo.detach() 
    print("서보 모터 동작 완료 및 비활성화")
    
if is_time.hour == target_hour and is_time.minute == target_minute:
    buzzer.play(Tone(240))  
    sleep(1)
    CCTV()  
    sleep(1)
    harr()
    set_angle()


