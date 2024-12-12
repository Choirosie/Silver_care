import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)
cascade_filename = "/home/pi/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_filename)

def imgDetector(img, cascade):
    img = cv2.resize(img, dsize=None, fx=1.0, fy=1.0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = cascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(20, 20),
    )
    for box in results:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
    return img    

def gen1():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        success, frame = camera.read()
        if not success:
            break
        retImg = imgDetector(frame, cascade)
        
        # 메모리 버퍼에 저장하여 효율적으로 처리
        _, buffer = cv2.imencode('.jpg', retImg)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
    camera.release()
    cv2.destroyAllWindows()

def gen2():
    camera = cv2.VideoCapture(2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        #카메라 뜨도록 
        success, frame = camera.read()
        if not success:
            break
        retImg = imgDetector(frame, cascade)
    
        # 메모리 버퍼에 저장하여 효율적으로 처리
        _, buffer = cv2.imencode('.jpg', retImg)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
    camera.release()
    cv2.destroyAllWindows()

@app.route("/")
def index():
    return render_template("index_LIVE.html")

@app.route("/room")
def video_feed_room():
    return Response(gen1(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/living")
def video_feed_living():
    return Response(gen2(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, debug=True, threaded=True)
