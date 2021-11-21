import argparse
import base64
import socketio
import cv2
import numpy as np


def img_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    string = base64.b64encode(buffer)
    return string


def base64_to_img(string):
    buffer = base64.b64decode(string)
    img_np = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(img_np, flags=1)
    return img


# define client
sio = socketio.Client()


@sio.event
def connect():
    print("server connected, sending frames")


@sio.event
def disconnect():
    print("server disconnected.")


@sio.event
def response(data):
    print("got response from server: ", data["processed"][-10:])

    # read data
    string = data["processed"]

    # convert from base64 to cv2 format
    img = base64_to_img(string)

    # show image
    # while True:
    #     cv2.imshow("result", img)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, help="host address", required=True)
    parser.add_argument("--port", type=int, help="port number", required=True)

    args = parser.parse_args()

    # client config
    HOST = args.host
    PORT = args.port

    # start connection
    sio.connect(f"http://{HOST}:{PORT}")

    # emit event
    cap = cv2.VideoCapture("testvideo2.mp4")
    while cap.isOpened():
        success, img = cap.read()

        if success:
            string = img_to_base64(img)
            sio.call("request", {"frame": string})

    # disconnect
    sio.disconnect()
