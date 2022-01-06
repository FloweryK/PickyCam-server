import time
import base64
import argparse
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
    # check response arrival time
    # date_res_arrive = time.time()

    # read data
    string = data["frame"]
    # date_req_depart = data["date_req_depart"]
    # date_req_arrive = data["date_req_arrive"]
    # date_res_depart = data["date_res_depart"]
    # interval_base2img = data["interval_base2img"]
    # interval_inference = data["interval_inference"]
    # interval_img2base = data["interval_img2base"]

    # make text
    text = f"got response from server: {string[-10:]}"
    # text += f"\nclient->server: {(date_req_arrive-date_req_depart)*1000:4.1f}ms"
    # text += f" | base2img: {interval_base2img:4.1f}ms"
    # text += f" | inference: {interval_inference:4.1f}ms"
    # text += f" | img2base: {interval_img2base:4.1f}ms"
    # text += f" | server->client: {(date_res_arrive-date_res_depart)*1000:4.1f}ms"
    print(text)

    # convert from base64 to cv2 format
    img = base64_to_img(string)
    print(text)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, help="host address", required=True)
    parser.add_argument("--port", type=int, help="port number", required=True)
    parser.add_argument("--video", required=True, type=str)

    args = parser.parse_args()

    # client config
    HOST = args.host
    PORT = args.port
    video_path = args.video

    # start connection
    sio.connect(f"http://{HOST}:{PORT}", wait_timeout=10)

    # emit event
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, img = cap.read()

        if success:
            # make data
            string = img_to_base64(img)
            date_req_depart = time.time()

            # make json
            json = {
                "frame": string,
                # "date_req_depart": date_req_depart,
            }

            sio.call("request", json)

    # disconnect
    sio.disconnect()
