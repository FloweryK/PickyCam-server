import os
import math
import cv2
import torch
import numpy as np
from model.segmentation import SegModel
from model.inpainting import EdgeConnectModel
from model.human_detection import HumanDetectionModel
from model.face_detection import FaceDetectionModel
from timer import Timer
from recoder import Recoder
from config import *


def resize_without_distortion(img, width):
    h, w = img.shape[:2]
    height = int((h / w) * width)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def putText_with_newline(img, text, pos):
    for i, line in enumerate(text.split('\n')):
        x = pos[0]
        y = pos[1] + i * 40
        img = cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img


def main():
    # capture webcam
    # if using webcam, the argument in VideoCapture represent the index of video device.
    # if using video file, just replace the argument as file path.
    cap = cv2.VideoCapture(0)

    # get cap width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # model definition
    humanDetectionModel = HumanDetectionModel()
    faceDetectionModel = FaceDetectionModel()
    segModel = SegModel(device=DEVICE, pad=PAD)
    inpaintModel = EdgeConnectModel(
        device=DEVICE,
        edge_checkpoint=MODEL_EDGE_CHECKPOINT_PATH,
        inpaint_checkpoint=MODEL_INPAINT_CHECKPOINT_PATH,
    )

    # video recoder
    if IS_RECORDING:
        recoder = Recoder(framerate=FRAME_RATE, width=width, height=2 * height)

    # timer
    timer = Timer()

    # final result canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # run on webcam
    while cap.isOpened():
        # cv2 format: np.array(H*W*C) with C as BGR
        success, img = cap.read()
        h, w, c = img.shape

        if success:
            # measure fps
            timer.initialize()

            # resize image and convert into RGB
            img_resized = resize_without_distortion(img, RESIZE_WIDTH)
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            timer.check('resizing')

            # predict human mask with segmentation model
            mask = segModel.predict(img_resized)
            timer.check('human masking')

            # human detection
            detected_humans = humanDetectionModel.predict(img)
            timer.check('human detection')

            # known face detecting
            faces = []
            for human in detected_humans:
                xmin = math.floor(human['xmin'])
                xmax = math.ceil(human['xmax'])
                ymin = math.floor(human['ymin'])
                ymax = math.ceil(human['ymax'])
                img_human = img[ymin:ymax, xmin:xmax]

                face_match = faceDetectionModel.predict(img_human[:, :, ::-1])
                faces.append(
                    {
                        'name': face_match['name'],
                        'distance': face_match['distance'],
                        'xmin': xmin,
                        'xmax': xmax,
                        'ymin': ymin,
                        'ymax': ymax,
                    }
                )

                if face_match['distance'] < MAX_DISTANCE:
                    shrink_rate = RESIZE_WIDTH / w
                    xmin = max(0, math.floor(xmin * shrink_rate) - PAD)
                    xmax = math.ceil(xmax * shrink_rate) + PAD
                    ymin = max(0, math.floor(ymin * shrink_rate) - PAD)
                    ymax = math.ceil(ymax * shrink_rate) + PAD

                    mask[ymin:ymax, xmin:xmax] = 0
            timer.check('known face detecting')

            # generate paint
            img_generated = inpaintModel.predict(img_resized, mask)
            timer.check('inpainting')

            # now merge all results
            # resize mask into original size
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(bool).astype(np.uint8)
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            # resize inpainted image
            img_generated = cv2.resize(img_generated, (width, height), interpolation=cv2.INTER_CUBIC)
            # img_generated = cv2.medianBlur(img_generated, 3)  # not good!

            # firstly, make masked image of original frame
            canvas[mask == 0] = img[mask == 0]

            # secondly, resize and crop inpainted image
            canvas[mask == 1] = img_generated[mask == 1]

            # measure end time and calculate fps
            timer.check('merging')

            # add human detection info
            for human in detected_humans:
                xmin = int(human['xmin'])
                xmax = int(human['xmax'])
                ymin = int(human['ymin'])
                ymax = int(human['ymax'])
                img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

            # add face info
            for face in faces:
                distance = face['distance']
                name = face['name'] if distance < MAX_DISTANCE else 'unknown'
                xmax = face['xmax'] - 150
                ymin = face['ymin'] + 100
                img = cv2.putText(img, f'{name}:{distance:.2f}', (xmax, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, )

            timer.check('writing info')

            # show final webcam live video
            result = np.vstack((img, canvas))
            result = putText_with_newline(result, timer.get_result_as_text(), (10, 30))

            # record the frame
            if IS_RECORDING:
                recoder.write(result)

            # show image
            cv2.imshow('result', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
