import os
import math
import cv2
import torch
import numpy as np
import face_recognition
from model.human_detection import HumanDetectionModel
from model.segmentation import SegModel
from model.inpainting import EdgeConnectModel
from timer import Timer
from recoder import Recoder
from config import *


def resize(img, width, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    height = int((h / w) * width)
    return cv2.resize(img, (width, height), interpolation=interp)


def putText_with_newline(img, text, pos):
    for i, line in enumerate(text.split('\n')):
        x = pos[0]
        y = pos[1] + i * 40
        img = cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img


def pad(mask, pad=3):
    # pad masked area
    kernel = np.ones((1+2*pad, 1+2*pad))
    mask = cv2.filter2D(mask.astype(np.float32), -1, kernel)
    mask = mask >= 1
    return mask


def main():
    # timer
    timer = Timer()

    # recoder
    if IS_RECORDING:
        recoder = Recoder(framerate=frame_RATE, width=width, height=2 * height)

    known_faces = []
    for img_name in os.listdir('faces'):
        if img_name[-4:] != '.png':
            continue
        img = cv2.imread(f'faces/{img_name}')
        img = img[:, :, ::-1]
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        for face_encoding in face_encodings:
            known_faces.append(face_encoding)

    humanDetectionModel = HumanDetectionModel()
    segModel = SegModel(DEVICE)
    inpaintModel = EdgeConnectModel(DEVICE, MODEL_EDGE_CHECKPOINT_PATH, MODEL_INPAINT_CHECKPOINT_PATH)

    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        # cv2 is in BGR format
        success, img = cap.read()

        if success:
            timer.initialize()

            # convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # human segmantation
            mask_seg = segModel.predict(
                img=resize(img, 128))
            timer.check('human segmentation')

            humans = humanDetectionModel.predict(
                img=resize(img, 256))
            timer.check('human detection')

            for human in humans:
                r = width / 256
                xmin = math.floor(r * human['xmin'])
                ymin = math.floor(r * human['ymin'])
                xmax = math.ceil(r * human['xmax'])
                ymax = math.ceil(r * human['ymax'])
                img_human_crop = img[ymin:ymax, xmin:xmax, :]
                img_human_crop = resize(img_human_crop, 128)

                face_locations = face_recognition.face_locations(img_human_crop)
                face_encodings = face_recognition.face_encodings(img_human_crop, face_locations)

                min_dist = 1
                for face_encoding in face_encodings:
                    dists = face_recognition.face_distance(known_faces, face_encoding)
                    min_dist = min(min_dist, min(dists))

                if min_dist < MAX_DISTANCE:
                    r = 128 / 256
                    xmin = math.floor(r * human['xmin'])
                    ymin = math.floor(r * human['ymin'])
                    xmax = math.ceil(r * human['xmax'])
                    ymax = math.ceil(r * human['ymax'])
                    mask_seg[ymin:ymax, xmin:xmax] = 0
            timer.check('known face detection')

            mask_seg = pad(mask_seg, 5)
            mask_seg = mask_seg.astype(np.uint8)

            # mask inpainting
            img_gen = inpaintModel.predict(
                img=resize(img, 128), 
                mask=resize(mask_seg, 128))
            timer.check('inpainting')

            # resize to original size
            mask_seg = resize(mask_seg, width, interp=cv2.INTER_NEAREST)
            mask_seg = np.repeat(mask_seg[:, :, np.newaxis], 3, axis=2)
            mask_seg = (mask_seg > 0).astype(bool)
            img_gen = resize(img_gen, width, interp=cv2.INTER_CUBIC)

            # merge all results
            img_result = img.copy()
            img_result[mask_seg == 1] = img_gen[mask_seg == 1]
            timer.check('merging')

            # convert from RGB to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_result = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)

            # put info on original img
            img = putText_with_newline(img, timer.get_result_as_text(), (10, 30))

            # show result
            cv2.imshow('result', np.vstack((img, img_result)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



if __name__ == '__main__':
    main()
