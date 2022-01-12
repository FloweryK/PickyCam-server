import os
import cv2
from face_recognition.api import face_distance, face_encodings, face_locations
import numpy as np
import face_recognition as fr
from models.segmentation.yolact.model import SegModel
from models.inpainting.edgeconnect.model import InpaintModel
from utils.timer import Timer
from utils.images import *


class ServeModel:
    def __init__(self):
        # TODO: leave out direct configs
        DEVICE = "cuda"

        # TODO: split network and pre/post process codes in each inferencers
        self.model_seg = SegModel(DEVICE)
        self.model_inp = InpaintModel(DEVICE)

        # load known_faces
        self.known_faces = []
        self.load_known_faces()

        # utils
        self.timer = Timer()

    def load_known_faces(self, dir='faces/'):
        for file_name in os.listdir(dir):
            img = fr.load_image_file(os.path.join(dir, file_name))
            face_locations = fr.face_locations(img)
            face_encodings = fr.face_encodings(img, face_locations)
            self.known_faces += face_encodings

    def human_segmentation(self, img, shape):
        # preprocess
        img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)

        # net forward
        masks = self.model_seg(img)

        # postprocess
        if type(masks) != type(None):  # TODO: code cleaning
            masks = masks.detach().cpu().numpy()
        else:
            masks = []

        return masks

    def face_recognition(self, img, masks, shape):
        knowns = [False for _ in range(len(masks))]

        # preprocess
        img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)

        # extract all faces
        face_locations = fr.face_locations(img)
        face_encodings = fr.face_encodings(img, face_locations)

        # match face locations to mask
        if len(masks) > 0:
            face_to_mask = {}
            shape_mask = masks[0].shape[::-1]
            r = (shape_mask[0] / shape[0])

            for i, location in enumerate(face_locations):
                ymin, xmax, ymax, xmin = (np.array(location) * r).astype(int).tolist()

                areas = [np.sum(mask[xmin:xmax, ymin:ymax]) for mask in masks]
                face_to_mask[i] = np.argmax(areas)

            # find known face
            for i, encoding in enumerate(face_encodings):
                face_distances = fr.face_distance(self.known_faces, encoding)
                
                known = (np.array(face_distances) < 0.5).any()

                if known:
                    knowns[face_to_mask[i]] = True

        return knowns

    def inpaint(self, img, mask, shape):
        # preprocess
        img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, shape, interpolation=cv2.INTER_AREA)

        # net forward
        img = self.model_inp(img, mask)
        return img

    def inference(self, img, settings={
        "width_seg": 480,
        "width_fcr": 240,
        "width_inp": 100,
        "pad_ratio_known": 0.01,
        "pad_ratio_unknown": 0.04,
        "isDebug": False,
        "faceDetect": True
    }):
        # config
        WIDTH_SEG = settings["width_seg"]
        WIDTH_FCR = settings["width_fcr"]
        WIDTH_INP = settings["width_inp"]
        PAD_RATIO_KNOWN = settings["pad_ratio_known"]
        PAD_RATIO_UNKNOWN = settings["pad_ratio_unknown"]
        IS_DEBUG = settings["isDebug"]
        FACE_DETECT = settings["faceDetect"]

        # settings
        shape_org = img.shape[:2][::-1]
        shape_seg = cal_shape(shape_org, w_target=WIDTH_SEG)
        shape_fcr = cal_shape(shape_org, w_target=WIDTH_FCR)
        shape_inp = cal_shape(shape_org, w_target=WIDTH_INP, by4=True)

        # timer
        self.timer.initialize()

        # convert from BGR to RGB
        img = cv2.resize(img, shape_org, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # human segmantation
        masks = self.human_segmentation(img, shape_seg)
        self.timer.check("human segmentation")

        # known face recognition
        if FACE_DETECT:
            knowns = self.face_recognition(img, masks, shape_fcr)
        else:
            knowns = [False for _ in range(len(masks))]
        self.timer.check("known face recognition")

        # post process 
        mask_unknown = [np.zeros(shape_seg[::-1], dtype=np.float32)]
        mask_known = [np.zeros(shape_seg[::-1], dtype=np.float32)]
        for i, known in enumerate(knowns):
            if known:
                mask_known.append(masks[i])
            else:
                mask_unknown.append(masks[i])

        mask_unknown = sum(mask_unknown)
        mask_known = sum(mask_known)

        mask_unknown = padding(mask_unknown, pad=int(PAD_RATIO_UNKNOWN * img.shape[1]))
        mask_known = padding(mask_known, pad=int(PAD_RATIO_KNOWN * img.shape[1]))
        mask_unknown -= mask_known * 100
        mask_unknown = mask_unknown > 0
        mask_unknown = mask_unknown.astype(np.uint8)
        mask_unknown = fill_mask_hole(mask_unknown)
        mask = mask_unknown
        self.timer.check("unknown mask merging")

        # inpainting
        img_inp = self.inpaint(img, mask, shape_inp)
        img_inp = img_inp.detach().cpu().numpy()
        img_inp = cv2.convertScaleAbs(img_inp, alpha=(255.0))
        self.timer.check("inpainting")

        # resize to original size
        mask = cv2.resize(mask, shape_org, interpolation=cv2.INTER_NEAREST)
        img_inp = cv2.resize(img_inp, shape_org, interpolation=cv2.INTER_CUBIC)

        # replace human into inpainted background
        img_erased = replace_masked_area(img, img_inp, mask)
        self.timer.check("merging")

        # CODES BELOW ARE SOLELY FOR DEV OPTION
        # make img with mask color
        img_mask = overlay_mask(img, mask)

        # tetris display
        if IS_DEBUG:
            result = merge_4by4(img, img_mask, img_inp, img_erased, width=500)
        else:
            result = img_erased
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        self.timer.check("dev merging")

        # write fps info
        if IS_DEBUG:
            result = write_text_on_image(result, self.timer.get_result_as_text())

        return result


if __name__ == "__main__":
    # tmp import only for testing
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--debug", default=True, type=bool)
    parser.add_argument("--faceDetect", default=True, type=bool)
    parser.add_argument("--width_seg", default=480, type=int)
    parser.add_argument("--width_fcr", default=480, type=int)
    parser.add_argument("--width_inp", default=100, type=int)
    parser.add_argument("--pad_ratio_known", default=0.01, type=float)
    parser.add_argument("--pad_ratio_unknown", default=0.04, type=float)

    args = parser.parse_args()

    video_path = args.video
    isDebug = args.debug
    faceDetect = args.faceDetect
    width_seg = args.width_seg
    width_fcr = args.width_fcr
    width_inp = args.width_inp
    pad_ratio_known = args.pad_ratio_known
    pad_ratio_unknown = args.pad_ratio_unknown

    settings = {
        "width_seg": width_seg,
        "width_inp": width_inp,
        "width_fcr": width_fcr,
        "pad_ratio_known": pad_ratio_known,
        "pad_ratio_unknown": pad_ratio_unknown,
        "isDebug": isDebug,
        "faceDetect": faceDetect
    }

    serve_model = ServeModel()

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        # cv2 is in BGR format
        success, img = cap.read()

        if success:
            result = serve_model.inference(img, settings=settings)

            # show result
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
