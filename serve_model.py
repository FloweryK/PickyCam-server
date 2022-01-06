import cv2
import numpy as np
from models.segmentation.yolact.model import SegModel
from models.inpainting.edgeconnect.model import InpaintModel
from utils.timer import Timer


def cal_shape(shape, w_target, by4=False):
    w, h = shape
    h_target = int(h * (w_target / w))

    if by4:
        r = h_target % 4
        h_target += (4 - r) if r > 2 else (-r)

    return (w_target, h_target)


def pad(mask, pad=3):
    # pad masked area
    kernel = np.ones((1 + 2 * pad, 1 + 2 * pad))
    mask = cv2.filter2D(mask.astype(np.float32), -1, kernel)
    mask = (mask >= 1).astype(np.float32)
    return mask


def fill_mask_hole(mask):
    contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(mask, [cnt], 0, 255, -1)
    mask = (mask > 0).astype(np.uint8)
    return mask


def replace_masked_area(img1, img2, mask):
    result = img1.copy()  # deepcopy doesnt work if you directly use img1
    result[mask == True] = img2[mask == True]
    return result


def overlay_mask(img, mask):
    result = img.copy()  # deepcopy doesnt work if you directly use img
    mask_color = np.zeros(result.shape, result.dtype)
    mask_color[:, :] = (0, 255, 0)
    mask_color = cv2.bitwise_and(mask_color, mask_color, mask=mask)
    cv2.addWeighted(mask_color, 0.7, result, 1, 0, result)
    return result


def merge_4by4(img11, img12, img21, img22, width):
    result = np.vstack(
        (
            np.hstack((img11, img12)),
            np.hstack((img21, img22)),
        )
    )
    result = cv2.resize(result, (width, int(width * result.shape[0] / result.shape[1])))
    return result


def write_text_on_image(img, text):
    # write fps info
    for i, line in enumerate(text.split("\n")):
        pos = (5, 15 + i * 23)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        color = (0, 0, 0)  # black
        thickness = 2

        img = cv2.putText(img, line, pos, font, fontScale, color, thickness)

    return img


class ServeModel:
    def __init__(self):
        # TODO: leave out direct configs
        DEVICE = "cuda"

        # TODO: split network and pre/post process codes in each inferencers
        self.model_seg = SegModel(DEVICE)
        self.model_inp = InpaintModel(DEVICE)

        # utils
        self.timer = Timer()

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
        # config
        PAD_RATIO = 0.04

        # preprocess
        img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)

        mask_unknown = [np.zeros(img.shape[:2], dtype=np.float32)]
        mask_known = [np.zeros(img.shape[:2], dtype=np.float32)]

        for i, mask in enumerate(masks):
            # TODO: face recognition
            # known = True if i == 0 else False
            known = False

            if known:
                mask_known.append(mask)
            else:
                mask_unknown.append(mask)

        mask_unknown = sum(mask_unknown)
        mask_known = sum(mask_known)

        # postprocess
        mask_unknown = pad(mask_unknown, pad=int(PAD_RATIO * img.shape[1]))
        mask_unknown -= mask_known * 100
        mask_unknown = mask_unknown > 0
        mask_unknown = mask_unknown.astype(np.uint8)
        mask_unknown = fill_mask_hole(mask_unknown)

        return mask_unknown

    def inpaint(self, img, mask, shape):
        # preprocess
        img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, shape, interpolation=cv2.INTER_AREA)

        # net forward
        img = self.model_inp(img, mask)
        return img

    def inference(self, img):
        # config
        WIDTH_SEG = 480
        WIDTH_INP = 100

        # settings
        shape_org = img.shape[:2][::-1]
        shape_seg = cal_shape(shape_org, w_target=WIDTH_SEG, by4=False)
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
        mask = self.face_recognition(img, masks, shape_seg)
        self.timer.check("known face recognition")

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
        result = merge_4by4(img, img_mask, img_inp, img_erased, width=500)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        self.timer.check("dev merging")

        # write fps info
        result = write_text_on_image(result, self.timer.get_result_as_text())

        return result


if __name__ == "__main__":
    # tmp import only for testing
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    args = parser.parse_args()

    video_path = args.video

    serve_model = ServeModel()

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        # cv2 is in BGR format
        success, img = cap.read()

        if success:
            result = serve_model.inference(img)

            # show result
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
