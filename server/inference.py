import cv2
import numpy as np
from models.segmentation.yolact.model import SegModel
from models.inpainting.edgeconnect.model import InpaintModel

def pad(mask, pad=3):
    # pad masked area
    kernel = np.ones((1 + 2 * pad, 1 + 2 * pad))
    mask = cv2.filter2D(mask.astype(np.float32), -1, kernel)
    mask = (mask >= 1).astype(np.float32)
    return mask


class Inferencer:
    def __init__(self):
        DEVICE = "cuda"

        self.model_seg = SegModel(DEVICE)
        self.model_inpaint = InpaintModel(DEVICE)

    def inference(self, img):
        RESIZE_ORG = (480, 1016)
        RESIZE_INP = (108, 192)
        PAD = 15

        # tmp
        img = cv2.resize(img, RESIZE_ORG, cv2.INTER_CUBIC)

        # convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # human segmantation
        masks = self.model_seg(img)
        masks = masks.detach().cpu().numpy()

        # known face recognition
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

        mask_unknown = pad(mask_unknown, PAD)
        mask_unknown -= mask_known * 100
        mask_unknown = (mask_unknown > 0)
        mask_unknown = mask_unknown.astype(np.uint8)

        # fill holes inside each mask
        contour, _ = cv2.findContours(mask_unknown, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(mask_unknown, [cnt], 0, 255, -1)
        mask_unknown = (mask_unknown > 0)
        mask_unknown = mask_unknown.astype(np.uint8)

        # inpainting
        img_inpaint = cv2.resize(img, RESIZE_INP, interpolation=cv2.INTER_AREA)
        mask_inpaint = cv2.resize(mask_unknown, RESIZE_INP, interpolation=cv2.INTER_AREA)

        img_inpaint = self.model_inpaint(img_inpaint, mask_inpaint)
        img_inpaint = img_inpaint.detach().cpu().numpy()

        img_inpaint = cv2.convertScaleAbs(img_inpaint, alpha=(255.0))

        
        # resize to original size
        mask_unknown = cv2.resize(mask_unknown, RESIZE_ORG, interpolation=cv2.INTER_NEAREST)
        img_inpaint = cv2.resize(img_inpaint, RESIZE_ORG, interpolation=cv2.INTER_CUBIC)

        # replace human into inpainted background
        img_erased = img.copy()
        img_erased[mask_unknown == True] = img_inpaint[mask_unknown == True]

        return img_erased