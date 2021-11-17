import cv2
import numpy as np
from models.segmentation.yolact.model import SegModel
from models.inpainting.edgeconnect.model import InpaintModel
from models.superresolution.espcn.model import SuperResModel
from utils.timer import Timer


def pad(mask, pad=3):
    # pad masked area
    kernel = np.ones((1 + 2 * pad, 1 + 2 * pad))
    mask = cv2.filter2D(mask.astype(np.float32), -1, kernel)
    mask = (mask >= 1).astype(np.float32)
    return mask


def main():
    # settings
    DEVICE = "cuda"
    RESIZE_ORG = (480, 1016)
    RESIZE_INP = (108, 192)
    RESIZE_SPR = (120, 254)
    PAD = 15

    # timer
    timer = Timer()

    model_seg = SegModel(DEVICE)
    model_inpaint = InpaintModel(DEVICE)
    model_supres = SuperResModel()

    cap = cv2.VideoCapture("testvideo2.mp4")

    while cap.isOpened():
        # cv2 is in BGR format
        success, img = cap.read()

        if success:
            timer.initialize()

            # convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # human segmantation
            masks = model_seg(img)
            if type(masks) != type(None): # TODO: code cleaning
                masks = masks.detach().cpu().numpy()
            else:
                masks = []
            timer.check("human segmentation")
            
            # known face recognition
            mask_unknown = [np.zeros(img.shape[:2], dtype=np.float32)]
            mask_known = [np.zeros(img.shape[:2], dtype=np.float32)]

            for i, mask in enumerate(masks):
                # TODO: face recognition
                known = True if i == 0 else False

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

            timer.check("known face recognition")

            # inpainting
            img_inpaint = cv2.resize(img, RESIZE_INP, interpolation=cv2.INTER_AREA)
            mask_inpaint = cv2.resize(mask_unknown, RESIZE_INP, interpolation=cv2.INTER_AREA)

            img_inpaint = model_inpaint(img_inpaint, mask_inpaint)
            img_inpaint = img_inpaint.detach().cpu().numpy()

            img_inpaint = cv2.convertScaleAbs(img_inpaint, alpha=(255.0))
            timer.check("inpainting")

            # super resolution
            # img_supres = cv2.resize(img_inpaint, RESIZE_SPR, interpolation=cv2.INTER_CUBIC)
            # img_supres = model_supres(img_supres)
            # timer.check("super resolution")

            # resize to original size
            mask_unknown = cv2.resize(mask_unknown, RESIZE_ORG, interpolation=cv2.INTER_NEAREST)
            img_inpaint = cv2.resize(img_inpaint, RESIZE_ORG, interpolation=cv2.INTER_CUBIC)

            # replace human into inpainted background
            img_erased = img.copy()
            img_erased[mask_unknown == True] = img_inpaint[mask_unknown == True]

            # tetris display
            img_with_mask = img.copy()
            mask_color = np.zeros(img_with_mask.shape, img_with_mask.dtype)
            mask_color[:, :] = (0, 255, 0)
            mask_color = cv2.bitwise_and(mask_color, mask_color, mask=mask_unknown)
            cv2.addWeighted(mask_color, 0.7, img_with_mask, 1, 0, img_with_mask)

            display = np.vstack((
                np.hstack((img, img_with_mask)), 
                np.hstack((img_inpaint, img_erased))
                ))
            display = cv2.resize(display, (500, int(500 * display.shape[0]/display.shape[1])))
            display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            timer.check("merging")

            # write fps info
            for i, line in enumerate(timer.get_result_as_text().split("\n")):
                display = cv2.putText(display, line, (5, 15 + i * 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,)

            # show result
            cv2.imshow("result", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break



if __name__ == "__main__":
    main()
