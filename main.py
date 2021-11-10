import cv2
import numpy as np
from models.segmentation.yolact.model import SegModel
from models.inpainting.edgeconnect.model import InpaintModel
from utils.timer import Timer


def pad(mask, pad=3):
    # pad masked area
    kernel = np.ones((1 + 2 * pad, 1 + 2 * pad))
    mask = cv2.filter2D(mask.astype(np.float32), -1, kernel)
    return mask


def main():
    # settings
    DEVICE = "cuda"
    RESIZE_ORG = (480, 1016)
    RESIZE_INP = (216, 384)
    PAD = 5

    # timer
    timer = Timer()

    model_seg = SegModel(DEVICE)
    model_inpaint = InpaintModel(DEVICE)

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
            masks = masks.detach().cpu().numpy()
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
            timer.check("known face recognition")

            # inpainting
            img_inpaint = cv2.resize(img, RESIZE_INP, interpolation=cv2.INTER_AREA)
            mask_inpaint = cv2.resize(mask_unknown, RESIZE_INP, interpolation=cv2.INTER_AREA)

            img_inpaint = model_inpaint(img_inpaint, mask_inpaint)
            img_inpaint = img_inpaint.detach().cpu().numpy()

            img_inpaint = cv2.convertScaleAbs(img_inpaint, alpha=(255.0))
            timer.check("inpainting")

            # resize to original size
            mask_unknown = cv2.resize(mask_unknown, RESIZE_ORG, interpolation=cv2.INTER_NEAREST)
            img_inpaint = cv2.resize(img_inpaint, RESIZE_ORG, interpolation=cv2.INTER_CUBIC)
            timer.check("resizing")

            # merge all results
            img_erased = img.copy()
            img_erased[mask_unknown == True] = img_inpaint[mask_unknown == True]
            result = np.vstack((img, img_erased))
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            timer.check("merging")

            # write fps info and resize
            for i, line in enumerate(timer.get_result_as_text().split("\n")):
                result = cv2.putText(result, line, (10, 30 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,)
            result = cv2.resize(result, (270, 960), cv2.INTER_AREA)

            # show result
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break



if __name__ == "__main__":
    main()
