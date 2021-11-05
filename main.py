import cv2
import torch
import numpy as np
from models.segmentation.deeplabv3.model import SegModel
from models.inpainting.edgeconnect.model import InpaintModel
from utils.timer import Timer


def putText_with_newline(img, text, pos):
    for i, line in enumerate(text.split("\n")):
        x = pos[0]
        y = pos[1] + i * 40
        img = cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img


def pad(mask, pad=3):
    # pad masked area
    kernel = np.ones((1 + 2 * pad, 1 + 2 * pad))
    mask = cv2.filter2D(mask.astype(np.float32), -1, kernel)
    mask = mask >= 1
    return mask


def main():
    # settings
    DEVICE = "cuda"

    # timer
    timer = Timer()

    model_seg = SegModel(DEVICE)
    model_inpaint = InpaintModel(DEVICE)

    cap = cv2.VideoCapture("testvideo.mp4")

    while cap.isOpened():
        # cv2 is in BGR format
        success, img = cap.read()

        if success:
            timer.initialize()

            # convert from BGR to RGB
            img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # human segmantation
            img_seg = cv2.resize(img, (128, 72), interpolation=cv2.INTER_AREA)
            mask_seg = model_seg(img_seg)
            mask_seg = mask_seg.detach().cpu().numpy()
            mask_seg = pad(mask_seg, 5)
            mask_seg = mask_seg.astype(np.uint8)
            timer.check("human segmentation")

            # mask inpainting
            img_inpaint = cv2.resize(img, (128, 72), interpolation=cv2.INTER_AREA)
            mask_inpaint = cv2.resize(mask_seg, (128, 72), interpolation=cv2.INTER_AREA)
            img_gen = model_inpaint(img_inpaint, mask_inpaint)
            img_gen = torch.moveaxis(img_gen, 0, -1)
            img_gen = img_gen.detach().cpu().numpy()
            img_gen = cv2.convertScaleAbs(img_gen, alpha=(255.0))
            timer.check("inpainting")

            # resize to original size
            mask_seg = cv2.resize(
                mask_seg, (1280, 720), interpolation=cv2.INTER_NEAREST
            )
            img_gen = cv2.resize(img_gen, (1280, 720), interpolation=cv2.INTER_CUBIC)
            timer.check("resizing")

            # merge all results
            img_erased = img.copy()
            img_erased[mask_seg == True] = img_gen[mask_seg == True]
            result = np.vstack((img, img_erased))
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            timer.check("merging")

            # write fps info and resize
            for i, line in enumerate(timer.get_result_as_text().split("\n")):
                result = cv2.putText(
                    result,
                    line,
                    (10, 30 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )
            result = cv2.resize(result, (640, 720), cv2.INTER_AREA)

            # show result
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
