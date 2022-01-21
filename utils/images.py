import cv2
import base64
import numpy as np

def cal_shape(shape, w_target, by4=False):
    w, h = shape
    h_target = int(h * (w_target / w))

    if by4:
        r = h_target % 4
        h_target += (4 - r) if r > 2 else (-r)

    return (w_target, h_target)


def padding(mask, pad=3):
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


def overlay_mask(img, mask, color=(0, 255, 0), contour=False):
    result = img.copy()  # deepcopy doesnt work if you directly use img
    mask_color = np.zeros(result.shape, result.dtype)
    mask_color[:, :] = color
    mask_color = cv2.bitwise_and(mask_color, mask_color, mask=mask)
    cv2.addWeighted(mask_color, 0.7, result, 1, 0, result)

    if contour:
        mask_gray = np.repeat(mask[..., np.newaxis], 3, axis=2)
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_RGB2GRAY)
        cnt, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, cnt, -1, (255), 2)
    
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


def write_text_on_image(img, text, color_text=(0, 255, 0), color_boundary=(0, 0, 0)):
    for i, line in enumerate(text.split("\n")):
        pos = (5, 20 + i * 23)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        color = color_boundary
        thickness = 4

        img = cv2.putText(img, line, pos, font, fontScale, color, thickness)

    for i, line in enumerate(text.split("\n")):
        pos = (5, 20 + i * 23)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        color = color_text
        thickness = 1

        img = cv2.putText(img, line, pos, font, fontScale, color, thickness)
    
    return img



def base64_to_img(string):
    buffer = base64.b64decode(string)
    img_np = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(img_np, flags=1)
    return img


def img_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
    string = str(base64.b64encode(buffer))
    return string