import time
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from skimage.feature import canny
from skimage.color import rgb2gray
from mmdet.apis import init_detector, inference_detector
from edgeconnect.src.networks import EdgeGenerator, InpaintGenerator
from config import *

# TODO: 
# dataset
# segmentation
# 	backbone 모델 결정하기
# 	segmentation class 제한하기
# inpainting
# 	학습 더 해서 성능 높이기 -> dataset?
# 	resize할 때 super-resolution 해보기
#	mask를 resize할 때 anti-aliasing 적용하기
# mobile
# 	mobile GPU 사용하기
#	fps 
# motivation -> mother


def main():
	# segmentation model
	model_seg = init_detector(MODEL_SEG_CONFIG_FILE, MODEL_SEG_CHECKPOINT_FILE, DEVICE)

	# inpainting model
	model_edge = EdgeGenerator()
	model_edge.load_state_dict(torch.load(MODEL_EDGE_CHECKPOINT_FILE, map_location=torch.device(DEVICE))['generator'])
	model_edge.eval()
	model_inpaint = InpaintGenerator()
	model_inpaint.load_state_dict(torch.load(MODEL_INPAINT_CHECKPOINT_FILE, map_location=torch.device(DEVICE))['generator'])
	model_inpaint.eval()

	# capture webcam
	# if using webcam, the argument in VideoCapture represent the index of video device.
	# if using video file, just replace the argument as file path.
	cap = cv2.VideoCapture(0)

	# run on webcam
	while cap.isOpened():
		# cv2 uses BGR as np.array(H*W*C)
		success, img = cap.read()

		if success:
			# measure fps
			start = time.time()

			### IMAGE SEGMENTATION START ###
			# resize image in order to put it into the segmentation model
			# here we use INTER_AREA interpolation for we shrink our frame here
			# for more info on iterpolation, read this: https://076923.github.io/posts/Python-opencv-8/
			img_resized = cv2.resize(img, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)

			# predict segmentation bboxes and masks
			# predict_seg info
			# predict_seg 2 <class 'tuple'>					 -> 0 for bbox, 1 for mask
			# predict_seg[0] 80 <class 'list'>
			# predict_seg[0][0] 0 <class 'numpy.ndarray'>
			# predict_seg[1] 80 <class 'list'>
			# predict_seg[1][0] 0 <class 'list'>
			predict_seg = inference_detector(model_seg, img_resized)

			# we only care about masks(as [1]) of person class(as [0])
			masks = predict_seg[1][0]

			# merge multi-instance masks
			if len(masks) == 0:
				continue
			elif len(masks) == 1:
				mask = masks[0]
			else:
				mask = np.logical_or.reduce(np.array(masks, dtype=bool))

			# avoiding some weird error: cv2.error: OpenCV(4.5.3) /private/var/folders/24/8k48jl6d249_n_qfxwsl6xvm0000gn/T/pip-req-build-vy_omupv/opencv/modules/highgui/src/precomp.hpp:155: error: (-215:Assertion failed) src_depth != CV_16F && src_depth != CV_32S in function 'convertToShow'
			# see: https://stackoverflow.com/questions/55759535/integral-image-error-code-cv2-imshowimageintegral-imageintegral
			mask = mask.astype(np.uint8)

			### IMAGE INPAINTING START ###
			img_gray = rgb2gray(img_resized)
			edge = canny(img_gray, sigma=2, mask=(1-mask).astype(bool)).astype(float) # is .astype(float) necessary?

			# transfrom all inputs into tensor batches
			img_resized = F.to_tensor(Image.fromarray(img_resized)).float()
			img_gray = F.to_tensor(Image.fromarray(img_gray)).float()
			edge = F.to_tensor(Image.fromarray(edge)).float()
			mask = F.to_tensor(Image.fromarray(mask)).float()

			imgs_resized = torch.unsqueeze(img_resized, 0)
			imgs_gray = torch.unsqueeze(img_gray, 0)
			edges = torch.unsqueeze(edge, 0)
			masks = torch.unsqueeze(mask, 0)

			# erase areas in mask
			imgs_gray_masked = imgs_gray * (1 - masks.bool().int()) + masks * 255
			edges_masked = edges * (1 - masks.bool().int())

			# edge generation
			inputs = torch.cat((imgs_gray_masked, edges_masked, masks), dim=1)
			edges = model_edge(inputs).detach()

			# inpaint generation
			imgs_masked = imgs_resized * (1 - masks.bool().int()) + masks * 255
			inputs = torch.cat((imgs_masked, edges), dim=1)
			outputs = model_inpaint(inputs).detach()
			# outputs = outputs * masks.bool().int()

			output = outputs[0]
			output = torch.swapaxes(output, 0, 1)
			output = torch.swapaxes(output, 1, 2)
			output = output.numpy()
			output = cv2.convertScaleAbs(output, alpha=(255.0))

			# rewind mask in order to merge inpaints with original images
			mask = masks[0]
			mask = torch.swapaxes(mask, 0, 1)
			mask = torch.swapaxes(mask, 1, 2)
			mask = mask.numpy()
			mask = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_LINEAR)
			mask = (mask > 0).astype(bool).astype(np.uint8)
			mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

			# resize to original size
			output = cv2.resize(output, (1280, 720), interpolation=cv2.INTER_LINEAR)
			result = mask * output + img * (1 - mask)

			# measure end time and calculate fps
			end = time.time()
			processtime = end - start
			fps = 1 / processtime
			print(f'fps: {fps:.4f} (time: {processtime*1000:.2f}ms)')

			# show final webcam live video
			cv2.imshow('result', result)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break


if __name__ == '__main__':
	main()

