import os
import time
import cv2
import torch
import numpy as np
from models.seg_model import SegModel
from models.inpaint_model import InpaintModel
from config import *
from recoder import Recoder


def resize_without_distortion(img, width):
	h, w = img.shape[:2]
	height = int((h/w) * width)
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
	segModel = SegModel(device=DEVICE, pad=PAD)
	inpaintModel = InpaintModel(device=DEVICE,
								edge_checkpoint=MODEL_EDGE_CHECKPOINT_PATH,
								inpaint_checkpoint=MODEL_INPAINT_CHECKPOINT_PATH)

	# video recoder
	if IS_RECORDING:
		recoder = Recoder(framerate=FRAME_RATE, 
						  width=2*width, 
						  height=height)

	# final result canvas
	canvas = np.zeros((height, width, 3), dtype=np.uint8)

	# run on webcam
	while cap.isOpened():
		# cv2 format: np.array(H*W*C) with C as BGR
		success, img = cap.read()

		if success:
			# measure fps
			t_start = time.time()

			# resize image and convert into RGB
			img_resized = resize_without_distortion(img, RESIZE_WIDTH)
			img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
			t_resizing = time.time()

			# predict human mask with segmentation model
			mask = segModel.predict(img_resized)
			t_seg = time.time()

			# generate paint
			img_generated = inpaintModel.predict(img_resized, mask)
			t_inpaint = time.time()

			# now merge all results
			# resize mask into original size
			mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
			mask = (mask > 0).astype(bool).astype(np.uint8)
			mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

			# resize inpainted image
			img_generated = cv2.resize(img_generated, (width, height), interpolation=cv2.INTER_LINEAR)

			# firstly, make masked image of original frame
			canvas[mask == 0] = img[mask == 0]

			# secondly, resize and crop inpainted image
			canvas[mask == 1] = img_generated[mask == 1]

			# measure end time and calculate fps
			t_end = time.time()
			processtime = t_end - t_start
			fps = 1 / processtime
			fps_text = f'fps: {fps:.4f} (time: {processtime*1000:.2f}ms)'	
			fps_text += f'\nresizing: {(t_resizing - t_start)*1000:.2f}ms'	# mostly ~3ms
			fps_text += f'\nsegmenting: {(t_seg - t_resizing)*1000:.2f}ms'	# mostly ~800ms
			fps_text += f'\ninpainting: {(t_inpaint - t_seg)*1000:.2f}ms'	# mostly ~1800ms
			fps_text += f'\nending: {(t_end - t_inpaint)*1000:.2f}ms'		# mostly ~24ms
			print(fps_text)

			# show final webcam live video
			result = np.vstack((img, canvas))
			result = putText_with_newline(result, fps_text, (10, 30))
			# result = cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
			
			# record the frame
			if IS_RECORDING:
				recoder.write(result)

			# show image
			cv2.imshow('result', result)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break


if __name__ == '__main__':
	main()

