import platform
import time
import datetime
import cv2
import torch
import numpy as np
from models.seg_model import SegModel
from models.inpaint_model import InpaintModel
from config import *

class Recoder:
	def __init__(self, framerate, width, height):
		# codec type
		if platform.system() == 'Darwin':
			codec = 'MJPG'
		elif platform.system() == 'Windows':
			codec = 'DIVX'
		else:
			raise TypeError('invalid platform system')
		fourcc = cv2.VideoWriter_fourcc(*codec)

		# filename
		filename = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S') + '.avi'

		# define videowriter
		self.out = cv2.VideoWriter(filename, fourcc, framerate, (width, height))

	def write(self, frame):
		self.out.write(frame)


class Memory:
	def __init__(self, height, width, th=10):
		self.height = height
		self.width = width
		self.th = th

		self.has_memory = np.zeros((height, width, 3), dtype=bool)
		self.background = np.zeros((height, width, 3), dtype=np.uint8)

	def update(self, img, mask):
		# resize the mask into original image size
		mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
		mask = (mask > 0).astype(bool)
		mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

		# check whether there's new background to memorize
		is_changed = (np.abs(self.background - img) > self.th)
		is_background = np.invert(mask)

		# memorize relevant area
		self.has_memory = np.logical_or(self.has_memory, is_changed & is_background)
		self.background[is_changed & is_background] = img[is_changed & is_background]


def boundaries(mask):
	pad = 1
	mask = np.pad(mask, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
	mask_right = np.roll(mask, pad, axis=1)
	mask_left = np.roll(mask, -pad, axis=1)
	mask_up = np.roll(mask, pad, axis=0)
	mask_down = np.roll(mask, -pad, axis=0)

	outer = np.logical_or.reduce(((mask_right-mask).astype(bool), (mask_left-mask).astype(bool), (mask_up-mask).astype(bool), (mask_down-mask).astype(bool))).astype(np.uint8)
	inner = np.logical_or.reduce(((mask-mask_right).astype(bool), (mask-mask_left).astype(bool), (mask-mask_up).astype(bool), (mask-mask_down).astype(bool))).astype(np.uint8)

	outer = outer[pad:-pad, pad:-pad, :]
	inner = inner[pad:-pad, pad:-pad, :]

	return outer, inner



def main():
	# capture webcam
	# if using webcam, the argument in VideoCapture represent the index of video device.
	# if using video file, just replace the argument as file path.
	cap = cv2.VideoCapture(0)

	# get cap width and height
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# model defination
	segModel = SegModel(device=DEVICE)
	inpaintModel = InpaintModel(device=DEVICE,
								edge_checkpoint=MODEL_EDGE_CHECKPOINT_PATH,
								inpaint_checkpoint=MODEL_INPAINT_CHECKPOINT_PATH)

	# background memorizatoin preparing
	memory = Memory(height, width)

	# video recoder
	recoder = Recoder(framerate=FRAME_RATE,
					  width=2*width, height=height)

	# final result canvas
	canvas = np.zeros((height, width, 3), dtype=np.uint8)

	# run on webcam
	while cap.isOpened():
		# cv2 format: np.array(H*W*C) with C as BGR
		success, img = cap.read()

		if success:
			# measure fps
			start = time.time()

			# resize image and convert into RGB
			img_resized = cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)
			img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

			# predict human mask with segmentation model
			mask = segModel.predict(img_resized)

			# update background memory
			memory.update(img, mask)

			# generate paint
			img_generated = inpaintModel.predict(img_resized, mask)

			# the result is in C*H*W, so convert it into H*W*C
			img_generated = np.moveaxis(img_generated, 0, -1)
			img_generated[:, :, [0, 2]] = img_generated[:, :, [2, 0]]
			img_generated = cv2.resize(img_generated, (width, height), interpolation=cv2.INTER_LINEAR)
			img_generated = cv2.convertScaleAbs(img_generated, alpha=(255.0))

			# now merge all results
			# resize mask into original size
			mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
			mask = (mask > 0).astype(bool).astype(np.uint8)
			mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

			# firstly, make masked image of original frame
			canvas[mask == 0] = img[mask == 0]

			# secondly, paint background with memory if there's some relavant background memory
			canvas[(mask == 1) & memory.has_memory] = memory.background[(mask == 1) & memory.has_memory]

			# third, resize and crop inpainted image
			canvas[(mask == 1) & (~memory.has_memory)] = img_generated[(mask == 1) & (~memory.has_memory)]

			# balance alpha
			# outer, inner = boundaries(mask)

			# measure end time and calculate fps
			end = time.time()
			processtime = end - start
			fps = 1 / processtime
			fps_text = f'fps: {fps:.4f} (time: {processtime*1000:.2f}ms)'
			print(fps_text)

			# show final webcam live video
			result = np.hstack((img, canvas))
			result = cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
			recoder.write(result)

			cv2.imshow('result', result)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break


if __name__ == '__main__':
	main()

