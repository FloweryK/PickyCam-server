import cv2
import time
import numpy as np
from recoder import Recoder


def main():
	# capture webcam
	# if using webcam, the argument in VideoCapture represent the index of video device.
	# if using video file, just replace the argument as file path.
	cap = cv2.VideoCapture(0)

	# get cap width and height
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	recoder = Recoder(framerate=1/0.1, width=1279, height=959)

	memory = None

	# run on webcam
	while cap.isOpened():
		# cv2 format: np.array(H*W*C) with C as BGR
		success, img = cap.read()

		time.sleep(0.3)

		if memory is not None:
			print('hit')

			canvas = []

			for i in range(3):
				img_r = img[:, :, i]
				memory_r = memory[:, :, i]

				img_rx = np.mean(img_r, axis=0)
				img_ry = np.mean(img_r, axis=1)
				memory_rx = np.mean(memory_r, axis=0)
				memory_ry = np.mean(memory_r, axis=1)

				denom_x = np.array([(i+1) if i <= (len(img_rx)-1) else len(img_rx) - i for i in range(2*len(img_rx)-1)])
				denom_y = np.array([(i+1) if i <= (len(img_ry)-1) else len(img_ry) - i for i in range(2*len(img_ry)-1)])
				conv_x = np.convolve(img_rx, memory_rx) / (denom_x**2)
				conv_y = np.convolve(img_ry, memory_ry) / (denom_y**2)

				x = np.argmin(conv_x)
				y = np.argmin(conv_y)

				# c = np.zeros((len(conv_y), len(conv_x)))
				# c[y, x] = 1
				# c = cv2.filter2D(c, -1, np.ones((10, 10)))
				# canvas.append(c)

				a = np.tile(conv_x, (len(conv_y), 1))
				b = np.tile(conv_y, (len(conv_x), 1))

				M = a * b.T
				# M = (M - np.min(M)) / (np.max(M) - np.min(M))
				canvas.append(M)

			canvas = np.array(canvas) * 255
			canvas = np.moveaxis(canvas, 0, -1)
			canvas = canvas.astype(np.uint8)

			text = f'max conv position = ({x}, {y})'
			print(text)
			print(canvas.shape)

			cv2.imshow('result', canvas)
			recoder.write(canvas)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		memory = img


if __name__ == '__main__':
	main()	