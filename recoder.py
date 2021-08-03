import os
import platform
import datetime
import cv2


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
		self.out = cv2.VideoWriter(os.path.join('result', filename), fourcc, framerate, (width, height))

	def write(self, frame):
		self.out.write(frame)