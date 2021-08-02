import time
import cv2
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import canny
from torchvision import models
from torchvision import transforms as T
from edgeconnect.src.networks import EdgeGenerator, InpaintGenerator

DEVICE = 'cpu'
TARGET_CLASS = 15
MODEL_EDGE_CHECKPOINT_PATH = 'edgeconnect/checkpoints/places2/EdgeModel_gen.pth'
MODEL_INPAINT_CHECKPOINT_PATH = 'edgeconnect/checkpoints/places2/InpaintingModel_gen.pth'


class SegModel:
	def __init__(self, device):
		self.device = device

		# torch/hub.py line 127 and 423 had been modified due to ssl error.
		# please check: https://gentlesark.tistory.com/57
		self.model = torch.hub.load('pytorch/vision', 'fcn_resnet101', pretrained=True)
		self.model = self.model.eval()
		self.model = self.model.to(device)

	@staticmethod
	def preprocess(img):
		t = T.Compose([
			T.ToTensor(),
			T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

		img = t(img)
		img = img.unsqueeze(0)
		return img

	def predict(self, img):
		# preprocess
		inp = self.preprocess(img)

		# move input to device
		inp = inp.to(self.device)

		# predict
		masks = self.model(inp)['out']

		# postprocess
		mask = torch.argmax(masks.squeeze(), dim=0).detach().cpu().numpy()
		mask = mask == 15

		# some weird error
		mask = mask.astype(np.uint8)
		pad = 7
		mask_right = np.roll(mask, pad, axis=1)
		mask_left = np.roll(mask, -pad, axis=1)
		mask_up = np.roll(mask, pad, axis=0)
		mask_down = np.roll(mask, -pad, axis=0)

		mask = np.logical_or.reduce((mask, mask_up, mask_down, mask_left, mask_right))
		mask = mask.astype(np.uint8)

		# for i, j in np.argwhere(mask == 1):
		# 	if mask[i, j] == 1:
		# 		pad = 5
		# 		left = i - pad if i - pad > 0 else 0
		# 		right = i + pad if i + pad < 256 else 256
		# 		down = j - pad if j - pad > 0 else 0
		# 		up = j + pad if j + pad < 256 else 256
		#
		# 		mask[left:right, down:up] = 1

		return mask


class InpaintModel:
	def __init__(self, device, edge_checkpoint, inpaint_checkpoint):
		self.device = device

		self.edge_model = EdgeGenerator()
		self.edge_model.load_state_dict(torch.load(edge_checkpoint, map_location=torch.device(device))['generator'])
		self.edge_model = self.edge_model.eval()
		self.edge_model = self.edge_model.to(device)

		self.inpaint_model = InpaintGenerator()
		self.inpaint_model.load_state_dict(torch.load(inpaint_checkpoint, map_location=torch.device(device))['generator'])
		self.inpaint_model = self.inpaint_model.eval()
		self.inpaint_model = self.inpaint_model.to(device)

	@staticmethod
	def preprocess(img, grayscale=False):
		if grayscale:
			t = T.Compose([
				T.Grayscale(),
				T.ToTensor()
			])
		else:
			t = T.Compose([
				T.ToTensor()
			])

		img = Image.fromarray(img)
		img = t(img)
		img = img.float()
		img = img.unsqueeze(0)
		return img

	def predict(self, img, mask):
		# preprocess of edge generation
		img_gray = self.preprocess(img, grayscale=True)
		edge = self.preprocess(canny(rgb2gray(img), sigma=1, mask=(1-mask).astype(bool)))
		mask = self.preprocess(mask)

		img_gray_masked = img_gray * (1 - mask.bool().int()) + mask * 255
		edge_masked = edge * (1-mask.bool().int())
		inp_edge = torch.cat((img_gray_masked, edge_masked, mask), dim=1)

		# move input to device
		inp_edge = inp_edge.to(self.device)

		# generate edges
		edge_generated = self.edge_model(inp_edge)

		# preprocess of inpainting
		img = self.preprocess(img)
		img_masked = img * (1 - mask.bool().int()) + mask * 255
		img_masked = img_masked.to(self.device)
		inp_inpaint = torch.cat((img_masked, edge_generated), dim=1)

		# generate paint
		paint_generated = self.inpaint_model(inp_inpaint)

		# postprocess
		paint_generated = paint_generated.detach().cpu().numpy()[0]

		return paint_generated


def main():
	segModel = SegModel(device=DEVICE)
	inpaintModel = InpaintModel(device=DEVICE,
								edge_checkpoint=MODEL_EDGE_CHECKPOINT_PATH,
								inpaint_checkpoint=MODEL_INPAINT_CHECKPOINT_PATH)

	canvas = np.zeros((480, 640, 3), dtype=np.uint8)
	has_memory = np.zeros((480, 640, 3), dtype=bool)
	memory = np.zeros((480, 640, 3), dtype=np.uint8)

	# capture webcam
	# if using webcam, the argument in VideoCapture represent the index of video device.
	# if using video file, just replace the argument as file path.
	cap = cv2.VideoCapture(0)

	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	out = cv2.VideoWriter('output.avi', fourcc, 6, (640*2, 480))

	# run on webcam
	while cap.isOpened():
		# cv2 format: np.array(H*W*C) with C as BGR
		success, img = cap.read()

		if success:
			# measure fps
			start = time.time()

			# resize image and convert into RGB
			img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
			img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

			# predict human mask with segmentation model
			mask = segModel.predict(img_resized)

			# memorize background
			tmp_mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_LINEAR)
			tmp_mask = (tmp_mask > 0).astype(bool)
			tmp_mask = np.repeat(tmp_mask[:, :, np.newaxis], 3, axis=2)
			is_changed = (np.abs(memory - img) > 10)
			is_background = np.invert(tmp_mask)
			memory[is_changed & is_background] = img[is_changed & is_background]

			# memory = np.where((memory != img) & (has_memory == 0) & (is_background == 1), img, memory)
			has_memory = np.logical_or(has_memory, is_changed & is_background)

			# generate paint
			img_generated = inpaintModel.predict(img_resized, mask)

			# the result is in C*H*W, so convert it into H*W*C
			# img_generated = np.moveaxis(img_generated, 0, -1)

			# now merge all results
			# first, resize mask into original size
			mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_LINEAR)
			mask = (mask > 0).astype(bool).astype(np.uint8)
			mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

			# second, make masked image of original frame
			canvas[mask == 0] = img[mask == 0]

			# third, resize and crop inpainted image
			# img_generated = cv2.resize(img_generated, (640, 480), interpolation=cv2.INTER_LINEAR)
			# img_generated = cv2.convertScaleAbs(img_generated, alpha=(255.0))

			canvas[has_memory & (mask == 1)] = memory[has_memory & (mask == 1)]
			# canvas[(~has_memory) & mask == 1] = img_generated[(~has_memory) & mask == 1]

			# finally, merge all
			# result = img_masked + img_generated

			# measure end time and calculate fps
			end = time.time()
			processtime = end - start
			fps = 1 / processtime
			print(f'fps: {fps:.4f} (time: {processtime*1000:.2f}ms)')

			# show final webcam live video
			result = np.hstack((img, canvas))
			out.write(result)
			cv2.imshow('result', result)
			# cv2.imshow('result', canvas)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break


if __name__ == '__main__':
	main()

