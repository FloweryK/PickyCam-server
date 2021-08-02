import torch
import numpy as np
from torchvision import models
from torchvision import transforms as T


class SegModel:
	def __init__(self, device, pad=7):
		self.device = device
		self.pad = pad

		# torch/hub.py line 127 and 423 had been modified due to ssl error.
		# please check: https://gentlesark.tistory.com/57
		# self.model = torch.hub.load('pytorch/vision', 'fcn_resnet101', pretrained=True)
		self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
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

		# pad masked area
		mask_right = np.roll(mask, self.pad, axis=1)
		mask_left = np.roll(mask, -self.pad, axis=1)
		mask_up = np.roll(mask, self.pad, axis=0)
		mask_down = np.roll(mask, -self.pad, axis=0)

		mask = np.logical_or.reduce((mask, mask_up, mask_down, mask_left, mask_right))
		mask = mask.astype(np.uint8)

		return mask
