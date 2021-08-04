import os
import torch
import numpy as np
from PIL import Image
from torchvision import models
from torchvision import transforms as T
from skimage.color import rgb2gray
from skimage.feature import canny

# check if there's edgeconnect
if os.path.exists('../edgeconnect'):
	import setup
	setup.main()
from edgeconnect.src.networks import EdgeGenerator, InpaintGenerator

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