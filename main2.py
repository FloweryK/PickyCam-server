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


def trf_seg(img_resized):
	t = T.Compose([
			# T.Resize(256), 
			# T.CenterCrop(224), 
			T.ToTensor(), 
			T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
		])

	return t(img_resized)


def trf_edge(img_resized, grayscale=False):
	if grayscale:
		t = T.Compose([
			T.Grayscale(),
			T.ToTensor()
		])
	else:
		t = T.Compose([
			T.ToTensor()
		])
	

	return t(img_resized)


def main():
	# cuda configuration
	DEVICE = 'cuda'

	# segmentation configuration
	TARGET_CLASS = 15

	# inpaint configuration
	MODEL_EDGE_CHECKPOINT_FILE = 'edgeconnect/checkpoints/places2/EdgeModel_gen.pth'
	MODEL_INPAINT_CHECKPOINT_FILE = 'edgeconnect//checkpoints/places2/InpaintingModel_gen.pth'

	# capture webcam
	# if using webcam, the argument in VideoCapture represent the index of video device.
	# if using video file, just replace the argument as file path.
	cap = cv2.VideoCapture(0)

	# segmentation model
	model_seg = models.segmentation.deeplabv3_resnet50(pretrained=True)
	model_seg = model_seg.eval()
	model_seg = model_seg.to(DEVICE)

	# inpaint model
	model_edge = EdgeGenerator()
	model_edge.load_state_dict(torch.load(MODEL_EDGE_CHECKPOINT_FILE, map_location=torch.device(DEVICE))['generator'])
	model_edge.eval()
	model_edge.to(DEVICE)
	model_inpaint = InpaintGenerator()
	model_inpaint.load_state_dict(torch.load(MODEL_INPAINT_CHECKPOINT_FILE, map_location=torch.device(DEVICE))['generator'])
	model_inpaint.eval()
	model_inpaint.to(DEVICE)


	# run on webcam
	while cap.isOpened():
		# cv2 uses BGR as np.array(H*W*C)
		success, img = cap.read()

		if success:
			# measure fps
			start = time.time()

			img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
			img_resized_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))

			### SEGMENTATION ###
			# perform segmentation
			inp_seg = trf_seg(img_resized_pil).unsqueeze(0)
			inp_seg = inp_seg.to(DEVICE)
			out_seg = model_seg(inp_seg)['out']

			# post process segmentation
			mask = torch.argmax(out_seg.squeeze(), dim=0).detach().cpu().numpy()
			mask = mask == 15
			mask = mask.astype(np.uint8)

			### INPAINTING ###
			# generate edge
			inp_edge_img_resized 			= trf_edge(img_resized_pil).float().unsqueeze(0)
			inp_edge_img_resized_gray = trf_edge(img_resized_pil, grayscale=True).float().unsqueeze(0)
			inp_edge_edge 		= trf_edge(canny(rgb2gray(img_resized), sigma=2, mask=(1-mask).astype(bool))).float().unsqueeze(0)
			inp_edge_mask 		= trf_edge(mask).float().unsqueeze(0)
			inp_edge = torch.cat((inp_edge_img_resized_gray * (1 - inp_edge_mask.bool().int()) + inp_edge_mask * 255, inp_edge_edge * (1 - inp_edge_mask), inp_edge_mask), dim=1)
			inp_edge = inp_edge.to(DEVICE)
			out_edge = model_edge(inp_edge)

			# generate paint
			inp_inpaint_img_resized_masked = inp_edge_img_resized * (1 - inp_edge_mask.bool().int()) + inp_edge_mask * 255
			inp_inpaint_img_resized_masked = inp_inpaint_img_resized_masked.to(DEVICE)
			inp_inpaint = torch.cat((inp_inpaint_img_resized_masked, out_edge), dim=1)
			inp_inpaint = inp_inpaint.to(DEVICE)
			out_inpaint = model_inpaint(inp_inpaint)

			mask = inp_edge_mask.detach()[0]
			mask = torch.swapaxes(mask, 0, 1)
			mask = torch.swapaxes(mask, 1, 2)
			mask = mask.cpu().numpy()
			mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_LINEAR)
			mask = (mask > 0).astype(bool).astype(np.uint8)
			mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

			output = out_inpaint.detach()[0]
			output = torch.swapaxes(output, 0, 1)
			output = torch.swapaxes(output, 1, 2)
			output = output.cpu().numpy()
			output = cv2.resize(output, (640, 480), interpolation=cv2.INTER_LINEAR)
			output = cv2.convertScaleAbs(output, alpha=(255.0))
			
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

