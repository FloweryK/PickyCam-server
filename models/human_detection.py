import torch


class HumanDetectionModel:
	def __init__(self):
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
		self.model.classes = [0]

	def predict(self, img):
		result = []

		objects = self.model([img]).pandas().xyxy[0]
		for row in objects.values.tolist():
			xmin, ymin, xmax, ymax, confidence, class_num, name = row

			result.append({
				'xmin': xmin,
				'xmax': xmax,
				'ymin': ymin,
				'ymax': ymax,
				'confidence': confidence,
				'class_num': class_num,
				'name': name
				})

		return result