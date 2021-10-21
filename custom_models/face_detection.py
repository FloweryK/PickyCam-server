import os
import cv2
import numpy as np
import face_recognition


class FaceDetectionModel:
	def __init__(self):
		self.known_faces = {}

		for image_name in os.listdir('known_faces'):
			if image_name[-4:] != '.png':
				continue
			image = cv2.imread(f'known_faces/{image_name}')

			# convert from BGR to RGB
			image = image[:, :, ::-1]

			face_locations = face_recognition.face_locations(image)
			face_encodings = face_recognition.face_encodings(image, face_locations)

			for face_encoding in face_encodings:
				self.known_faces[f'person_{len(self.known_faces)}'] = face_encoding

	def predict(self, img):
		result = {
			'name': 'Unknown', 
			# 'location': None, 
			'distance': 1
		}
		h, w, c = img.shape
		face_locations = face_recognition.face_locations(img)
		face_encodings = face_recognition.face_encodings(img, face_locations)

		c_min = 1

		for i, face_encoding in enumerate(face_encodings):
			confidence = face_recognition.face_distance(list(self.known_faces.values()), face_encoding)
			if min(confidence) < c_min:
				name = list(self.known_faces.keys())[np.argmin(confidence)]
				c_min = min(confidence)
				# y_min, x_max, y_max, x_min = face_locations[i]

				result['name'] = name
				result['distance'] = c_min
				# result['location'] = (x_min, x_max - w, y_min, y_max - h)

		return result


if __name__ == '__main__':
	model_face_detection = FaceDetectionModel()
	print(model_face_detection.known_faces)