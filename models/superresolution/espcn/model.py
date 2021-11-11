import cv2

class SuperResModel:
    def __init__(self):
        self.model = cv2.dnn_superres.DnnSuperResImpl_create()
        self.model.readModel("models/superresolution/espcn/weights/ESPCN_x4.pb")
        self.model.setModel("espcn", 4)

    def __call__(self, img):
        return self.model.upsample(img)