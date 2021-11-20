import torch
import numpy as np
from torchvision import models
from torchvision import transforms as T


class SegModel:
    def __init__(self, device):
        self.device = device

        # torch/hub.py line 127 and 423 had been modified due to ssl error.
        # please check: https://gentlesark.tistory.com/57
        # self.net = torch.hub.load('pytorch/vision', 'fcn_resnet101', pretrained=True)
        self.net = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.net = self.net.eval()
        self.net = self.net.to(device)

    def t(self, img):
        t = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                lambda x: x.unsqueeze(0),
            ]
        )

        return t(img)

    def __call__(self, img):
        # preprocess
        img = self.t(img)
        img = img.to(self.device)

        # forward
        masks = self.net(img)["out"]

        # postprocess
        mask = torch.argmax(masks.squeeze(), dim=0)
        mask = mask == 15

        return mask
