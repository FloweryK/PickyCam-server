import re

import torch
from models.segmentation.yolact.modules.yolact import Yolact
from models.segmentation.yolact.utils.augmentations import val_aug
from models.segmentation.yolact.utils.config import get_config
from models.segmentation.yolact.utils.output_utils import after_nms, nms

class Args:
    def __init__(self):
        self.img_size = 544
        self.weight = 'models/segmentation/yolact/weights/best_30.4_res101_coco_340000.pth'
        self.traditional_nms = False
        self.save_lincomb = False
        self.no_crop = False
        self.visual_thre = 0.3

        prefix = re.findall(r'best_\d+\.\d+_', self.weight)[0]
        suffix = re.findall(r'_\d+\.pth', self.weight)[0]
        self.cfg = self.weight.split(prefix)[-1].split(suffix)[0]
    

class SegModel:
    def __init__(self):
        self.cfg = get_config(Args(), 'detect')
        self.net = Yolact(self.cfg)
        self.net.load_weights(self.cfg.weight, self.cfg.cuda)
        self.net.eval()

        if self.cfg.cuda:
            self.net = self.net.cuda()
    
    def __call__(self, img_origin):
        img_normed = val_aug(img_origin, self.cfg.img_size)
        img_tensor = torch.tensor(img_normed, dtype=torch.float32).unsqueeze(0)

        if self.cfg.cuda:
            img_tensor = img_tensor.cuda()
        
        with torch.no_grad():
            class_p, box_p, coef_p, proto_p = self.net(img_tensor)
        
        img_h, img_w = img_origin.shape[0:2]
        ids_p, class_p, box_p, coef_p, proto_p = nms(class_p, box_p, coef_p, proto_p, self.net.anchors, self.cfg)
        ids_p, class_p, boxes_p, masks_p = after_nms(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, self.cfg)

        return masks_p