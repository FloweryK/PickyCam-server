# mmdetection-segmentation settings
MODEL_SEG_CONFIG_FILE = 'mmdetection/configs/yolact/yolact_r101_1x8_coco.py'
MODEL_SEG_CHECKPOINT_FILE = 'mmdetection/checkpoints/yolact_r101_1x8_coco_20200908-4cbe9101.pth'
MODEL_EDGE_CHECKPOINT_FILE = 'edgeconnect/checkpoints/places2/EdgeModel_gen.pth'
MODEL_INPAINT_CHECKPOINT_FILE = 'edgeconnect//checkpoints/places2/InpaintingModel_gen.pth'
DEVICE = 'cpu'

# cv2 settings
WIDTH = 128
HEIGHT = 128