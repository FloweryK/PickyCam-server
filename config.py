# Recording option
IS_RECORDING = False
FRAME_RATE = 0.9

# CPU / GPU option
DEVICE = 'cpu'

# Model checkpoints
MODEL_EDGE_CHECKPOINT_PATH = 'model/checkpoints/places2/EdgeModel_gen.pth'
MODEL_INPAINT_CHECKPOINT_PATH = 'model/checkpoints/places2/InpaintingModel_gen.pth'

# Segmentation, Inpainting
RESIZE_WIDTH = 128
PAD = 5
TARGET_CLASS = 15

# Face detection
MAX_DISTANCE = 0.4
