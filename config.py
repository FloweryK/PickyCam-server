import torch

FRAME_RATE = 0.2
DEVICE = 'cuda' if (torch.cuda.is_available and torch.cuda.device_count() > 0) else 'cpu'
PAD = 5
RESIZE = 256
TARGET_CLASS = 15
MODEL_EDGE_CHECKPOINT_PATH = 'checkpoints/places2/EdgeModel_gen.pth'
MODEL_INPAINT_CHECKPOINT_PATH = 'checkpoints/places2/InpaintingModel_gen.pth'