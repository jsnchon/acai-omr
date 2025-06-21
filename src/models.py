import torch
from torch import nn
from torchvision.models import VisionTransformer

EMBEDDING_DIM = 768

# TODO: surgery on default ViT to change positional embedding scheme
class Encoder(VisionTransformer):
    def __init__(self):
        return
