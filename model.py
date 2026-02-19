import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class BaselineTactileEncoder(nn.Module):
    def __init__(
        self,
        vision_resnet_emb_size: int,
        tactile_resnet_emb_size: int,
        model_emb_size: int):
        
        super(BaselineTactileEncoder, self).__init__()
        self.vision_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.tactile_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.vision_resnet_emb_size = vision_resnet_emb_size
        self.tactile_resnet_emb_size = tactile_resnet_emb_size
        self.model_emb_size = model_emb_size
        
        self.resnet_weights = ResNet50_Weights.DEFAULT
        self.resnet_transforms = self.resnet_weights.transforms()

        
        ## Vision part
