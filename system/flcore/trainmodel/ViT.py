import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from torch import nn

import timm
from transformers import AutoImageProcessor, AutoModelForImageClassification,AutoConfig
from transformers import ViTForImageClassification
from torchvision import transforms

class ViT(nn.Module):
    def __init__(self, num_classes):
        super(ViT, self).__init__()
        self.num_classes = num_classes
        # self.model = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.processor = AutoImageProcessor.from_pretrained("./flcore/trainmodel/vit-tiny-patch16-224")
        config = AutoConfig.from_pretrained("./flcore/trainmodel/vit-tiny-patch16-224")
        self.vit = AutoModelForImageClassification.from_config(config)
        # self.model = ViTForImageClassification.from_pretrained('google/vit-small-patch16-224-in21k', num_labels=self.num_classes)
        
        # processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        # self.model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.fc = nn.Linear(self.vit.classifier.in_features, num_classes)
        self.vit.classifier = nn.Identity()
        self.resize = transforms.Resize((224, 224))

    def forward(self, x):
        x = self.resize(x)

        outputs = self.vit(x)
        
        logits = outputs.logits

        logits = self.fc(logits)
        
        return logits