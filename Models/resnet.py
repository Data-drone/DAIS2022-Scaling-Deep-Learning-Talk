import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

import pytorch_lightning as pl
import torchvision.models as models


class ResnetClassification(pl.LightningModule):
    """
    
    Our primary model class this has been hardcoded to Resnet50 for now
    
    """
    def __init__(self, channels, width, height, num_classes, pretrain:bool=True, learning_rate=2e-4):
        super().__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()

        
        self.model_tag = 'resnet'

        self.model = models.resnet50(pretrained=pretrain)

        ### TRANSFER LEARNING STEPS
        # change the final layer
        linear_size = list(self.model.children())[-1].in_features
        self.model.fc = nn.Linear(linear_size, self.num_classes)

        # change the input channels as needed
        if self.channels != 3:
            self.model.conv1 = nn.Conv2d(self.channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        ### END TRANSFER LEARNING STEPS


    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer