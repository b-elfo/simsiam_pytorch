from turtle import forward
import torch
import torch.nn as nn

import timm

###

class Projector(nn.Module):
    def __init__(self,
                 device: str = 'cuda',
                 ):
        super().__init__()
        self.fcn_1  = nn.Linear(1000, 2048, device=device)
        self.relu_1 = nn.ReLU()
        self.btnm_1 = nn.BatchNorm1d(2048, device=device)

        self.fcn_2  = nn.Linear(2048, 2048, device=device)
        self.relu_2 = nn.ReLU()
        self.btnm_2 = nn.BatchNorm1d(2048, device=device)

        self.fcn_3  = nn.Linear(2048, 2048, device=device)
        self.btnm_3 = nn.BatchNorm1d(2048, device=device)

    def forward(self,
                X: torch.Tensor,
                ):
        X = self.fcn_1(X)
        X = self.relu_1(X)
        X = self.btnm_1(X)
        
        X = self.fcn_2(X)
        X = self.relu_2(X)
        X = self.btnm_2(X)

        X = self.fcn_3(X)
        X = self.btnm_3(X)

        return X


class Predictor(nn.Module):
    def __init__(self,
                 device: str = 'cuda',
                 ):
        super().__init__()
        self.fcn_1  = nn.Linear(2048, 512, device=device)
        self.relu_1 = nn.ReLU()
        self.btnm_1 = nn.BatchNorm1d(512, device=device)
        
        self.fcn_2  = nn.Linear(512, 2048, device=device)

    def forward(self,
                X: torch.Tensor,
                ):
        X = self.fcn_1(X)
        X = self.relu_1(X)
        X = self.btnm_1(X)
        X = self.fcn_2(X)
        return X
        

class SimSiamNet(nn.Module):
    def __init__(self,
                 model_name: str = 'resnet50',
                 pretrained: bool = False,
                 device: str = 'cuda',
                 ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained).to(device)
        self.projector = Projector(device=device)
        self.predictor = Predictor(device=device)

    def forward(self,
                X: torch.Tensor,
                ):
        X = self.backbone(X)
        X_proj = self.projector(X)
        X_proj = X_proj.detach()
        X_pred = self.predictor(X_proj)
        return X_proj, X_pred

###

class DownStreamNet(SimSiamNet):
    def __init__(self,
                 model_name: str = 'resnet50',
                 pretrained: bool = False,
                 device: str = 'cuda',
                 ):
        super().__init__()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.head = Classifier(device=device)
    
    def forward(self,
                X: torch.Tensor,
                ):
        X = self.backbone(X)
        X = self.head(X)
        return X

###

class Classifier(nn.Module):
    def __init__(self,
                 num_classes: int = 10,
                 device: str = 'cuda',
                 ):
        super().__init__()
        self.head = nn.Linear(1000, num_classes, device=device)

    def forward(self,
                X: torch.Tensor,
                ):
        X = self.head(X)
        return X