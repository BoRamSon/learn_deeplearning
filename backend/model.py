import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=6):
        super(CNNLSTM, self).__init__()
        
        # CNN 특징 추출기
        self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        
        # LSTM 시퀀스 모델
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        
        # 분류기
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        hidden = None
        
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x
