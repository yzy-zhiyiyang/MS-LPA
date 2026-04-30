import torch
import torch.nn as nn

# from .visualEncoder import visualFrontend, visualTCN, visualConv1D, DynamicGate
# from .TCN3D import visualFrontend, visualTCN, visualConv1D, DynamicGate
from .ResNet import ResNet
# from .EEGNet import EEGNet
# from .DenseNet import DenseNet

class LPSModel(nn.Module):
    def __init__(self, name):
        super(LPSModel, self).__init__()
        # Visual Temporal Encoder
        self.ResNet = ResNet()
        self.f_type = name
        # self.EEGNet = EEGNet()
        # self.DenseNet = DenseNet()
        # self.visualFrontend  = visualFrontend() # Visual Frontend
        # self.visualTCN       = visualTCN()      # Visual Temporal Network TCN
        # self.visualConv1D    = visualConv1D()   # Visual Temporal Network Conv1d
        # self.gate            = DynamicGate(tcn_dim=256, conv_dim=256)

        self.FC1 = nn.Linear(256, 128).cuda()
        self.FC2 = nn.Linear(128, 2).cuda()
        self.FC3 = nn.Linear(512, 2).cuda()
        self.FC4 = nn.Linear(16, 2).cuda()
        self.FC5 = nn.Linear(1024, 2).cuda()

    def forward_visual_frontend(self, x):
        B, T, H, W = x.shape
        x = x.view(B*T, 1, 1, H, W)
        x = self.visualFrontend(x)
        x = x.view(B, T, 256)
        x = x.transpose(1, 2)
        x1 = self.visualTCN(x)
        x2 = self.visualConv1D(x)
        x = self.gate(x1, x2)
        x = x.mean(dim=2)
        x = self.FC1(x)
        x = self.FC2(x)

        return x

    def forward_ResNet_frontend(self, x):
        x = x.unsqueeze(1).unsqueeze(3)
        x = self.ResNet(x)
        # x = x.mean(dim=-1)
        x = x.reshape(x.size(0), -1)
        x = self.FC3(x)

        return x

    def forward_EEGNet_frontend(self, x):
        B, T, C = x.shape
        x = x.view(B, 1, T, C)
        x = x.transpose(2, 3)
        x = self.EEGNet(x)
        # x = x.mean(dim=-1)
        x = x.reshape(x.size(0), -1)
        x = self.FC4(x)

        return x

    def forward_DenseNet_frontend(self, x):
        # B, T, C = x.shape
        # x = x.view(B, 1, T, C)
        # x = x.transpose(2, 3)
        x = self.DenseNet(x)
        # x = x.mean(dim=-1)
        x = x.reshape(x.size(0), -1)
        x = self.FC5(x)

        return x

    def forward(self, x):
        if self.f_type == 'vis':
            return self.forward_visual_frontend(x)
        elif self.f_type == 'dense':
            return self.forward_DenseNet_frontend(x)
        elif self.f_type == 'eeg':
            return self.forward_EEGNet_frontend(x)
        elif self.f_type == 'res':
            return self.forward_ResNet_frontend(x)









