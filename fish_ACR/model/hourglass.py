import torch
import torch.nn as nn
import torch.nn.functional as F

class HourglassModule(nn.Module):
    def __init__(self, num_modules, depth, num_features, num_joints):
        super(HourglassModule, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.num_joints = num_joints  # Added number of keypoints parameter
        self._generate_modules()
        self.init_conv = nn.Conv2d(3, 256, kernel_size=1, stride=1, padding=0, bias=False)

    def _generate_modules(self):
        hg = []
        for _ in range(self.num_modules):
            hg.append(self._make_hourglass(self.depth, self.features))
        self.hgs = nn.ModuleList(hg)

        self.res = nn.ModuleList([self._make_residual(self.features, self.features) for _ in range(self.num_modules)])
        self.fc = nn.ModuleList([self._make_fc(self.features, self.features) for _ in range(self.num_modules)])
        self.score = nn.ModuleList([nn.Conv2d(self.features, self.num_joints, kernel_size=1, bias=True) for _ in range(self.num_modules)])
        self.merge = nn.ModuleList([nn.Conv2d(self.features, self.features, kernel_size=1, bias=True) for _ in range(self.num_modules - 1)])

    def _make_hourglass(self, depth, features):
        downsample = nn.MaxPool2d(2, stride=2)
        upsample = nn.Upsample(scale_factor=2)

        hg = []
        for i in range(depth):
            res = self._make_residual(features, features)
            if i == 0 or i == depth-1:
                hg.append(res)
            else:
                hg.append(nn.Sequential(res, downsample))
        hg = hg[::-1]
        for i in range(depth-1):
            hg.append(nn.Sequential(res, upsample))
        return nn.Sequential(*hg)

    def _make_residual(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes // 2, planes // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes // 2, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def _make_fc(self, in_features, out_features):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.init_conv(x)  # Adjust the number of channels in the input image
        out = []
        for i in range(self.num_modules):
            x = self.hgs[i](x)
            x = self.res[i](x)
            x = self.fc[i](x)
            score = self.score[i](x)
            out.append(score)
            if i < self.num_modules - 1:
                x = x + self.merge[i](x)
        return out[-1]  # Return the output of the last module for simplified processing

# Use the modified class to create the model
num_joints = 22
num_modules = 2
depth = 4
num_features = 256

model = HourglassModule(num_modules=num_modules, depth=depth, num_features=num_features, num_joints=num_joints)
