import torch
import torch.nn as nn
import torchvision.models as models


class ResNetAdaptedForKeypoints(nn.Module):
    def __init__(self, num_joints, pretrained=True):
        super(ResNetAdaptedForKeypoints, self).__init__()

        # Load the pretrained ResNet50, remove the last two layers
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # Add a transposed convolution layer to increase the resolution of the feature map
        self.upsample = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn_upsample = nn.BatchNorm2d(256)

        # Final convolution layer to adjust the number of channels to the number of keypoints
        self.final_conv = nn.Conv2d(256, num_joints, kernel_size=1, stride=1, padding=0, bias=True)

        # Initialize the weights of the transposed convolution layer and the final convolution layer
        self._initialize_weights()

    def forward(self, x):
        x = self.resnet(x)
        x = self.upsample(x)
        x = self.bn_upsample(x)
        x = torch.relu(x)
        x = self.final_conv(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Create a model instance, specifying the number of keypoints
num_joints = 22  # Assume there are 22 keypoints
model = ResNetAdaptedForKeypoints(num_joints=num_joints)
