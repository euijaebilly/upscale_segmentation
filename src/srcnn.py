import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward_single_scale(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def forward(self, x):
        # Create image pyramid
        x1 = x
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)

        # Process each scale
        y1 = self.forward_single_scale(x1)
        y2 = self.forward_single_scale(x2)
        y3 = self.forward_single_scale(x3)

        # Upsample lower resolutions back to original size
        y2 = F.interpolate(y2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        y3 = F.interpolate(y3, size=x1.shape[2:], mode='bilinear', align_corners=False)

        # Combine all scales
        y = (y1 + y2 + y3) / 3
        return y
