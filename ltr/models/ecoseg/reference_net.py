import torch.nn as nn
import torch
from ltr.models.layers.blocks import LinearBlock
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


def valid_roi(roi: torch.Tensor, image_size: torch.Tensor):
    valid = all(0 <= roi[:, 1]) and all(0 <= roi[:, 2]) and all(roi[:, 3] <= image_size[0]-1) and \
            all(roi[:, 4] <= image_size[1]-1)
    return valid


class ReferenceNet(nn.Module):
    # Network module for feature channel weight
    def __init__(self, input_dim=(256,)):
        super().__init__()
        resolution = 5
        backbone_stride = 16
        self.conv1 = conv(input_dim[0], 256, kernel_size=3, stride=1)
        self.prroi_pool = PrRoIPool2D(resolution, resolution, 1 / backbone_stride)
        self.fc = LinearBlock(256, input_dim[0], resolution)

    def forward(self, feat, bbox):

        # Add batch_index to rois
        batch_size = bbox.size()[0]
        batch_index = torch.Tensor([x for x in range(batch_size)]).view(batch_size, 1).to(bbox.device)

        bbox[:, 2:4] = bbox[:, 0:2] + bbox[:, 2:4]      # xyxy
        rois = torch.cat((batch_index, bbox), dim=1)

        feat = self.conv1(feat)
        feat = self.prroi_pool(feat, rois)
        return self.fc(feat)




