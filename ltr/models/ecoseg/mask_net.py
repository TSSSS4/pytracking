import torch.nn as nn
import torch
import torch.nn.functional as F
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from ltr.models.layers.blocks import BottleneckWithFixedBatchNorm
from ltr.models.layers.layer import Conv2d, ConvTranspose2d


class MaskNet(nn.Module):
    def __init__(self, input_dim=(256,)):
        super(MaskNet, self).__init__()
        # RoI Pooler
        resolution = 7
        backbone_stride = 16
        self.prroi_pool = PrRoIPool2D(resolution, resolution, 1/backbone_stride)

        self.feature_extractor = ResNet18Conv4FeatureExtractor(
            in_channels=input_dim[0], bottleneck_channels=128, out_channels=input_dim[0], block_count=3)

        self.predictor = MaskRCNNC4Predictor(in_channels=input_dim[0])
        # self.post_processor = MaskPostProcessor()   # Only for evaluation

    def forward(self, feat, bbox):

        # Add batch_index to rois
        batch_size = feat.size()[0]
        boxes_per_frame = int(bbox.size()[0] / batch_size)
        batch_index = torch.Tensor([x for x in range(batch_size)]).view(batch_size, 1).to(bbox.device)
        batch_index = batch_index.repeat_interleave(boxes_per_frame, dim=0)

        bbox[:, 2:4] = bbox[:, 0:2] + bbox[:, 2:4]
        rois = torch.cat((batch_index, bbox), dim=1)

        # Extract RoI feat
        feat = self.prroi_pool(feat, rois)

        # Mask
        feat = self.feature_extractor(feat)
        mask_logits = self.predictor(feat)

        if self.training:
            return mask_logits
        else:
            return mask_logits.sigmoid()


class ResNet18Conv4FeatureExtractor(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, block_count):
        super(ResNet18Conv4FeatureExtractor, self).__init__()

        block_module = BottleneckWithFixedBatchNorm
        blocks = []
        for _ in range(block_count):
            blocks.append(
                block_module(
                    in_channels=in_channels,
                    bottleneck_channels=bottleneck_channels,
                    out_channels=out_channels,
                )
            )
            in_channels = out_channels
        self.ResNetHead = nn.Sequential(*blocks)

    def forward(self, feat):
        return self.ResNetHead(feat)


class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, in_channels):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = 1     # fg
        dim_reduced = 256

        self.conv5_mask = ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


