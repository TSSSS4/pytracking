import torch.nn as nn
import ltr.models.backbone as backbones
import ltr.models.ecoseg as esmodels
from ltr import model_constructor


class ECOSeg(nn.Module):
    """ ATOM network module"""
    def __init__(self, feature_extractor, reference, masker, feat_layers, backbone_grad=True):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(ECOSeg, self).__init__()

        self.feature_extractor = feature_extractor
        self.reference = reference
        self.masker = masker

        self.feat_layers = feat_layers
        if not backbone_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        # num_sequences = train_imgs.shape[-4]    # This should be batch_size ??
        # num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
        # num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1

        # Extract backbone features tensor(b,c,w,h)
        train_feat = self.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.extract_backbone_features(
            test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))
        # # (1,batch_size,c,w,h)
        # train_feat = [feat.view(num_train_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
        #               for feat in train_feat.values()]
        # test_feat = [feat.view(num_test_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
        #              for feat in test_feat.values()]
        train_feat = [feat for feat in train_feat.values()]     # {(b,c,w,h)} -> (b,c,w,h)
        train_feat = train_feat[0]
        test_feat = [feat for feat in test_feat.values()]
        test_feat = test_feat[0]
        train_bb = train_bb.squeeze()                           # (1,b,4) -> (b,4)
        test_proposals = test_proposals.view(-1, test_proposals.shape[-1])  # (1,b,n,4)->(bn,4)

        # Reference weight calculate
        reference_weight = self.reference(train_feat, train_bb)
        test_feat = test_feat.mul(reference_weight.unsqueeze(2).unsqueeze(3))

        # Mask
        mask = self.masker(test_feat, test_proposals)

        return mask

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.feat_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)


@model_constructor
def ecoseg_resnet18(backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    # reference
    reference_net = esmodels.ReferenceNet(input_dim=(256,))

    # mask
    mask_net = esmodels.MaskNet(input_dim=(256,))

    net = ECOSeg(feature_extractor=backbone_net, reference=reference_net, masker=mask_net,
                 feat_layers=('layer3',), backbone_grad=False)

    return net
