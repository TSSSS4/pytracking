from . import BaseActor


class ECOSegActor(BaseActor):
    """ Actor for training the ReferenceNet in ECOSeg"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_mask',
             'test_mask', 'train_anno', 'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        mask_pred = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_proposals'])

        target = data['test_masks']     # (num_img, batch, num_proposals, w, h)
        target = target.view(-1, 1, target.shape[-2], target.shape[-1])
        # Compute loss
        loss = self.objective(mask_pred, target)

        # Return training stats
        stats = {'Loss': loss.item()}

        return loss, stats
