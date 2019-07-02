from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pytracking import complex, dcf, fourier, TensorList
from pytracking.libs.tensorlist import tensor_operation
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor
from pytracking.libs.optimization import GaussNewtonCG
from .optim import FilterOptim, FactorizedConvProblem
from pytracking.features import augmentation


class ECOSeg(BaseTracker):

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.features.initialize()
        self.features_initialized = True

    def initialize_mask(self, im):
        # feats[0]:(1,96,60,60) feats[1]:(1,256,15,15)
        feats = self.params.features.extract(im, self.pos.round(), self.target_scale, self.img_sample_sz)

        # bbox ((x0,y0,w,h))
        # bbox = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]])).unsqueeze(0).cuda()
        pos_sample = self.img_sample_sz / 2
        target_sz = self.base_target_sz[[1, 0]]
        bbox = torch.cat((pos_sample - (target_sz - 1) / 2, target_sz)).unsqueeze(0).cuda()

        # reference weight (1,c,w,h)
        self.reference_weight = self.params.reference_net(feats[1], bbox).unsqueeze(2).unsqueeze(3)
        del self.params.reference_net

    def initialize(self, image, state, *args, **kwargs):

        # Initialize some stuff
        self.frame_num = 1
        if not hasattr(self.params, 'device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Image size
        self.img_sz = torch.Tensor(image.shape[:2])

        # Initialize features
        self.initialize_features()

        # Chack if image is color
        self.params.features.set_is_color(image.shape[2] == 3)

        # Get feature specific params
        self.fparams = self.params.features.get_fparams('feature_params')

        # Get position and size
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Set search area
        self.target_scale = 1.0
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        if search_area > self.params.max_image_sample_size:
            self.target_scale =  math.sqrt(search_area / self.params.max_image_sample_size)
        elif search_area < self.params.min_image_sample_size:
            self.target_scale =  math.sqrt(search_area / self.params.min_image_sample_size)

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Use odd square search area and set sizes
        feat_max_stride = max(self.params.features.stride())
        self.img_sample_sz = torch.round(torch.sqrt(torch.prod(self.base_target_sz * self.params.search_area_scale))) * torch.ones(2)
        self.img_sample_sz += feat_max_stride - self.img_sample_sz % (2 * feat_max_stride)

        # Set other sizes (corresponds to ECO code)
        self.img_support_sz = self.img_sample_sz
        self.feature_sz = self.params.features.size(self.img_sample_sz)
        self.filter_sz = self.feature_sz + (self.feature_sz + 1) % 2
        self.output_sz = self.params.score_upsample_factor * self.img_support_sz    # Interpolated size of the output
        self.compressed_dim = self.fparams.attribute('compressed_dim')

        # Number of filters
        self.num_filters = len(self.filter_sz)

        # Get window function
        self.window = TensorList([dcf.hann2d(sz).to(self.params.device) for sz in self.feature_sz])

        # Get interpolation function
        self.interp_fs = TensorList([dcf.get_interp_fourier(sz, self.params.interpolation_method,
                                                self.params.interpolation_bicubic_a, self.params.interpolation_centering,
                                                self.params.interpolation_windowing, self.params.device) for sz in self.filter_sz])

        # Get regularization filter
        self.reg_filter = TensorList([dcf.get_reg_filter(self.img_support_sz, self.base_target_sz, fparams).to(self.params.device)
                                      for fparams in self.fparams])
        self.reg_energy = self.reg_filter.view(-1) @ self.reg_filter.view(-1)

        # Get label function
        output_sigma_factor = self.fparams.attribute('output_sigma_factor')
        sigma = (self.filter_sz / self.img_support_sz) * torch.sqrt(self.base_target_sz.prod()) * output_sigma_factor
        self.yf = TensorList([dcf.label_function(sz, sig).to(self.params.device) for sz, sig in zip(self.filter_sz, sigma)])

        # Optimization options
        self.params.precond_learning_rate = self.fparams.attribute('learning_rate')
        if self.params.CG_forgetting_rate is None or max(self.params.precond_learning_rate) >= 1:
            self.params.direction_forget_factor = 0
        else:
            self.params.direction_forget_factor = (1 - max(self.params.precond_learning_rate))**self.params.CG_forgetting_rate


        # Convert image
        im = numpy_to_torch(image)

        # MaskNet initialization
        self.initialize_mask(im)

        # Setup bounds
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        x = self.generate_init_samples(im)

        # Initialize projection matrix
        x_mat = TensorList([e.permute(1,0,2,3).reshape(e.shape[1], -1).clone() for e in x])
        x_mat -= x_mat.mean(dim=1, keepdim=True)
        cov_x = x_mat @ x_mat.t()
        self.projection_matrix = TensorList([torch.svd(C)[0][:,:cdim].clone() for C, cdim in zip(cov_x, self.compressed_dim)])

        # Transform to get the training sample
        train_xf = self.preprocess_sample(x)

        # Shift the samples back
        if 'shift' in self.params.augmentation:
            for xf in train_xf:
                if xf.shape[0] == 1:
                    continue
                for i, shift in enumerate(self.params.augmentation['shift']):
                    shift_samp = 2 * math.pi * torch.Tensor(shift) / self.img_support_sz
                    xf[1+i:2+i,...] = fourier.shift_fs(xf[1+i:2+i,...], shift=shift_samp)

        # Shift sample
        shift_samp = 2*math.pi * (self.pos - self.pos.round()) / (self.target_scale * self.img_support_sz)
        train_xf = fourier.shift_fs(train_xf, shift=shift_samp)

        # Initialize first-frame training samples
        num_init_samples = train_xf.size(0)
        self.init_sample_weights = TensorList([xf.new_ones(1) / xf.shape[0] for xf in train_xf])
        self.init_training_samples = train_xf.permute(2, 3, 0, 1, 4)


        # Sample counters and weights
        self.num_stored_samples = num_init_samples
        self.previous_replace_ind = [None]*len(self.num_stored_samples)
        self.sample_weights = TensorList([xf.new_zeros(self.params.sample_memory_size) for xf in train_xf])
        for sw, init_sw, num in zip(self.sample_weights, self.init_sample_weights, num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [xf.new_zeros(xf.shape[2], xf.shape[3], self.params.sample_memory_size, cdim, 2) for xf, cdim in zip(train_xf, self.compressed_dim)])

        # Initialize filter
        self.filter = TensorList(
            [xf.new_zeros(1, cdim, xf.shape[2], xf.shape[3], 2) for xf, cdim in zip(train_xf, self.compressed_dim)])

        # Do joint optimization
        self.joint_problem = FactorizedConvProblem(self.init_training_samples, self.yf, self.reg_filter, self.projection_matrix, self.params, self.init_sample_weights)
        joint_var = self.filter.concat(self.projection_matrix)
        self.joint_optimizer = GaussNewtonCG(self.joint_problem, joint_var, debug=(self.params.debug>=3))

        if self.params.update_projection_matrix:
            self.joint_optimizer.run(self.params.init_CG_iter // self.params.init_GN_iter, self.params.init_GN_iter)

        # Re-project samples with the new projection matrix
        compressed_samples = complex.mtimes(self.init_training_samples, self.projection_matrix)
        for train_samp, init_samp in zip(self.training_samples, compressed_samples):
            train_samp[:,:,:init_samp.shape[2],:,:] = init_samp

        # Initialize optimizer
        self.filter_optimizer = FilterOptim(self.params, self.reg_energy)
        self.filter_optimizer.register(self.filter, self.training_samples, self.yf, self.sample_weights, self.reg_filter)
        self.filter_optimizer.sample_energy = self.joint_problem.sample_energy
        self.filter_optimizer.residuals = self.joint_optimizer.residuals.clone()

        if not self.params.update_projection_matrix:
            self.filter_optimizer.run(self.params.init_CG_iter)

        # Post optimization
        self.filter_optimizer.run(self.params.post_init_CG_iter)

        self.symmetrize_filter()

    def track(self, image):

        self.frame_num += 1

        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Get sample
        sample_pos = self.pos.round()
        sample_scales = self.target_scale * self.params.scale_factors
        feats = self.extract_sample(im, sample_pos, sample_scales, self.img_sample_sz)
        test_xf = self.preprocess_sample(self.project_sample(feats))

        # Stage 1
        # Compute scores
        sf = self.apply_filter(test_xf)
        translation_vec, scale_ind, s = self.localize_target(sf)
        scale_change_factor = self.params.scale_factors[scale_ind]

        # Update position and scale
        # self.update_state(sample_pos + translation_vec, self.target_scale * scale_change_factor)

        # Stage 2
        # Mask and Update position and scale
        translation_vec, mask = self.mask_locate(feats, translation_vec)

        # Update position and scale
        self.update_state(sample_pos + translation_vec, self.target_scale * scale_change_factor)

        if self.params.debug >= 2:
            show_tensor(s[scale_ind,...], 5)
        if self.params.debug >= 3:
            for i, hf in enumerate(self.filter):
                show_tensor(fourier.sample_fs(hf).abs().mean(1), 6+i)


        # ------- UPDATE ------- #

        # Get train sample
        train_xf = TensorList([xf[scale_ind:scale_ind+1, ...] for xf in test_xf])

        # Shift the sample
        shift_samp = 2*math.pi * (self.pos - sample_pos) / (sample_scales[scale_ind] * self.img_support_sz)
        train_xf = fourier.shift_fs(train_xf, shift=shift_samp)

        # Update memory
        self.update_memory(train_xf)

        # Train filter
        if self.frame_num % self.params.train_skipping == 1:
            self.filter_optimizer.run(self.params.CG_iter, train_xf)
            self.symmetrize_filter()

        # Return new state
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        return {'bbox': new_state.tolist(), 'mask': mask}

    def apply_filter(self, sample_xf: TensorList) -> torch.Tensor:
        return complex.mult(self.filter, sample_xf).sum(1, keepdim=True)

    def localize_target(self, sf: TensorList):
        if self.params.score_fusion_strategy == 'sum':
            scores = fourier.sample_fs(fourier.sum_fs(sf), self.output_sz)
        elif self.params.score_fusion_strategy == 'weightedsum':
            weight = self.fparams.attribute('translation_weight')
            scores = fourier.sample_fs(fourier.sum_fs(weight * sf), self.output_sz)
        elif self.params.score_fusion_strategy == 'transcale':
            alpha = self.fparams.attribute('scale_weight')
            beta = self.fparams.attribute('translation_weight')
            sample_sz = torch.round(self.output_sz.view(1,-1) * self.params.scale_factors.view(-1,1))
            scores = 0
            for sfe, a, b in zip(sf, alpha, beta):
                sfe = fourier.shift_fs(sfe, math.pi*torch.ones(2))
                scores_scales = []
                for sind, sz in enumerate(sample_sz):
                    pd = (self.output_sz-sz)/2
                    scores_scales.append(F.pad(fourier.sample_fs(sfe[sind:sind+1,...], sz),
                                        (math.floor(pd[1].item()), math.ceil(pd[1].item()),
                                         math.floor(pd[0].item()), math.ceil(pd[0].item()))))
                scores_cat = torch.cat(scores_scales)
                scores = scores + (b - a) * scores_cat.mean(dim=0, keepdim=True) + a * scores_cat
        else:
            raise ValueError('Unknown score fusion strategy.')

        # Get maximum
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp.float().cpu()

        # Convert to displacements in the base scale
        if self.params.score_fusion_strategy in ['sum', 'weightedsum']:
            disp = (max_disp + self.output_sz / 2) % self.output_sz - self.output_sz / 2
        elif self.params.score_fusion_strategy == 'transcale':
            disp = max_disp - self.output_sz / 2

        # Compute translation vector and scale change factor
        translation_vec = disp[scale_ind, ...].view(-1) * (self.img_support_sz / self.output_sz) * self.target_scale
        if self.params.score_fusion_strategy in ['sum', 'weightedsum']:
            translation_vec *= self.params.scale_factors[scale_ind]

        return translation_vec, scale_ind, scores

    def extract_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        return self.params.features.extract(im, pos, scales, sz)

    def extract_fourier_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor) -> TensorList:
        x = self.extract_sample(im, pos, scales, sz)
        return self.preprocess_sample(self.project_sample(x))

    def preprocess_sample(self, x: TensorList) -> TensorList:
        x *= self.window
        sample_xf = fourier.cfft2(x)
        return TensorList([dcf.interpolate_dft(xf, bf) for xf, bf in zip(sample_xf, self.interp_fs)])

    def project_sample(self, x: TensorList):
        @tensor_operation
        def _project_sample(x: torch.Tensor, P: torch.Tensor):
            if P is None:
                return x
            return torch.matmul(x.permute(2, 3, 0, 1), P).permute(2, 3, 0, 1)

        return _project_sample(x, self.projection_matrix)

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        # Do data augmentation
        transforms = [augmentation.Identity()]
        if 'shift' in self.params.augmentation:
            transforms.extend([augmentation.Translation(shift) for shift in self.params.augmentation['shift']])
        if 'fliplr' in self.params.augmentation and self.params.augmentation['fliplr']:
            transforms.append(augmentation.FlipHorizontal())
        if 'rotate' in self.params.augmentation:
            transforms.extend([augmentation.Rotate(angle) for angle in self.params.augmentation['rotate']])
        if 'blur' in self.params.augmentation:
            transforms.extend([augmentation.Blur(sigma) for sigma in self.params.augmentation['blur']])

        init_samples = self.params.features.extract_transformed(im, self.pos.round(), self.target_scale, self.img_sample_sz, transforms)

        # Remove augmented samples for those that shall not have
        for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
            if not use_aug:
                init_samples[i] = init_samples[i][0:1, ...]

        if 'dropout' in self.params.augmentation:
            num, prob = self.params.augmentation['dropout']
            for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
                if use_aug:
                    init_samples[i] = torch.cat([init_samples[i], F.dropout2d(init_samples[i][0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        return init_samples

    def update_memory(self, sample_xf: TensorList):
        # Update weights and get index to replace
        replace_ind = self.update_sample_weights()
        for train_samp, xf, ind in zip(self.training_samples, sample_xf, replace_ind):
            train_samp[:,:,ind:ind+1,:,:] = xf.permute(2, 3, 0, 1, 4)

    def update_sample_weights(self):
        replace_ind = []
        for sw, prev_ind, num_samp, fparams in zip(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.fparams):
            if num_samp == 0 or fparams.learning_rate == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                _, r_ind = torch.min(sw, 0)
                r_ind = r_ind.item()

                # Update weights
                if prev_ind is None:
                    sw /= 1 - fparams.learning_rate
                    sw[r_ind] = fparams.learning_rate
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - fparams.learning_rate)

            sw /= sw.sum()
            replace_ind.append(r_ind)

        self.previous_replace_ind = replace_ind.copy()
        self.num_stored_samples += 1
        return replace_ind

    def update_state(self, new_pos, new_scale):
        # Update scale
        self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
        self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = 0.2
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def symmetrize_filter(self):
        for hf in self.filter:
            hf[:,:,:,0,:] /= 2
            hf[:,:,:,0,:] += complex.conj(hf[:,:,:,0,:].flip((2,)))

    def mask_locate(self, feats, translation_vec):

        # (cy,cx,h,w) -> (cx,cy,w,h)
        translation_vec = translation_vec[[1, 0]]
        pos = self.pos[[1, 0]]
        target_sz = self.target_sz[[1, 0]]
        pos_sample = self.img_sample_sz / 2 + translation_vec

        # roi feat
        mask_feat = feats[1].mul(self.reference_weight)

        # mask (14,14)
        bbox_sample = torch.cat((pos_sample - (self.target_sz - 1) / 2, self.target_sz)).cuda()  # (cx,cy,w,h)
        mask = self.params.mask_net(mask_feat, bbox_sample.unsqueeze(0)).squeeze()

        # mask on image (h,w)
        bbox_img = torch.cat((pos + translation_vec, target_sz)).cuda()  # (cx,cy,w,h)
        im_mask = self.masker(mask, bbox_img, padding=1, thresh=0.5, im_sz=self.img_sz)

        # bbox calculate
        mask = im_mask.detach().cpu().numpy().squeeze()
        # box = self.mask_to_box(mask, mode='polygon')

        # locate

        # if self.params.visualize:
        #     self.plot(sample, mask, box)

        return translation_vec[[1, 0]], mask

    def plot(self, sample, mask, box):
        # bbox plot
        sample = sample.permute(2, 3, 1, 0).squeeze().numpy().astype(np.uint8)
        res = cv2.polylines(sample, [box], True, (0, 255, 0))
        contours, hierarchy = self.find_contours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # contour plot
        res = cv2.drawContours(res, contours, -1, (0, 255, 0), 3)
        plt.ion()
        plt.imshow(res)
        plt.pause(0.1)

    def visualize(self, image, state):
        self.ax.cla()
        self.ax.imshow(image)
        bbox = state['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(rect)

        if state.__contains__('mask'):
            plt.contour(state['mask'], colors='red', linewidths=1.0)

        if hasattr(self, 'gt_state') and False:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g',
                                     facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        plt.draw()
        plt.pause(0.001)

        if self.pause_mode:
            plt.waitforbuttonpress()

    @staticmethod
    def masker(mask, bbox, padding, thresh, im_sz):
        im_w = int(im_sz[1])
        im_h = int(im_sz[0])

        # expand mask
        M = mask.shape[-1]
        pad2 = 2 * padding
        scale = float(M + pad2) / M
        padded_mask = mask.new_zeros((M + pad2, M + pad2))
        padded_mask[padding:-padding, padding:-padding] = mask
        mask = padded_mask

        # # expand box
        # w_half = (bbox[2] - bbox[0]) * .5
        # h_half = (bbox[3] - bbox[1]) * .5
        # x_c = (bbox[2] + bbox[0]) * .5
        # y_c = (bbox[3] + bbox[1]) * .5
        #
        # w_half *= scale
        # h_half *= scale
        #
        # bbox_exp = torch.zeros_like(bbox)
        # bbox_exp[0] = x_c - w_half
        # bbox_exp[2] = x_c + w_half
        # bbox_exp[1] = y_c - h_half
        # bbox_exp[3] = y_c + h_half
        # bbox = bbox_exp.to(dtype=torch.int32)
        #
        # #
        # TO_REMOVE = 1
        # w = int(bbox[2] - bbox[0] + TO_REMOVE)
        # h = int(bbox[3] - bbox[1] + TO_REMOVE)
        # w = max(w, 1)
        # h = max(h, 1)

        # expand box
        w_half = bbox[2] * .5
        h_half = bbox[3] * .5
        x_c = bbox[0]
        y_c = bbox[1]

        w_half *= scale
        h_half *= scale

        bbox_exp = torch.zeros_like(bbox)
        bbox_exp[0] = x_c - w_half
        bbox_exp[2] = x_c + w_half
        bbox_exp[1] = y_c - h_half
        bbox_exp[3] = y_c + h_half
        bbox = bbox_exp.to(dtype=torch.int32)

        #
        TO_REMOVE = 1
        w = int(bbox[2] - bbox[0] + TO_REMOVE)
        h = int(bbox[3] - bbox[1] + TO_REMOVE)
        w = max(w, 1)
        h = max(h, 1)

        # Resize mask to bbox size
        mask = mask.expand((1, 1, -1, -1))
        mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        mask = mask.squeeze()
        # mask threshold
        if thresh >= 0:
            mask = mask > thresh
        else:
            # for visualization and debugging, we also
            # allow it to return an unmodified mask
            mask = (mask * 255).to(torch.uint8)

        # paste mask to sample image
        im_mask = torch.zeros((im_h, im_w), dtype=torch.uint8)
        x_0 = max(bbox[0], 0)
        x_1 = min(bbox[2] + 1, im_w)
        y_0 = max(bbox[1], 0)
        y_1 = min(bbox[3] + 1, im_h)

        im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - bbox[1]): (y_1 - bbox[1]),
                                         (x_0 - bbox[0]): (x_1 - bbox[0])]
        return im_mask

    @staticmethod
    def mask_to_box(mask, mode):
        """
        Calculate bounding box according to binary mask.
        :param mask: binary mask (w,h)
        :param mode: rectangle or polygon
        :return: 4 corners of bounding box ndarray([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
        """
        idx = np.where(mask == 1)               # (2,n)
        if mode == 'polygon':
            idx = np.array(idx).transpose(1, 0)     # (n,2)
            rect = cv2.minAreaRect(idx)             # tuple((cx,cy), (w,h), angle)
            box = cv2.boxPoints(rect)               # (4,2)
            box = np.int0(box)
            box = np.array([p[-1::-1] for p in box])
        elif mode == 'rectangle':
            box = np.array([[idx[0].min(), idx[1].min()],
                            [idx[0].max(), idx[1].min()],
                            [idx[0].max(), idx[1].max()],
                            [idx[0].min(), idx[1].max()]])
        return box

    @staticmethod
    def find_contours(*args, **kwargs):
        """
        Wraps cv2.findContours to maintain compatiblity between versions
        3 and 4

        Returns:
            contours, hierarchy
        """
        if cv2.__version__.startswith('4'):
            contours, hierarchy = cv2.findContours(*args, **kwargs)
        elif cv2.__version__.startswith('3'):
            _, contours, hierarchy = cv2.findContours(*args, **kwargs)
        else:
            raise AssertionError(
                'cv2 must be either version 3 or 4 to call this method')

        return contours, hierarchy

