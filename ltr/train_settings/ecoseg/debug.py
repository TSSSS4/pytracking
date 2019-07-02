import torch.nn as nn
import torch.optim as optim
import torchvision.transforms

from ltr.dataset import DAVIS, VOT2016, YouTubeVOS_2018, dataset_merge_split
from ltr.data import processing, sampler, LTRLoader
import ltr.models.ecoseg.ecoseg as ecoseg_models
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as dltransforms
from ltr.solver import make_optimizer


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.description = 'ECOSeg debug.'
    settings.print_interval = 1                                 # How often to print loss and other info
    settings.batch_size = 32                                    # Batch size
    settings.iter_per_epoch = 1000
    settings.num_workers = 8                                    # Number of workers for image loading
    settings.normalize_mean = [0.485, 0.456, 0.406]             # Normalize mean (default pytorch ImageNet values)
    settings.normalize_std = [0.229, 0.224, 0.225]              # Normalize std (default pytorch ImageNet values)
    settings.search_area_factor = 5.0                           # Image patch size relative to target size
    settings.feature_sz = 18                                    # Size of feature map
    settings.output_sz = settings.feature_sz * 16               # Size of input image patches
    settings.mask_sz = 7 * 2

    # Settings for the image sample and proposal generation
    settings.center_jitter_factor = {'train': 0, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.5}
    settings.proposal_params = {'min_iou': 0.5, 'boxes_per_frame': 16, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}

    # Setting for training
    settings.base_lr = 0.1
    settings.weight_decay = 0.0001
    settings.base_lr_factor = 2
    settings.weight_decay_bias = 0
    settings.momentum = 0.9

    # Train datasets
    train_ratio = 1
    davis_train = DAVIS(mode='train', train_ratio=train_ratio)
    vot2016_train = VOT2016(mode='train', train_ratio=0)
    youtubevos_train = YouTubeVOS_2018(mode='train', train_ratio=train_ratio)

    # Validation datasets
    davis_val = DAVIS(mode='val', train_ratio=train_ratio)
    vot2016_val = VOT2016(mode='val', train_ratio=0)
    youtubevos_val = YouTubeVOS_2018(mode='val', train_ratio=train_ratio)

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = dltransforms.ToGrayscale(probability=0.05)

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
                                                      torchvision.transforms.Normalize(mean=settings.normalize_mean,
                                                                                       std=settings.normalize_std)])

    # The augmentation transform applied to the validation set (individually to each image in the pair)
    transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=settings.normalize_mean,
                                                                                     std=settings.normalize_std)])

    # Data processing to do on the training pairs
    data_processing_train = processing.ECOSegProcessing(search_area_factor=settings.search_area_factor,
                                                        output_sz=settings.output_sz,
                                                        mask_size=settings.mask_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        proposal_params=settings.proposal_params,
                                                        transform=transform_train,
                                                        joint_transform=transform_joint)

    data_processing_val = processing.ECOSegProcessing(search_area_factor=settings.search_area_factor,
                                                        output_sz=settings.output_sz,
                                                        mask_size=settings.mask_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        proposal_params=settings.proposal_params,
                                                        transform=transform_val,
                                                        joint_transform=transform_joint)

    # The sampler for training
    dataset_train = sampler.ECOSegSampler([davis_train, youtubevos_train], [1,1], samples_per_epoch=1000*settings.batch_size,
                                          max_gap=50, processing=data_processing_train)

    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # The sampler for validation
    dataset_val = sampler.ECOSegSampler([vot2016_val], [1], samples_per_epoch=100 * settings.batch_size,
                                        max_gap=50, processing=data_processing_val)

    # The loader for validation
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=1, stack_dim=1)

    # Create network
    net = ecoseg_models.ecoseg_resnet18(backbone_pretrained=True)

    # Set objective
    objective = nn.BCEWithLogitsLoss()

    # Create actor, which wraps network and objective
    actor = actors.ECOSegActor(net=net, objective=objective)

    # Optimizer
    optimizer = make_optimizer(settings, model=actor.net)

    # Learning rate scheduler
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(100, load_latest=True, fail_safe=False)
