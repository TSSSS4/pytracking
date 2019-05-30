import torch


def make_optimizer(config, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = config.base_lr
        weight_decay = config.weight_decay
        if "bias" in key:
            lr = config.base_lr * config.base_lr_factor
            weight_decay = config.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=config.momentum)
    return optimizer

