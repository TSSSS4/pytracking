import torch


def dataset_merge_split(datasets, val_ratio=0.1, val_max=500):
    train = []
    val = []
    for dataset in datasets:
        seq_num = len(dataset.sequence_list)

    return train, val