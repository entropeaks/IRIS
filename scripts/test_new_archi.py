from pathlib import Path
import wandb
import hydra
from omegaconf import OmegaConf

from transformers import AutoModel, AutoImageProcessor
from src.models.deepmodels import SiameseDino
from src.eval import Recall
from src.config import Config
from src.utils import set_device
from src.data import CachedCollection, LazyLoadCollection, DataPreparator, PKSampler, make_transform

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2


@hydra.main(config_path="../config", config_name="base_config", version_base=None)
def main(config: Config):

    train_transforms = v2.Compose([                           
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0), 
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 0.9)),
        v2.RandomRotation(degrees=15),
    ])

    device = set_device(config.base.device)

    RANDOM_STATE = 42
    RESIZE_SIZE = 224
    
    processor = AutoImageProcessor.from_pretrained(config.model.backbone_name)
    backbone = AutoModel.from_pretrained(config.model.backbone_name, dtype=torch.float32)

    metric = Recall(config.eval.recall_k)
    EXPERIMENT_DATA_PATH = Path(config.dataset.augmented_dataset_path)
    ORIGINAL_DATA_PATH = Path(config.dataset.original_dataset_path)
    
    data_preparator = DataPreparator(ORIGINAL_DATA_PATH, EXPERIMENT_DATA_PATH, RANDOM_STATE)

    training_size = 1 - (config.eval.val_size + config.eval.test_size)
    data_splits = data_preparator.train_val_test_split(training_size, config.eval.val_size, config.eval.k_query)
    train_paths, train_labels = data_splits["train"]
    gallery_paths, gallery_labels = data_splits["gallery"]
    val_query_paths, val_query_labels = data_splits["val_query"]
    test_query_paths, test_query_labels = data_splits["test_query"]

    train_dataset = CachedCollection(train_paths, train_labels, transform=v2.Compose([train_transforms, make_transform(RESIZE_SIZE)]))
    pksampler = PKSampler(train_dataset, config.train.sampler_P, config.train.sampler_K)
    train_dataloader = DataLoader(train_dataset, batch_sampler=pksampler)

    gallery_dataset = CachedCollection(gallery_paths, gallery_labels, transform=make_transform(RESIZE_SIZE))
    gallery_dataloader = DataLoader(gallery_dataset, batch_size=32)

    val_dataset = CachedCollection(val_query_paths, val_query_labels, transform=make_transform(RESIZE_SIZE))
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    test_dataset = CachedCollection(test_query_paths, test_query_labels, transform=make_transform(RESIZE_SIZE))
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    # --------------------------------------------------------------------------
    siamese_model = SiameseDino(backbone,
                                processor,
                                config,
                                )
    for param in siamese_model.backbone.parameters():
        param.requires_grad = False
    for param in siamese_model.backbone.layer[-2:].parameters():
        param.requires_grad = True
        
    optim_params = [
        {"params": siamese_model.projection_head.parameters(), "lr": config.train.head_lr},
        {"params": siamese_model.backbone.layer[-2:].parameters(), "lr": config.train.backbone_lr}
    ]

    optimizer = torch.optim.Adam(optim_params, weight_decay=config.train.weight_decay)
    siamese_model.set_optimizer(optimizer)
    _ = siamese_model.to(device)

    metrics = siamese_model.fit_and_evaluate(train_dataloader, gallery_dataloader, val_dataloader, metric)

    print(metrics)


if __name__ == "__main__":
    main()