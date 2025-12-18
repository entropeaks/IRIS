from pathlib import Path
import wandb
import hydra
from omegaconf import OmegaConf

from src.model import SiameseDino
from src.data import PKSampler, CachedCollection, LazyLoadCollection,  make_transform, DataPreparator
from src.train import train_and_evaluate, evaluate
from src.utils import set_device
from src.config import Config

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
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
    siamese_model = SiameseDino(backbone, hidden_dim=config.model.hidden_dim, output_dim=config.model.output_dim, normalize=config.model.normalize, dropout=config.model.dropout)
    for param in siamese_model.backbone.parameters():
        param.requires_grad = False
    for param in siamese_model.backbone.layer[-2:].parameters():
        param.requires_grad = True
    _ = siamese_model.to(device)

    optim_params = [
        {"params": siamese_model.projection_head.parameters(), "lr": config.train.head_lr},
        {"params": siamese_model.backbone.layer[-2:].parameters(), "lr": config.train.backbone_lr}
    ]

    run = wandb.init(project=config.base.wandb_project_name, entity=config.base.wandb_entity, config=OmegaConf.to_container(config, resolve=True))
    
    train_and_evaluate(run,
                    siamese_model,
                    processor,
                    train_dataloader,
                    gallery_dataloader,
                    val_dataloader,
                    optim_params,
                    config.train.epochs,
                    config.train.weight_decay,
                    config.train.margin,
                    config.eval.recall_k
                    )
    
    final_scores = evaluate(siamese_model,
                            processor,
                            gallery_dataloader,
                            test_dataloader,
                            config.eval.recall_k)
    
    run.log({f"test_{k}": v for k, v in final_scores.items()})
    run.finish()


if __name__ == "__main__":
    print("Starting experiment")
    main()