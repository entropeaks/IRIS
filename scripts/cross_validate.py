import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import wandb
import hydra
from omegaconf import OmegaConf

from src.data import CachedCollection, PKSampler, DataPreparator, make_transform
from src.model import SiameseDino
from src.train import train_and_evaluate
from src.config import Config
from src.utils import set_device

import pandas as pd

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

    random_state = 42
    
    processor = AutoImageProcessor.from_pretrained(config.model.backbone_name)
    backbone = AutoModel.from_pretrained(config.model.backbone_name, dtype=torch.float32)

    EXPERIMENT_DATA_PATH = Path(config.dataset.augmented_dataset_path)
    ORIGINAL_DATA_PATH = Path(config.dataset.original_dataset_path)

    data_preparator = DataPreparator(ORIGINAL_DATA_PATH, EXPERIMENT_DATA_PATH, random_state)
    n_folds = 4
    folds = data_preparator.get_k_folds(n_folds, config.eval.k_query)

    folds_metrics = []
    
    group_name = f"cv-{wandb.util.generate_id()}"

    for fold_id, fold in enumerate(folds):
        print(f"-------------------------------------- STARTING FOLD {fold_id+1} --------------------------------------")

        train_paths, train_labels = fold["train"]
        gallery_paths, gallery_labels = fold["gallery"]
        val_query_paths, val_query_labels = fold["val_query"]

        train_dataset = CachedCollection(train_paths, train_labels, transform=v2.Compose([train_transforms, make_transform()]))
        pksampler = PKSampler(train_dataset, config.train.sampler_P, config.train.sampler_K)
        train_dataloader = DataLoader(train_dataset, batch_sampler=pksampler)

        gallery_dataset = CachedCollection(gallery_paths, gallery_labels, transform=make_transform())
        gallery_dataloader = DataLoader(gallery_dataset, batch_size=32)

        val_dataset = CachedCollection(val_query_paths, val_query_labels, transform=make_transform())
        val_dataloader = DataLoader(val_dataset, batch_size=32)

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

        run = wandb.init(project=config.base.wandb_project_name,
                         entity=config.base.wandb_entity,
                         config=OmegaConf.to_container(config, resolve=True),
                         group=group_name,
                         name=f"fold-{fold_id+1}",
                         job_type="train_fold",
                         reinit=True)
        
        best_metrics = train_and_evaluate(run,
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
        
        print(f"-------------------------------------- END FOLD {fold_id} --------------------------------------")

        run.finish()
        folds_metrics.append(best_metrics)
    
    folds_metrics_df = pd.DataFrame(folds_metrics)
    avg_metrics = folds_metrics_df.mean()
    std_metrics = folds_metrics_df.std()
    print(f"========================== AVERAGE METRICS ON {n_folds} FOLDS ===========================")
    print(avg_metrics)
    print(f"==================== AVERAGE VARIANCE PERCENTAGES ON {n_folds} FOLDS ====================")
    print(std_metrics)

    wandb.init(project=config.base.wandb_project_name,
            entity=config.base.wandb_entity,
            config=OmegaConf.to_container(config, resolve=True),
            group=group_name,
            name=f"cv-summary",
            job_type="summary")
    
    wandb.log({**{f"avg_{k}": v for k, v in avg_metrics.to_dict().items()},
               **{f"std_{k}": v for k, v in std_metrics.to_dict().items()}}
               )
    
if __name__ == "__main__":
    main()