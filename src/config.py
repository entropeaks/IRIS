from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    backbone_name: str="facebook/dinov3-vitb16-pretrain-lvd1689m"
    hidden_dim: int=0
    output_dim: int=128
    normalize: bool=True
    dropout: int=0


@dataclass
class TrainConfig:
    epochs: int=20
    head_lr: float=0.00005
    backbone_lr: float=0.000005
    weight_decay: float=0.0001
    margin: float=0.5
    sampler_P: int=8
    sampler_K: int=4


@dataclass
class EvalConfig:
    recall_k: list[int]=field(default_factory=lambda: [1, 3, 10])
    k_query: int=1
    val_size: float=0.2
    test_size: float=0.1


@dataclass
class DatasetConfig:
    original_dataset_path: str="data/augmented_data16"
    augmented_dataset_path: str="data/original_data"


@dataclass
class BaseConfig:
    wandb_project_name: str=""
    wandb_entity: str=""
    device: str="cpu"
    model_checkpoints_path: str="model_checkpoints"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    base: BaseConfig = field(default_factory=BaseConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)