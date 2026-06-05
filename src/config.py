from dataclasses import dataclass, field

@dataclass
class ImgResize:
    height: int=448
    width: int=448

@dataclass
class ModelConfig:
    backbone_name: str="facebook/dinov3-vitb16-pretrain-lvd1689m"
    hidden_dim: int=0
    output_dim: int=128
    normalize: bool=True
    dropout: float=0.0


@dataclass
class TrainConfig:
    epochs: int=20
    head_lr: float=0.00005
    backbone_lr: float=0.000005
    weight_decay: float=0.0001
    margin: float=0.5
    sampler_P: int=8
    sampler_K: int=4
    resize: ImgResize = field(default_factory=ImgResize)


@dataclass
class EvalConfig:
    recall_k: list[int]=field(default_factory=lambda: [1, 3, 10])
    k_query: int=1
    val_size: float=0.2
    test_size: float=0.1


@dataclass
class DatasetConfig:
    original_dataset_path: str="data/original_data"
    augmented_dataset_path: str="data/augmented_data16"


@dataclass
class BaseConfig:
    wandb_project_name: str=None
    wandb_entity: str=None
    device: str="cpu"
    model_checkpoints_path: str="model_checkpoints"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    base: BaseConfig = field(default_factory=BaseConfig)


def load_config(path: str = "config/config.yaml") -> Config:
    import yaml
    import dacite

    with open(path) as f:
        raw = yaml.safe_load(f)

    return dacite.from_dict(Config, raw, config=dacite.Config(strict=True))