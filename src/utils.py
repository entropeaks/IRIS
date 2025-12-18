import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from abc import ABC, abstractmethod
from typing import Tuple, Generator, List
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
from transformers.image_utils import load_image
from PIL import Image, ImageOps, ImageFilter
from pathlib import Path
import random
import pandas as pd
import yaml
import numpy as np

from src.data import DataPreparator, CachedCollection, make_transform
from src.models.base import BaseModel, DeepModel
from src.eval import Metric, Recall


from src.data import RANDOM_SEED

def set_device(device: str) -> torch.device:
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps:0":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        return torch.device("cpu")
    

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class Transform(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_transformed(image: Image) -> Image:
        pass


class CropTransform(Transform):
    
    def __init__(self, device: str, objects: List[str], model_id: str="IDEA-Research/grounding-dino-tiny"):
        self.device = set_device(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.objects = objects

    def get_cropbox(self, image: Image):
        inputs = self.processor(images=image, text=self.objects, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=0.4,
            text_threshold=0.3,
            target_sizes=[(image.height, image.width)]
        )
        result = results[0]

        if result["boxes"].shape[0] == 0:
            return []
        
        highest_score_box = result["boxes"][0]
        box = [round(x, 2) for x in highest_score_box.tolist()]

        return box

    def get_transformed(self, image):
        box = self.get_cropbox(image)
        return image.crop(box)


class SegmentationModel:

    def __init__(self, device: str, model_id: str="facebook/sam-vit-base"):
        self.device = set_device(device)
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device)

    def get_mask(self, image: Image, input_boxes: List[List[int]]) -> np.ndarray:
        inputs = self.processor(image, input_boxes=[input_boxes], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )

        best_mask = np.argmax(outputs.iou_scores.cpu())
        masks_tensors = masks[0].squeeze()

        return masks_tensors[best_mask].numpy()


class BackgroundRandomSampler:

    def __init__(self, bg_file_path: str):
        self.bg_file_path = bg_file_path
        self.df = pd.read_csv(self.bg_file_path, sep='\t', header=0)

    def sample(self):
        idx = random.randint(0, len(self.df)-1)
        url = self.df["photo_image_url"][idx]
        return load_image(url)


class BackgroundTransform(Transform):

    def __init__(self, device: str,
                 crop_model: CropTransform,
                 segmentation_model: SegmentationModel,
                 background_sampler: BackgroundRandomSampler,
                 instance_transforms: v2.Compose,
                 background_blur_max_level: int=15,
                 blur_probability: float=0.5
                 ):
        self.device = set_device(device)
        self.crop_model = crop_model
        self.segmentation_model = segmentation_model
        self.background_sampler = background_sampler
        self.instance_transforms = instance_transforms
        self.background_blur_max_level = background_blur_max_level
        self.blur_probability = blur_probability

    def blur_bg(self, bg: Image) -> Image:
        if random.random() > self.blur_probability:
            return bg
        
        bg_blur = random.randint(0, self.background_blur_max_level)
        blurred_bg = bg.filter(ImageFilter.GaussianBlur(bg_blur))
    
        return blurred_bg

    def get_transformed(self, image: Image):
        image = ImageOps.exif_transpose(image)
        image = self.instance_transforms(image)
        boxes = self.crop_model.get_cropbox(image)
        if len(boxes) == 0 or type(boxes[0]) != list:
            boxes = [boxes]
        mask = self.segmentation_model.get_mask(image, boxes)
        bg = self.background_sampler.sample().resize(image.size)
        bg = self.blur_bg(bg)

        return Image.composite(image, bg, Image.fromarray(mask))
        

class Browser:

    def __init__(self, path: Path):
        self.path = path
        self.samples_num = self._get_samples_num()

    def _get_samples_num(self) -> int:
        count = 0
        for _, _ in self._iterate_on_files():
            count += 1
        return count

    def extract_paths_and_labels(self) -> Tuple[List[str], List[int]]:
        paths = []
        labels = []
        for path, label in self._iterate_on_files():
            paths.append(path.as_posix())
            labels.append(int(label))

        return paths, labels
    
    def _iterate_on_files(self) -> Generator[Tuple[Path, str], None, None]:
        for class_dir in self._iterate_on_classes():
            for path in class_dir.iterdir():
                yield path, class_dir.name
        
    def _iterate_on_classes(self) -> Generator[Path, None, None]:
        for class_dir in self.path.iterdir():
            if class_dir.is_dir():
                yield class_dir

    def _construct_filename(self, filename: str, n: int) -> str:
        base_name, extension = filename.split('.')
        return base_name + f"_{str(n)}." + extension

    def generate_transformed_dataset(self,
                                     destination_path: str,
                                     transform: Transform,
                                     multiplier: int=1
                                     ) -> None:
        destinationPath = Path(destination_path)
        destinationPath.mkdir(exist_ok=True)
        for class_dir in self._iterate_on_classes():
            label = class_dir.name
            destinationPath.joinpath(label).mkdir(exist_ok=True)
        
        for path, label in tqdm(self._iterate_on_files(), total=self.samples_num):
            src_img = Image.open(path.as_posix())
            for i in range(multiplier):
                filename = self._construct_filename(path.name, i)
                if destinationPath.joinpath(label).joinpath(filename).exists():
                    continue
                new_img = transform.get_transformed(src_img)
                new_img.save(destinationPath.joinpath(label).joinpath(filename))

    def sample_k_per_class(self, k: int, random_state: int=RANDOM_SEED) -> Tuple[List, List]:
        paths = []
        labels = []
        for class_dir in self._iterate_on_classes():
            class_paths = [path.as_posix() for path in class_dir.iterdir()]
            class_path = random.sample(class_paths, k)
            paths.extend(class_path)
            labels.extend([int(class_dir.name)]*len(class_path))

        return paths, labels
    
    def sample_leave_k_out(self, k: int):

        gallery_paths, gallery_labels = [], []
        query_paths, query_labels = [], []

        for class_dir in self._iterate_on_classes():
            class_paths = [p.as_posix() for p in class_dir.iterdir()]
            n = len(class_paths)

            if n <= k:
                continue  # ou autre politique explicite

            n_gallery = n - k
            gallery_indices = set(random.sample(range(n), n_gallery))

            to_gallery = [class_paths[i] for i in sorted(gallery_indices)]
            left_out = [p for i, p in enumerate(class_paths) if i not in gallery_indices]

            label = int(class_dir.name)

            gallery_paths.extend(to_gallery)
            gallery_labels.extend([label] * len(to_gallery))
            query_paths.extend(left_out)
            query_labels.extend([label] * len(left_out))

        return gallery_paths, gallery_labels, query_paths, query_labels
                

class ModelReport:

    def __init__(self, models: dict[str, BaseModel], fit_model: bool=False):
        self.models = models
        self.fit_model = fit_model
        self.col = ["training time",
                    "training energy",
                    "inference time",
                    "inference energy",
                    "metrics",
                    "total test time",
                    "total test energy"
                    ]

    def generate_report(self, dataset_path: str, export_path: str, metric: Metric=Recall()) -> pd.DataFrame:
        data = []

        EXPERIMENT_DATA_PATH = Path("../data/augmented_data16")
        ORIGINAL_DATA_PATH = Path(dataset_path)

        data_preparator = DataPreparator(ORIGINAL_DATA_PATH, EXPERIMENT_DATA_PATH, 42)

        training_size = 0.7
        data_splits = data_preparator.train_val_test_split(train_ratio=0, val_ratio=1, n_query=1)
        train_paths, train_labels = data_splits["train"]
        gallery_paths, gallery_labels = data_splits["gallery"]
        val_query_paths, val_query_labels = data_splits["val_query"]
        test_query_paths, test_query_labels = data_splits["test_query"]

        browser = Browser(ORIGINAL_DATA_PATH)
        gallery_paths, gallery_labels, query_paths, query_labels = browser.sample_leave_k_out(1)

        RESIZE_SIZE = 224
        
        print("Loading dataloaders...")
        gallery_dataset = CachedCollection(gallery_paths, gallery_labels, transform=make_transform(RESIZE_SIZE))
        gallery_dataloader = DataLoader(gallery_dataset, batch_size=32)

        val_dataset = CachedCollection(query_paths, query_labels, transform=make_transform(RESIZE_SIZE))
        val_dataloader = DataLoader(val_dataset, batch_size=32)

        """ if self.fit_model:
            for model in self.models:
                if self.models[model].issubclass(DeepModel): """
        
        # TODO: add size of index in the report

        for model in self.models:
            model_data = {}
            estimator = self.models[model]
            metrics = estimator.evaluate(gallery_dataloader, val_dataloader, metric)
            model_data["evaluation time"] = estimator.time
            estimator.find_nearest_neighbors(val_query_paths[0], 3)
            model_data["inference time"] = estimator.time
            model_data.update(metrics)

            data.append(model_data)

        df = pd.DataFrame(data, index=list(self.models.keys()))

        return df