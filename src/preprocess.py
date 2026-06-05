from abc import ABC, abstractmethod
from typing import List, Tuple, Generator
import random
from tqdm import tqdm
from pathlib import Path
import logging

from PIL import Image, ImageOps, ImageFilter
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamProcessor, SamModel
from transformers.image_utils import load_image
import torch
from torchvision.transforms import v2
from ultralytics import YOLO
import cv2

import numpy as np
import pandas as pd

from src.utils import set_device


class Transform(ABC):

    def __init__(self):
        pass

    def __call__(self, img_rgb):
        return self.get_transformed(img_rgb)

    @abstractmethod
    def get_transformed(self, img_rgb: Image) -> Image:
        pass

class PolarTransform(Transform):

    def __init__(self, detect_center: bool=True):
        self._detect_center = detect_center

    def get_transformed(self, img_rgb):
        img_arr = np.array(img_rgb)
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        if self._detect_center:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.GaussianBlur(img_gray, (9, 9), 2)
            circles = cv2.HoughCircles(
                img_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w),
                param1=80, param2=40,
                minRadius=int(0.30 * min(h, w)),
                maxRadius=int(0.50 * min(h, w))
            )
            if circles is not None:
                c = circles[0, 0]
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            else:
                cx, cy, r = w // 2, h // 2, int(0.45 * min(h, w))
        else:
            cx, cy, r = w // 2, h // 2, int(0.45 * min(h, w))

        radius = int(r * 0.95)

        polar = cv2.warpPolar(img_arr, (w, h), (cx, cy), radius,
                            cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)
        
        return Image.fromarray(polar)

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
        
        if len(boxes) == 0:
            return image

        if type(boxes[0]) != list:
            boxes = [boxes]
        
        mask = self.segmentation_model.get_mask(image, boxes)
        bg = self.background_sampler.sample().resize(image.size)
        bg = self.blur_bg(bg)

        return Image.composite(image, bg, Image.fromarray(mask))
    

class YOLOCustomCrop(Transform):

    def __init__(self, model_path: str, bg_color: Tuple=(128, 128, 128)):
        self._model_path = model_path
        self._bg_color = bg_color
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        self._model = YOLO(model_path)
        
    def get_transformed(self, image: Image) -> Image:
        results =  self._model(image)
        cropped_img = self._get_cropped(image, results[0])

        return cropped_img

    def _get_cropped(self, image: Image, result) -> Image:
        img = np.array(image.convert("RGB"))
        h, w = img.shape[:2]

        polygon = result.masks.xyn[0]
        points = (polygon * [w, h]).astype(np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        background = np.full_like(img, self._bg_color)
        result_img = np.where(mask[:, :, None] == 255, img, background)

        x, y, bw, bh = cv2.boundingRect(points)
        cropped = result_img[y:y+bh, x:x+bw]

        return Image.fromarray(cropped)
    

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
        components = filename.split('.')
        base_name = '.'.join(components[:-1])
        extension = components[-1]
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

    """ def sample_k_per_class(self, k: int, random_state: int=RANDOM_SEED) -> Tuple[List, List]:
        paths = []
        labels = []
        for class_dir in self._iterate_on_classes():
            class_paths = [path.as_posix() for path in class_dir.iterdir()]
            class_path = random.sample(class_paths, k)
            paths.extend(class_path)
            labels.extend([int(class_dir.name)]*len(class_path))

        return paths, labels """
    
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