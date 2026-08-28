"""Image transforms applied before a backbone sees an image.

The zero-shot teachers that used to live here moved to `src.distillation`:
they cost about a second per image, which is why they were distilled into the
YOLO student `YOLOCustomCrop` loads.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import logging

from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO


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

class YOLOCustomCrop(Transform):

    def __init__(self, model_path: str, bg_color: Optional[Tuple]=(128, 128, 128)):
        """`bg_color=None` crops to the mask's bounding box but keeps the original
        pixels, which isolates the crop from the background removal."""
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

        if self._bg_color is None:
            result_img = img
        else:
            background = np.full_like(img, self._bg_color)
            result_img = np.where(mask[:, :, None] == 255, img, background)

        x, y, bw, bh = cv2.boundingRect(points)
        cropped = result_img[y:y+bh, x:x+bw]

        return Image.fromarray(cropped)
