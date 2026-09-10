"""Distil slow zero-shot teachers into a fast student.

GroundingDINO locates an object from a text prompt and SAM turns that box into a
mask. Together they label an image without any training, at roughly a second per
image -- fine for building a dataset once, far too slow to run in a pipeline.
So they label a corpus, a YOLO segmentation model trains on it, and that student
does the work afterwards at a few milliseconds per image.

The teachers are wrong sometimes, and a wrong label is worse than a missing one:
the student learns the mistake and nothing downstream ever says so. Both stages
report a confidence, so anything the teachers are unsure about is set aside in
`rejected/` with its score rather than written into the training set.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                          SamProcessor, SamModel)
from transformers.image_utils import load_image
from torchvision.transforms import v2

from src.preprocess import Transform
from src.utils import set_device


@dataclass
class Detection:
    """One located object: pixel box, detector confidence, matched prompt term."""
    box: tuple[float, float, float, float]
    score: float
    label: str


class ZeroShotDetector:
    """Locates objects named in a text prompt, without training on them."""

    def __init__(self, device: str=None, model_id: str="IDEA-Research/grounding-dino-tiny"):
        self.device = set_device(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model_id = model_id

    @torch.no_grad()
    def detect(self, image: Image, labels: list[str],
               box_threshold: float=0.4, text_threshold: float=0.3) -> list[Detection]:
        """Every match above threshold, best first.

        Returns all of them rather than the best one: a photo holding several
        objects is worth several training instances, and YOLO stores as many per
        image as it is given.
        """
        inputs = self.processor(images=image, text=labels, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]

        detections = [
            Detection(box=tuple(round(v, 2) for v in box.tolist()),
                      score=float(score),
                      label=str(label))
            for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"])
        ]
        return sorted(detections, key=lambda d: d.score, reverse=True)


class SamSegmenter:
    """Turns a box into a mask."""

    def __init__(self, device: str=None, model_id: str="facebook/sam-vit-base"):
        self.device = set_device(device)
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device)
        self.model_id = model_id

    @torch.no_grad()
    def segment(self, image: Image, boxes: list[list[float]]) -> list[tuple[np.ndarray, float]]:
        """One mask per box, with the confidence SAM assigns it.

        SAM proposes three masks per box and scores each; the best of the three
        is taken per box. Reducing over the whole tensor instead, as a single-box
        implementation can get away with, picks an index into the box axis once
        there is more than one box.
        """
        if not boxes:
            return []

        inputs = self.processor(image, input_boxes=[[list(b) for b in boxes]],
                                return_tensors="pt")
        original_sizes = inputs["original_sizes"]
        reshaped_sizes = inputs["reshaped_input_sizes"]

        # the processor emits boxes as float64 and MPS has no such dtype, so cast
        # rather than move wholesale; integer sizes must stay integers
        inputs = {name: (tensor.to(self.device, dtype=torch.float32)
                         if tensor.dtype == torch.float64 else tensor.to(self.device))
                  for name, tensor in inputs.items()}
        outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), original_sizes, reshaped_sizes,
        )[0]                                   # (n_boxes, n_proposals, H, W)
        scores = outputs.iou_scores.cpu()[0]   # (n_boxes, n_proposals)

        best = scores.argmax(dim=-1)
        return [(masks[i, best[i]].numpy(), float(scores[i, best[i]]))
                for i in range(len(boxes))]


def mask_to_polygon(mask: np.ndarray, width: int, height: int,
                    epsilon_factor: float=0.002) -> list[float] | None:
    """Largest contour of a binary mask as a normalised polygon, or None if empty.

    Only the largest is kept: SAM occasionally returns speckle alongside the
    object, and YOLO reads one polygon per instance.
    """
    contours, _ = cv2.findContours((mask.astype(np.uint8) * 255),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(float)
    if len(points) < 3:
        return None

    points[:, 0] /= width
    points[:, 1] /= height
    return np.clip(points, 0.0, 1.0).ravel().tolist()


@dataclass
class LabellingReport:
    """What the teachers produced, and what they refused to commit to."""
    labelled: int = 0
    instances: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    scores: list[dict] = field(default_factory=list)

    @property
    def seen(self) -> int:
        return self.labelled + sum(self.rejected.values())

    @property
    def rejection_rate(self) -> float:
        return sum(self.rejected.values()) / max(self.seen, 1)

    def summary(self) -> str:
        lines = [f"labelled {self.labelled}/{self.seen} images "
                 f"({self.instances} instances), "
                 f"rejected {self.rejection_rate:.1%}"]
        for reason, count in sorted(self.rejected.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {reason:22} {count}")
        return "\n".join(lines)


class Distiller:
    """Labels a corpus with the zero-shot teachers, for a YOLO student to train on.

    Labelling and training are kept apart on purpose. Labelling is the slow half
    and its product is a dataset worth looking at before anything trains on it --
    auto-labels are wrong often enough that hiding them behind a single call
    removes the one moment you could notice.
    """

    def __init__(self,
                 detector: ZeroShotDetector=None,
                 segmenter: SamSegmenter=None,
                 box_threshold: float=0.4,
                 text_threshold: float=0.3,
                 mask_threshold: float=0.85,
                 max_instances: int=None,
                 device: str=None):
        self.detector = detector or ZeroShotDetector(device=device)
        self.segmenter = segmenter or SamSegmenter(device=device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.mask_threshold = mask_threshold
        self.max_instances = max_instances

    def label(self,
              image_paths: list[str | Path],
              labels: list[str],
              out_dir: str | Path,
              val_ratio: float=0.2,
              seed: int=42) -> LabellingReport:
        """Write a YOLO segmentation dataset, setting aside what the teachers doubt.

        `labels` is the text prompt, one term per class, in class-id order, so
        the same distiller can label a different object without being rebuilt.
        """
        out = Path(out_dir)
        report = LabellingReport()
        class_ids = {name.lower(): i for i, name in enumerate(labels)}

        rng = random.Random(seed)
        paths = [Path(p) for p in image_paths]
        rng.shuffle(paths)
        split = int(len(paths) * (1 - val_ratio))
        assignment = {p: ("train" if i < split else "val") for i, p in enumerate(paths)}

        for path in tqdm(sorted(paths), desc="Labelling"):
            record = self._label_one(path, labels, class_ids)
            report.scores.append({"image": str(path), **record["scores"]})

            if record["reason"]:
                report.rejected[record["reason"]] = report.rejected.get(record["reason"], 0) + 1
                self._write_rejected(out, path, record["reason"])
                continue

            self._write_labelled(out, path, assignment[path], record["lines"])
            report.labelled += 1
            report.instances += len(record["lines"])

        self._write_dataset_yaml(out, labels)
        self._write_manifest(out, labels, report)
        return report

    def _label_one(self, path: Path, labels: list[str], class_ids: dict) -> dict:
        image = ImageOps.exif_transpose(load_image(str(path)))
        width, height = image.size

        detections = self.detector.detect(image, labels,
                                          box_threshold=self.box_threshold,
                                          text_threshold=self.text_threshold)
        if not detections:
            return {"reason": "no_detection", "lines": [], "scores": {}}
        if self.max_instances:
            detections = detections[:self.max_instances]

        masks = self.segmenter.segment(image, [d.box for d in detections])
        scores = {"detector_best": detections[0].score,
                  "mask_best": max((s for _, s in masks), default=0.0),
                  "instances_found": len(detections)}

        lines = []
        for detection, (mask, iou) in zip(detections, masks):
            if iou < self.mask_threshold:
                continue
            polygon = mask_to_polygon(mask, width, height)
            if polygon is None:
                continue
            class_id = class_ids.get(detection.label.lower(), 0)
            lines.append(" ".join([str(class_id)] + [f"{v:.6f}" for v in polygon]))

        if not lines:
            reason = ("low_mask_score" if scores["mask_best"] < self.mask_threshold
                      else "no_contour")
            return {"reason": reason, "lines": [], "scores": scores}
        return {"reason": None, "lines": lines, "scores": scores}

    def _write_labelled(self, out: Path, path: Path, split: str, lines: list[str]) -> None:
        """Store the image with its EXIF rotation already applied.

        Copying the file as-is leaves the orientation for the reader to resolve,
        and readers disagree: cv2 applies the tag, PIL does not. The polygons are
        normalised against the rotated frame, so a reader that skips the tag gets
        labels that silently do not match its pixels.
        """
        image_dir = out / "images" / split
        label_dir = out / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        self._save_upright(path, image_dir / path.name)
        (label_dir / path.with_suffix(".txt").name).write_text("\n".join(lines) + "\n")

    def _write_rejected(self, out: Path, path: Path, reason: str) -> None:
        target = out / "rejected" / reason
        target.mkdir(parents=True, exist_ok=True)
        self._save_upright(path, target / path.name)

    @staticmethod
    def _save_upright(source: Path, target: Path) -> None:
        with Image.open(source) as image:
            ImageOps.exif_transpose(image).convert("RGB").save(target, quality=95)

    def _write_dataset_yaml(self, out: Path, labels: list[str]) -> None:
        names = "\n".join(f"  {i}: {name}" for i, name in enumerate(labels))
        (out / "data.yaml").write_text(
            f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n{names}\n")

    def _write_manifest(self, out: Path, labels: list[str], report: LabellingReport) -> None:
        """Record what produced this dataset; in six months nothing else will say."""
        (out / "manifest.json").write_text(json.dumps({
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prompt": labels,
            "detector": self.detector.model_id,
            "segmenter": self.segmenter.model_id,
            "thresholds": {"box": self.box_threshold, "text": self.text_threshold,
                           "mask": self.mask_threshold},
            "max_instances": self.max_instances,
            "labelled": report.labelled,
            "instances": report.instances,
            "rejected": report.rejected,
        }, indent=2) + "\n")
        pd.DataFrame(report.scores).to_csv(out / "scores.csv", index=False)

    def train(self, dataset_yaml: str | Path, model: str="yolov8n-seg.pt", **kwargs):
        """Train the student. A thin pass-through to ultralytics, which owns this."""
        from ultralytics import YOLO

        return YOLO(model).train(data=str(dataset_yaml), **kwargs)


class BackgroundRandomSampler:
    """Draws a random background image from a TSV of URLs."""

    def __init__(self, bg_file_path: str, seed: int=None):
        self.bg_file_path = bg_file_path
        self.df = pd.read_csv(self.bg_file_path, sep="\t", header=0)
        self._rng = random.Random(seed)

    def sample(self) -> Image:
        idx = self._rng.randrange(len(self.df))
        return load_image(self.df["photo_image_url"][idx])


class BackgroundTransform(Transform):
    """Cuts the object out and drops it onto an unrelated background.

    Augmentation, not distillation, but it leans on the same teachers, so it
    lives beside them. It is the slow path by construction; use it to generate a
    dataset offline, never inside a training loop.
    """

    def __init__(self,
                 detector: ZeroShotDetector,
                 segmenter: SamSegmenter,
                 background_sampler: BackgroundRandomSampler,
                 labels: list[str],
                 instance_transforms: v2.Compose=None,
                 background_blur_max_level: int=15,
                 blur_probability: float=0.5,
                 seed: int=None):
        self.detector = detector
        self.segmenter = segmenter
        self.background_sampler = background_sampler
        self.labels = labels
        self.instance_transforms = instance_transforms
        self.background_blur_max_level = background_blur_max_level
        self.blur_probability = blur_probability
        self._rng = random.Random(seed)

    def blur_bg(self, bg: Image) -> Image:
        if self._rng.random() > self.blur_probability:
            return bg
        return bg.filter(ImageFilter.GaussianBlur(self._rng.randint(0, self.background_blur_max_level)))

    def get_transformed(self, image: Image) -> Image:
        image = ImageOps.exif_transpose(image)
        if self.instance_transforms:
            image = self.instance_transforms(image)

        detections = self.detector.detect(image, self.labels)
        if not detections:
            return image

        masks = self.segmenter.segment(image, [detections[0].box])
        if not masks:
            return image

        background = self.blur_bg(self.background_sampler.sample().resize(image.size))
        return Image.composite(image, background, Image.fromarray(masks[0][0]))
