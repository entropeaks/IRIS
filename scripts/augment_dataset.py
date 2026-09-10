from pathlib import Path

from transformers.image_utils import load_image
from rembg import remove
from PIL import Image, ImageOps, ImageFilter
import pandas as pd
import random

def replace_background(img: Image, bg: Image, bg_blur_max: int=30):
    bg_blur = random.randint(0, bg_blur_max)
    img = ImageOps.exif_transpose(img)
    bg = bg.resize(img.size)
    blurred_bg = bg.filter(ImageFilter.GaussianBlur(bg_blur))
    output_mask = remove(img, only_mask=True)
    composite = Image.composite(img, blurred_bg, output_mask)

    return composite


bgs = pd.read_csv("resources/clustered_backgrounds.csv")
clusters = []
for _, cluster in bgs.groupby("cluster"):
    clusters.append(cluster)

original_path = Path("data/original_data")
new_source_path = Path("data/augmented_data16-v2")
image_num_multiplier = 4
base_image_num = 4
counter = 0
n_clusters = len(clusters)

for cap_dir in original_path.iterdir():
    label = cap_dir.as_posix().split("/")[-1]
    subpath = new_source_path.joinpath(label)
    if cap_dir.is_dir() and (not subpath.exists() or sum([1 for _ in subpath.iterdir()]) < image_num_multiplier*base_image_num):
        new_source_path.joinpath(label).mkdir(exist_ok=True)
        for image_path in cap_dir.iterdir():
            print(image_path)
            image_name = image_path.as_posix().split("/")[-1]
            image = Image.open(image_path.as_posix())
            
            for n in range(image_num_multiplier):
                cluster_rank = counter%len(clusters)
                bg_rank_within_cluster = (counter//n_clusters)%len(clusters[cluster_rank])
                bg_url = clusters[cluster_rank]["url"].iloc[bg_rank_within_cluster]
                bg = load_image(bg_url)
                counter += 1
                blur_probability = 0.66
                max_blur = 0 if random.random() < blur_probability else 20
                augmented_image = replace_background(image, bg, max_blur)
                augmented_image = augmented_image.resize((224, 224))
                augmented_image.save(subpath.joinpath('-'.join([str(n), image_name])).as_posix())