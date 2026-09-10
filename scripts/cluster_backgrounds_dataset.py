import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from src.data import make_transform
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image


class BackgroundDataset(Dataset):
        def __init__(self, urls: pd.Series, transforms: v2):
            self.urls = urls
            self.transforms = transforms

        def __len__(self):
            return self.urls.size
        
        def __getitem__(self, idx):
            return self.transforms(load_image(self.urls[idx]))
        
     
def get_background_embeddings(processor: AutoImageProcessor, model: AutoModel, background_dataloader: DataLoader) -> np.ndarray:
    all_embeddings = []
    with torch.no_grad():
        for image_batch in tqdm(background_dataloader):
            inputs = processor(images=image_batch, return_tensors="pt")
            outputs = model(**inputs)
            embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings).numpy()


def main(path_to_csv: str, cutoff: int, batch_size: int, n_clusters: int):
    
    df = pd.read_csv(path_to_csv, sep='\t', header=0)
    background_dataset = BackgroundDataset(df[:cutoff]["photo_image_url"], make_transform())
    background_dataloader = DataLoader(background_dataset, batch_size=batch_size)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
    model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m", dtype=torch.float32)

    bg_embeddings = get_background_embeddings(processor, model, background_dataloader)

    kmeans = KMeans(n_clusters)
    labels = kmeans.fit_predict(bg_embeddings)

    clustered = []
    for i in range(len(labels)):
        sample = {}
        sample["url"] = df["photo_image_url"][i]
        sample["cluster"] = labels[i]
        clustered.append(sample)

    clustered_df = pd.DataFrame(clustered)
    clustered_df.to_csv("resources/clustered_backgrounds.csv", index=False)

    print("Clusters file successfully saved.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset Augmentation")

    parser.add_argument("path_to_csv", type=str, help="The csv file from the backgrounds dataset.\nSee https://github.com/unsplash/datasets?tab=readme-ov-file.")
    parser.add_argument("cutoff", type=int, help="The number of backgrounds to be sampled from the augmentation dataset.")
    parser.add_argument("batch_size", type=int, help="Batch size (reduce if low VRAM).")
    parser.add_argument("n_clusters", type=int, help="Num clusters to sample backgrounds from.")
    args = parser.parse_args()

    main(args.path_to_csv, args.cutoff, args.batch_size, args.n_clusters)