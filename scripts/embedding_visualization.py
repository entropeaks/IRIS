import fiftyone.brain as fob
import fiftyone as fo
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.model import SiameseDino
from src.data import LazyLoadCollection, make_transform
from src.utils import set_device, Browser
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
from umap import UMAP

def compute_embeddings(model, images_paths: list[Path], labels: list[int]):
    dataset = LazyLoadCollection(images_paths, labels, transform=make_transform())
    dataloader = DataLoader(dataset, batch_size=16)
    embeddings = []

    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Computing embeddings"):
            inputs = processor(images=images, return_tensors="pt").to(model.device)
            outputs = model(**inputs)

            if isinstance(outputs, dict):
                outputs = outputs["last_hidden_state"].mean(dim=1)

            elif isinstance(outputs, torch.Tensor):
                pass

            embeddings.append(outputs.cpu().numpy())
            torch.mps.empty_cache() if torch.backends.mps.is_available() else torch.cuda.empty_cache()

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

checkpoint = "sandy-fire-218"
processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
dinov3_model = AutoModel.from_pretrained(
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    dtype=torch.float32
)
hidden_dim = 0
output_dim = 128
device = set_device("cuda")
siamese_model = SiameseDino(dinov3_model, hidden_dim, output_dim)
siamese_model.load_state_dict(torch.load(f"model_checkpoints/{checkpoint}.pth"))
_ = siamese_model.to(device)

data_root_path = Path("data/augmented_data16_v3")
browser = Browser(data_root_path)
images_paths, labels = browser.sample_k_per_class(4)

samples = []
print("Loading samples...")
for img_path, label in tqdm(zip(images_paths, labels)):
    sample = fo.Sample(filepath=img_path)
    sample["ground_truth"] = label
    samples.append(sample)

if "dasdataset" in fo.list_datasets():
    fo.delete_dataset("dasdataset")
dataset = fo.Dataset("dasdataset")
dataset.add_samples(samples)


embeddings = compute_embeddings(siamese_model, images_paths, labels)
# Initialize UMAP
umap_model = UMAP(
    n_neighbors=15,      # controls local vs global structure
    min_dist=0.1,        # controls how tight clusters are
    n_components=2,      # 2 for 2D visualization, 3 for 3D
    metric="cosine",     # cosine distance works well for embeddings
    random_state=42
)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
# Compute the low-dimensional embedding
embeddings_umap = umap_model.fit_transform(embeddings)
print(embeddings_umap.shape)

results = fob.compute_visualization(
    dataset,
    brain_key=checkpoint.replace("-", "_")+"_umap",
    method="manual",
    points=embeddings_umap
)

session = fo.launch_app(dataset)
session.show(results)
session.wait()