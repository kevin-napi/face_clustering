import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR    = Path("outputs")
MANIFEST      = OUTPUT_DIR / "manifest.csv"
EMBEDDINGS_OUT = OUTPUT_DIR / "embeddings.npy"
METADATA_OUT  = OUTPUT_DIR / "metadata.csv"
BATCH_SIZE    = 32

def build_model() -> nn.Module:
    """
    ResNet50 pretrained on ImageNet, final FC layer removed.
    Output: 2048-dim feature vector per image.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

def get_transform():
    """Standard ImageNet preprocessing that ResNet expects."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # manga is B&W; repeat to 3ch
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats
            std=[0.229, 0.224, 0.225]
        ),
    ])

def load_batch(paths: list[str], transform) -> torch.Tensor:
    """Load a list of image paths into a batched tensor."""
    tensors = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(transform(img))
        except Exception as e:
            print(f"  [warn] Could not load {p}: {e}")
            tensors.append(torch.zeros(3, 224, 224))
    return torch.stack(tensors)

def main():
    manifest = pd.read_csv(MANIFEST)
    print(f"Loaded manifest: {len(manifest)} images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model().to(device)
    transform = get_transform()

    all_embeddings = []
    paths = manifest["image_path"].tolist()

    # Process in batches
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Extracting features"):
        batch_paths = paths[i : i + BATCH_SIZE]
        batch_tensor = load_batch(batch_paths, transform).to(device)
 
        with torch.no_grad():
            features = model(batch_tensor) # (B, 2048, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # (B, 2048)
 
        all_embeddings.append(features.cpu().numpy())
 
    embeddings = np.vstack(all_embeddings)  # (N, 2048)
    print(f"\nEmbeddings shape: {embeddings.shape}")

    np.save(EMBEDDINGS_OUT, embeddings)
    manifest.to_csv(METADATA_OUT, index=False)

    print(f"Embeddings saved: {EMBEDDINGS_OUT}")
    print(f"Metadata saved: {METADATA_OUT}")


if __name__ == "__main__":
    main()
