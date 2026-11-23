import json
import time
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

torch.set_num_threads(20)

class ISICFeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.base_dir / row["relative_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=0)
        if self.transform:
            image = self.transform(image)
        label = int(row["class_index"]) if "class_index" in row and not pd.isna(row["class_index"]) else -1
        return image, row["image_id"], label

def build_feature_extractor():
    weights_path = r"C:\Users\Dishitha Ramesh\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth"
    model = resnet18()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    # Replace fc with Identity to get 512-dim representation
    model.fc = nn.Identity()
    model.eval()
    return model

def main():
    prep_dir = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\data\isic2019\prepared")
    base_dir = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\data\isic2019")
    out_dir = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al\outputs\cached_features")
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = build_feature_extractor()

    # Load metadata and labels
    train_df = pd.read_csv(prep_dir / "public/train.csv")
    train_labels = pd.read_csv(prep_dir / "private/train_oracle.csv")
    train_df = train_df.merge(train_labels[["image_id", "class_index"]], on="image_id", how="left")

    val_df = pd.read_csv(prep_dir / "public/validation.csv")
    val_labels = pd.read_csv(prep_dir / "private/validation_labels.csv")
    val_df = val_df.merge(val_labels[["image_id", "class_index"]], on="image_id", how="left")

    test_df = pd.read_csv(prep_dir / "public/test.csv")
    test_labels = pd.read_csv(prep_dir / "private/test_labels.csv")
    test_df = test_df.merge(test_labels[["image_id", "class_index"]], on="image_id", how="left")

    splits = {
        "val": val_df,
        "test": test_df,
        "train": train_df,
    }

    for split_name, split_df in splits.items():
        save_path = out_dir / f"{split_name}_features.pt"
        if save_path.exists():
            print(f"[{split_name}] Already cached at {save_path}, skipping.")
            continue

        print(f"[{split_name}] Starting extraction for {len(split_df)} images...")
        ds = ISICFeatureDataset(split_df, base_dir, transform=transform)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

        feats_list = []
        ids_list = []
        labels_list = []

        t0 = time.time()
        with torch.no_grad():
            for i, (imgs, ids, lbls) in enumerate(loader):
                emb = model(imgs)
                feats_list.append(emb)
                ids_list.extend(ids)
                labels_list.append(lbls)
                if (i + 1) % 25 == 0 or (i + 1) == len(loader):
                    done = (i + 1) * 64
                    elapsed = time.time() - t0
                    speed = done / max(1e-5, elapsed)
                    print(f"  [{split_name}] Batch {i+1}/{len(loader)}: {done} images ({speed:.1f} img/s)")

        feats = torch.cat(feats_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        torch.save({
            "features": feats,
            "labels": labels,
            "image_ids": ids_list,
        }, save_path)
        print(f"[{split_name}] Saved {feats.shape} to {save_path} in {time.time()-t0:.1f}s")

    print("\nFeature extraction complete for all splits!")

if __name__ == "__main__":
    main()
