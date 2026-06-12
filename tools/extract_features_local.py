#!/usr/bin/env python3
"""Lokální extrakce ResNet50 featur z NIH malaria datasetu (běží na Apple MPS).

Kanonický generátor datových souborů. DŮLEŽITÉ: pořadí obrázků je dané SETŘÍDĚNÍM
podle názvu souboru, takže split je deterministický napříč stroji — instruktorský
notebook i soběstačná větev studentského notebooku dají STEJNÝ split, a tím pádem
`test_labels.npz` sedí na predikce studentů bez ohledu na to, kde se featury počítaly.

Vstup:  _data/cell_images.zip  (stažený z https://data.lhncbc.nlm.nih.gov/public/Malaria/)
Výstup: train.npz, val.npz, test_features.npz, test_labels.npz, samples.zip
"""
import glob
import os
import shutil
import zipfile

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

DATA_DIR = "_data"
ZIP_PATH = os.path.join(DATA_DIR, "cell_images.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "cell_images")

prep = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class BunkyDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


def collect_items():
    if not os.path.isdir(EXTRACT_DIR):
        print("rozbaluji zip ...")
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(DATA_DIR)
    png = glob.glob(os.path.join(DATA_DIR, "**", "*.png"), recursive=True)
    items = []
    for p in png:
        parent = os.path.basename(os.path.dirname(p)).lower()
        if "parasitized" in parent:
            items.append((p, 1))
        elif "uninfected" in parent:
            items.append((p, 0))
    # DETERMINISTICKÉ pořadí nezávislé na stroji: podle názvu souboru, tie-break label
    items.sort(key=lambda t: (os.path.basename(t[0]), t[1]))
    return items


def main():
    device = "mps" if torch.backends.mps.is_available() else \
             ("cuda" if torch.cuda.is_available() else "cpu")
    print("zařízení:", device)

    items = collect_items()
    print("obrázků:", len(items),
          "| napadených:", sum(l for _, l in items),
          "| zdravých:", sum(1 - l for _, l in items))

    loader = DataLoader(BunkyDataset(items, prep), batch_size=128,
                        shuffle=False, num_workers=4)

    encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    encoder.fc = nn.Identity()
    encoder.eval().to(device)

    feats, labels = [], []
    with torch.no_grad():
        for k, (xb, yb) in enumerate(loader):
            f = encoder(xb.to(device)).cpu().numpy().astype(np.float16)
            feats.append(f)
            labels.append(np.asarray(yb))
            if k % 20 == 0:
                print(f"  {k*128:6d}/{len(items)}")
    X = np.concatenate(feats).astype(np.float16)
    y = np.concatenate(labels).astype(np.int64)
    print("featury:", X.shape, X.dtype)

    from sklearn.model_selection import train_test_split
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=42)
    for name, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"  {name:5s} {len(yy):6d}  (napad. {int(yy.sum())}, zdrav. {int((1-yy).sum())})")

    np.savez_compressed("train.npz", X=X_train.astype(np.float16), y=y_train)
    np.savez_compressed("val.npz", X=X_val.astype(np.float16), y=y_val)
    np.savez_compressed("test_features.npz", X=X_test.astype(np.float16))
    np.savez_compressed("test_labels.npz", y=y_test)
    for f in ["train.npz", "val.npz", "test_features.npz", "test_labels.npz"]:
        print(f, round(os.path.getsize(f) / 1e6, 1), "MB")

    # samples.zip
    os.makedirs("samples/Parasitized", exist_ok=True)
    os.makedirs("samples/Uninfected", exist_ok=True)
    paras = [p for p, l in items if l == 1][:15]
    heal = [p for p, l in items if l == 0][:15]
    for p in paras:
        shutil.copy(p, "samples/Parasitized/")
    for p in heal:
        shutil.copy(p, "samples/Uninfected/")
    with zipfile.ZipFile("samples.zip", "w") as z:
        for f in glob.glob("samples/**/*.png", recursive=True):
            z.write(f)
    print("samples.zip", round(os.path.getsize("samples.zip") / 1e6, 2), "MB")

    # ---- rychlá kontrola kvality na REÁLNÝCH datech ----
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve
    sc = StandardScaler().fit(X_train.astype(np.float32))
    Xtr = torch.tensor(sc.transform(X_train.astype(np.float32)), dtype=torch.float32)
    Xva = torch.tensor(sc.transform(X_val.astype(np.float32)), dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    head = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(25):
        head.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(head(Xtr[idx]).squeeze(1), ytr[idx])
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        prob = torch.sigmoid(head(Xva).squeeze(1)).numpy()
    acc = ((prob > 0.5).astype(int) == y_val).mean()
    fpr, tpr, _ = roc_curve(y_val, prob)
    sens95 = tpr[fpr <= 0.05].max()
    print(f"\nKONTROLA na reálných datech: val accuracy={acc:.4f} | "
          f"senzitivita@spec>=95%={sens95:.4f}")
    print("hotovo.")


if __name__ == "__main__":
    main()
