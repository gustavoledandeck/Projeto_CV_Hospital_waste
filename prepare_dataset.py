#!/usr/bin/env python3
import pathlib, shutil, random, cv2, numpy as np, albumentations as A
from albumentations.pytorch import ToTensorV2

ROOT = pathlib.Path("data/raw")
OUT  = pathlib.Path("data/processed")
OUT.mkdir(exist_ok=True)

# load calibration
cal = np.load("calibration/calib.npz")
mtx, dist = cal["mtx"], cal["dist"]
def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

tf_train = A.Compose([
    A.Lambda(image=undistort),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    A.Resize(224, 224),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

tf_val = A.Compose([
    A.Lambda(image=undistort),
    A.Resize(224, 224),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

classes = ["low", "medium", "high", "normal"]
for cls in classes:
    (OUT / "train" / cls).mkdir(parents=True, exist_ok=True)
    (OUT / "val"   / cls).mkdir(parents=True, exist_ok=True)
    (OUT / "test"  / cls).mkdir(parents=True, exist_ok=True)

    imgs = list((ROOT / cls).glob("*"))
    random.shuffle(imgs)
    n = len(imgs)
    nt, nv = int(0.7*n), int(0.15*n)
    splits = {"train": imgs[:nt], "val": imgs[nt:nt+nv], "test": imgs[nt+nv:]}
    for split, lst in splits.items():
        for p in lst:
            shutil.copy(p, OUT / split / cls / p.name)

print("✅  Dataset prepared.")
