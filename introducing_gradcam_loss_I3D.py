# ============================================================
# Imports
# ============================================================
import os
import glob
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import pandas as pd
from numpy import random

from torchvision.io import read_video
import torchvision.transforms as T

from pytorchvideo.models.hub import i3d_r50
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# Utility functions
# ============================================================

def normalize(x, method="standard", axis=None):
    """
    Normalize an array using different strategies.

    Parameters
    ----------
    x : array-like
        Input data.
    method : str
        - 'standard': zero mean, unit variance
        - 'range': scaled to [0, 1]
        - 'sum': normalized so that sum equals 1
    axis : int or None
        Axis along which to normalize. If None, the array is flattened.

    Returns
    -------
    np.ndarray
        Normalized array.
    """
    x = np.asarray(x, copy=False)

    if axis is None:
        if method == "standard":
            return (x - np.mean(x)) / np.std(x)
        elif method == "range":
            return (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == "sum":
            return x / float(np.sum(x))
        else:
            raise ValueError("Unknown normalization method")

    # Axis-wise normalization
    y = np.rollaxis(x, axis).reshape(x.shape[axis], -1)
    shape = np.ones(len(x.shape), dtype=int)
    shape[axis] = x.shape[axis]

    if method == "standard":
        return (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
    elif method == "range":
        return (x - np.min(y, axis=1).reshape(shape)) / (
            np.max(y, axis=1) - np.min(y, axis=1)
        ).reshape(shape)
    elif method == "sum":
        return x / np.sum(y, axis=1).reshape(shape)
    else:
        raise ValueError("Unknown normalization method")


# ============================================================
# AUC-Judd metric for saliency evaluation
# ============================================================

def AUC_Judd(saliency_map, fixation_map, jitter=True):
    """
    Compute AUC-Judd score between a saliency map and a fixation map.

    AUC = 0.5 corresponds to chance level.

    Parameters
    ----------
    saliency_map : np.ndarray
        Predicted saliency map.
    fixation_map : np.ndarray
        Binary human fixation map.
    jitter : bool
        Add small random noise to break ties.

    Returns
    -------
    float
        AUC-Judd score in [0, 1].
    """
    saliency_map = np.asarray(saliency_map, copy=False)
    fixation_map = (np.asarray(fixation_map, copy=False) > 0.5)

    if not np.any(fixation_map):
        return np.nan

    if jitter:
        saliency_map = saliency_map + random.rand(*saliency_map.shape) * 1e-7

    saliency_map = normalize(saliency_map, method="range")

    S = saliency_map.ravel()
    F = fixation_map.ravel()

    S_fix = S[F]
    n_fix = len(S_fix)
    n_pixels = len(S)

    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds) + 2)
    fp = np.zeros(len(thresholds) + 2)

    tp[-1] = 1
    fp[-1] = 1

    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh)
        tp[k + 1] = (k + 1) / float(n_fix)
        fp[k + 1] = (above_th - k - 1) / float(n_pixels - n_fix)

    return np.trapezoid(tp, fp)


# ============================================================
# Video loading and preprocessing
# ============================================================

def load_video_tensor(path, num_frames=16, resize=(224, 224)):
    """
    Load a video file and uniformly sample frames.

    Output tensor shape: [C, T, H, W]
    """
    video, _, _ = read_video(path, pts_unit="sec")
    total_frames = video.shape[0]

    if total_frames == 0:
        raise ValueError(f"Empty or corrupted video: {path}")

    if total_frames >= num_frames:
        idxs = torch.linspace(0, total_frames - 1, num_frames).long()
    else:
        idxs = torch.arange(total_frames)
        padding = torch.full((num_frames - total_frames,), total_frames - 1)
        idxs = torch.cat([idxs, padding])

    video = video[idxs]                       # [T, H, W, C]
    video = video.permute(3, 0, 1, 2) / 255.0  # [C, T, H, W]

    transform = T.Compose([
        T.Resize(resize, antialias=True),
        T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

    frames = [transform(video[:, t]) for t in range(num_frames)]
    return torch.stack(frames, dim=1)


# ============================================================
# Dataset definition
# ============================================================

class VideoDataset(Dataset):
    """
    Dataset that loads videos from class-specific folders.
    """

    def __init__(self, root, num_frames=16, verbose=False):
        self.samples = []
        self.num_frames = num_frames

        classes = sorted(os.listdir(root))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            class_dir = os.path.join(root, c)
            if not os.path.isdir(class_dir):
                continue
            for v in glob.glob(os.path.join(class_dir, "*.mp4")):
                self.samples.append((v, self.class_to_idx[c]))

        if verbose:
            print(f"Loaded {len(self.samples)} videos from {len(classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        video = load_video_tensor(path, self.num_frames)
        return video, label


# ============================================================
# Custom loss: Classification + GradCAM regularization
# ============================================================

class CustomLoss(nn.Module):
    """
    Cross-entropy loss combined with a GradCAM-based regularizer.
    """

    def __init__(self, lambda_cam=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_cam = lambda_cam

    def forward(self, logits, targets, cam_score=None):
        ce_loss = self.ce(logits, targets)

        cam_loss = 0.0
        if cam_score is not None:
            cam_loss = 1.0 - cam_score.mean()

        return ce_loss + self.lambda_cam * cam_loss


# ============================================================
# Model, optimizer, scheduler
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 4
model = i3d_r50(pretrained=True)
model.blocks[-1].proj = nn.Linear(
    model.blocks[-1].proj.in_features, num_classes
)
model.to(device)

criterion = CustomLoss(lambda_cam=1.0)
optimizer = optim.SGD(
    model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# ============================================================
# Dataset split
# ============================================================

root = "_DATASET_fixation_videos_WS/"
dataset = VideoDataset(root, num_frames=8, verbose=True)

train_size = int(len(dataset) * 7 / 9)
test_size = len(dataset) - train_size

train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


# ============================================================
# GradCAM setup
# ============================================================

target_layers = [model.blocks[-2]]
cam = GradCAM(model=model, target_layers=target_layers)


# ============================================================
# Training loop (high-level)
# ============================================================

model.train()
for epoch in range(60):
    running_loss = 0.0

    for videos, labels in train_loader:
        videos = videos.to(device)
        labels = labels.to(device)

        logits = model(videos)

        # Compute GradCAM for ground-truth classes
        cams = cam(
            input_tensor=videos,
            targets=[ClassifierOutputTarget(int(l)) for l in labels],
        )

        # Placeholder: CAM score must be computed externally
        cam_scores = None

        loss = criterion(logits, labels, cam_scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    print(
        f"{datetime.now()} | Epoch {epoch + 1:03d} | "
        f"Loss: {running_loss:.3f}"
    )


# ============================================================
# ============================================================
# Evaluation (classification + GradCAM + AUC-Judd + TOI weighting)
# ============================================================

model.eval()

test_correct = 0
test_total = 0

video_auc_values = []          # mean AUC per video
video_TOI_auc_values = []      # TOI-weighted mean AUC per video

with torch.no_grad():
    for i, (videos, labels) in enumerate(test_loader):
        videos = videos.to(device)
        labels = labels.to(device)

        batch_size = videos.size(0)

        # Forward pass
        logits = model(videos)
        preds = logits.argmax(dim=1)

        # Classification accuracy
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

        # GradCAM on predicted class
        cams = cam(
            input_tensor=videos,
            targets=[ClassifierOutputTarget(int(p)) for p in preds],
        )  # shape: [B, T, H, W]

        for b in range(batch_size):
            predicted_label = preds[b].item()
            true_label = labels[b].item()

            # Recover original video path
            original_index = test_set.indices[i * test_loader.batch_size + b]
            video_path, _ = test_set.dataset.samples[original_index]
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            print(f"\nVideo: {video_name}")
            print(f"  Predicted label: {predicted_label}")
            print(f"  Ground-truth label: {true_label}")

            # Remove rotation prefixes if present
            if video_name.startswith(("90rot_", "-90rot_", "180rot_")):
                video_name = video_name.split("rot_")[-1]

            # Load fixation map
            fixation_path = os.path.join("_FIXATION_DATA", f"{video_name}_GT.npy")
            if not os.path.exists(fixation_path):
                print("  Fixation map not found – skipping AUC")
                continue

            fixation_map = np.load(fixation_path)  # [H, W, 16]
            cam_video = cams[b]                    # [T, H, W]

            # Build GT fixation maps (16 frames)
            gt_maps_16 = np.array([fixation_map[:, :, t] for t in range(16)])

            # ------------------------------------------------------------
            # Frame-by-frame AUC-Judd
            # ------------------------------------------------------------
            auc_per_frame = []

            for t in range(cam_video.shape[0]):
                cam_frame = cam_video[t]
                cam_frame = cam_frame / (cam_frame.max() + 1e-8)

                fixation_frame = gt_maps_16[t]
                auc_t = AUC_Judd(cam_frame, fixation_frame)

                if not np.isnan(auc_t):
                    auc_per_frame.append(auc_t)

            # ------------------------------------------------------------
            # Video-level AUC aggregation
            # ------------------------------------------------------------
            if len(auc_per_frame) == 0:
                continue

            video_auc = np.mean(auc_per_frame)
            video_auc_values.append(video_auc)
            print(f"  Mean AUC-Judd: {video_auc:.4f}")

            # ------------------------------------------------------------
            # TOI-weighted AUC aggregation
            # ------------------------------------------------------------
            df = pd.read_excel("social TOIs_16values.xlsx")
            df = df[df.iloc[:, 2].apply(lambda x: x in video_name)]

            if len(df) != 1:
                print("  TOI weighting not applied (missing or ambiguous entry)")
                continue

            toi_values = df.iloc[0, 4:20].values

            if np.sum(toi_values) == 0:
                print("  TOI weighting skipped (all TOIs are zero)")
                continue

            video_auc_weighted = np.average(auc_per_frame, weights=toi_values)
            video_TOI_auc_values.append(video_auc_weighted)
            print(f"  TOI-weighted AUC-Judd: {video_auc_weighted:.4f}")

# ============================================================
# Final test statistics
# ============================================================

final_accuracy = test_correct / test_total if test_total > 0 else 0.0

print("\n================ FINAL TEST RESULTS ================")
print(f"Classification Accuracy: {final_accuracy:.4f} ({test_correct}/{test_total})")

if len(video_auc_values) > 0:
    print(f"Mean AUC-Judd over test set: {np.mean(video_auc_values):.4f}")

if len(video_TOI_auc_values) > 0:
    print(f"Mean TOI-weighted AUC-Judd: {np.mean(video_TOI_auc_values):.4f}")
else:
    print("No TOI-weighted AUC values computed")
