import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_video
import torchvision.transforms as T

import glob
import os
import numpy as np
import pandas as pd
from numpy import random
from datetime import datetime

from pytorchvideo.models.hub import x3d_m
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# Utility functions
# ============================================================

def normalize(x, method='standard', axis=None):
    """
    Normalize an array using a specified method.

    Parameters
    ----------
    x : array-like
        Input data
    method : str
        - 'standard': mean = 0, std = 1
        - 'range': min = 0, max = 1
        - 'sum': sum = 1
    axis : int or None
        Axis along which normalization is applied

    Returns
    -------
    res : numpy.ndarray
        Normalized array
    """
    x = np.array(x, copy=False)

    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]

        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (
                np.max(y, axis=1) - np.min(y, axis=1)
            ).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError("Invalid normalization method")
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError("Invalid normalization method")

    return res


def AUC_Judd(saliency_map, fixation_map, jitter=True):
    """
    Compute AUC Judd between a saliency map and a human fixation map.

    AUC = Area Under the ROC Curve
    - 0.5 : chance level
    - 1.0 : perfect prediction
    """
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5

    # If there are no fixations, return NaN
    if not np.any(fixation_map):
        print("No fixation to predict")
        return np.nan

    # Add small noise to avoid ties
    if jitter:
        saliency_map += random.rand(*saliency_map.shape) * 1e-7

    # Normalize saliency map to [0, 1]
    saliency_map = normalize(saliency_map, method='range')

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
    Load a video and uniformly sample num_frames frames.

    Output tensor shape:
    [C, T, H, W]
    """
    video, _, _ = read_video(path, pts_unit='sec')
    total_frames = video.shape[0]

    if total_frames == 0:
        raise ValueError(f"Empty or corrupted video: {path}")

    # Uniform temporal sampling
    if total_frames >= num_frames:
        idxs = torch.linspace(0, total_frames - 1, num_frames).long()
    else:
        idxs = torch.linspace(0, total_frames - 1, total_frames).long()
        padding = torch.tensor([total_frames - 1] * (num_frames - total_frames))
        idxs = torch.cat((idxs, padding))

    video = video[idxs]                       # [T, H, W, C]
    video = video.permute(3, 0, 1, 2).float() / 255.0

    transform = T.Compose([
        T.Resize(resize, antialias=True),
        T.Normalize(mean=[0.45, 0.45, 0.45],
                    std=[0.225, 0.225, 0.225])
    ])

    frames = [transform(video[:, t]) for t in range(num_frames)]
    video = torch.stack(frames, dim=1)

    return video


# ============================================================
# Dataset definition
# ============================================================

class VideoDataset(Dataset):
    """
    Dataset for video classification.
    Videos are expected to be organized in subfolders by class.
    """
    def __init__(self, root, num_frames=16, verbose=False):
        self.samples = []
        self.num_frames = num_frames

        classes = sorted(os.listdir(root))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            class_path = os.path.join(root, c)
            if not os.path.isdir(class_path):
                continue
            for v in glob.glob(os.path.join(class_path, "*.mp4")):
                self.samples.append((v, self.class_to_idx[c]))

        if verbose:
            print(f"Dataset loaded: {len(self.samples)} videos, {len(classes)} classes")
            for path, label in self.samples:
                print(f" - {os.path.basename(path)} -> class {label}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        video = load_video_tensor(path, self.num_frames)
        return video, label


# ============================================================
# Custom loss function
# ============================================================

class CustomLoss(nn.Module):
    """
    Combined loss:
    CrossEntropy + GradCAM-based term
    """
    def __init__(self, lambda_cam):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_cam = lambda_cam

    def forward(self, preds, targets, cam_score=None):
        ce_loss = self.ce(preds, targets)

        cam_loss = 0.0
        if cam_score is not None:
            cam_loss = (1 - cam_score.mean())

        print("CrossEntropy:", round(ce_loss.item(), 3),
              "| GradCAM loss:", round(cam_loss.item(), 3))

        return ce_loss + self.lambda_cam * cam_loss


# ============================================================
# Model, optimizer, scheduler
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 4

model = x3d_m(pretrained=True)

in_features = model.blocks[-1].proj.in_features
model.blocks[-1].proj = nn.Linear(in_features, num_classes)

model = model.to(device)

criterion = CustomLoss(lambda_cam=1.5)

optimizer = optim.SGD(
    model.parameters(),
    lr=5e-2,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.5
)


# ============================================================
# Dataset split and loaders
# ============================================================

train_root = "_DATASET_fixation_videos/"
full_dataset = VideoDataset(train_root, num_frames=16, verbose=True)

train_size = int(len(full_dataset) * 7 / 9)
test_size = len(full_dataset) - train_size

train_subset, test_subset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)


# ============================================================
# GradCAM setup
# ============================================================

target_layers = [model.blocks[-2].res_blocks[-1]]
cam = GradCAM(model=model, target_layers=target_layers)


# ============================================================
# Training loop
# ============================================================

model.train()
train_corrects = 0
train_total_samples = 0

for epoch in range(50):
    running_loss = 0.0

    for i, (video, label) in enumerate(train_loader):
        video = video.to(device)
        label = label.to(device)

        preds = model(video)
        pred_classes = preds.argmax(dim=1)

        train_corrects += (pred_classes == label).sum().item()
        train_total_samples += label.size(0)

        all_cams = cam(
            input_tensor=video,
            targets=[ClassifierOutputTarget(l.item()) for l in label]
        )

        cam_scores = []

        for b in range(len(label)):
            video_cams = all_cams[b]

            original_index = train_subset.indices[i * train_loader.batch_size + b]
            video_path, _ = train_subset.dataset.samples[original_index]
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            if video_name.startswith(("90rot_", "-90rot_", "180rot_")):
                video_name = video_name.split("rot_")[-1]

            fixation_path = os.path.join("_FIXATION_DATA", f"{video_name}_GT.npy")

            if not os.path.exists(fixation_path):
                continue

            fixation_map_16th = np.load(fixation_path)
            gt_maps_16th = np.array([fixation_map_16th[:, :, i] for i in range(16)])

            auc_per_frame = []

            for t in range(gt_maps_16th.shape[0]):
                cam_frame = video_cams[t]
                cam_frame = cam_frame / (cam_frame.max() + 1e-8)
                fixation_frame = gt_maps_16th[t]

                auc_t = AUC_Judd(cam_frame, fixation_frame)
                if not np.isnan(auc_t):
                    auc_per_frame.append(auc_t)

        if len(auc_per_frame) > 0:
            df = pd.read_excel('social TOIs_16values.xlsx')
            df = df[df.iloc[:, 2].apply(lambda x: x in video_name)]

            if len(df) != 1:
                video_auc = 1
            else:
                toi_values = df.iloc[0, 4:20].values
                video_auc = np.average(auc_per_frame, weights=toi_values)

            cam_scores.append(video_auc)
            cam_scores = torch.tensor(cam_scores, dtype=torch.float32, device=device)

        loss = criterion(preds, label, cam_score=cam_scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    print(f"{datetime.now()} | Epoch {epoch+1} | Loss: {round(running_loss, 3)}")
    print(f"Training accuracy: {train_corrects / train_total_samples}")


# ============================================================
# Test + GradCAM evaluation
# ============================================================

model.eval()
cam = GradCAM(model=model, target_layers=target_layers)

test_corrects = 0
test_total_samples = 0
video_auc_values = []
video_TOI_auc_values = []

for i, (video, label) in enumerate(test_loader):
    video = video.to(device)
    label = label.to(device)

    preds = model(video)
    pred_classes = preds.argmax(dim=1)

    test_corrects += (pred_classes == label).sum().item()
    test_total_samples += label.size(0)

    all_cams = cam(
        input_tensor=video,
        targets=[ClassifierOutputTarget(int(c.item())) for c in pred_classes]
    )

    for b in range(video.shape[0]):
        original_index = test_subset.indices[i]
        video_path, _ = test_subset.dataset.samples[original_index]
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if video_name.startswith(("90rot_", "-90rot_", "180rot_")):
            video_name = video_name.split("rot_")[-1]

        fixation_path = os.path.join("_FIXATION_DATA", f"{video_name}_GT.npy")
        if not os.path.exists(fixation_path):
            continue

        fixation_map_16th = np.load(fixation_path)
        gt_maps_16th = np.array([fixation_map_16th[:, :, i] for i in range(16)])

        video_cams_test = all_cams[b]

        auc_per_frame = []

        for t in range(video_cams_test.shape[0]):
            cam_frame = video_cams_test[t]
            cam_frame = cam_frame / (cam_frame.max() + 1e-8)
            auc_t = AUC_Judd(cam_frame, gt_maps_16th[t])

            if not np.isnan(auc_t):
                auc_per_frame.append(auc_t)

        if len(auc_per_frame) > 0:
            video_auc = np.mean(auc_per_frame)
            video_auc_values.append(video_auc)

            df = pd.read_excel('social TOIs_16values.xlsx')
            df = df[df.iloc[:, 2].apply(lambda x: x in video_name)]

            if len(df) == 1:
                toi_values = df.iloc[0, 4:20].values
                if np.sum(toi_values) > 0:
                    video_auc_weighted = np.average(auc_per_frame, weights=toi_values)
                    video_TOI_auc_values.append(video_auc_weighted)


# ============================================================
# Final results
# ============================================================

final_acc = test_corrects / test_total_samples
print(f"\nFinal classification accuracy: {final_acc:.2f}")

if len(video_auc_values) > 0:
    print(f"Mean AUC Judd (test set): {np.mean(video_auc_values):.4f}")

if len(video_TOI_auc_values) > 0:
    print(f"Mean AUC Judd weighted by TOIs: {np.mean(video_TOI_auc_values):.4f}")
else:
    print("No AUC values computed.")
