# ============================================================
# Imports
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
import torchvision.transforms as T
import glob, os
from pytorchvideo.models.hub import x3d_m

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import random_split

import numpy as np
import pandas as pd
from numpy import random

from datetime import datetime

# ============================================================
# Utility functions
# ============================================================
def normalize(x, method='standard', axis=None):
    '''Normalizes the input with specified method.

    Parameters
    ----------
    x : array-like
    method : string, optional
        Valid values for method are:
        - 'standard': mean=0, std=1
        - 'range': min=0, max=1
        - 'sum': sum=1
    axis : int, optional
        Axis perpendicular to which array is sliced and normalized.
        If None, array is flattened and normalized.

    Returns
    -------
    res : numpy.ndarray
        Normalized array.
    '''
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res



def AUC_Judd(saliency_map, fixation_map, jitter=True):
    '''
    AUC stands for Area Under ROC Curve.
    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.

    ROC curve is created by sweeping through threshold values
    determined by range of saliency map values at fixation locations.
    True positive (tp) rate correspond to the ratio of saliency map values above threshold
    at fixation locations to the total number of fixation locations.
    False positive (fp) rate correspond to the ratio of saliency map values above threshold
    at all other locations to the total number of possible other (non-fixated image pixels).

    AUC=0.5 is chance level.

    Parameters
    ----------
    saliency_map : real-valued matrix
    fixation_map : binary matrix
        Human fixation map.
    jitter : boolean, optional
        If True (default), a small random number would be added to each pixel of the saliency map.
        Jitter saliency maps that come from saliency models that have a lot of zero values.
        If the saliency map is made with a Gaussian then it does not need to be jittered
        as the values vary and there is not a large patch of the same value.
        In fact, jittering breaks the ordering in the small values!

    Returns
    -------
    AUC : float, between [0,1]
    '''
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        if video_name: print(f'no fixation to predict for {video_name}')
        else: print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='nearest')
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        saliency_map += random.rand(*saliency_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
        tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
        #NTELA il numero per cui divido è", n_pixels )
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
        
    return np.trapezoid(tp, fp) # y, x


def load_video_tensor(path, num_frames=16, resize=(224, 224)):
    """
    Carica il video intero e estrae num_frames equidistanti nel tempo.
    Ignora la fluidità del movimento (stride alto) per coprire tutto il contesto.
    """
    # 1. Legge tutto il video
    video, _, info = read_video(path, pts_unit='sec')
    total_frames = video.shape[0]

    if total_frames == 0:
        raise ValueError(f"Video vuoto o corrotto: {path}")

    # 2. Calcola indici equidistanti (Uniform Sampling)
    # Se il video ha meno frame di quelli richiesti, li prende tutti e poi duplica (padding) o crasha.
    # Qui usiamo linspace per spalmare i frame su tutto il video.
    if total_frames >= num_frames:
        idxs = torch.linspace(0, total_frames - 1, num_frames).long()
    else:
        # Fallback: se il video è cortissimo (es. 10 frame ma ne chiedi 16)
        # Prendiamo tutto e ripetiamo l'ultimo frame fino a riempire
        idxs = torch.linspace(0, total_frames - 1, total_frames).long()
        padding = torch.tensor([total_frames - 1] * (num_frames - total_frames)).long()
        idxs = torch.cat((idxs, padding))

    # 3. Estrazione
    video = video[idxs]  # [num_frames, H, W, C]

    # 4. Trasformazioni (Standard I3D/ResNet)
    video = video.permute(3, 0, 1, 2).float() / 255.0
    
    transform = T.Compose([
        T.Resize(resize, antialias=True),
        T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])
    
    frames = [transform(video[:, t, :, :]) for t in range(num_frames)]
    video = torch.stack(frames, dim=1)  # [C, T, H, W]

    return video 



#  Dataset definition Class
class VideoDataset(Dataset):
    def __init__(self, root, num_frames=16, verbose=False):
        self.samples = []
        self.num_frames = num_frames
        self.verbose = verbose

        classes = sorted(os.listdir(root))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            class_path = os.path.join(root, c)
            if not os.path.isdir(class_path):
                continue
            videos = glob.glob(os.path.join(class_path, "*.mp4"))
            for v in videos:
                self.samples.append((v, self.class_to_idx[c]))

        if verbose:
            print(f"📦 Dataset: {len(self.samples)} video trovati in {len(classes)} classi")
            # --- AGGIUNGI QUESTO BLOCCO ---
            print("\nLista dei video caricati:")
            for path, label in self.samples:
                video_name = os.path.basename(path)
                class_name = [name for name, idx in self.class_to_idx.items() if idx == label][0]
                print(f"   - {video_name} -> Classe: {class_name} (Label: {label})")
            print("---------------------------------")
            # ------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        video = load_video_tensor(path, num_frames=self.num_frames)
        return video, label

#  Custom Loss Function
class CustomLoss(nn.Module):
    def __init__(self, lambda_cam):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_cam = lambda_cam

    def forward(self, preds, targets, cam_score=None):
        ce_loss = self.ce(preds, targets)

        # score gradcam
        cam_loss = 0.0
        if cam_score is not None:
            cam_loss = (1 - cam_score.mean())  # the higher is the gradcam score the lower is the loss
        print("Cross entropy loss:", np.round(ce_loss.item(), 3), " _ _ _ _ Grad_CAM loss:", np.round(cam_loss.item(),3)) 
        return ce_loss + self.lambda_cam * cam_loss


# ============================================================
# Model, optimizer, scheduler
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4  # NEW LINE for 3. trainings

model = x3d_m(pretrained=True)
#for x3d 
in_features = model.blocks[-1].proj.in_features
model.blocks[-1].proj = nn.Linear(in_features, num_classes)

print("--- Checking weight (requires_grad) ---")
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"⚠️ Warning: The parameter'{name}' has requires_grad=False")
print("---------------------------------------")

model = model.to(device)
criterion = CustomLoss(lambda_cam=1.5)
#criterion = nn.CrossEntropyLoss()
initial_lr = 5e-2      
momentum_value = 0.9      
weight_decay_value = 1e-4 

optimizer = optim.SGD(
    model.parameters(),
    lr=initial_lr,             
    momentum=momentum_value,   
    weight_decay=weight_decay_value 
)

# SCHEDULER 
step_size = 10  # every 10 epochs
gamma = 0.5     # halve lr
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


train_root = "_DATASET_fixation_videos/"
full_sampled_dataset = VideoDataset(train_root, num_frames=16, verbose=True)

train_ratio = 7 / 9
test_ratio = 2 / 9

total_size = len(full_sampled_dataset)
train_size = int(total_size * train_ratio)
test_size = total_size - train_size

train_subset, test_subset = random_split(full_sampled_dataset, [train_size, test_size])

train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
test_loader  = DataLoader(test_subset, batch_size=1, shuffle=False)

print(f"📦 Total dataset: {total_size} videos")
print(f"🧠 Training: {train_size} videos")
print(f"🧪 Test: {test_size} videos")



from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


#target_layers = [model.blocks[-2]]
target_layers = [model.blocks[-2].res_blocks[-1]]
cam = GradCAM(model=model, target_layers=target_layers)


train_corrects=0
train_total_samples=0
model.train()
for epoch in range(50):
    running_loss = 0.0
    for i, (video, label) in enumerate(train_loader):
        video = video.to(device)
        label = label.to(device)

        preds = model(video)
        pred_classes = preds.argmax(dim=1)

        train_corrects += (pred_classes == label).sum().item()
        train_total_samples += label.size(0)

        all_cams = cam(input_tensor=video, targets=[ClassifierOutputTarget(l.item()) for l in label])

        cam_scores = []
        for b in range(len(label)):
            video_cams = all_cams[b] # [H, W] for all the 16 gradcams

            original_index = train_subset.indices[i * train_loader.batch_size + b]
            video_path, _ = train_subset.dataset.samples[original_index]
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            if video_name.startswith(("90rot_", "-90rot_", "180rot_")):
                video_name = video_name.split("rot_")[-1]

            fixation_path = os.path.join("_FIXATION_DATA", f"{video_name}_GT.npy")

            if not os.path.exists(fixation_path):
                print(f"⚠️ Fixation mancante per {video_name}")
                continue

            fixation_map_16th = np.load(fixation_path)
            gt_maps_16th = np.array([fixation_map_16th[:, :, i] for i in range(16)])

            auc_per_frame=[]

            for t in range(gt_maps_16th.shape[0]): 
                #normalising
                cam_frame = video_cams[t]
                cam_frame = cam_frame / (cam_frame.max() + 1e-8)
                
                fixation_frame = gt_maps_16th[t]
                
                auc_t = AUC_Judd(cam_frame, fixation_frame)
                
                if not np.isnan(auc_t):
                    auc_per_frame.append(auc_t)
    
    # --- Final average on the video ---
    if len(auc_per_frame) > 0:
        
        #open the excel file social TOIs_16values.xlsx  and take the row with third column value which is contianed in video_name
        df = pd.read_excel('social TOIs_16values.xlsx')
        #df became only the row with third column value which is contians video_name
        df = df[df.iloc[:, 2].apply(lambda x: x in video_name)]
        #check if the number of rows is 1, otherwise print a warning
        if len(df) != 1:
            print(f"⚠️ Warning: more than one row or no row found for {video_name} in social TOIs_16values.xlsx")
            print(f"‼️FOR THIS VIDEO AVERAGE IS NOT WEIGHTED ‼️")
            video_auc=1
        else:
            toi_values = df.iloc[0, 4:20].values 
            video_auc = np.average(auc_per_frame, weights=toi_values)
        
        
        cam_scores.append(video_auc)
        cam_scores = torch.tensor(cam_scores, dtype=torch.float32, device=device)

    #    loss and backpropagation
        loss = criterion(preds, label, cam_score=cam_scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        scheduler.step()

    current_dateTime = datetime.now()
    print(f"{current_dateTime}******* Epoch {epoch+1}, loss totale = {np.round(running_loss, 3)} *******")
    print(f"Accuracy of this epoch {train_corrects / train_total_samples}")


"""
#########################################
###          TEST + GRADCAM           ###
#########################################
"""
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

model.eval()

target_layers = [model.blocks[-2].res_blocks[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

test_corrects = 0
test_total_samples = 0
video_auc_values = []   # contains 1 values for the average AUC for each video
video_TOI_auc_values = []   # contains 1 values for the average AUC weighted on the social TOI for each video

for i, (video, label) in enumerate(test_loader):
    video = video.to(device)
    label = label.to(device)

    batch_size = video.shape[0]

    # Predictions
    preds = model(video)
    pred_classes = preds.argmax(dim=1)
    
    # Calculating accuracy
    test_corrects += (pred_classes == label).sum().item()
    test_total_samples += label.size(0)
    

    # GradCAM entire batch
    all_cams = cam(
        input_tensor=video,
        targets=[ClassifierOutputTarget(int(c.item())) for c in pred_classes]
    )
    # -> shape (B, T, H, W)

    for b in range(batch_size):
        
        predicted_label = pred_classes[b].item()
        true_label = label[b].item()
        
        original_index = test_subset.indices[i * test_loader.batch_size + b]
        video_path, _ = test_subset.dataset.samples[original_index]

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"\n🎥 {video_name}")
        print(f"   - **Model Prediction:** {predicted_label}")
        print(f"   - **Real label (GT):** {true_label}")
        # ---------------------------------
        
        # Remove rotation prefix if present
        if video_name.startswith(("90rot_", "-90rot_", "180rot_")):
            video_name = video_name.split("rot_")[-1]

        fixation_path = os.path.join("_FIXATION_DATA", f"{video_name}_GT.npy")

        if not os.path.exists(fixation_path):
            print(f"⚠️ Missing fixation for {video_name}")
            continue

        fixation_map_16th = np.load(fixation_path) 
        video_cams_test = all_cams[b]

        gt_maps_16th = np.array([fixation_map_16th[:, :, i] for i in range(16)])


        #############################################
        # AUC frame-by-frame
        #############################################
        auc_per_frame = []

        for t in range(video_cams_test.shape[0]):
            cam_frame = video_cams_test[t]
            cam_frame = cam_frame / (cam_frame.max() + 1e-8)

            fixation_frame = gt_maps_16th[t]
            
            auc_t = AUC_Judd(cam_frame, fixation_frame)

            if not np.isnan(auc_t):
                auc_per_frame.append(auc_t)

        #############################################
        # Final AUC for the video
        #############################################
        if len(auc_per_frame) > 0:
            video_auc = np.mean(auc_per_frame)
            video_auc_values.append(video_auc)

            print(f"   - **AVg AUC Judd:** {video_auc:.4f}")

            df = pd.read_excel('social TOIs_16values.xlsx')
            df = df[df.iloc[:, 2].apply(lambda x: x in video_name)]

            if len(df) != 1:
                print(f"⚠️ It was no possible to compute the wighted average for {video_name}")
            else:
                toi_values = df.iloc[0, 4:20].values
                if np.sum(toi_values) == 0:
                    print(f"   - **Avg AUC Judd Medio Weighted (TOIs):** Cannot be computed because all TOIs are zero.")
                    continue
                video_auc_weighted = np.average(auc_per_frame, weights=toi_values)
                video_TOI_auc_values.append(video_auc_weighted)
                print(f"   - ** Avg AUC Judd Medio Weighted:** {video_auc_weighted:.4f}")


#############################################
# MEDIA AUC e ACC SUL TEST SET
#############################################

print(f"\n📊 FINAL RESULTS:")
final_acc = test_corrects / test_total_samples
print(f"   - Classification Accuracy: {final_acc:.2f}% ({test_corrects}/{test_total_samples})")
if len(video_auc_values) > 0:
    mean_auc = np.mean(video_auc_values)
    print(f"\n📊 Avg AUC_Judd on the test set = {mean_auc:.4f} (N={len(video_auc_values)} video)")
if len(video_TOI_auc_values) > 0:
    mean_TOI_auc = np.mean(video_TOI_auc_values)
    print(f"\n📊 Avg AUC_Judd Weighted (TOIs) on the test set = {mean_TOI_auc:.4f} (N={len(video_TOI_auc_values)} video)")
else:
    print("No AUC values computed!")