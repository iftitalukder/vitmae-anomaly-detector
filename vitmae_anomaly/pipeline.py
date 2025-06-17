import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
import timm
import torchvision.transforms as transforms
from skimage.transform import resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_window_timeseries(csv_path, window_size=256, stride=64, column=0):
    data = pd.read_csv(csv_path)
    series = data.iloc[:, column].values
    scaler = StandardScaler()
    series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    windows = [series[i:i+window_size] for i in range(0, len(series)-window_size+1, stride)]
    return np.array(windows)

def generate_rp_image(window, epsilon=0.1, img_size=(224, 224)):
    W = len(window)
    R = np.zeros((W, W))
    for i in range(W):
        for j in range(W):
            R[i, j] = np.abs(window[i] - window[j])
    R = (R < epsilon).astype(float)
    R_resized = resize(R, img_size, anti_aliasing=True)
    return R_resized

def get_cls_embeddings(rp_images, patch_size=16, batch_size=32):
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model.head = torch.nn.Identity()
    model.to(device)
    model.eval()

    rp_images_rgb = np.repeat(rp_images[:, np.newaxis, :, :], 3, axis=1)
    tensor_data = torch.tensor(rp_images_rgb, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size)

    embeddings = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            emb = model.forward_features(x)
            embeddings.append(emb[:, 0].cpu().numpy())  # [CLS] token
    return np.vstack(embeddings)

def fit_mahalanobis_model(embeddings):
    model = EmpiricalCovariance()
    model.fit(embeddings)
    return model

def score_mahalanobis(model, embeddings):
    return model.mahalanobis(embeddings)

def run_pipeline(csv_path, window_size=256, stride=64, epsilon=0.1, img_size=(224, 224), column=0):
    print("Loading and windowing time-series...")
    windows = load_and_window_timeseries(csv_path, window_size, stride, column)

    print("Generating recurrence plots...")
    rp_images = np.array([generate_rp_image(w, epsilon, img_size) for w in windows])

    print("Extracting ViT-MAE embeddings...")
    embeddings = get_cls_embeddings(rp_images)

    print("Fitting Mahalanobis model...")
    model = fit_mahalanobis_model(embeddings)

    print("Scoring test embeddings...")
    scores = score_mahalanobis(model, embeddings)

    return scores
