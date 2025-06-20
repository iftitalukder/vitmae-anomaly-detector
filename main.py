import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from skimage.transform import resize
import matplotlib.pyplot as plt
import timm

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_window_timeseries(csv_path, window_size=256, stride=64, columns=None, use_pca=True):
    data = pd.read_csv(csv_path, sep=None, engine="python")
    if columns is None:
        selected = data.select_dtypes(include=[np.number]).values
    else:
        selected = data.iloc[:, columns].values

    scaler = StandardScaler()
    normed = scaler.fit_transform(selected)

    if normed.ndim == 1:
        normed = normed[:, None]

    if use_pca and normed.shape[1] > 1:
        pca = PCA(n_components=1)
        normed = pca.fit_transform(normed)

    normed = normed.flatten()
    windows = [normed[i:i+window_size] for i in range(0, len(normed)-window_size+1, stride)]
    return np.array(windows)

def generate_rp_image(window, epsilon=0.1, img_size=(224, 224)):
    R = np.abs(window.reshape(-1,1) - window.reshape(1,-1))
    R = (R < epsilon).astype(float)
    return resize(R, img_size, anti_aliasing=True)

def get_cls_embeddings(rp_images, batch_size=32, model_name="vit_base_patch16_224.mae"):
    try:
        model = timm.create_model(model_name, pretrained=True)
    except:
        print(f"Model '{model_name}' not found. Falling back to 'vit_base_patch16_224'")
        model = timm.create_model("vit_base_patch16_224", pretrained=True)

    model.head = torch.nn.Identity()
    model.to(device).eval()

    rp_rgb = np.repeat(rp_images[:, None, :, :], 3, axis=1)
    tensor_data = torch.tensor(rp_rgb, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size)

    embeddings = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            feats = model.forward_features(x)
            embeddings.append(feats[:, 0].cpu().numpy())  # [CLS] token
    return np.vstack(embeddings)

def fit_mahalanobis_model(embeddings):
    model = EmpiricalCovariance()
    model.fit(embeddings)
    return model

def score_mahalanobis(model, embeddings):
    return model.mahalanobis(embeddings)

def plot_scores(scores, output_plot="score_plot.png"):
    plt.figure(figsize=(12, 4))
    plt.plot(scores, label="Anomaly Score")
    thresh = np.percentile(scores, 99)
    plt.axhline(thresh, color='red', linestyle='--', label="99th Percentile")
    plt.title("Anomaly Score Timeline")
    plt.xlabel("Window Index")
    plt.ylabel("Mahalanobis Distance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.show()

def run_pipeline(csv_path, columns=None, window_size=256, stride=64, epsilon=0.1, img_size=(224,224)):
    print("Loading and windowing time-series...")
    windows = load_and_window_timeseries(csv_path, window_size, stride, columns)

    print("Generating recurrence plots...")
    rp_images = np.array([generate_rp_image(w, epsilon, img_size) for w in windows])

    print("Extracting ViT-MAE embeddings...")
    embeddings = get_cls_embeddings(rp_images)

    print("Fitting Mahalanobis model...")
    model = fit_mahalanobis_model(embeddings)

    print("Scoring embeddings...")
    scores = score_mahalanobis(model, embeddings)

    np.savetxt("anomaly_scores.csv", scores, delimiter=",")
    print("Saved anomaly_scores.csv")

    plot_scores(scores)
    print("Saved score_plot.png")
    return scores

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <your_file.csv> [column1 column2 ...]")
        sys.exit(1)

    csv_file = sys.argv[1]
    columns = list(map(int, sys.argv[2:])) if len(sys.argv) > 2 else None
    run_pipeline(csv_file, columns)
