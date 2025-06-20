#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
import timm
from skimage.transform import resize
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_window_timeseries(csv_path, window_size=256, stride=64, column=0):
    """
    Load a CSV (auto-detects comma or semicolon), select a numeric column,
    normalize it, and split into overlapping windows.
    """
    data = pd.read_csv(csv_path, sep=None, engine="python")  # auto-detects delimiter
    series = data.iloc[:, column].values
    scaler = StandardScaler()
    series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    windows = [series[i:i+window_size] 
               for i in range(0, len(series)-window_size+1, stride)]
    return np.array(windows)

def generate_rp_image(window, epsilon=0.1, img_size=(224, 224)):
    """
    Convert a 1D window into a 2D recurrence plot (RP) image.
    """
    W = len(window)
    R = np.abs(window.reshape(-1,1) - window.reshape(1,-1))
    R = (R < epsilon).astype(float)
    R_resized = resize(R, img_size, anti_aliasing=True)
    return R_resized

def get_cls_embeddings(rp_images, batch_size=32):
    """
    Use a pretrained ViT-B/16 model to extract [CLS] embeddings from RP images.
    """
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model.head = torch.nn.Identity()
    model.to(device).eval()

    # Expand to 3-channel and build DataLoader
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
    """
    Fit a Gaussian (Mahalanobis) model on embeddings of normal data.
    """
    model = EmpiricalCovariance()
    model.fit(embeddings)
    return model

def score_mahalanobis(model, embeddings):
    """
    Compute Mahalanobis distance scores for each embedding.
    """
    return model.mahalanobis(embeddings)

def plot_scores(scores, output_plot="score_plot.png"):
    """
    Plot the anomaly scores over window index and save the figure.
    """
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

def run_pipeline(csv_path, column, window_size=256, stride=64, epsilon=0.1, img_size=(224,224)):
    """
    Full pipeline:
    1. Load & window time-series
    2. Generate recurrence plots
    3. Extract ViT-MAE embeddings
    4. Fit Mahalanobis model on embeddings
    5. Score embeddings
    6. Save scores & plot
    """
    print("Loading and windowing time-series...")
    windows = load_and_window_timeseries(csv_path, window_size, stride, column)

    print("Generating recurrence plots...")
    rp_images = np.array([generate_rp_image(w, epsilon, img_size) for w in windows])

    print("Extracting ViT-MAE embeddings...")
    embeddings = get_cls_embeddings(rp_images)

    print("Fitting Mahalanobis model...")
    model = fit_mahalanobis_model(embeddings)

    print("Scoring embeddings...")
    scores = score_mahalanobis(model, embeddings)

    # Save scores
    np.savetxt("anomaly_scores.csv", scores, delimiter=",")
    print("Saved anomaly_scores.csv")

    # Plot and save
    plot_scores(scores)
    print("Saved score_plot.png")

    return scores

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pipeline.py <your_file.csv> <column_index>")
        print("Example: python pipeline.py valve1.csv 1")
        sys.exit(1)

    csv_file = sys.argv[1]
    try:
        col_idx = int(sys.argv[2])
    except ValueError:
        print("Error: <column_index> must be an integer (e.g., 1 for second column).")
        sys.exit(1)

    # Execute
    run_pipeline(csv_file, column=col_idx)
