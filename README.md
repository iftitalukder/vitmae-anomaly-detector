# vitmae-anomaly-detector
# ViT-MAE for Unsupervised Anomaly Detection in Time Series

This repository contains the code for my IEEE paper:  
**"Vision Transformer Masked Autoencoders for Unsupervised Anomaly Detection in Time Series"**

## 🔍 Overview
- Converts time-series to recurrence plots
- Uses ViT-MAE (Vision Transformer Masked Autoencoder)
- Scores anomalies using Mahalanobis distance on [CLS] token embeddings
- Supports real-time inference, cross-domain generalization

## 🚀 Getting Started

```bash
pip install -r requirements.txt
