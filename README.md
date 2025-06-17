# ViT-MAE Anomaly Detector

A plug-and-play, unsupervised anomaly detection toolkit for univariate time-series.  
It converts your CSV data into recurrence plots, extracts features with a pretrained ViT-MAE, and flags outliers via Mahalanobis distance.

---
## Prepare your CSV ###3

Place your time-series in the repo folder (comma- or semicolon-separated).
Make sure it has at least one numeric column.

#### Run the pipeline ####
bash

Copy

Edit

python pipeline.py <your_file.csv> <column_index>

<your_file.csv>: name of your CSV

<column_index>: zero-based index of the numeric column to analyze

### View results ####
anomaly_scores.csv: Mahalanobis score per window
score_plot.png: timeline of anomaly scores

## 📦 Installation & Dependencies

1. **Clone the repo**  
   ```bash
   git clone https://github.com/iftitalukder/vitmae-anomaly-detector.git
   cd vitmae-anomaly-detector

## 📦 Dependencies

Install all required Python packages:

```bash
pip install torch torchvision timm scikit-learn scikit-image pandas numpy matplotlib



