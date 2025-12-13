import os
import pandas as pd
import matplotlib.pyplot as plt


FILE_PATH = "./model_artifacts/training_metrics.csv"

def load_metrics(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext} (use .csv)")
    return df


df = load_metrics(FILE_PATH)

df = df.copy()
df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
df = df.dropna(subset=["epoch"]).sort_values("epoch")

plt.figure(figsize=(9, 5))
plt.plot(df["epoch"], df["train_loss"], marker="o", linewidth=2, label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], marker="o", linewidth=2, label="Validation Loss")

plt.title("Training vs Validation Loss by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.legend()

loss_img_name = "./model_artifacts/fig_training_loss_curves.png"
plt.tight_layout()
plt.savefig(loss_img_name, dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(df["epoch"], df["accuracy"], marker="o", linewidth=2, label="Accuracy")
plt.plot(df["epoch"], df["precision"], marker="o", linewidth=2, label="Precision")
plt.plot(df["epoch"], df["recall"], marker="o", linewidth=2, label="Recall")
plt.plot(df["epoch"], df["f1"], marker="o", linewidth=2, label="F1 Score")

plt.title("Classification Metrics by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend(ncol=2)

metrics_img_name = "./model_artifacts/fig_training_classification_metrics.png"
plt.tight_layout()
plt.savefig(metrics_img_name, dpi=300, bbox_inches="tight")
plt.close()
