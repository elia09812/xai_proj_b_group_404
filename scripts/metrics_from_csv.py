import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

CSV_FILES = [
    "/Users/dominicschlegel/Documents/WiSe25_26/ProjectML/eval_outputs/largernet_predictions_aug.csv",
    "/Users/dominicschlegel/Documents/WiSe25_26/ProjectML/eval_outputs/largernet_predictions.csv",
    "/Users/dominicschlegel/Documents/WiSe25_26/ProjectML/eval_outputs/resnet18_predictions_aug.csv",
    "/Users/dominicschlegel/Documents/WiSe25_26/ProjectML/eval_outputs/resnet18_predictions.csv",
    "/Users/dominicschlegel/Documents/WiSe25_26/ProjectML/eval_outputs/simplenet_predictions_aug.csv",
    "/Users/dominicschlegel/Documents/WiSe25_26/ProjectML/eval_outputs/simplenet_predictions.csv",
]

IDX_TO_CLASS = {
    0: "coffee-mug",
    1: "notebook",
    2: "remote-control",
    3: "soup-bowl",
    4: "teapot",
    5: "wooden-spoon",
    6: "computer-keyboard",
    7: "mouse",
    8: "binder",
    9: "toilet-tissue",
}
N_CLASSES = 10
CLASS_NAMES = [IDX_TO_CLASS[i] for i in range(N_CLASSES)]

OUT_DIR = "/Users/dominicschlegel/Documents/WiSe25_26/ProjectML/eval_outputs/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# Optional: set False if you only want AUC numbers, no ROC curve plots
SAVE_ROC_PLOTS = True


def infer_model_tag(csv_path: str) -> str:
    base = os.path.basename(csv_path).lower()
    base = base.replace(".csv", "").replace("predictions", "").strip("_")
    return base.replace("_", "-")


def load_preds(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, sep=";")


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    thresh = cm.max() * 0.6 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, str(val),
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=12,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_class_accuracy(per_class_acc: Dict[str, float], title: str, out_path: str) -> None:
    labels = list(per_class_acc.keys())
    values = list(per_class_acc.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc_ovr(y_true: np.ndarray, y_score: np.ndarray, title: str, out_path: str) -> None:
    """
    ROC curves for each class (OvR).
    y_score shape: (N, C)
    """
    y_true_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))

    fig, ax = plt.subplots(figsize=(9, 6))
    for c in range(N_CLASSES):
        # need both positives & negatives
        if len(np.unique(y_true_bin[:, c])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_score[:, c])
        ax.plot(fpr, tpr, label=CLASS_NAMES[c])

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def summarize_one(df: pd.DataFrame):
    """
    Returns:
    - summary dict (accuracy, f1, auc...)
    - confusion matrix
    - per-class accuracy dict
    """
    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()

    acc = accuracy_score(y_true, y_pred)

    # You can keep these (or remove later)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))

    per_class_acc = {}
    for i in range(N_CLASSES):
        row_sum = cm[i, :].sum()
        per_class_acc[IDX_TO_CLASS[i]] = cm[i, i] / row_sum if row_sum > 0 else float("nan")

    # --- ROC-AUC (multiclass OvR) ---
    prob_cols = [f"p{i}" for i in range(N_CLASSES)]
    has_probs = all(c in df.columns for c in prob_cols)

    auc_macro = float("nan")
    auc_micro = float("nan")
    auc_per_class = {IDX_TO_CLASS[i]: float("nan") for i in range(N_CLASSES)}
    y_score = None

    if has_probs:
        y_score = df[prob_cols].to_numpy(dtype=float)

        # If they are proper probabilities, rows sum to ~1; otherwise, still works as "scores"
        y_true_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))

        try:
            auc_macro = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")
            auc_micro = roc_auc_score(y_true_bin, y_score, average="micro", multi_class="ovr")
        except ValueError:
            # can happen if a class is missing in y_true
            auc_macro = float("nan")
            auc_micro = float("nan")

        for c in range(N_CLASSES):
            if len(np.unique(y_true_bin[:, c])) < 2:
                auc_per_class[IDX_TO_CLASS[c]] = float("nan")
            else:
                auc_per_class[IDX_TO_CLASS[c]] = roc_auc_score(y_true_bin[:, c], y_score[:, c])

    summary = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "roc_auc_macro_ovr": float(auc_macro),
        "roc_auc_micro_ovr": float(auc_micro),
        "n_samples": int(len(df)),
    }

    return summary, cm, per_class_acc, auc_per_class, y_true, y_score, has_probs


def main():
    all_summaries = []
    per_class_table_rows = []
    auc_per_class_rows = []

    for path in CSV_FILES:
        if not os.path.exists(path):
            print("Missing file, skipping:", path)
            continue

        tag = infer_model_tag(path)
        df = load_preds(path)

        summary, cm, per_class_acc, auc_per_class, y_true, y_score, has_probs = summarize_one(df)
        all_summaries.append({"model": tag, **summary})

        # Confusion matrix plot
        cm_path = os.path.join(OUT_DIR, f"{tag}_confusion_matrix.png")
        plot_confusion_matrix(cm, CLASS_NAMES, f"Confusion Matrix — {tag}", cm_path)

        # Per-class accuracy plot
        pca_path = os.path.join(OUT_DIR, f"{tag}_per_class_accuracy.png")
        plot_per_class_accuracy(per_class_acc, f"Per-class Accuracy — {tag}", pca_path)

        # ROC plot (optional)
        if SAVE_ROC_PLOTS and has_probs and y_score is not None:
            roc_path = os.path.join(OUT_DIR, f"{tag}_roc_ovr.png")
            plot_roc_ovr(y_true, y_score, f"ROC (OvR) — {tag}", roc_path)
        elif SAVE_ROC_PLOTS and not has_probs:
            print(f"[{tag}] No p0..p9 columns found -> skipping ROC plot & AUC.")

        # Long tables
        for cls_name, cls_acc in per_class_acc.items():
            per_class_table_rows.append({"model": tag, "class": cls_name, "class_accuracy": cls_acc})

        for cls_name, cls_auc in auc_per_class.items():
            auc_per_class_rows.append({"model": tag, "class": cls_name, "roc_auc_ovr": cls_auc})

    # Write summary
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"), index=False, sep=";", encoding="utf-8-sig")

    # Write per-class accuracy
    pca_df = pd.DataFrame(per_class_table_rows)
    pca_df.to_csv(os.path.join(OUT_DIR, "per_class_accuracy.csv"), index=False, sep=";", encoding="utf-8-sig")

    # Write per-class AUC
    auc_pc_df = pd.DataFrame(auc_per_class_rows)
    auc_pc_df.to_csv(os.path.join(OUT_DIR, "auc_per_class.csv"), index=False, sep=";", encoding="utf-8-sig")

    print("Done. All outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
 