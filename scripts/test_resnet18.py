import torch
import csv
import os
from torch.utils.data import DataLoader

from ml_shared import build_resnet18_for_10_classes, ImageDataset, get_device

# 1) die absoluten Pfade zu Testdaten und Checkpoint
TEST_ROOT = "/Users/dominicschlegel/Documents/WiSe25_26/ProjectML/xai_proj_b_group_404/data/collected_dataset"
CKPT_PATH = "/Users/dominicschlegel/Documents/WiSe25_26/ProjectML/kaggl results/ResNet18/Without Augmentation/ResNet18.pth"

NUM_CLASSES = 10


def main():
    device = get_device()
    print("Device:", device)

    model, preprocess = build_resnet18_for_10_classes()

    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    test_ds = ImageDataset(root=TEST_ROOT, transform=preprocess, train=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    correct = 0
    total = 0
    all_rows = []  # pro Bild eine Zeile für CSV

    with torch.no_grad():
        for images, labels, filenames in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)  # (B, 10)

            pred = probs.argmax(dim=1)
            conf = probs.max(dim=1).values
            p_true = probs[torch.arange(probs.size(0)), labels]
            p_pred = probs[torch.arange(probs.size(0)), pred]  # == conf

            correct += (pred == labels).sum().item()
            total += labels.size(0)

            probs_cpu = probs.cpu().tolist()

            for i, (fn, yt, yp, cf, pt, pp) in enumerate(
                zip(
                    filenames,
                    labels.cpu().tolist(),
                    pred.cpu().tolist(),
                    conf.cpu().tolist(),
                    p_true.cpu().tolist(),
                    p_pred.cpu().tolist(),
                )
            ):
                # row: base cols + p0..p9
                row = [fn, yt, yp, cf, pt, pp] + probs_cpu[i]
                all_rows.append(row)

    acc = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {acc:.4f} ({correct}/{total})")

    # CSV speichern (Excel-DE friendly: Semikolon + utf-8-sig)
    out_dir = "eval_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "resnet18_predictions.csv")

    header = ["filename", "y_true", "y_pred", "confidence", "p_true", "p_pred"] + [f"p{i}" for i in range(NUM_CLASSES)]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(all_rows)

    print("Saved predictions to:", out_csv)


if __name__ == "__main__":
    main()
