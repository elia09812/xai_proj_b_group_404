import torch
import csv
import os
from torch.utils.data import DataLoader

from ml_shared import build_largernet_for_10_classes, ImageDataset, get_device

# 1) Ich setze hier die absoluten Pfade zu Testdaten und Checkpoint
TEST_ROOT = "/Users/dominicschlegel/Documents/WiSe25_26/ProjectML/xai_proj_b_group_404/data/collected_dataset"
CKPT_PATH = "/Users/dominicschlegel/Documents/WiSe25_26/ProjectML/kaggl results/LargerNet/Without Augmentation/LargerNet.pth"

NUM_CLASSES = 10


def main():
    device = get_device()
    print("Device:", device)

    model, preprocess = build_largernet_for_10_classes()

    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    test_ds = ImageDataset(root=TEST_ROOT, transform=preprocess, train=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    correct = 0
    total = 0
    all_rows = []

    with torch.no_grad():
        for images, labels, filenames in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)  # (B, 10)

            pred = probs.argmax(dim=1)
            conf = probs.max(dim=1).values
            p_true = probs[torch.arange(probs.size(0)), labels]

            correct += (pred == labels).sum().item()
            total += labels.size(0)

            probs_cpu = probs.cpu().tolist()

            for i, (fn, yt, yp, cf, pt) in enumerate(
                zip(
                    filenames,
                    labels.cpu().tolist(),
                    pred.cpu().tolist(),
                    conf.cpu().tolist(),
                    p_true.cpu().tolist(),
                )
            ):
                row = [fn, yt, yp, cf, pt] + probs_cpu[i]  # p0..p9
                all_rows.append(row)

    acc = correct / total if total else 0.0
    print(f"Test Accuracy: {acc:.4f} ({correct}/{total})")

    os.makedirs("eval_outputs", exist_ok=True)
    out_csv = "eval_outputs/largernet_predictions.csv"

    header = ["filename", "y_true", "y_pred", "confidence", "p_true"] + [f"p{i}" for i in range(NUM_CLASSES)]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(all_rows)

    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
