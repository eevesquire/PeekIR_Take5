
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = Path(__file__).resolve().parent
DATASET = BASE / "knife_dataset"
OUTPUT = BASE / "ir_knife_model.pth"
LABELS = ["no_knife", "knife"]

class ThermalDataset(Dataset):
    def __init__(self):
        self.samples = []
        for label_id, label in enumerate(LABELS):
            for path in sorted((DATASET / label).glob("*.png")):
                self.samples.append((path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("L").resize((160, 120))
            pixels = np.asarray(image, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(pixels).unsqueeze(0)
        return tensor, label

class ThermalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def main():
    dataset = ThermalDataset()
    labels = np.array([label for _, label in dataset.samples])
    indices = np.arange(len(dataset))

    if len(dataset) < 20 or len(set(labels)) != 2:
        raise ValueError("Need sufficient images from both classes.")

    # Initial experimental split.
    # Replace with session-based splitting when session IDs are available.
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=labels, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=labels[temp_idx],
        random_state=SEED
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=8, shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=8
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=8
    )

    counts = np.bincount(labels[train_idx], minlength=2)
    weights = len(train_idx) / (2 * counts)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32)
    )

    model = ThermalCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_state = None

    print(f"Total: {len(dataset)}")
    print(f"Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}")

    for epoch in range(20):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(targets)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                loss = loss_fn(model(images), targets)
                val_loss += loss.item() * len(targets)

        val_loss /= len(val_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch + 1:02d}/20 | "
            f"Train loss: {running_loss / len(train_idx):.4f} | "
            f"Validation loss: {val_loss:.4f}"
        )

    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": model.state_dict(),
        "labels": LABELS,
        "image_size": (160, 120),
        "preprocessing": "grayscale_0_to_1"
    }, OUTPUT)

    actual, predicted = [], []
    model.eval()

    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images).argmax(dim=1)
            actual.extend(targets.tolist())
            predicted.extend(outputs.tolist())

    print("\nExperimental test results:")
    print(classification_report(
        actual, predicted, labels=[0, 1],
        target_names=LABELS, zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))
    print(f"\nSaved model: {OUTPUT}")

if __name__ == "__main__":
    main()
