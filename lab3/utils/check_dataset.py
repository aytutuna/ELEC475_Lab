"""
Pre-Lab Task 3, Step 2: checks every sample of both splits and plots eight
training images with their nose labels. Run from the lab folder:
    python -m utils.check_dataset
"""
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.dataset import PetNoseDataset, denormalize, IMG_SIZE

if __name__ == "__main__":
    for split in ["train", "test"]:
        dataset = PetNoseDataset(split=split)
        loader = DataLoader(dataset, batch_size=64)
        for images, labels, orig_size in loader:
            assert images.shape[1:] == (3, IMG_SIZE, IMG_SIZE), images.shape
            assert (labels >= 0).all() and (labels <= IMG_SIZE).all(), labels
        print(f"{split}: {len(dataset)} samples OK")

    dataset = PetNoseDataset(split="train")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        image, label, orig_size = dataset[i]
        ax.imshow(denormalize(image).permute(1, 2, 0))
        ax.plot(label[0], label[1], "go")
        ax.set_title(dataset.samples[i][0], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("plots/dataset_check.png")
    print("Saved plots/dataset_check.png")
