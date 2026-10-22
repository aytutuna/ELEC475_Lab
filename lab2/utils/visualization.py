"""
ELEC 475: Lab 2 - Plotting & Visualization Utilities
"""
import os
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "./plots/training_curves.png",
    model_name: str = "Model"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r--s', label='Val Loss')
    plt.title(f"{model_name.upper()} - Cross-Entropy Loss", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'g--^', label='Val Accuracy')
    plt.title(f"{model_name.upper()} - Top-1 Accuracy (%)", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training curves successfully saved to: {save_path}")

def plot_ablation_summary(
    ablation_results: Dict[str, Dict[str, List[float]]],
    save_path: str = "./plots/ablation_summary.png"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    styles = ['-', '--', '-.', ':', '-', '--', '-.']

    for idx, (model_label, data) in enumerate(ablation_results.items()):
        val_accs = data.get("val_accs", [])
        epochs = range(1, len(val_accs) + 1)
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        plt.plot(epochs, val_accs, linestyle=style, marker='o', color=color, label=model_label, linewidth=2)

    plt.title("ELEC 475: Master Ablation Study (Validation Accuracy)", fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Master ablation summary plot saved to: {save_path}")


# History files written by train.py --history-file, in master table order
ABLATION_RUNS = [
    ("1. Base LeNet-5", "lenet.json"),
    ("2. Modernized LeNet", "lenet_modern.json"),
    ("3. AlexNet (No Norm)", "alexnet_none.json"),
    ("4. AlexNet + LRN", "alexnet_lrn.json"),
    ("5. AlexNet + BatchNorm", "alexnet_bn.json"),
    ("6. AlexNet + BN + Aug", "alexnet_with_aug.json"),
    ("7. PyTorch Zoo AlexNet", "zoo_alexnet.json"),
]

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="ELEC 475: Lab 2 plotting utilities")
    parser.add_argument("--plot-ablation", action="store_true", help="Plot validation accuracy of all master table runs")
    parser.add_argument("--history-dir", type=str, default="./history/cifar10", help="Directory with train.py history JSON files")
    parser.add_argument("--save-path", type=str, default="./plots/ablation_summary.png")
    args = parser.parse_args()

    if args.plot_ablation:
        results = {}
        for label, fname in ABLATION_RUNS:
            path = os.path.join(args.history_dir, fname)
            if not os.path.exists(path):
                print(f"Skipping '{label}': {path} not found")
                continue
            with open(path) as f:
                results[label] = {"val_accs": json.load(f)["val_accs"]}
        plot_ablation_summary(results, save_path=args.save_path)
