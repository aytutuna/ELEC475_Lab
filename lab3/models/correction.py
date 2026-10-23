"""
ELEC 475: Lab 3 - Correction Model (Starter Template)
A small CNN that predicts the offset from a patch centre to the true nose.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrectionNet(nn.Module):
    """
    CorrectionNet for (3, 64, 64) ImageNet-normalized patches.

    Spatial dimension arithmetic (each block halves the size with 2x2 max-pooling):
        Input        : (B, 3, 64, 64)
        Block 1      : (B, 32, 32, 32)
        Block 2      : (B, 64, 16, 16)
        Block 3      : (B, 128, 8, 8)
        Block 4      : (B, 128, 4, 4)
        Flatten      : 128 * 4 * 4 = 2,048 features
        FC1          : 2,048 -> 256
        FC2          : 256 -> 2, the normalized offset (du, dv) / HALF_PATCH
    """
    def __init__(self):
        super(CorrectionNet, self).__init__()

        self.block1 = self._block(3, 32)
        self.block2 = self._block(32, 64)
        self.block3 = self._block(64, 128)
        self.block4 = self._block(128, 128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 2)

    @staticmethod
    def _block(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block4(self.block3(self.block2(self.block1(x))))
        x = torch.flatten(x, start_dim=1)
        return self.fc2(F.relu(self.fc1(x)))


class CorrectionNetModified(nn.Module):
    """
    Your modified CorrectionNet (Post-Lab Step 5). Same input (B, 3, 64, 64) and
    output (B, 2) normalized offsets as CorrectionNet.

    Changes from CorrectionNet: two 3x3 conv layers per block instead of one
    (VGG style, larger receptive field before each pooling step) and dropout
    of 0.3 before fc2. The 4x4 spatial map is still flattened, not pooled away,
    because the offset depends on where the nose is in the patch.
    """
    def __init__(self):
        super(CorrectionNetModified, self).__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 128),
        )
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 2)

    @staticmethod
    def _block(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.features(x), start_dim=1)
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


if __name__ == "__main__":
    from torchinfo import summary
    for model in [CorrectionNet(), CorrectionNetModified()]:
        print(f"Testing {type(model).__name__} Architecture:")
        try:
            summary(model, input_size=(1, 3, 64, 64))
        except Exception as e:
            print(f"Model not fully implemented yet: {e}")
