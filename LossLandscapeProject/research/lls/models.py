import torch
import torch.nn as nn


class MLP(nn.Module):
    """Same family as the original study: in -> 128 -> 64 -> 10, ReLU."""

    def __init__(self, in_dim, hidden=(128, 64), n_classes=10):
        super().__init__()
        dims = [in_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


class MLPBN(nn.Module):
    """in -> 128 -> 64 -> 10 with BatchNorm after each hidden linear layer (pre-BN weights are
    scale-invariant per output unit; BN affine and the next layer form a ReLU rescaling pair)."""

    def __init__(self, in_dim, hidden=(128, 64), n_classes=10):
        super().__init__()
        dims = [in_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b, bias=False), nn.BatchNorm1d(b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


class SmallCNN(nn.Module):
    """Small conv net without normalization layers (keeps the landscape analysis clean)."""

    def __init__(self, in_ch=3, n_classes=10, width=32, img=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(2 * width * (img // 4) ** 2, 128), nn.ReLU(), nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.head(self.features(x))


def build(arch, dataset):
    in_ch, img = (1, 28) if dataset in ("mnist", "fashion") else (3, 32)
    if arch == "mlp":
        return MLP(in_ch * img * img)
    if arch == "mlpbn":
        return MLPBN(in_ch * img * img)
    if arch == "cnn":
        return SmallCNN(in_ch=in_ch, img=img)
    raise ValueError(arch)
