# Paper: https://arxiv.org/pdf/1501.00092
from torch import nn


class SRCNN(nn.Module):
    def __init__(
        self,
        n1: int = 64,
        n2: int = 32,
        f1: int = 9,
        f2: int = 1,
        f3: int = 5,
    ):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                3,
                n1,
                kernel_size=f1,
                padding=f1 // 2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                n1,
                n2,
                kernel_size=f2,
                padding=f2 // 2,
            ),
            nn.ReLU(),
        )
        self.block_final = nn.Conv2d(
            n2,
            3,
            kernel_size=f3,
            padding=f3 // 2,
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_final(x)
        x = x.clamp(min=0.0, max=1.0)
        return x
