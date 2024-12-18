# Peak Signal-to-Noise Ratio (PSNR)
# Definition: https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/

import torch
from torch import nn
from torchvision import models
from torchvision.models import feature_extraction as fe

mse_loss = nn.MSELoss()
bce_logit_loss = nn.BCEWithLogitsLoss()


def calculate_psnr(mse_score: torch.Tensor) -> torch.Tensor:
    if mse_score.item() == 0:
        return 100

    psnr = 20 * (1 / mse_score.sqrt()).log10()
    return psnr


class VGGLoss(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = fe.create_feature_extractor(
            vgg,
            # i_th_maxpooling: int = 5, j_th_convolution: int = 4 => features.34
            return_nodes={"features.34": "features"},
        ).to(device)
        self.feature_extractor.requires_grad_(False)
        self.feature_extractor.eval()

    def forward(self, sr_img: torch.Tensor, hr_img: torch.Tensor) -> torch.Tensor:
        sr_ft_map = self.feature_extractor(sr_img)["features"]
        hr_ft_map = self.feature_extractor(hr_img)["features"]

        # return torch.cdist(sr_ft_map, hr_ft_map)
        # return ((sr_ft_map-hr_ft_map)**2).sum(axis=1).mean()
        return torch.norm(sr_ft_map - hr_ft_map, p=2)
