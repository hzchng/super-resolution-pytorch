from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from src import loss, metrics
from src.device import device


class UtilSRCNN:
    transforms = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    detransforms = T.Compose(
        [
            # T.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            # T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            T.ToPILImage(),
        ]
    )
    scale = 2
    gaussian_radius = 0.55

    def train(
        srcnn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
    ):
        loss_arr = []
        psnr_arr = []

        srcnn.train()
        for lr, hr in loader:
            lr = lr.to(device)
            hr = hr.to(device)

            pred_hr = srcnn(lr)
            loss_score = loss.mse_loss(hr, pred_hr)

            optimizer.zero_grad()
            loss_score.backward()
            optimizer.step()

            loss_item = loss_score.detach()
            loss_arr.append(loss_item.item())
            psnr_arr.append(loss.calculate_psnr(loss_item).item())

        mean_loss = np.mean(loss_arr)
        mean_psnr = np.mean(psnr_arr)
        return mean_loss, mean_psnr

    def eval(
        srcnn: torch.nn.Module,
        loader: DataLoader,
        metric: metrics.MetricSRCNN,
    ):
        loss_arr = []
        psnr_arr = []

        srcnn.eval()
        with torch.no_grad():
            for lr, hr in loader:
                lr = lr.to(device)
                hr = hr.to(device)

                pred_hr = srcnn(lr).detach()
                loss_item = loss.mse_loss(hr, pred_hr)

                loss_arr.append(loss_item.item())
                psnr_arr.append(loss.calculate_psnr(loss_item).item())

        mean_loss = np.mean(loss_arr)
        mean_psnr = np.mean(psnr_arr)
        metric.add_eval(psnr_arr)
        return mean_loss, mean_psnr

    @classmethod
    def inference(
        cls,
        srcnn: torch.nn.Module,
        img_path: Path,
    ) -> torch.Tensor:
        srcnn.eval()
        img = Image.open(img_path)
        lr, hr = cls.downscale(img, scale=cls.scale, gaussian_radius=cls.gaussian_radius)
        with torch.no_grad():
            lr_t = cls.transforms(lr)
            hr_t = cls.transforms(hr)
            pred_hr = srcnn(lr_t.to(device)).detach()
            mse_score = loss.mse_loss(hr_t.cuda(), pred_hr.cuda())
            psnr_score = loss.calculate_psnr(mse_score)
            print(f"mse: {mse_score:.4f}, psnr: {psnr_score:.4f}")

        return lr, hr, pred_hr

    @staticmethod
    def downscale(img: Image.Image, scale: int, gaussian_radius: float = 0.55):
        new_height = (img.height // scale) * scale
        new_width = (img.width // scale) * scale
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

        lr_img = img.filter(ImageFilter.GaussianBlur(radius=gaussian_radius))
        lr_img = lr_img.resize((lr_img.width // scale, lr_img.height // scale), Image.Resampling.BICUBIC)
        lr_img = lr_img.resize((lr_img.width * scale, lr_img.height * scale), Image.Resampling.BICUBIC)

        return lr_img, img
