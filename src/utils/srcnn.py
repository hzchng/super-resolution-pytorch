from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from src import device, loss, metrics


class UtilSRCNN:
    transforms = T.Compose(
        [
            # T.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
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
            lr = lr.to(device.type)
            hr = hr.to(device.type)

            with torch.autocast(device_type=device.type.type, dtype=device.dtype):
                pred_hr = srcnn(lr)
                loss_score = loss.get_mse_loss(hr, pred_hr)

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
        with torch.inference_mode(), torch.autocast(device_type=device.type.type, dtype=device.dtype):
            for lr, hr in loader:
                lr = lr.to(device.type)
                hr = hr.to(device.type)

                pred_hr = srcnn(lr)
                loss_item = loss.get_mse_loss(hr, pred_hr)

                loss_arr.append(loss_item.item())
                psnr_arr.append(loss.calculate_psnr(loss_item).item())

        mean_loss = np.mean(loss_arr)
        mean_psnr = np.mean(psnr_arr)
        metric.add_eval(psnr_arr)
        return mean_loss, mean_psnr

    @classmethod
    def inference_blurred(
        cls,
        srcnn: torch.nn.Module,
        img_path: Path,
    ) -> torch.Tensor:
        srcnn.eval()
        img = Image.open(img_path)
        lr, hr = cls.downscale(img, scale=cls.scale, gaussian_radius=cls.gaussian_radius)
        pred_hr, mse_score, psnr_score = cls._infer(srcnn, lr, hr)
        return lr, hr, pred_hr, mse_score, psnr_score

    @classmethod
    def inference(
        cls,
        srcnn: torch.nn.Module,
        img_path: Path,
    ) -> torch.Tensor:
        srcnn.eval()
        img = Image.open(img_path)
        pred_hr, mse_score, psnr_score = cls._infer(srcnn, img)
        return img, pred_hr, mse_score, psnr_score

    @classmethod
    def _infer(
        cls,
        srcnn: torch.nn.Module,
        lr_img: Image.Image,
        hr_img: Image.Image | None = None,
    ) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(device_type=device.type.type, dtype=device.dtype):
            base_img = cls.transforms(lr_img).to(device.type)
            truth_img = base_img
            if hr_img is not None:
                truth_img = cls.transforms(lr_img).to(device.type)

            pred_img = srcnn(base_img)
            mse_score = loss.get_mse_loss(truth_img, pred_img)
            psnr_score = loss.calculate_psnr(mse_score)
            print(f"mse: {mse_score:.4f}, psnr: {psnr_score:.4f}")

        return pred_img, mse_score, psnr_score

    @staticmethod
    def downscale(img: Image.Image, scale: int, gaussian_radius: float = 0.55):
        new_height = (img.height // scale) * scale
        new_width = (img.width // scale) * scale
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

        lr_img = img.filter(ImageFilter.GaussianBlur(radius=gaussian_radius))
        lr_img = lr_img.resize((lr_img.width // scale, lr_img.height // scale), Image.Resampling.BICUBIC)
        lr_img = lr_img.resize((lr_img.width * scale, lr_img.height * scale), Image.Resampling.BICUBIC)

        return lr_img, img
