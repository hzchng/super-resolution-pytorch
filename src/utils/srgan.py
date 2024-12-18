from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from src import loss, metrics
from src.device import device


class UtilSRGAN:
    lr_transforms = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    lr_detransforms = T.Compose(
        [
            # T.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            # T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            T.ToPILImage(),
        ]
    )
    hr_transforms = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    hr_detransforms = T.Compose(
        [
            T.Normalize(mean=[0, 0, 0], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
            T.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1]),
            T.ToPILImage(),
        ]
    )
    init_transform = T.Compose(
        [
            T.RandomCrop(96),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ]
    )
    init_eval_transform = T.Compose(
        [
            T.RandomCrop(96),
        ]
    )
    scale = 4
    gaussian_radius = 0.55
    vgg_loss = loss.VGGLoss(device=device)

    def train(
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        gen_optimizer: torch.optim.Optimizer,
        disc_optimizer: torch.optim.Optimizer,
        loader: DataLoader,
    ):
        gen_loss_arr = []
        disc_loss_arr = []
        psnr_arr = []

        generator.train()
        discriminator.train()
        for lr, hr in loader:
            lr = lr.to(device)
            hr = hr.to(device)

            label_real = torch.ones((hr.shape[0], 1), device=device)
            label_fake = torch.zeros((hr.shape[0], 1), device=device)

            # training generator
            gen_optimizer.zero_grad()

            sr = generator(lr)
            discriminator_fake = discriminator(sr)
            adversarial_loss = loss.bce_logit_loss(discriminator_fake, label_real)
            # gen_loss = vgg_loss(sr, hr) / 12.75  # perceptual loss
            mse_loss = loss.mse_loss(sr, hr)
            gen_loss = mse_loss + (adversarial_loss * 1e-3)
            gen_loss.backward()
            gen_optimizer.step()

            # training discriminator
            disc_optimizer.zero_grad()

            ## real
            discriminator_real = discriminator(hr)
            discriminator_real_loss = loss.bce_logit_loss(discriminator_real, label_real)

            ## fake
            discriminator_fake = discriminator(sr.detach())
            discriminator_fake_loss = loss.bce_logit_loss(discriminator_fake, label_fake)
            discriminator_loss = (discriminator_real_loss + discriminator_fake_loss) / 2
            discriminator_loss.backward()
            disc_optimizer.step()

            discriminator_loss_item = discriminator_loss.detach()
            gen_loss_item = gen_loss.detach()
            disc_loss_arr.append(discriminator_loss_item.item())
            gen_loss_arr.append(gen_loss_item.item())
            psnr_arr.append(loss.calculate_psnr(mse_loss.detach()).item())

        mean_gen_loss = np.mean(gen_loss_arr)
        mean_disc_loss = np.mean(disc_loss_arr)
        mean_psnr = np.mean(psnr_arr)
        return mean_gen_loss, mean_disc_loss, mean_psnr

    def eval(
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        loader: DataLoader,
        metric: metrics.MetricSRCNN,
    ):
        gen_loss_arr = []
        psnr_arr = []

        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            for lr, hr in loader:
                lr = lr.to(device)
                hr = hr.to(device)

                label_real = torch.ones((hr.shape[0], 1), device=device)

                sr = generator(lr)
                discriminator_fake = discriminator(sr)
                adversarial_loss = loss.bce_logit_loss(discriminator_fake, label_real)
                mse_loss = loss.mse_loss(sr, hr)
                gen_loss = mse_loss + (adversarial_loss * 1e-3)

                gen_loss_item = gen_loss.detach()
                gen_loss_arr.append(gen_loss_item.item())
                psnr_arr.append(loss.calculate_psnr(mse_loss.detach()).item())

        gen_loss_arr = np.mean(gen_loss_arr)
        mean_psnr = np.mean(psnr_arr)
        metric.add_eval(psnr_arr)
        return gen_loss_arr, mean_psnr

    @classmethod
    def inference(
        cls,
        generator: torch.nn.Module,
        img_path: Path,
    ) -> torch.Tensor:
        img = Image.open(img_path)
        lr, hr = cls.downscale(img, scale=cls.scale, gaussian_radius=cls.gaussian_radius)

        generator.eval()
        with torch.no_grad():
            lr_t = cls.lr_transforms(lr).to(device).unsqueeze(0)
            hr_t = cls.hr_transforms(hr).to(device)
            hr_pred = generator(lr_t).squeeze(0)
            mse_score = loss.mse_loss(hr_t, hr_pred)
            psnr_score = loss.calculate_psnr(mse_score)
            print(f"mse: {mse_score:.4f}, psnr: {psnr_score:.4f}")

        return lr_t.squeeze(0), hr_t, hr_pred

    @staticmethod
    def downscale(img: Image.Image, scale: int, gaussian_radius: float = 0.55):
        new_height = img.height // scale
        new_width = img.width // scale

        lr_img = img.filter(ImageFilter.GaussianBlur(radius=gaussian_radius))
        lr_img = lr_img.resize((new_width, new_height), Image.Resampling.BICUBIC)

        return lr_img, img
