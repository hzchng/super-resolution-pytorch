import re
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from src.utils import UtilSRCNN, UtilSRGAN


def get_images_files(path: Path, suffix: str = "", is_recursive: bool = False) -> list[Path]:
    if is_recursive:
        files = [file for file in path.rglob(f"*{suffix}") if file.is_file()]
    else:
        files = [file for file in path.glob(f"*{suffix}") if file.is_file()]

    if not files:
        raise FileNotFoundError(f"No images found in {path.resolve()}")

    return files


class SRCNNData(Dataset):
    def __init__(
        self,
        root_dir: Path | str,
        transform=None,
        scale: int = 2,
        gaussian_radius: float = 2,
    ):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir
        self.files = get_images_files(
            root_dir,
            suffix=".png",
            is_recursive=True,
        )
        self.transform = transform
        self.scale = scale
        self.gaussian_radius = gaussian_radius

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        img = Image.open(self.files[index])
        lr_img, img = UtilSRCNN.downscale(img, self.scale, self.gaussian_radius)

        if self.transform:
            img = self.transform(img)
            lr_img = self.transform(lr_img)

        return lr_img, img


class SRGANData(Dataset):
    def __init__(
        self,
        root_dir: Path,
        scale: int,
        hr_name: str,
        init_transfrom: T.Compose | None,
        hr_transform: T.Compose,
        lr_transform: T.Compose,
        gaussian_radius: float = 2,
    ):
        super().__init__()

        self.scale = scale
        self.root_dir = root_dir
        self.hr_files = get_images_files(root_dir / hr_name, suffix=".png", is_recursive=True)

        self.init_transfrom = init_transfrom
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        self.gaussian_radius = gaussian_radius

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        img = Image.open(self.hr_files[idx])
        if self.init_transfrom:
            img = self.init_transfrom(img)
        lr_img = UtilSRGAN.downscale(img, self.scale, self.gaussian_radius)

        img = self.hr_transform(img)
        lr_img = self.lr_transform(lr_img)
        return lr_img, img
