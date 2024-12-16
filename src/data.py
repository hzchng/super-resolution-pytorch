from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T

from . import image


def get_images_files(path: Path, suffix: str = "", is_recursive: bool = False) -> list[Path]:
    if is_recursive:
        files = [file for file in path.rglob(f"*{suffix}") if file.is_file()]
    else:
        files = [file for file in path.glob(f"*{suffix}") if file.is_file()]

    if not files:
        raise FileNotFoundError(f"No images found in {path.resolve()}")

    return files


class SRCNNData(Dataset):
    def __init__(self, root_dir: Path | str, transform=None):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir
        self.transform = transform
        self.files = get_images_files(
            root_dir,
            suffix=".png",
            is_recursive=True,
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        img = Image.open(self.files[index])
        lr_img, img = image.srcnn_img_preprocess(img)

        if self.transform:
            img = self.transform(img)
            lr_img = self.transform(lr_img)

        return lr_img, img


class SRGANData(Dataset):
    def __init__(
        self,
        image_path: Path,
        hr_name: str,
        lr_name: str,
        hr_transform: T.Compose | None = None,
        lr_transform: T.Compose | None = None,
    ):
        super().__init__()

        if not hr_transform:
            hr_transform = T.Compose(
                [
                    T.Resize((256, 256)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        if not lr_transform:
            lr_transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        self.hr_transform = hr_transform
        self.lr_transform = lr_transform

        self.hr_images = get_images_files(image_path / hr_name)
        self.lr_images = get_images_files(image_path / lr_name)

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        hr_image = Image.open(self.hr_images[idx])
        lr_image = Image.open(self.lr_images[idx])

        hr_image = self.hr_transform(hr_image)
        lr_image = self.lr_transform(lr_image)

        return hr_image, lr_image
