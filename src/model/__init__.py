from pathlib import Path

from .srcnn import SRCNN
from .srgan import Discriminator, Generator

export_folder = Path(__file__).parent / "export"


def get_trained_models(model_name: str):
    models_folder = export_folder.rglob(f"*{model_name}*")
    return [folder.name for folder in models_folder if folder.is_dir()]


def get_model_weights(model_name: str):
    models_folder = (export_folder / model_name).rglob("*.pt")
    return [file.name for file in models_folder if file.is_file()]
