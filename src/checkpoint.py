from pathlib import Path

import torch

base_dir = Path(__file__).parent
export_dir = base_dir / "model" / "export"
export_dir.mkdir(exist_ok=True)


def save(
    name: str,
    **kwargs,
):
    save_to_file = export_dir / name
    save_to_file.parent.mkdir(exist_ok=True)
    torch.save(kwargs, save_to_file)
    print(f"Model saved to {save_to_file.relative_to(base_dir)}")


def load(name: str, device: torch.device):
    saved_file = export_dir / name
    if not saved_file.exists():
        raise FileNotFoundError(f"Dir not found: {saved_file}")

    checkpoint = torch.load(
        saved_file,
        map_location=device,
        weights_only=False,
    )
    return checkpoint
