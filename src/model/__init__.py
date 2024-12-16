from pathlib import Path

import torch
from torch import nn

from .srcnn import SRCNN
from .srgan import SRGAN

export_dir = Path(__file__).parent / "export"
export_dir.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export(
    name: str,
    model: nn.Module,
    **kwargs,
):
    export_path = export_dir / name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            **kwargs,
        },
        export_path,
    )
    print(f"Model {name} saved to {export_path}")


def load_checkpoint(name: str, device: torch.device = device):
    export_path = export_dir / name
    if not export_path.exists():
        raise FileNotFoundError(f"Model {name} not found")

    checkpoint = torch.load(
        export_path,
        map_location=device,
        # weights_only=True, # need to update on export function too
        weights_only=False,
    )
    return checkpoint
