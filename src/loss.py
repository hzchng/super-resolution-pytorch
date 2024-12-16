# Peak Signal-to-Noise Ratio (PSNR)
# Definition: https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/

from torch import Tensor, nn

mse_loss = nn.MSELoss()


def calculate_psnr(mse_score: float) -> Tensor:
    if mse_score == 0:
        return 100

    psnr = 20 * (1 / mse_score.sqrt()).log10()
    return psnr
