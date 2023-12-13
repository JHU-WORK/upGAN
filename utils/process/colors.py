#TAKEN FROM https://github.com/TheZino/pytorch-color-conversions/blob/master/colors.py
import numpy as np
import torch
from scipy import linalg


##### RGB - YCbCr

# Helper for the creation of module-global constant tensors
def _t(data):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # TODO inherit this
    device = torch.device("cpu") # TODO inherit this
    return torch.tensor(data, requires_grad=False, dtype=torch.float32, device=device)

# Helper for color matrix multiplication
def _mul(coeffs, image):
    # This is implementation is clearly suboptimal.  The function will
    # be implemented with 'einsum' when a bug in pytorch 0.4.0 will be
    # fixed (Einsum modifies variables in-place #7763).
    return torch.einsum("dc,bcij->bdij", (coeffs.to(image.device), image))

_RGB_TO_YCBCR = _t([[0.257, 0.504, 0.098], [-0.148, -0.291, 0.439], [0.439 , -0.368, -0.071]])
_YCBCR_OFF = _t([0.063, 0.502, 0.502]).view(1, 3, 1, 1)


def rgb2ycbcr(rgb):
    """sRGB to YCbCr conversion."""
    clip_rgb=False
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    return _mul(_RGB_TO_YCBCR, rgb) + _YCBCR_OFF.to(rgb.device)


def ycbcr2rgb(rgb):
    """YCbCr to sRGB conversion."""
    clip_rgb=False
    rgb = _mul(torch.inverse(_RGB_TO_YCBCR), rgb - _YCBCR_OFF.to(rgb.device))
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    return rgb


