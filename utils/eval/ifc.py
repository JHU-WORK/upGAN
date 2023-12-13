import torch
import pywt
import numpy as np

def ifc(outputs, targets):
    score = 0
    for out, tar in zip(outputs, targets):
        score += ifc_score(out, tar)
    return score / len(outputs)

def ifc_score(image1, image2):
    # Step 0: Preprocessing Tensor Images
    # Ensure the input images are PyTorch tensors
    if not (isinstance(image1, torch.Tensor) and isinstance(image2, torch.Tensor)):
        raise TypeError("image1 and image2 should be PyTorch tensors")
    
    # Ensure the both images have the same dimentions
    if image1.shape != image2.shape:
        raise ValueError("image1 and image2 should have the same dimentions")

    # Convert PyTorch tensors to NumPy arrays
    img1 = image1.permute(1, 2, 0).cpu().detach().numpy()
    img2 = image2.permute(1, 2, 0).cpu().detach().numpy()

    # Step 1: Wavelet Decomposition
    # Decompose both images using a wavelet transform.
    # PyWavelets can be used, but you need to integrate it with PyTorch tensors.
    coeffs1 = pywt.dwt2(img1, 'haar')
    coeffs2 = pywt.dwt2(img2, 'haar')

    # Extract the LL, LH, HL, HH components
    LL1, (LH1, HL1, HH1) = coeffs1
    LL2, (LH2, HL2, HH2) = coeffs2

    # Convert components to PyTorch tensors
    LL1, LL2 = torch.tensor(LL1), torch.tensor(LL2)
    LH1, LH2 = torch.tensor(LH1), torch.tensor(LH2)
    HL1, HL2 = torch.tensor(HL1), torch.tensor(HL2)
    HH1, HH2 = torch.tensor(HH1), torch.tensor(HH2)

    # Step 2: Calculate Error Sensitivity Functions for each component
    # You might need to convert the arrays to PyTorch tensors again
    LL_sensitivity = torch.var(LL1 - LL2, dim=[1, 2], unbiased=False).unsqueeze(-1).unsqueeze(-1)
    LH_sensitivity = torch.var(LH1 - LH2, dim=[1, 2], unbiased=False).unsqueeze(-1).unsqueeze(-1)
    HL_sensitivity = torch.var(HL1 - HL2, dim=[1, 2], unbiased=False).unsqueeze(-1).unsqueeze(-1)
    HH_sensitivity = torch.var(HH1 - HH2, dim=[1, 2], unbiased=False).unsqueeze(-1).unsqueeze(-1)

    # Step 3: Calculate Mutual Information for each component
    # Similar process for each component
    LL_mutual_info = LL_sensitivity * ((LL1 - LL2)**2)
    LH_mutual_info = LH_sensitivity * ((LH1 - LH2)**2)
    HL_mutual_info = HL_sensitivity * ((HL1 - HL2)**2)
    HH_mutual_info = HH_sensitivity * ((HH1 - HH2)**2)

    # Step 4: Aggregate IFC Score
    # Sum or average the mutual information values from all components
    ifc_score = torch.mean(LL_mutual_info + LH_mutual_info + HL_mutual_info + HH_mutual_info)

    return ifc_score
