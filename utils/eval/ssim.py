import torch
from skimage.metrics import structural_similarity

def ssim(outputs, targets):
    score = 0
    for out, tar in zip(outputs, targets):
        score += ssim_score(out, tar)
    return score / len(outputs)

def ssim_score(image1, image2):
    # Ensure the input images are PyTorch tensors
    if not (isinstance(image1, torch.Tensor) and isinstance(image2, torch.Tensor)):
        raise TypeError("image1 and image2 should be PyTorch tensors")
    
    # Ensure the both images have the same dimentions
    if image1.shape != image2.shape:
        raise ValueError("image1 and image2 should have the same dimentions")

    # Convert PyTorch tensors to NumPy arrays
    img1 = image1.permute(1, 2, 0).cpu().detach().numpy()
    img2 = image2.permute(1, 2, 0).cpu().detach().numpy()

    # Calculate the Structural Similarity Index (SSIM) between two images
    return structural_similarity(img1, img2, channel_axis=-1, data_range=255)
