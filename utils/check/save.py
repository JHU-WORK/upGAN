import torch

# Function to save the model state
def save_model(model, epoch, optimizer, filename, checkpoint=False):
    if checkpoint:
        filename = "chk_" + filename
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filename)