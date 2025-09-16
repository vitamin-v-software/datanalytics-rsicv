import torch
import torchvision.models as models
import os

# Define save path
save_dir = './samples/pretrained_models/'

os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'epoch_001.pth.tar')

# Load pretrained VGG11 model
model = models.vgg11(pretrained=True)

# Optional: create a dummy optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Save model checkpoint
torch.save({
    'epoch': 1,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),  # Optional, can be removed if not needed
}, save_path)

print(f"Checkpoint saved to: {save_path}")

