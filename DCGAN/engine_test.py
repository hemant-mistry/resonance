import torch
import matplotlib.pyplot as plt
from engine import Generator

# 1. Initialize the untrained Generator
gen = Generator(latent_dim=100, channels=3)
# gen.eval() # Set to evaluation mode

# 2. Create 16 random noise vectors
noise = torch.randn(16, 100)

# 3. Generate the dummy images (turn off gradients for speed)
with torch.no_grad():
    fake_images = gen(noise)

# 4. Set up the Matplotlib UI
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
fig.suptitle("Untrained DCGAN Output (Colored Static)", fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flatten()):
    # Grab a single image tensor from the batch
    img = fake_images[i]
    
    # 🚨 CRITICAL MATH STEP 🚨
    # Tanh outputs values from -1.0 to 1.0. 
    # Matplotlib strictly requires RGB colors to be between 0.0 and 1.0.
    # We mathematically shift the values back to a positive range.
    img = (img + 1.0) / 2.0
    
    # PyTorch structures images as (Channels, Height, Width) -> [3, 32, 32]
    # Matplotlib expects (Height, Width, Channels) -> [32, 32, 3]
    # .permute() rearranges the dimensions to keep Matplotlib happy!
    img_np = img.permute(1, 2, 0).numpy()
    
    # Draw it
    ax.imshow(img_np)
    ax.axis('off')

plt.tight_layout()
plt.show()