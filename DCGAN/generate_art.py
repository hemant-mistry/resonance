import torch
import matplotlib.pyplot as plt
# NOTE: If your file is named engine.py instead of models.py, change this import!
from engine import Generator 

def generate_and_show():
    print("Loading Trained Generator...")
    
    # 1. Setup
    latent_dim = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Initialize the brain and load the trained memories
    gen = Generator(latent_dim=latent_dim, channels=3).to(device)
    
    # map_location ensures it loads safely regardless of hardware
    gen.load_state_dict(torch.load("weights/resonance_generator.pth", map_location=device, weights_only=True))
    
    # 🌟 CRITICAL: Turn eval mode ON! 
    # The model is trained now, so we want it to use the variances it memorized.
    gen.eval() 

    # 3. Create 16 brand new random seed vectors
    print("Forging new images...")
    noise = torch.randn(16, latent_dim).to(device)

    # 4. Generate the images (No gradients needed for inference)
    with torch.no_grad():
        fake_images = gen(noise)
        
        # Move the images out of the GPU and back to system RAM for Matplotlib
        fake_images = fake_images.cpu() 

    # 5. Set up the Matplotlib UI
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    fig.suptitle("Trained DCGAN Output (CIFAR-10 Fakes)", fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flatten()):
        # Grab a single image tensor from the batch
        img = fake_images[i]
        
        # Un-normalize from [-1.0, 1.0] back to [0.0, 1.0]
        img = (img + 1.0) / 2.0
        
        # Permute from PyTorch (Channels, Height, Width) to Matplotlib (Height, Width, Channels)
        img_np = img.permute(1, 2, 0).numpy()
        
        # Draw it
        ax.imshow(img_np)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_and_show()