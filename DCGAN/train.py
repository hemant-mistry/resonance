import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from engine import Generator, Discriminator
import os

latent_dim = 100
batch_size = 128
num_epochs = 50
learning_rate = 0.0002

# Auto-detect hardware (This will now grab your RTX 3050!)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

print("Downloading/Loading CIFAR-10 dataset...")
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 3. Initialize Models ---
generator = Generator(latent_dim=latent_dim, channels=3).to(device)
discriminator = Discriminator(channels=3).to(device)

# --- 4. Loss Function and Optimizers ---
criterion = nn.BCELoss() # Binary Cross Entropy (Real vs Fake probability)

# Custom Adam parameters (betas) to prevent the Discriminator from learning too fast
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

print("Starting Training Loop..")
for epoch in range(num_epochs):
    for i,data in enumerate(dataloader):

        real_images = data[0].to(device)
        current_batch_size = real_images.size(0)

        real_labels = torch.ones(current_batch_size, 1).to(device)
        fake_labels = torch.zeros(current_batch_size, 1).to(device)

        optimizer_D.zero_grad()

        outputs_real = discriminator(real_images)
        d_loss_real = criterion(outputs_real, real_labels)

        noise = torch.randn(current_batch_size, latent_dim, 1,1).to(device)
        fake_images = generator(noise)

        outputs_fake = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # --- Logging ---
        if i % 200 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
            
# --- 6. Save the final model ---
os.makedirs("weights", exist_ok=True)
torch.save(generator.state_dict(), "weights/resonance_generator.pth")
print("\nTraining Complete! Generator weights saved to weights/resonance_generator.pth")