# gan_mnist.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# --- 1. Setup and Hyperparameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
lr = 0.0002
batch_size = 128
image_size = 64 # We'll resize the 28x28 MNIST images to 64x64
img_channels = 1
latent_dim = 100 # Size of the noise vector for the Generator
num_epochs = 10

# --- 2. Data Loading ---
# We'll use the MNIST dataset
transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(img_channels)], [0.5 for _ in range(img_channels)]),
])

dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 3. Define the Models ---

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 8 x 8
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 16 x 16
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size. 32 x 32 x 32
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (img_channels) x 64 x 64
        )

    def forward(self, x):
        return self.net(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # input is (img_channels) x 64 x 64
            nn.Conv2d(img_channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 32 x 32 x 32
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 4 x 4
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)


# --- 4. Instantiate Models, Loss, and Optimizers ---
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

# For saving generated images
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
if not os.path.exists('gan_images'):
    os.makedirs('gan_images')

# --- 5. Training Loop ---
print("--- Starting Training ---")
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        
        # --- Train Discriminator ---
        discriminator.zero_grad()
        # Train with real images
        label = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
        output = discriminator(real_images)
        lossD_real = criterion(output, label)
        lossD_real.backward()

        # Train with fake images
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        label.fill_(0.0)
        output = discriminator(fake_images.detach())
        lossD_fake = criterion(output, label)
        lossD_fake.backward()
        
        lossD = lossD_real + lossD_fake
        optimizerD.step()

        # --- Train Generator ---
        generator.zero_grad()
        label.fill_(1.0) # Generator wants the discriminator to think the fake images are real
        output = discriminator(fake_images)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], '
                  f'Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}')

    # Save generated images at the end of each epoch
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    torchvision.utils.save_image(fake, f'gan_images/fake_images_epoch_{epoch}.png', normalize=True)

print("--- Training Finished ---")
