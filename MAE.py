import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        
        # Recursively find all images in the root directory
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        for ext in valid_extensions:
            self.image_paths.extend(list(self.root_dir.rglob(f'*{ext}')))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        # Convert to grayscale if image is RGB
        if image.mode != 'L':
            image = image.convert('L')
            
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # Return 0 as dummy label

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        H, W = self.window_size
        B, C, h, w = x.shape
        
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        x = x.transpose(1, 2).reshape(B, C, h, w)
        return x

class SwinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial convolution adjusted for 64x64 input
        self.init_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        
        # Adjusted window sizes for 64x64 input
        self.stage1 = nn.Sequential(
            SwinTransformerBlock(dim=64, num_heads=4, window_size=(64, 64)),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # Output: 48x48
        )
        
        self.stage2 = nn.Sequential(
            SwinTransformerBlock(dim=64, num_heads=8, window_size=(48, 48)),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Output: 24x24
        )
        
        self.stage3 = SwinTransformerBlock(dim=128, num_heads=16, window_size=(24, 24))

    def forward(self, x):
        x = self.init_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

class MaskingLayer(nn.Module):
    def __init__(self, mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
    
    def forward(self, x):
        mask = torch.rand_like(x) > self.mask_ratio
        return torch.where(mask, x, torch.zeros_like(x))

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 24x24 -> 48x48
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 48x48 -> 64x64
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(x)

class MaskedAutoencoder(nn.Module):
    def __init__(self, mask_ratio=0.5):
        super().__init__()
        self.masking = MaskingLayer(mask_ratio)
        self.encoder = SwinEncoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        masked = self.masking(x)
        latent = self.encoder(masked)
        return self.decoder(latent)

def train_model(model, train_loader, val_loader, num_epochs=10):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Slightly lower learning rate
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                output = model(data)
                val_loss += criterion(output, data).item()
        val_loss /= len(val_loader)
        
        print(f'Epoch: {epoch+1}, Training loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}')

def visualize_results(model, test_loader):
    model.eval()
    with torch.no_grad():
        data = next(iter(test_loader))[0][:8].to(device)
        masked_data = model.masking(data)
        reconstructed = model(data)
        
        data = data.cpu()
        masked_data = masked_data.cpu()
        reconstructed = reconstructed.cpu()
        
        plt.figure(figsize=(12, 4))
        for i in range(8):
            plt.subplot(3, 8, i + 1)
            plt.imshow(data[i][0], cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Original')
            
            plt.subplot(3, 8, i + 9)
            plt.imshow(masked_data[i][0], cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Masked')
            
            plt.subplot(3, 8, i + 17)
            plt.imshow(reconstructed[i][0], cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Reconstructed')
        
        plt.tight_layout()
        plt.show()

# Data transforms with 64x64 size
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Usage example
if __name__ == "__main__":
    # Replace these paths with your actual data directories
    train_dir = "data/train"
    val_dir = "data/train"
    
    # Create datasets
    train_dataset = CustomImageDataset(train_dir, transform=transform)
    val_dataset = CustomImageDataset(val_dir, transform=transform)
    
    # Create data loaders with larger batch size due to smaller images
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Initialize and train the model
    model = MaskedAutoencoder()
    train_model(model, train_loader, val_loader, num_epochs=10)
    
    # Visualize results
    visualize_results(model, val_loader)