import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import copy
# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Data transforms with more augmentations
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageFilter
import random

class GaussianBlur:
    def __init__(self, radius_min=0.1, radius_max=2.0):
        self.radius_min = radius_min
        self.radius_max = radius_max
    
    def __call__(self, img):
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_pretraining=True, is_training=True):
        self.root_dir = Path(root_dir)
        self.is_pretraining = is_pretraining
        self.is_training = is_training
        
        # Define separate transforms for training and validation
        if transform is None:
            if is_training:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),  # Larger size for random crops
                    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1),
                        shear=5
                    ),
                    GaussianBlur(0.1, 2.0),
                    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                    transforms.RandomAutocontrast(p=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
        else:
            self.transform = transform
        
        # For classification, get class names from directory names
        if not is_pretraining:
            self.classes = [d for d in self.root_dir.iterdir() if d.is_dir()]
            self.class_to_idx = {cls.name: idx for idx, cls in enumerate(self.classes)}
        
        # Recursively find all images
        self.image_paths = []
        self.labels = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        if is_pretraining:
            for ext in valid_extensions:
                self.image_paths.extend(list(self.root_dir.rglob(f'*{ext}')))
        else:
            for class_dir in self.classes:
                for ext in valid_extensions:
                    for img_path in class_dir.rglob(f'*{ext}'):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_dir.name])
    
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
            
            # Add random noise during training
            if self.is_training:
                noise = torch.randn_like(image) * 0.02
                image = image + noise
                image = torch.clamp(image, -1, 1)
        
        if self.is_pretraining:
            return image, image  # Return same image for input and target
        else:
            return image, self.labels[idx]

# MAE Components
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
        
        self.init_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        
        self.stage1 = nn.Sequential(
            SwinTransformerBlock(dim=64, num_heads=4, window_size=(64, 64)),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        )
        
        self.stage2 = nn.Sequential(
            SwinTransformerBlock(dim=64, num_heads=8, window_size=(32, 32)),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        )

        self.stage3 = nn.Sequential(
            SwinTransformerBlock(dim=128, num_heads=16, window_size=(16, 16)),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )

        self.stage4 = nn.Sequential(
            SwinTransformerBlock(dim=256, num_heads=16, window_size=(8, 8)),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        )
        
        self.stage5 = SwinTransformerBlock(dim=512, num_heads=16, window_size=(4, 4))

    def forward(self, x):
        x = self.init_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
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
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(x)
    
class ClassificationHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, num_classes)
        )
    
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class PreTrainingModel(nn.Module):
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.encoder = SwinEncoder()
        self.masking = MaskingLayer(mask_ratio)
        self.decoder = Decoder()
    
    def forward(self, x):
        masked = self.masking(x)
        features = self.encoder(masked)
        reconstruction = self.decoder(features)
        return reconstruction, masked

class FineTuningModel(nn.Module):
    def __init__(self, pretrained_encoder, num_classes):
        super().__init__()
        self.encoder = pretrained_encoder
        self.classifier = ClassificationHead(num_classes)
        
        # Freeze encoder initially
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self, num_layers=None):
        """Unfreeze encoder layers gradually during training"""
        if num_layers is None:
            # Unfreeze all layers
            for param in self.encoder.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific stages from the end
            stages = [self.encoder.stage5, self.encoder.stage4, 
                     self.encoder.stage3, self.encoder.stage2, 
                     self.encoder.stage1]
            for stage in stages[:num_layers]:
                for param in stage.parameters():
                    param.requires_grad = True
    
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

def pretrain_mae(model, train_loader, val_loader, num_epochs=50):
    """Pretrain the encoder using MAE"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [MAE Pretrain]')
        for data, _ in pbar:
            data = data.to(device)
            
            optimizer.zero_grad()
            reconstruction, masked = model(data)
            loss = criterion(reconstruction, data)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in tqdm(val_loader, desc='[Validation]'):
                data = data.to(device)
                reconstruction, masked = model(data)
                loss = criterion(reconstruction, data)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f'Epoch: {epoch+1}')
        print(f'Training Loss: {train_loss:.6f}')
        print(f'Validation Loss: {val_loss:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.encoder.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print("Early stopping triggered!")
                break
    
    return best_model_state

def finetune_classifier(model, train_loader, val_loader, num_epochs=50):
    """Finetune the model for classification"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Start with only classifier parameters
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.1, verbose=True)
    
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Gradually unfreeze encoder layers
        if epoch == 10:
            model.unfreeze_encoder(num_layers=2)  # Unfreeze last 2 stages
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=5e-5, weight_decay=0.01)
        elif epoch == 20:
            model.unfreeze_encoder()  # Unfreeze all layers
            optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        
        # Training
        model.train()
        correct = 0
        total = 0
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Finetune]')
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100.*correct/total
            })
        
        train_loss /= len(train_loader)
        train_acc = 100.*correct/total
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc='[Validation]'):
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100.*correct/total
        
        scheduler.step(val_acc)
        
        print(f'Epoch: {epoch+1}')
        print(f'Training - Loss: {train_loss:.6f}, Acc: {train_acc:.2f}%')
        print(f'Validation - Loss: {val_loss:.6f}, Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping triggered!")
                break
    
    model.load_state_dict(best_model_state)
    return model

if __name__ == "__main__":
    # Create datasets
    pretrain_dir = "data/train"  # Directory with unlabeled images
    train_dir = "data/train"  # Directory with labeled images
    
    # Create pretraining datasets
    pretrain_dataset = CustomImageDataset(pretrain_dir, is_pretraining=True, is_training=True)
    pretrain_val_dataset = CustomImageDataset(pretrain_dir, is_pretraining=True, is_training=False)
    
    # Split pretraining dataset
    pretrain_train_size = int(0.9 * len(pretrain_dataset))
    pretrain_val_size = len(pretrain_dataset) - pretrain_train_size
    pretrain_train_dataset, pretrain_val_dataset = random_split(
        pretrain_dataset, [pretrain_train_size, pretrain_val_size]
    )
    
    # Create pretraining data loaders
    pretrain_train_loader = DataLoader(
        pretrain_train_dataset, batch_size=8, shuffle=True, num_workers=4
    )
    pretrain_val_loader = DataLoader(
        pretrain_val_dataset, batch_size=8, shuffle=False, num_workers=4
    )
    
    # Create finetuning datasets
    train_dataset = CustomImageDataset(train_dir, is_pretraining=False, is_training=True)
    val_dataset = CustomImageDataset(train_dir, is_pretraining=False, is_training=False)
    
    # Split finetuning dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create finetuning data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Phase 1: Pretraining
    print("Starting MAE pretraining...")
    pretrain_model = PreTrainingModel()
    best_encoder_state = pretrain_mae(
        pretrain_model, pretrain_train_loader, pretrain_val_loader
    )
    
    # Save pretrained encoder
    torch.save(best_encoder_state, 'pretrained_encoder.pth')
    
    # Phase 2: Finetuning
    print("Starting classification finetuning...")
    encoder = SwinEncoder()
    encoder.load_state_dict(best_encoder_state)
    num_classes = len(train_dataset.dataset.classes)
    finetune_model = FineTuningModel(encoder, num_classes)
    
    finetune_model = finetune_classifier(
        finetune_model, train_loader, val_loader
    )
    
    # Save final model
    torch.save(finetune_model.state_dict(), 'final_finetuned_model.pth')