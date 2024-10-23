import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class MAEClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_encoder):
        super().__init__()
        self.encoder = pretrained_encoder
        self.classifier = ClassificationHead(num_classes)
    
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

def train_mae(model, train_loader, val_loader, num_epochs=10):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for data, _ in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
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
                output = model(data)
                val_loss += criterion(output, data).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f'Epoch: {epoch+1}')
        print(f'Training Loss: {train_loss:.6f}')
        print(f'Validation Loss: {val_loss:.6f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping triggered!")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def train_classifier(model, train_loader, val_loader, num_epochs=10):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.1, verbose=True)
    
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
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
        val_loss = 0
        correct = 0
        total = 0
        
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
        print(f'Training Loss: {train_loss:.6f}, Training Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.6f}, Validation Acc: {val_acc:.2f}%')
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping triggered!")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

if __name__ == "__main__":
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    # Create pretraining dataset
    pretrain_dir = "data/train"
    # Create pretraining datasets with different transforms for train and val
    pretrain_dataset = CustomImageDataset(pretrain_dir, is_pretraining=True, is_training=True)
    val_dataset_pretrain = CustomImageDataset(pretrain_dir, is_pretraining=True, is_training=False)

    # Split pretraining dataset
    pretrain_size = int(0.8 * len(pretrain_dataset))
    val_size = len(pretrain_dataset) - pretrain_size
    pretrain_dataset, val_dataset = random_split(pretrain_dataset, [pretrain_size, val_size])
    
    # Create pretraining data loaders
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Initialize and train MAE
    print("Starting MAE pretraining...")
    mae_model = MaskedAutoencoder()
    mae_model = train_mae(mae_model, pretrain_loader, val_loader, num_epochs=50)
    
    # Save pretrained MAE
    torch.save(mae_model.state_dict(), 'pretrained_mae.pth')
    
    # Create classification dataset
    train_dir = "data/train"
    train_dataset = CustomImageDataset(train_dir, is_pretraining=False, is_training=True)
    val_dataset = CustomImageDataset(train_dir, is_pretraining=False, is_training=False)
    
    # Split training dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create classification data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Initialize classifier with pretrained encoder
    num_classes = len(train_dataset.dataset.classes)  # Access classes through dataset attribute
    classifier = MAEClassifier(num_classes, mae_model.encoder)
    
    # Train classifier
    print("\nStarting classifier training...")
    classifier = train_classifier(classifier, train_loader, val_loader, num_epochs=50)
    
    # Save final classifier
    torch.save(classifier.state_dict(), 'final_classifier.pth')
    
    # Optional: Create test dataset and evaluate
    test_dir = "data/test"
    if os.path.exists(test_dir):
        print("\nEvaluating on test set...")
        test_dataset = CustomImageDataset(test_dir, transform=transform, is_pretraining=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
        
        classifier.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc='Testing'):
                data, targets = data.to(device), targets.to(device)
                outputs = classifier(data)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = 100. * correct / total
        print(f'\nTest Accuracy: {accuracy:.2f}%')
        
        # Plot confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=test_dataset.classes,
                   yticklabels=test_dataset.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

        # Save detailed metrics
        from sklearn.metrics import classification_report
        report = classification_report(all_targets, all_preds, 
                                    target_names=test_dataset.classes,
                                    output_dict=True)
        
        # Convert to DataFrame and save
        import pandas as pd
        df_metrics = pd.DataFrame(report).transpose()
        df_metrics.to_csv('classification_metrics.csv')
        
        print("\nDetailed metrics saved to 'classification_metrics.csv'")
        print("Confusion matrix plot saved as 'confusion_matrix.png'")