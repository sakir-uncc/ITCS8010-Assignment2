import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import scipy.stats as stats
from ssat import CombinedModel, CustomImageDataset

def calculate_reconstruction_metrics(original, reconstruction):
    """Calculate various reconstruction quality metrics"""
    # Convert tensors to numpy arrays
    orig = original.cpu().numpy()
    recon = reconstruction.cpu().numpy()
    
    # Mean Squared Error
    mse = np.mean((orig - recon) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Structural Similarity Index (simplified version)
    def ssim(img1, img2):
        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2
        
        mu1 = img1.mean()
        mu2 = img2.mean()
        
        sigma1 = np.sqrt(((img1 - mu1) ** 2).mean())
        sigma2 = np.sqrt(((img2 - mu2) ** 2).mean())
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2))
        return ssim
    
    ssim_value = ssim(orig, recon)
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim_value
    }

def plot_learning_curves(train_losses, val_losses, save_dir):
    """Plot training and validation learning curves"""
    plt.figure(figsize=(12, 5))
    
    # Classification loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses['clf'], label='Train')
    plt.plot(val_losses['clf'], label='Validation')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # MAE loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses['mae'], label='Train')
    plt.plot(val_losses['mae'], label='Validation')
    plt.title('MAE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/learning_curves.png')
    plt.close()

def plot_roc_curves(y_true, y_score, classes, save_dir):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))
    
    # Binarize labels for ROC curve
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(fpr[i], tpr[i], label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curves.png')
    plt.close()
    
    return roc_auc

def plot_precision_recall_curves(y_true, y_score, classes, save_dir):
    """Plot Precision-Recall curves for each class"""
    plt.figure(figsize=(10, 8))
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    
    # Compute Precision-Recall curve for each class
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f'{classes[i]} (AP = {ap:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/precision_recall_curves.png')
    plt.close()

def plot_reconstruction_distribution(recon_metrics, save_dir):
    """Plot distribution of reconstruction metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MSE distribution
    sns.histplot(recon_metrics['mse'], ax=axes[0], kde=True)
    axes[0].set_title('MSE Distribution')
    axes[0].set_xlabel('MSE')
    
    # PSNR distribution
    sns.histplot(recon_metrics['psnr'], ax=axes[1], kde=True)
    axes[1].set_title('PSNR Distribution')
    axes[1].set_xlabel('PSNR (dB)')
    
    # SSIM distribution
    sns.histplot(recon_metrics['ssim'], ax=axes[2], kde=True)
    axes[2].set_title('SSIM Distribution')
    axes[2].set_xlabel('SSIM')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/reconstruction_metrics_distribution.png')
    plt.close()

def plot_latent_space(features, labels, classes, save_dir):
    """Plot t-SNE visualization of the latent space"""
    from sklearn.manifold import TSNE
    
    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Visualization of Latent Space')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, 
              title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/latent_space_tsne.png')
    plt.close()

def evaluate_combined_model(model, test_loader, device, save_dir='results'):
    """
    Comprehensive evaluation of the combined model
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Initialize metrics storage
    all_preds = []
    all_targets = []
    all_scores = []
    all_features = []
    reconstruction_metrics = {
        'mse': [], 'psnr': [], 'ssim': []
    }
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc='Evaluating')):
            data, targets = data.to(device), targets.to(device)
            
            # Get model outputs
            clf_output, reconstruction = model(data)
            
            # Store classification results
            scores = F.softmax(clf_output, dim=1)
            _, predicted = clf_output.max(1)
            
            all_scores.extend(scores.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Store features for t-SNE
            features = model.encoder(data)
            all_features.extend(features.flatten(1).cpu().numpy())
            
            # Calculate reconstruction metrics
            batch_metrics = calculate_reconstruction_metrics(data, reconstruction)
            for metric, value in batch_metrics.items():
                reconstruction_metrics[metric].append(value)
            
            # Save example reconstructions (first batch only)
            if batch_idx == 0:
                n_examples = min(8, data.size(0))
                fig, axes = plt.subplots(2, n_examples, figsize=(2*n_examples, 4))
                for i in range(n_examples):
                    # Original
                    axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
                    axes[0, i].axis('off')
                    if i == 0:
                        axes[0, i].set_title('Original')
                    
                    # Reconstruction
                    axes[1, i].imshow(reconstruction[i].cpu().squeeze(), cmap='gray')
                    axes[1, i].axis('off')
                    if i == 0:
                        axes[1, i].set_title('Reconstructed')
                
                plt.tight_layout()
                plt.savefig(save_dir / 'example_reconstructions.png')
                plt.close()
    
    # Convert lists to arrays
    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_features = np.array(all_features)

    def get_classes(dataset):
        """
        Helper function to get classes from dataset, handling both direct datasets and subsets
        """
        if hasattr(dataset, 'classes'):
            return dataset.classes
        elif hasattr(dataset, 'dataset'):
            return dataset.dataset.classes
        else:
            # If using random_split, we need to go through the original dataset
            return get_original_dataset_classes(dataset)

    def get_original_dataset_classes(dataset):
        """
        Traverse through dataset splits to find original dataset with classes
        """
        current = dataset
        while hasattr(current, 'dataset'):
            current = current.dataset
        if hasattr(current, 'classes'):
            return current.classes
        else:
            raise AttributeError("Could not find classes attribute in dataset hierarchy")
    
    # Get class names
    classes = get_classes(test_loader.dataset)
    
    # Generate and save plots
    plot_roc_curves(all_targets, all_scores, classes, save_dir)
    plot_precision_recall_curves(all_targets, all_scores, classes, save_dir)
    plot_reconstruction_distribution(reconstruction_metrics, save_dir)
    plot_latent_space(all_features, all_targets, classes, save_dir)
    
    # Calculate and save metrics
    results = {
        'Classification': {
            'Accuracy': (all_preds == all_targets).mean() * 100,
            'Per-class Report': classification_report(all_targets, all_preds, 
                                                    target_names=classes, 
                                                    output_dict=True)
        },
        'Reconstruction': {
            'MSE': {
                'Mean': np.mean(reconstruction_metrics['mse']),
                'Std': np.std(reconstruction_metrics['mse'])
            },
            'PSNR': {
                'Mean': np.mean(reconstruction_metrics['psnr']),
                'Std': np.std(reconstruction_metrics['psnr'])
            },
            'SSIM': {
                'Mean': np.mean(reconstruction_metrics['ssim']),
                'Std': np.std(reconstruction_metrics['ssim'])
            }
        }
    }
    
    # Save results
    with open(save_dir / 'evaluation_results.txt', 'w') as f:
        f.write('Classification Results:\n')
        f.write(f"Overall Accuracy: {results['Classification']['Accuracy']:.2f}%\n\n")
        f.write('Per-class Results:\n')
        f.write(pd.DataFrame(results['Classification']['Per-class Report']).to_string())
        f.write('\n\nReconstruction Results:\n')
        for metric in ['MSE', 'PSNR', 'SSIM']:
            f.write(f"{metric}:\n")
            f.write(f"  Mean: {results['Reconstruction'][metric]['Mean']:.4f}\n")
            f.write(f"  Std:  {results['Reconstruction'][metric]['Std']:.4f}\n")
    
    return results

if __name__ == "__main__":
    # Load the saved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test dataset
    test_dir = "data/test"
    test_dataset = CustomImageDataset(test_dir, is_pretraining=False, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Initialize model
    num_classes = len(test_dataset.classes)
    model = CombinedModel(num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load('final_combined_model.pth'))
    model = model.to(device)
    
    # Evaluate the model
    results = evaluate_combined_model(model, test_loader, device)