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
from sklearn.manifold import TSNE
from torchvision import transforms

from mae import (  # Import from your model file
    SwinEncoder,
    FineTuningModel,
    CustomImageDataset
)

def evaluate_swin_classifier(model, test_loader, device, save_dir='mae_test_results'):
    """
    Comprehensive evaluation of the Swin Transformer classifier
    
    Args:
        model (FineTuningModel): The finetuned classification model
        test_loader (DataLoader): DataLoader for test dataset
        device (torch.device): Device to run evaluation on
        save_dir (str): Directory to save evaluation results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Initialize metrics storage
    all_preds = []
    all_targets = []
    all_scores = []
    all_features = []
    feature_maps = []
    
    print("Starting model evaluation...")
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc='Testing')):
            data, targets = data.to(device), targets.to(device)
            
            # Get features from encoder
            features = model.encoder(data)
            
            # Get classifier output
            outputs = model.classifier(features)
            scores = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store results
            all_scores.extend(scores.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_features.extend(features.flatten(1).cpu().numpy())
            
            # Store feature maps for visualization (first batch only)
            if batch_idx == 0:
                feature_maps = features.cpu().numpy()
    
    # Convert lists to arrays
    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_features = np.array(all_features)
    
    # Get class names
    classes = get_dataset_classes(test_loader.dataset)
    
    # Generate visualizations
    generate_evaluation_plots(all_targets, all_preds, all_scores, 
                            all_features, feature_maps, classes, save_dir)
    
    # Calculate and save metrics
    results = compile_evaluation_results(all_targets, all_preds, all_scores, classes)
    save_evaluation_report(results, save_dir / 'evaluation_results.txt')
    
    return results

def generate_evaluation_plots(targets, predictions, scores, features, 
                            feature_maps, classes, save_dir):
    """Generate all evaluation visualizations"""
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(targets, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png')
    plt.close()
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    y_true_bin = label_binarize(targets, classes=range(len(classes)))
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves.png')
    plt.close()
    
    # Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], scores[:, i])
        ap = average_precision_score(y_true_bin[:, i], scores[:, i])
        plt.plot(recall, precision, label=f'{class_name} (AP = {ap:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(save_dir / 'precision_recall_curves.png')
    plt.close()
    
    # t-SNE visualization of features
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=targets, cmap='tab10')
    
    # Create custom legend
    legend_elements = []
    for i, class_name in enumerate(classes):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=scatter.cmap(scatter.norm(i)),
                                        label=class_name, markersize=10))
    
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Visualization of Feature Space')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(handles=legend_elements,
              title="Classes",
              bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(save_dir / 'tsne_visualization.png', bbox_inches='tight')
    plt.close()
    
    # Feature map visualization
    visualize_feature_maps(feature_maps, save_dir / 'feature_maps.png')

def visualize_feature_maps(feature_maps, save_path, num_channels=16):
    """Visualize feature maps from the encoder"""
    # Select a subset of channels to visualize
    n = min(num_channels, feature_maps.shape[1])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i in range(n):
        row = i // 4
        col = i % 4
        feature_map = feature_maps[0, i]  # First sample, i-th channel
        
        # Normalize feature map
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        
        axes[row, col].imshow(feature_map, cmap='viridis')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_dataset_classes(dataset):
    """Helper function to get class names from dataset"""
    if hasattr(dataset, 'classes'):
        return [cls.name for cls in dataset.classes]  # Get class names
    elif hasattr(dataset, 'dataset'):
        return get_dataset_classes(dataset.dataset)
    else:
        current_dataset = dataset
        while hasattr(current_dataset, 'dataset'):
            current_dataset = current_dataset.dataset
        return [cls.name for cls in current_dataset.classes]

def compile_evaluation_results(targets, predictions, scores, classes):
    """Compile all evaluation metrics into a dictionary"""
    # Calculate per-class metrics
    classification_metrics = classification_report(
        targets, predictions,
        target_names=classes,
        output_dict=True
    )
    
    # Calculate ROC AUC for each class
    y_true_bin = label_binarize(targets, classes=range(len(classes)))
    roc_auc_scores = {}
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], scores[:, i])
        roc_auc_scores[class_name] = auc(fpr, tpr)
    
    return {
        'Classification': {
            'Overall Accuracy': (predictions == targets).mean() * 100,
            'Per-class Metrics': classification_metrics,
            'Confusion Matrix': confusion_matrix(targets, predictions).tolist(),
            'ROC AUC Scores': roc_auc_scores
        }
    }

def save_evaluation_report(results, save_path):
    """Save evaluation results to a text file"""
    with open(save_path, 'w') as f:
        f.write('Classification Results\n')
        f.write('=====================\n\n')
        
        f.write(f"Overall Accuracy: {results['Classification']['Overall Accuracy']:.2f}%\n\n")
        
        f.write('Per-class Metrics:\n')
        f.write('-----------------\n')
        metrics_df = pd.DataFrame(results['Classification']['Per-class Metrics'])
        f.write(metrics_df.to_string())
        f.write('\n\n')
        
        f.write('ROC AUC Scores:\n')
        f.write('--------------\n')
        for class_name, auc_score in results['Classification']['ROC AUC Scores'].items():
            f.write(f'{class_name}: {auc_score:.4f}\n')

def analyze_misclassifications(model, test_loader, device, save_dir, 
                             max_samples=10):
    """Analyze and visualize misclassified samples"""
    save_dir = Path(save_dir)
    model.eval()
    classes = get_dataset_classes(test_loader.dataset)
    
    misclassified = []
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            # Find misclassified samples
            mask = predicted.ne(targets)
            if mask.any():
                misclassified.extend([
                    (data[i], targets[i], predicted[i])
                    for i in range(len(data)) if mask[i]
                ])
            
            if len(misclassified) >= max_samples:
                break
    
    # Visualize misclassified samples
    n_samples = min(max_samples, len(misclassified))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(n_samples):
        img, true_label, pred_label = misclassified[i]
        axes[i].imshow(img.cpu().squeeze(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'True: {classes[true_label]}\nPred: {classes[pred_label]}')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'misclassified_samples.png')
    plt.close()

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test dataset and loader
    test_dir = "data/test"
    test_dataset = CustomImageDataset(test_dir, is_pretraining=False, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Initialize model
    encoder = SwinEncoder()
    num_classes = len(test_dataset.classes)
    model = FineTuningModel(encoder, num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load('final_finetuned_model.pth'))
    model = model.to(device)
    
    # Create results directory
    results_dir = Path('mae_test_results')
    results_dir.mkdir(exist_ok=True)
    
    # Evaluate model
    print("Starting model evaluation...")
    results = evaluate_swin_classifier(model, test_loader, device, results_dir)
    
    # Analyze misclassifications
    print("Analyzing misclassified samples...")
    analyze_misclassifications(model, test_loader, device, results_dir)