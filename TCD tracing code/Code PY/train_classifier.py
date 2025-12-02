import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import os
import matplotlib.pyplot as plt

# Import Models
from classifier_model import get_classifier_model
from cyclegan_model import Generator

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Handle LogSoftmax output from SelfResNet vs Linear from others
            # If outputs are log_probs, exp to get probs. If linear, softmax.
            # But SelfResNet returns log_softmax. ResNet returns linear.
            # Let's standardize: model should return raw logits or we handle it.
            # Current classifier_model.py: 
            # ResNet18_1D -> returns linear logits.
            # SelfResNet18_1D -> returns F.log_softmax.
            
            # Let's just apply exp if log_softmax, or softmax if linear. 
            # Heuristic: check if values are negative and sum to 1 (unlikely for logits).
            # Better: assume standard training loop with CrossEntropyLoss which expects Logits.
            # SelfResNet uses NLLLoss if it outputs log_softmax. ResNet uses CrossEntropy if logits.
            # To keep it simple, we'll use CrossEntropyLoss which takes Logits.
            # I should check classifier_model.py again. SelfResNet18_1D returns F.log_softmax.
            # So for SelfResNet we need NLLLoss. For ResNet, CrossEntropyLoss.
            
            # Just use argmax for preds.
            _, preds = torch.max(outputs, 1)
            
            # Probs for AUC
            # If log_softmax (all negative), exp it.
            if outputs.max() <= 0: # likely log_softmax
                 probs = torch.exp(outputs)[:, 1]
            else: # likely logits
                 probs = torch.softmax(outputs, dim=1)[:, 1]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0 # Handle edge case
        
    cm = confusion_matrix(all_labels, all_preds)
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return acc, auc, sensitivity, specificity, cm

def train_scenario(scenario_name, X, y, device, model_type='resnet18', epochs=20, batch_size=32):
    print(f"\n--- Training Scenario: {scenario_name} ({model_type}) ---")
    print(f"Data shape: {X.shape}")
    
    if len(X) == 0:
        print("No data for this scenario.")
        return {'acc': 0, 'auc': 0, 'sens': 0, 'spec': 0}
        
    # Prepare Data
    tensor_x = torch.Tensor(X).unsqueeze(1) # (N, 1, 1024)
    tensor_y = torch.LongTensor(y)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    
    # Split Train/Test (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = get_classifier_model(model_type=model_type, num_classes=2).to(device)
    
    # Loss Function
    if model_type == 'self_resnet18':
        criterion = nn.NLLLoss() # Expects log_probs
    else:
        criterion = nn.CrossEntropyLoss() # Expects logits
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    acc, auc, sens, spec, cm = evaluate_model(model, test_loader, device)
    
    print(f"Results for {scenario_name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Sensitivity (ICU Recall): {sens:.4f}")
    print(f"  Specificity (Healthy Recall): {spec:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    return {'acc': acc, 'auc': auc, 'sens': sens, 'spec': spec}

def normalize_segs(segments):
    norm = []
    for s in segments:
        min_val = np.min(s)
        max_val = np.max(s)
        if max_val > min_val:
            s_norm = (s - min_val) / (max_val - min_val)
            norm.append(s_norm)
        else:
            norm.append(np.zeros_like(s))
    return np.array(norm)

def main():
    parser = argparse.ArgumentParser(description="Train TCD Classifier (Healthy vs ICU).")
    parser.add_argument('--dataset_path', type=str, default='tcd_dataset.npz', help='Path to dataset.')
    parser.add_argument('--labels_path', type=str, default='sqi_labels.npz', help='Path to labels.')
    parser.add_argument('--recon_errors_path', type=str, default='recon_errors.npz', help='Path to recon errors (unused currently).')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    labels_path = args.labels_path
    generator_path = 'generator_AB.pth'
    
    if not os.path.exists(dataset_path) or not os.path.exists(labels_path):
        print("Error: Data files missing.")
        return
        
    data = np.load(dataset_path, allow_pickle=True)
    sqi_data = np.load(labels_path, allow_pickle=True)
    
    # Segments
    h_segs = data['healthy_valid']
    i_segs = data['icu_valid']
    
    # Detect scenarios from keys
    scenarios = []
    for key in sqi_data.files:
        if key.startswith('healthy_quality_labels_'):
            scenario_name = key.replace('healthy_quality_labels_', '')
            scenarios.append(scenario_name)
    
    if not scenarios:
        # Fallback for legacy format
        if 'healthy_quality_labels' in sqi_data:
            scenarios = ['Default']
        else:
            print("No label scenarios found.")
            return

    print(f"Found SQI Threshold Scenarios: {scenarios}")

    # Class Labels (0: Healthy, 1: ICU)
    h_y = np.zeros(len(h_segs))
    i_y = np.ones(len(i_segs))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_types_to_test = ['resnet18', 'self_resnet18']
    all_results = {} 

    # Baseline is independent of SQI threshold, run it once per model
    for model_type in model_types_to_test:
        print(f"\n--- Baseline Experiment (No SQA) - Model: {model_type} ---")
        X_base = normalize_segs(np.concatenate((h_segs, i_segs)))
        y_base = np.concatenate((h_y, i_y))
        res_base = train_scenario("Baseline (No SQA)", X_base, y_base, device, model_type=model_type, epochs=args.epochs, batch_size=args.batch_size)
        all_results[f"{model_type}_Baseline"] = res_base

    # Run SQA experiments for each scenario
    for scenario in scenarios:
        print(f"\n=== Processing SQA Scenario: {scenario} ===")
        
        if scenario == 'Default':
            h_q = sqi_data['healthy_quality_labels']
            i_q = sqi_data['icu_quality_labels']
        else:
            h_q = sqi_data[f'healthy_quality_labels_{scenario}']
            i_q = sqi_data[f'icu_quality_labels_{scenario}']

        for model_type in model_types_to_test:
            # --- Scenario 2: Proposed (SQA - Good Only) ---
            h_good_mask = h_q == 1
            i_good_mask = i_q == 1
            
            X_good_h = h_segs[h_good_mask]
            y_good_h = h_y[h_good_mask]
            X_good_i = i_segs[i_good_mask]
            y_good_i = i_y[i_good_mask]
            
            if len(X_good_h) == 0 or len(X_good_i) == 0:
                print(f"Skipping SQA Good for {scenario} {model_type} - Not enough GOOD data.")
                continue

            X_sqa = normalize_segs(np.concatenate((X_good_h, X_good_i)))
            y_sqa = np.concatenate((y_good_h, y_good_i))
            
            res_sqa = train_scenario(f"SQA ({scenario}) - Good Only", X_sqa, y_sqa, device, model_type=model_type, epochs=args.epochs, batch_size=args.batch_size)
            all_results[f"{model_type}_{scenario}_SQAGood"] = res_sqa
            
            # --- Scenario 3: Proposed + GAN Restoration (Optional) ---
            # GAN restoration logic would need to be adapted to use specific scenario labels if GAN was trained on them.
            # For simplicity, we'll skip GAN loop integration here or assume one GAN model.
            # If you want GAN per scenario, train_cyclegan needs update too.
            # We will skip GAN loop here to focus on threshold impact.

    # --- Summary Plot ---
    print("\n--- Final Summary Plot ---")
    # Determine number of bars: Models * (Baseline + Scenarios)
    plot_labels = ['Baseline'] + [f'SQA {s}' for s in scenarios]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(plot_labels))
    width = 0.35
    
    for i, model_type in enumerate(model_types_to_test):
        accuracies = []
        # Baseline
        accuracies.append(all_results.get(f"{model_type}_Baseline", {'acc': 0})['acc'])
        
        # Scenarios
        for s in scenarios:
            accuracies.append(all_results.get(f"{model_type}_{s}_SQAGood", {'acc': 0})['acc'])
            
        offset = width * (i - 0.5)
        ax.bar(x + offset, accuracies, width, label=model_type)

    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Performance: Baseline vs SQA Thresholds')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels)
    ax.set_ylim(0, 1.0)
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('classification_results_multiscenario.png')
    print("Summary plot saved to classification_results_multiscenario.png")

if __name__ == "__main__":
    main()
