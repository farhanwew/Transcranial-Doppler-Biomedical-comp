import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import os
import matplotlib.pyplot as plt

# Import Models
from classifier_model import TCDClassifier
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
            probs = torch.softmax(outputs, dim=1)[:, 1] # Prob of class 1 (ICU)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0 # Handle edge case if only one class present
        
    cm = confusion_matrix(all_labels, all_preds)
    
    # Sensitivity (Recall for class 1) & Specificity (Recall for class 0)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return acc, auc, sensitivity, specificity, cm

def train_scenario(scenario_name, X, y, device, epochs=20, batch_size=32):
    print(f"\n--- Training Scenario: {scenario_name} ---")
    print(f"Data shape: {X.shape}")
    
    if len(X) == 0:
        print("No data for this scenario.")
        return
        
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
    model = TCDClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
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
    # Simple min-max normalization to [0, 1] per segment
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
    dataset_path = 'tcd_dataset.npz'
    labels_path = 'sqi_labels.npz'
    generator_path = 'generator_AB.pth'
    
    if not os.path.exists(dataset_path) or not os.path.exists(labels_path):
        print("Error: Data files missing.")
        return
        
    data = np.load(dataset_path, allow_pickle=True)
    sqi_data = np.load(labels_path, allow_pickle=True)
    
    # Segments
    h_segs = data['healthy_valid']
    i_segs = data['icu_valid']
    
    # Quality Labels (-1, 0, 1)
    h_q = sqi_data['healthy_quality_labels']
    i_q = sqi_data['icu_quality_labels']
    
    # Class Labels (0: Healthy, 1: ICU)
    # Healthy
    h_y = np.zeros(len(h_segs))
    # ICU
    i_y = np.ones(len(i_segs))
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Scenario 1: Baseline (No SQA) ---
    # Use ALL valid segments (only hard filtered)
    X_base = normalize_segs(np.concatenate((h_segs, i_segs)))
    y_base = np.concatenate((h_y, i_y))
    
    res_base = train_scenario("Baseline (No SQA)", X_base, y_base, device)
    
    # --- Scenario 2: Proposed (SQA - Good Only) ---
    # Use only segments with Quality Label = 1 (GOOD)
    h_good_mask = h_q == 1
    i_good_mask = i_q == 1
    
    X_good_h = h_segs[h_good_mask]
    y_good_h = h_y[h_good_mask]
    
    X_good_i = i_segs[i_good_mask]
    y_good_i = i_y[i_good_mask]
    
    X_sqa = normalize_segs(np.concatenate((X_good_h, X_good_i)))
    y_sqa = np.concatenate((y_good_h, y_good_i))
    
    res_sqa = train_scenario("Proposed (SQA - Good Only)", X_sqa, y_sqa, device)
    
    # --- Scenario 3: Proposed + GAN Restoration (Optional) ---
    if os.path.exists(generator_path):
        print("\n--- Preparing GAN Restored Data ---")
        # Load Generator
        G = Generator().to(device)
        G.load_state_dict(torch.load(generator_path, map_location=device))
        G.eval()
        
        # Get Borderline Segments
        h_bord_mask = h_q == -1
        i_bord_mask = i_q == -1
        
        X_bord_h = normalize_segs(h_segs[h_bord_mask]) # Normalize for GAN input
        X_bord_i = normalize_segs(i_segs[i_bord_mask])
        
        # Restore Function
        def restore_segments(segments):
            if len(segments) == 0: return np.array([])
            # Map [0, 1] to [-1, 1] for GAN
            segs_gan = torch.Tensor(segments * 2 - 1).unsqueeze(1).to(device)
            with torch.no_grad():
                restored = G(segs_gan)
            # Map back to [0, 1]
            restored = (restored.squeeze(1).cpu().numpy() + 1) / 2
            return restored

        X_restored_h = restore_segments(X_bord_h)
        y_restored_h = h_y[h_bord_mask]
        
        X_restored_i = restore_segments(X_bord_i)
        y_restored_i = i_y[i_bord_mask]
        
        # Combine Good + Restored
        # Note: If arrays are empty, concatenation needs care.
        parts_X = [X_sqa]
        parts_y = [y_sqa]
        
        if len(X_restored_h) > 0:
            parts_X.append(X_restored_h)
            parts_y.append(y_restored_h)
        if len(X_restored_i) > 0:
            parts_X.append(X_restored_i)
            parts_y.append(y_restored_i)
            
        X_gan = np.concatenate(parts_X)
        y_gan = np.concatenate(parts_y)
        
        res_gan = train_scenario("Proposed + GAN Restoration", X_gan, y_gan, device)
    else:
        print("\nGenerator model not found. Skipping GAN scenario.")
        res_gan = None

    # --- Summary Plot ---
    scenarios = ['Baseline', 'SQA Good']
    accuracies = [res_base['acc'], res_sqa['acc']]
    
    if res_gan:
        scenarios.append('SQA + GAN')
        accuracies.append(res_gan['acc'])
        
    plt.figure(figsize=(8, 5))
    plt.bar(scenarios, accuracies, color=['gray', 'blue', 'green'])
    plt.ylabel('Accuracy')
    plt.title('Classification Performance Comparison')
    plt.ylim(0, 1.0)
    plt.savefig('classification_results.png')
    print("\nSummary plot saved to classification_results.png")

if __name__ == "__main__":
    main()
