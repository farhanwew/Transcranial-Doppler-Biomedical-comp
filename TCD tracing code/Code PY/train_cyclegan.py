import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import itertools
import os
import matplotlib.pyplot as plt

from cyclegan_model import Generator, Discriminator

import argparse

def train_cyclegan(
    train_data_path, 
    epochs, 
    batch_size, 
    lr, 
    b1, 
    b2
):
    
    if not os.path.exists(train_data_path):
        print(f"Error: Train data file not found at {train_data_path}. Run split_dataset.py first.")
        return

    # --- 1. Data Preparation ---
    print(f"Loading training data for CycleGAN from {train_data_path}...")
    data = np.load(train_data_path, allow_pickle=True)
    
    # Combine Healthy and ICU data to get full domain pools
    # Healthy
    h_segs = data['healthy_valid']
    
    # Determine which label key to use. 
    # split_dataset.py saves all scenarios. Let's default to the first available scenario or 'Default'.
    # We need to find a valid label key.
    available_keys = data.files
    label_suffix = None
    for key in available_keys:
        if key.startswith('healthy_quality_labels_'):
            label_suffix = key.replace('healthy_quality_labels_', '')
            break # Use the first found scenario
            
    if label_suffix:
        h_label_key = f'healthy_quality_labels_{label_suffix}'
        i_label_key = f'icu_quality_labels_{label_suffix}'
        print(f"Using quality labels from scenario: {label_suffix}")
    else:
        # Fallback if no scenario key found (legacy)
        h_label_key = 'healthy_quality_labels'
        i_label_key = 'icu_quality_labels'
        if h_label_key not in available_keys:
             print("Error: Quality labels not found in dataset.")
             return

    h_labels = data[h_label_key]
    
    # ICU
    i_segs = data['icu_valid']
    i_labels = data[i_label_key]
    
    # Filter domains
    # Domain A: Borderline (-1) - Source
    # Domain B: Good (1) - Target
    
    borderline_segs = np.concatenate((h_segs[h_labels == -1], i_segs[i_labels == -1]), axis=0)
    good_segs = np.concatenate((h_segs[h_labels == 1], i_segs[i_labels == 1]), axis=0)
    
    print(f"Domain A (Borderline) size: {len(borderline_segs)}")
    print(f"Domain B (Good) size: {len(good_segs)}")
    
    if len(borderline_segs) == 0 or len(good_segs) == 0:
        print("Error: Not enough data in one of the domains to train CycleGAN.")
        return

    # Normalize to [-1, 1] for Tanh activation in Generator
    # Assuming segments are roughly normalized per segment or globally.
    # Let's apply MinMax per segment to [0, 1] then shift to [-1, 1]
    def normalize_segs(segments):
        norm = []
        for s in segments:
            min_val = np.min(s)
            max_val = np.max(s)
            if max_val > min_val:
                s_norm = (s - min_val) / (max_val - min_val)
                s_scaled = s_norm * 2 - 1 # [0, 1] -> [-1, 1]
                norm.append(s_scaled)
        return np.array(norm)

    X_A = torch.Tensor(normalize_segs(borderline_segs)).unsqueeze(1) # (N, 1, 1024)
    X_B = torch.Tensor(normalize_segs(good_segs)).unsqueeze(1)
    
    # DataLoaders
    loader_A = DataLoader(TensorDataset(X_A), batch_size=batch_size, shuffle=True)
    loader_B = DataLoader(TensorDataset(X_B), batch_size=batch_size, shuffle=True)

    # --- 2. Model & Optimizer Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generators
    G_AB = Generator().to(device) # A -> B
    G_BA = Generator().to(device) # B -> A

    # Discriminators
    D_A = Discriminator().to(device) # Classifies Real A vs Fake A
    D_B = Discriminator().to(device) # Classifies Real B vs Fake B

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers
    optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

    # --- 3. Training Loop ---
    print(f"Starting CycleGAN training for {epochs} epochs with batch size {batch_size}...")
    
    loss_history = {'G': [], 'D': []}

    for epoch in range(epochs):
        for i, (batch_A, batch_B) in enumerate(zip(loader_A, itertools.cycle(loader_B))):
            # Set model input
            real_A = batch_A[0].to(device)
            real_B = batch_B[0].to(device)
            
            # Handle different batch sizes at end of epoch
            curr_batch_size = real_A.size(0)
            if real_B.size(0) != curr_batch_size:
                real_B = real_B[:curr_batch_size]

            # Adversarial ground truths
            valid = torch.ones((curr_batch_size, 1, 64), requires_grad=False).to(device) # PatchGAN output size approx
            fake = torch.zeros((curr_batch_size, 1, 64), requires_grad=False).to(device)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            rec_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(rec_A, real_A)
            rec_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(rec_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + 10.0 * loss_cycle + 5.0 * loss_identity
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), valid)
            loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), valid)
            loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

        # Log progress
        print(f"[Epoch {epoch+1}/{epochs}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")
        loss_history['G'].append(loss_G.item())
        loss_history['D'].append(loss_D.item())

    # Save Models
    torch.save(G_AB.state_dict(), 'generator_AB.pth') # Main restoration model
    torch.save(G_BA.state_dict(), 'generator_BA.pth')
    print("Models saved: generator_AB.pth (Borderline -> Good)")

    # Visualize Result
    G_AB.eval()
    with torch.no_grad():
        # Pick a random borderline sample
        sample_A = next(iter(loader_A))[0][:1].to(device)
        fake_B = G_AB(sample_A)
        
        plt.figure(figsize=(10, 4))
        plt.plot(sample_A.squeeze().cpu().numpy(), label='Original (Borderline)', alpha=0.7)
        plt.plot(fake_B.squeeze().cpu().numpy(), label='Restored (Good-like)', alpha=0.7)
        plt.title("CycleGAN Restoration Sample")
        plt.legend()
        plt.savefig('cyclegan_restoration_sample.png')
        print("Sample visualization saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CycleGAN for TCD signal restoration.")
    parser.add_argument('--train_data_path', type=str, default='tcd_train.npz', help='Path to the training dataset file.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate.')
    parser.add_argument('--b1', type=float, default=0.5, help='Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='Adam: decay of second order momentum of gradient')
    
    args = parser.parse_args()
    
    train_cyclegan(
        train_data_path=args.train_data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        b1=args.b1,
        b2=args.b2
    )
