import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

# Import VAE model
from vae_model import TCD_VAE, vae_loss_function

def train_vae(dataset_filepath='tcd_dataset.npz', model_save_path='vae_model.pth', errors_save_path='recon_errors.npy', num_epochs=50, batch_size=64, lr=1e-3):
    
    if not os.path.exists(dataset_filepath):
        print(f"Error: Dataset file not found at {dataset_filepath}")
        return

    # 1. Load Data
    print("Loading dataset...")
    data = np.load(dataset_filepath, allow_pickle=True)
    
    # Combine healthy and icu valid segments for unsupervised training
    # VAE learns the "general shape" of TCD signals
    healthy_segments = data['healthy_valid']
    icu_segments = data['icu_valid']
    
    if len(healthy_segments) == 0 and len(icu_segments) == 0:
        print("No valid segments found in dataset for VAE training.")
        return

    all_segments = np.concatenate((healthy_segments, icu_segments), axis=0)
    print(f"Total segments for VAE training: {all_segments.shape[0]}")

    # 2. Preprocessing: Normalize per segment to [0, 1]
    # X_norm = (X - min) / (max - min)
    # Handle potential division by zero if max == min (flat line - though filtered out, safety check)
    print("Normalizing data...")
    normalized_segments = []
    for seg in all_segments:
        min_val = np.min(seg)
        max_val = np.max(seg)
        if max_val - min_val > 1e-6:
            seg_norm = (seg - min_val) / (max_val - min_val)
        else:
            seg_norm = np.zeros_like(seg) # Or handle differently
        normalized_segments.append(seg_norm)
    
    normalized_segments = np.array(normalized_segments)
    
    # Convert to PyTorch Tensor and add Channel dimension: (N, 1024) -> (N, 1, 1024)
    tensor_x = torch.Tensor(normalized_segments).unsqueeze(1)

    # DataLoader
    dataset = TensorDataset(tensor_x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    vae = TCD_VAE(input_length=1024, latent_dim=32).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # 4. Training Loop
    print("Starting training...")
    vae.train()
    loss_history = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            
            # Annealing KL Divergence weight (optional, keeping it simple 1.0 for now)
            loss = vae_loss_function(recon_x, x, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save Model
    torch.save(vae.state_dict(), model_save_path)
    print(f"VAE model saved to {model_save_path}")
    
    # Plot Loss
    plt.figure()
    plt.plot(loss_history)
    plt.title('VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('vae_training_loss.png')
    print("Training loss plot saved.")

    # 5. Compute Reconstruction Errors for ALL segments
    print("Computing reconstruction errors...")
    vae.eval()
    recon_errors = []
    
    # Use a non-shuffled dataloader for sequential error calculation
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in eval_loader:
            x = batch[0].to(device)
            recon_x, _, _ = vae(x)
            
            # Calculate MSE per sample in batch
            # x: (Batch, 1, 1024), recon_x: (Batch, 1, 1024)
            # MSE over dim 2 (time), keep dim 0 (batch)
            mse = torch.mean((x - recon_x) ** 2, dim=[1, 2]) 
            recon_errors.extend(mse.cpu().numpy())
            
    recon_errors = np.array(recon_errors)
    
    # Save Reconstruction Errors
    # We need to know which error belongs to which segment. 
    # Since we concatenated healthy then ICU, the order is preserved.
    # Let's save them separately or with an index.
    # Better: Save as a dictionary or split back.
    
    num_healthy = len(healthy_segments)
    healthy_errors = recon_errors[:num_healthy]
    icu_errors = recon_errors[num_healthy:]
    
    np.savez(errors_save_path, healthy_errors=healthy_errors, icu_errors=icu_errors)
    print(f"Reconstruction errors saved to {errors_save_path}")
    
    # Visualize Reconstruction
    print("Visualizing sample reconstructions...")
    # Pick a random sample
    idx = np.random.randint(0, len(normalized_segments))
    original_sample = normalized_segments[idx]
    sample_tensor = torch.Tensor(original_sample).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        recon_sample, _, _ = vae(sample_tensor)
    
    recon_sample = recon_sample.squeeze().cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.plot(original_sample, label='Original (Normalized)', alpha=0.7)
    plt.plot(recon_sample, label='Reconstructed', alpha=0.7)
    plt.title(f'VAE Reconstruction (Sample Idx: {idx})')
    plt.legend()
    plt.savefig('vae_reconstruction_sample.png')
    print("Reconstruction visualization saved.")

if __name__ == "__main__":
    train_vae()
