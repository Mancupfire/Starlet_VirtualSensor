import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from Hybrid_DeepONet import HybridDeepONet
from DataLoader_PreProcessing import CO2Dataset
import os
import argparse
from tqdm import tqdm

# Hyperparameters & Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='Path to dataset CSV')
parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
args = parser.parse_args()

# Device Setup
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    DEVICE = torch.device('cuda')
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device('cpu')
    print("Training on CPU")

# Data Preparation
dataset = CO2Dataset(csv_path=args.data_path, scaler_path=args.save_dir, mode='train')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

# Model Initialization
model = HybridDeepONet(branch_in_dim=4, trunk_in_dim=3, latent_dim=128).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
criterion = nn.MSELoss()

# Training Loop
best_val_loss = float('inf')
os.makedirs(args.save_dir, exist_ok=True)

print("Start Training...")
for epoch in range(args.epochs):
    # Train
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
    
    for u, y, target in loop:
        u, y, target = u.to(DEVICE), y.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        pred = model(u, y)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for u, y, target in val_loader:
            u, y, target = u.to(DEVICE), y.to(DEVICE), target.to(DEVICE)
            pred = model(u, y)
            loss = criterion(pred, target)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)
    
    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
        print("--> Best Model Saved")

print("Training Completed.")