import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from data_provider.fusion_dataset import UnifiedDataset, FusionFeatureDataset, collate_fn, fusion_collate_fn
from models.fusion import FusionFeatureModel

# Example: How an expert model should be implemented or wrapped
class ExpertModelDemo(nn.Module):
    def __init__(self, target_cols=['observe_power', 'GHI_solargis'], d_model=128):
        super(ExpertModelDemo, self).__init__()
        self.target_cols = target_cols
        self.projection = nn.Linear(len(target_cols), d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )

    def forward_hidden(self, batch_dict):
        """
        Modified for dictionary-based batch format
        Each selected feature is (B, L)
        Returns (B, L, D_model)
        """
        # 1. Expert-specific column selection: Stack selected (B, L) features into (B, L, C)
        x_expert = torch.stack([batch_dict[col] for col in self.target_cols], dim=-1) # (B, L, C)
            
        # 2. Model forward pass (Projection + Transformer layer)
        hidden = self.projection(x_expert) # (B, L, d_model)
        hidden = self.transformer(hidden) # (B, L, d_model)
        return hidden 

    def forward(self, batch_dict):
        return self.forward_hidden(batch_dict)

def extract_and_save_features(model, dataloader, save_path, device='cpu'):
    """
    Extract hidden features and save to a .npy file using memory mapping (memmap)
    to avoid memory overflows with large datasets.
    """
    model.to(device)
    model.eval()
    
    num_samples = len(dataloader.dataset)
    
    # 1. Infer dimensions from a dummy pass of the first batch
    first_batch = next(iter(dataloader))
    first_batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in first_batch.items()}
    with torch.no_grad():
        dummy_output = model.forward_hidden(first_batch_device)
        # dummy_output shape is (B, L, D) or (B, C, P, D) -> we need (L, D) or (C, P, D)
        feature_shape = dummy_output.shape[1:] 
        d_model = dummy_output.shape[-1]
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 2. Create a memory-mapped array (pre-allocates disk space)
    # Final shape: (Num_Samples, *Feature_Shape)
    full_shape = (num_samples, *feature_shape)
    fp = np.lib.format.open_memmap(save_path, dtype='float32', mode='w+', shape=full_shape)
    
    print(f"Allocated memmap at {save_path} with shape {full_shape}")
    
    idx = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
            hidden = model.forward_hidden(batch_device).cpu().numpy()
            
            batch_size = hidden.shape[0]
            # 3. Direct write to disk slice
            fp[idx : idx + batch_size] = hidden
            idx += batch_size
            
            if (idx // batch_size) % 10 == 0:
                print(f"  Progress: {idx}/{num_samples} samples saved.")
    
    # 4. Flush and close
    fp.flush()
    del fp 
    print(f"Extraction complete. File size: {os.path.getsize(save_path)/1024/1024:.2f} MB")
    return full_shape

def run_demo():
    # 1. Create dummy data
    num_samples = 100
    seq_len = 672
    pred_len = 192
    data = {
        'timestamp_win': pd.date_range(start='2023-01-01', periods=num_samples, freq='H'),
        'observe_power': [np.random.randn(seq_len) for _ in range(num_samples)],
        'observe_power_future': [np.random.randn(pred_len) for _ in range(num_samples)],
        'GHI_solargis': [np.random.randn(seq_len) for _ in range(num_samples)],
        'GHI_solargis_future': [np.random.randn(pred_len) for _ in range(num_samples)],
    }
    df = pd.DataFrame(data)
    
    # 2. Initialize Row-based Unified Dataset for extraction
    dataset = UnifiedDataset(df)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=collate_fn)
    
    # 3. Phase 1: Feature Extraction
    expert_a = ExpertModelDemo(target_cols=['observe_power'], d_model=64)
    expert_b = ExpertModelDemo(target_cols=['GHI_solargis'], d_model=128)
    
    feature_paths = {
        'expert_a': 'data/features/expert_a.npy',
        'expert_b': 'data/features/expert_b.npy'
    }
    
    shape_a = extract_and_save_features(expert_a, dataloader, feature_paths['expert_a'])
    shape_b = extract_and_save_features(expert_b, dataloader, feature_paths['expert_b'])
    
    # 4. Phase 2: Fast Fusion Iteration
    print("\n>>> Starting Fast Fusion Iteration Demo...")
    
    # Initialize FusionFeatureDataset (mmap mode)
    fusion_dataset = FusionFeatureDataset(
        df, 
        feature_paths, 
        target_cols=['observe_power_future']
    )
    fusion_loader = DataLoader(
        fusion_dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=fusion_collate_fn
    )
    
    # Initialize FusionFeatureModel
    # Note: expert_dims should match extracted shapes (B, L, D) -> dim is D
    expert_dims = {
        'expert_a': shape_a[-1],
        'expert_b': shape_b[-1]
    }
    
    fusion_model = FusionFeatureModel(
        expert_dims=expert_dims,
        pred_len=pred_len,
        n_features=1, # predicting observe_power_future
        d_fusion=64,
        num_experts=2,
        use_dynamic_queries=True
    )
    
    # 5. Run a few training steps
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    fusion_model.train()
    for i, (feats, targets, x_input) in enumerate(fusion_loader):
        optimizer.zero_grad()
        
        # Forward pass with pre-computed features
        outputs = fusion_model(feats, x_input) # (B, P, n_features)
        
        # Loss calculation
        target_tensor = targets['observe_power_future'].unsqueeze(-1) # (B, P, 1)
        loss = criterion(outputs, target_tensor)
        
        loss.backward()
        optimizer.step()
        
        if i % 2 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")
            print(f"  Output shape: {outputs.shape}")
        if i >= 4: break

    print("\nDemo successful! Hidden vectors saved and fusion model iterated.")

if __name__ == "__main__":
    run_demo()
