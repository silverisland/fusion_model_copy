import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryGenerator(nn.Module):
    """
    Dynamically generates queries based on input data statistics (Mean, Std, Max, Min).
    """
    def __init__(self, n_features, num_queries, d_fusion):
        super().__init__()
        self.num_queries = num_queries
        self.d_fusion = d_fusion
        input_dim = n_features * 4
        
        self.generator = nn.Sequential(
            nn.Linear(input_dim, d_fusion),
            nn.GELU(),
            nn.Linear(d_fusion, num_queries * d_fusion),
            nn.LayerNorm(num_queries * d_fusion)
        )
        
    def forward(self, x):
        B, L, C = x.shape
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        max_val, _ = x.max(dim=1)
        min_val, _ = x.min(dim=1)
        stats = torch.cat([mean, std, max_val, min_val], dim=-1)
        queries = self.generator(stats)
        return queries.view(B, self.num_queries, self.d_fusion)

class SoftMoELayer(nn.Module):
    """
    Soft MoE implementation for feature-level fusion.
    Optimized for (Batch, Channel, Dim) token structure.
    """
    def __init__(self, d_model, num_experts, slots_per_expert):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.num_slots = num_experts * slots_per_expert
        
        self.phi = nn.Parameter(torch.randn(d_model, self.num_slots) * 0.02)
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(0.1)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        B, N, D = x.shape
        logits = torch.matmul(x, self.phi) 
        dispatch_weights = F.softmax(logits, dim=1) 
        combine_weights = F.softmax(logits, dim=2) 
        
        slots_input = torch.matmul(dispatch_weights.transpose(1, 2), x) 
        
        slots_output = []
        for i in range(self.num_experts):
            start = i * self.slots_per_expert
            end = (i + 1) * self.slots_per_expert
            expert_in = slots_input[:, start:end, :]
            expert_out = self.experts[i](expert_in)
            slots_output.append(expert_out)
        
        slots_output = torch.cat(slots_output, dim=1) 
        out = torch.matmul(combine_weights, slots_output) 
        return out

class FusionModelV3(nn.Module):
    """
    FusionModel V3 Tensor version (High Precision Edition):
    - Uses pre-computed expert hidden states from batch_tensor.
    - Integrates QueryGenerator for dynamic feature extraction.
    - Uses enhanced projectors (Linear-GELU-Linear + LayerNorm).
    - Includes Huber + Trend Loss for better generalization.
    """
    def __init__(self, models_dict, seq_len, pred_len, n_features, 
                 num_queries=16, d_fusion=512, num_experts=6, device='cuda'):
        super().__init__()
        self.device = device
        self.pred_len = pred_len
        self.n_features = n_features
        self.d_fusion = d_fusion
        self.num_queries = num_queries
        self.expert_names = list(models_dict.keys())

        # 1. Dynamic Query Generation
        self.query_gen = QueryGenerator(n_features, num_queries, d_fusion)

        # 2. Enhanced Projectors and Per-Expert Attention
        self.projectors = nn.ModuleDict()
        self.proj_norms = nn.ModuleDict()
        self.expert_attns = nn.ModuleDict()

        for name in self.expert_names:
            # Dimension mapping based on known expert outputs
            if name == 'm1': in_dim = 512
            elif name == 'm2': in_dim = 1024
            elif name == 'm3': in_dim = 62208
            elif name == 'm4': in_dim = 11520
            else: in_dim = d_fusion

            self.projectors[name] = nn.Sequential(
                nn.Linear(in_dim, d_fusion),
                nn.GELU(),
                nn.Linear(d_fusion, d_fusion)
            )
            self.proj_norms[name] = nn.LayerNorm(d_fusion)
            self.expert_attns[name] = nn.MultiheadAttention(d_fusion, num_heads=8, batch_first=True)

        # 3. Global Fusion components
        self.soft_moe = SoftMoELayer(d_fusion, num_experts=num_experts, slots_per_expert=4)
        self.norm = nn.LayerNorm(d_fusion)
        
        # 4. Output head logic from fusion.py
        self.output_head = nn.Linear(d_fusion, pred_len)
        self.total_tokens = num_queries * len(self.expert_names)
        self.aggregate = nn.Conv1d(self.total_tokens, self.n_features, 1)
        
        self.to(device)

    def forward(self, batch_tensor, batch, flag='test'):
        B = batch['observe_power'].shape[0]
        
        # 1. Generate dynamic queries
        q_gen = self.query_gen(batch['observe_power'].unsqueeze(-1)) # (B, Q, D)

        # 2. Distill features from each expert
        all_expert_summaries = []
        for name in self.expert_names:
            h = batch_tensor[name].flatten(1).unsqueeze(1) # (B, 1, D_in)
            
            # Project and Normalize
            h_proj = self.projectors[name](h)
            h_proj = self.proj_norms[name](h_proj)
            
            # Use dynamic queries to extract info from this expert
            summary, _ = self.expert_attns[name](q_gen, h_proj, h_proj) # (B, Q, D)
            all_expert_summaries.append(summary)

        # 3. Global Interaction with SoftMoE
        combined_tokens = torch.cat(all_expert_summaries, dim=1) # (B, total_tokens, D)
        fused_tokens = self.soft_moe(combined_tokens)
        fused_tokens = self.norm(combined_tokens + fused_tokens)
        
        # 4. Hierarchical Prediction
        out = self.output_head(fused_tokens) # (B, total_tokens, pred_len)
        output = self.aggregate(out) # (B, n_features, pred_len)

        if flag == 'test':
            return output
        else:
            return output, self.loss_func(output, batch['target_power'])

    def loss_func(self, pred, target):
        """
        Combined Loss: Huber (Robustness) + Trend (Gradient matching)
        """
        huber = nn.HuberLoss(delta=1.0)
        mse = nn.MSELoss()
        
        loss_val = huber(pred, target)
        
        if pred.shape[-1] > 1:
            diff_pred = pred[:, :, 1:] - pred[:, :, :-1]
            diff_target = target[:, :, 1:] - target[:, :, :-1]
            loss_trend = mse(diff_pred, diff_target)
            return loss_val + 0.5 * loss_trend
        
        return loss_val
