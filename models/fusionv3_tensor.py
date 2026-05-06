import torch
import torch.nn as nn
import torch.nn.functional as F

class FastSoftMoE(nn.Module):
    """
    Fully vectorized Soft MoE implementation using einsum.
    Eliminates Python loops for fast expert processing.
    """
    def __init__(self, d_model, num_experts, slots_per_expert):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.num_slots = num_experts * slots_per_expert
        
        self.phi = nn.Parameter(torch.randn(d_model, self.num_slots) * 0.02)
        
        # Vectorized expert weights: (Expert, In, Out)
        self.w1 = nn.Parameter(torch.randn(num_experts, d_model, d_model * 2) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, d_model * 2, d_model) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(num_experts, 1, d_model * 2))
        self.b2 = nn.Parameter(torch.zeros(num_experts, 1, d_model))
        
    def forward(self, x):
        B, N, D = x.shape
        
        # Compute routing logits
        logits = torch.matmul(x, self.phi) # (B, N, S)
        dispatch_weights = F.softmax(logits, dim=1) 
        combine_weights = F.softmax(logits, dim=2) 
        
        # 1. Dispatch tokens to slots
        slots_input = torch.matmul(dispatch_weights.transpose(1, 2), x) # (B, S, D)
        slots_input = slots_input.view(B, self.num_experts, self.slots_per_expert, D)
        
        # 2. Parallel processing by experts (Einsum)
        h = torch.einsum('besd,edh->besh', slots_input, self.w1) + self.b1
        h = F.gelu(h)
        slots_output = torch.einsum('besh,ehd->besd', h, self.w2) + self.b2
        
        # 3. Combine slots back
        slots_output = slots_output.reshape(B, self.num_slots, D)
        out = torch.matmul(combine_weights, slots_output) 
        return out

class FusionModelV3(nn.Module):
    """
    FusionModel V3 Tensor version (Clean Direct Regression):
    - Independent learnable queries for each expert.
    - Global SoftMoE fusion.
    - Direct regression on physical power values (No RevIN/Normalization).
    """
    def __init__(self, models_dict, seq_len, pred_len, n_features, 
                 queries_per_expert=8, d_fusion=512, num_experts=6, device='cuda'):
        super().__init__()
        self.device = device
        self.pred_len = pred_len
        self.n_features = n_features
        self.d_fusion = d_fusion
        self.queries_per_expert = queries_per_expert
        self.expert_names = list(models_dict.keys())
        num_exp = len(self.expert_names)

        # 1. Dedicated Learnable Queries
        self.expert_queries = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, queries_per_expert, d_fusion) * 0.02)
            for name in self.expert_names
        })

        # 2. Projectors and Independent Attention Heads
        self.projectors = nn.ModuleDict()
        self.expert_attns = nn.ModuleDict()
        for name in self.expert_names:
            # Dimension mapping based on expert hidden state output
            d_in = {'m1': 512, 'm2': 256, 'm3': 384, 'm4': 512}.get(name, d_fusion)
            self.projectors[name] = nn.Linear(d_in, d_fusion)
            self.expert_attns[name] = nn.MultiheadAttention(d_fusion, num_heads=8, batch_first=True)

        # 3. Global Fusion via Vectorized SoftMoE
        self.soft_moe = FastSoftMoE(d_fusion, num_experts=num_experts, slots_per_expert=4)
        self.norm = nn.LayerNorm(d_fusion)
        
        # 4. Output head
        self.output_head = nn.Linear(d_fusion, pred_len)
        self.total_tokens = queries_per_expert * num_exp
        self.aggregate = nn.Conv1d(self.total_tokens, self.n_features, 1)
        
        # Initialization
        nn.init.normal_(self.output_head.weight, std=0.01)
        nn.init.zeros_(self.output_head.bias)
        self.to(device)

    def forward(self, batch_tensor, batch, flag='test'):
        B = batch['observe_power'].shape[0]
        
        # 1. Distill features using per-expert dedicated queries
        all_expert_summaries = []
        for name in self.expert_names:
            h = batch_tensor[name]
            # Handle structured expert features
            if name == 'm2' and h.dim() == 4: h = h.flatten(1, 2)
            elif h.dim() == 2: h = h.unsqueeze(1)
            
            h_proj = self.projectors[name](h)
            q = self.expert_queries[name].expand(B, -1, -1)
            summary, _ = self.expert_attns[name](q, h_proj, h_proj)
            all_expert_summaries.append(summary)

        # 2. Global Interaction with SoftMoE
        combined_tokens = torch.cat(all_expert_summaries, dim=1) # (B, E*Q, D)
        fused_tokens = self.soft_moe(combined_tokens)
        fused_tokens = self.norm(combined_tokens + fused_tokens)
        
        # 3. Final Prediction (Direct physical scale)
        out = self.output_head(fused_tokens) # (B, E*Q, pred_len)
        output = self.aggregate(out) # (B, n_features, pred_len)

        if flag == 'test':
            return output
        else:
            # 4. Loss calculation directly on physical values
            target_raw = batch['target_power']
            if target_raw.dim() == 2:
                target_raw = target_raw.unsqueeze(1)
            
            loss = self.loss_func(output, target_raw)
            return output, loss

    def loss_func(self, pred, target):
        huber = nn.HuberLoss(delta=1.0)
        mse = nn.MSELoss()
        loss_val = huber(pred, target)
        if pred.shape[-1] > 1:
            diff_pred = pred[:, :, 1:] - pred[:, :, :-1]
            diff_target = target[:, :, 1:] - target[:, :, :-1]
            loss_trend = mse(diff_pred, diff_target)
            return loss_val + 0.5 * loss_trend
        return loss_val
