import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class QueryGenerator(nn.Module):
    """
    Dynamically generates queries based on input data statistics (Mean, Std, Max, Min).
    """
    def __init__(self, n_features, num_queries, d_fusion):
        super().__init__()
        self.num_queries = num_queries
        self.d_fusion = d_fusion
        
        # 4 stats per feature
        input_dim = n_features * 4
        
        self.generator = nn.Sequential(
            nn.Linear(input_dim, d_fusion),
            nn.GELU(),
            nn.Linear(d_fusion, num_queries * d_fusion),
            nn.LayerNorm(num_queries * d_fusion)
        )
        
    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        
        # 1. Global statistics extraction
        mean = x.mean(dim=1) # (B, C)
        std = x.std(dim=1)   # (B, C)
        max_val, _ = x.max(dim=1) # (B, C)
        min_val, _ = x.min(dim=1) # (B, C)
        
        stats = torch.cat([mean, std, max_val, min_val], dim=-1) # (B, 4C)
        
        # 2. Dynamic generation
        queries = self.generator(stats) # (B, num_queries * d_fusion)
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
        
        # Learnable parameters for the dispatch/combine routing
        self.phi = nn.Parameter(torch.randn(d_model, self.num_slots) * 0.02)
        
        # Experts: Small MLP specialized in different patterns
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
        
        # Compute routing logits
        logits = torch.matmul(x, self.phi) # (B, N, S)
        
        # Dispatch weights: Normalize over input tokens
        dispatch_weights = F.softmax(logits, dim=1) # (B, N, S)
        
        # Combine weights: Normalize over slots
        combine_weights = F.softmax(logits, dim=2) # (B, N, S)
        
        # 1. Dispatch tokens to slots
        slots_input = torch.matmul(dispatch_weights.transpose(1, 2), x) # (B, S, D)
        
        # 2. Parallel processing by experts
        slots_output = []
        for i in range(self.num_experts):
            start = i * self.slots_per_expert
            end = (i + 1) * self.slots_per_expert
            expert_in = slots_input[:, start:end, :]
            expert_out = self.experts[i](expert_in)
            slots_output.append(expert_out)
        
        slots_output = torch.cat(slots_output, dim=1) # (B, S, D)
        
        # 3. Combine slots back to original token shape
        out = torch.matmul(combine_weights, slots_output) # (B, N, D)
        return out



class FusionModel(nn.Module):
    def __init__(self, models_dict, seq_len, pred_len, n_features, 
                 num_queries=16, d_fusion=512, num_experts=6, device='cuda'):
        super().__init__()
        self.models_dict = nn.ModuleDict(models_dict)
        self.device = device
        self.pred_len = pred_len
        self.n_features = n_features
       
        self.num_queries = num_queries if num_queries is not None else n_features

        self.use_dynamic_queries = True 
        if self.use_dynamic_queries:
            self.query_gen = QueryGenerator(n_features, self.num_queries, d_fusion)
            self.queries = None 
        else:
            query_init_type = 'orthogonal'
            self.queries = nn.Parameter(torch.empty(1, self.num_queries, d_fusion))
            self._init_queries(query_init_type)

        self.cross_attn = nn.MultiheadAttention(d_fusion, num_heads = 8, batch_first = True)
        self.norm1 = nn.LayerNorm(d_fusion)

        # Individual projectors to summarize base models
        self.projectors = nn.ModuleDict()
        self.proj_norms = nn.ModuleDict() # For scale alignment
        
        for name, model in self.models_dict.items():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            
            # Use appropriate hidden dims - pooling handles the L dimension
            if name == 'm1': in_dim = 512
            elif name == 'm2': in_dim = 1024
            elif name == 'm3': in_dim = 512
            elif name == 'm4': in_dim = 512
            else: in_dim = d_fusion

            self.projectors[name] = nn.Sequential(
                nn.Linear(in_dim, d_fusion),
                nn.GELU(),
                nn.Linear(d_fusion, d_fion)
            )
            self.proj_norms[name] = nn.LayerNorm(d_fusion)

        self.soft_moe = SoftMoELayer(d_fusion, num_experts = num_experts, slots_per_expert = 4)
        self.norm2 = nn.LayerNorm(d_fusion)

        self.output_head = nn.Linear(d_fusion, pred_len)
        self.aggregate = nn.Conv1d(self.num_queries, self.n_features, 1)
        self.dropout = nn.Dropout(0.2)
        self.to(device)

    def _init_queries(self, init_type): 
        import math 
        q = self.queries.data 
        d_fusion = q.size(-1)

        if init_type == 'normal':
            nn.init.normal_(q, std = 0.02)
        elif init_type == 'orthogonal':
            flag_q = torch.empty(self.num_queries, d_fusion)
            nn.init.orthogonal_(flag_q)
            q.copy_(flag_q.unsqueeze(0))
        elif init_type == 'fourier':
            for i in range(self.num_queries):
                for j in range(d_fusion // 2):
                    val = i / math.pow(10000, 2 * j / d_fusion)
                    q[0, i, 2 * j] = math.sin(val)
                    q[0, i, 2 * j + 1] = math.cos(val) 
            q.mul_(0.02) 
        else:
            raise ValueError(f"Unknown query initialization type: {init_type}")                    

    def forward(self, batch):
        B = batch['observe_power'].shape[0]

        # 1. Distill features from each expert
        all_tokens = []
        for name, model in self.models_dict.items():
            model.eval() 
            with torch.no_grad():
                h = model.forward_hidden(batch) # Expected (B, L, D) or (B, D)
            
            # Global Average Pooling to prevent overfitting on specific time steps
            if h.dim() == 3:
                h_pooled = h.mean(dim=1) # (B, D)
            elif h.dim() == 4:
                h_pooled = h.mean(dim=(1, 2)) # Handle potential (B, C, L, D)
            else:
                h_pooled = h.flatten(1)
            
            # The projector must be outside no_grad to update its weights
            proj_h = self.projectors[name](h_pooled).unsqueeze(1) # (B, 1, d_fusion)
            proj_h = self.proj_norms[name](proj_h)
            all_tokens.append(proj_h)
        
        kv = torch.cat(all_tokens, dim = 1)
        if self.use_dynamic_queries:
            q = self.query_gen(batch['observe_power'].unsqueeze(-1))
        else:
            q = self.queries.expand(B, -1, -1)
        
        attn_out, _ = self.cross_attn(q, kv, kv)
        packed_tokens = self.norm1(q + self.dropout(attn_out))
        
        # 3. Global Fusion
        fused_tokens = self.soft_moe(packed_tokens)
        fused_tokens = self.norm2(packed_tokens + fused_tokens)
        
        # 4. Predict
        out = self.output_head(fused_tokens) 
        
        if self.num_queries == self.n_features:
            output = out.transpose(1, 2)
        else:
            output = self.aggregate(out).squeeze(1)
        
        return output
