import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CrossAttnMapper(nn.Module):
    """
    Expert-specific mapper that uses a fixed number of queries (dynamic or static)
    to aggregate information from the expert's hidden states.
    """
    def __init__(self, in_dim, d_fusion, n_features, queries_per_expert=16, 
                 use_dynamic_queries=True, num_heads=8, dropout=0.1):
        super().__init__()
        self.use_dynamic_queries = use_dynamic_queries
        
        # Feature alignment
        self.input_proj = nn.Linear(in_dim, d_fusion)
        
        # Query source
        if use_dynamic_queries:
            self.query_gen = QueryGenerator(n_features, queries_per_expert, d_fusion)
        else:
            self.static_queries = nn.Parameter(torch.randn(1, queries_per_expert, d_fusion) * 0.02)
        
        self.cross_attn = nn.MultiheadAttention(d_fusion, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_fusion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, x_raw=None):
        # Handle 4D or 3D expert outputs
        if h.dim() == 4:
            B, C, P, D = h.shape
            h = h.reshape(B, C * P, D)
        
        B, N, D = h.shape
        h_proj = self.input_proj(h)
        
        # Generate Queries
        if self.use_dynamic_queries:
            q = self.query_gen(x_raw)
        else:
            q = self.static_queries.expand(B, -1, -1)
            
        # Distill information
        attn_out, _ = self.cross_attn(q, h_proj, h_proj)
        return self.norm(q + self.dropout(attn_out))

class MultiHeadPredictor(nn.Module):
    """
    Inspired by TabM's multi-hypothesis ensemble. 
    Uses multiple independent heads to generate predictions.
    """
    def __init__(self, d_fusion, pred_len, n_heads=4, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_fusion, pred_len)
            ) for _ in range(n_heads)
        ])
        
    def forward(self, x):
        preds = torch.stack([head(x) for head in self.heads], dim=0)
        if self.training:
            return preds
        else:
            return preds.mean(dim=0)

class FusionModel(nn.Module):
    """
    Hierarchical Fusion Architecture:
    1. Each expert uses a CrossAttnMapper to distill information into K queries.
    2. Queries are concatenated across all experts (M * K tokens).
    3. Soft MoE performs cross-expert fusion on these balanced tokens.
    """
    def __init__(self, models_dict, seq_len, pred_len, n_features, 
                 queries_per_expert=16, d_fusion=256, num_experts=4, device='cpu', 
                 use_dynamic_queries=True, dropout=0.1):
        super().__init__()
        self.models_dict = nn.ModuleDict(models_dict)
        self.device = device
        self.pred_len = pred_len
        self.n_features = n_features
        self.use_dynamic_queries = use_dynamic_queries
        self.dropout_rate = dropout
        self.queries_per_expert = queries_per_expert

        # Individual projectors to summarize base models
        self.projectors = nn.ModuleDict()
        for name, model in self.models_dict.items():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            
            # Infer hidden dim
            dummy_batch = {
                'x': torch.zeros(1, seq_len, n_features).to(device),
                'observe_power': torch.zeros(1, seq_len, n_features).to(device)
            }
            with torch.no_grad():
                h = model.forward_hidden(dummy_batch)
                in_dim = h.shape[-1]
                self.projectors[name] = CrossAttnMapper(
                    in_dim, d_fusion, n_features, 
                    queries_per_expert=queries_per_expert,
                    use_dynamic_queries=use_dynamic_queries,
                    dropout=self.dropout_rate
                )
        
        self.num_queries = len(self.models_dict) * queries_per_expert

        # Fusion and Prediction
        self.soft_moe = SoftMoELayer(d_fusion, num_experts=num_experts, slots_per_expert=4)
        self.norm = nn.LayerNorm(d_fusion)
        self.output_head = MultiHeadPredictor(d_fusion, pred_len, n_heads=4, dropout=self.dropout_rate)
        self.aggregate = nn.Conv1d(self.num_queries, self.n_features, 1)
        
        self.to(device)

    def forward(self, batch):
        x_raw = batch['x']
        
        # 1. Distill features from each expert
        all_tokens = []
        with torch.no_grad():
            for name, model in self.models_dict.items():
                h = model.forward_hidden(batch) 
                proj_h = self.projectors[name](h, x_raw=x_raw)
                all_tokens.append(proj_h)
        
        # 2. Concatenate balanced tokens: (B, Num_Models * K, d_fusion)
        packed_tokens = torch.cat(all_tokens, dim=1)
        
        # 3. Global Fusion
        fused_tokens = self.soft_moe(packed_tokens)
        fused_tokens = self.norm(packed_tokens + fused_tokens)
        
        # 4. Predict
        out = self.output_head(fused_tokens) 
        
        # 5. Aggregate back to feature dimension
        if out.dim() == 4: # Training
            n_heads, B, Q, P = out.shape
            out_flat = out.view(n_heads * B, Q, P)
            output = self.aggregate(out_flat)
            output = output.view(n_heads, B, self.n_features, P)
            return output.transpose(2, 3)
        
        output = self.aggregate(out)
        return output.transpose(1, 2)

class FusionFeatureModel(nn.Module):
    """
    Feature-input version of the refactored FusionModel.
    """
    def __init__(self, expert_dims, pred_len, n_features, 
                 queries_per_expert=16, d_fusion=256, num_experts=4, device='cpu', 
                 use_dynamic_queries=True, dropout=0.1):
        super().__init__()
        self.device = device
        self.pred_len = pred_len
        self.n_features = n_features
        self.use_dynamic_queries = use_dynamic_queries
        self.dropout_rate = dropout
        self.expert_names = list(expert_dims.keys())
        self.queries_per_expert = queries_per_expert

        self.projectors = nn.ModuleDict()
        for name, in_dim in expert_dims.items():
            self.projectors[name] = CrossAttnMapper(
                in_dim, d_fusion, n_features,
                queries_per_expert=queries_per_expert,
                use_dynamic_queries=use_dynamic_queries,
                dropout=self.dropout_rate
            )
        
        self.num_queries = len(expert_dims) * queries_per_expert
        self.soft_moe = SoftMoELayer(d_fusion, num_experts=num_experts, slots_per_expert=4)
        self.norm = nn.LayerNorm(d_fusion)
        self.output_head = MultiHeadPredictor(d_fusion, pred_len, n_heads=4, dropout=self.dropout_rate)
        self.aggregate = nn.Conv1d(self.num_queries, self.n_features, 1)
        
        self.to(device)

    def forward(self, hidden_dict, x_input=None):
        # 1. Distill
        all_tokens = []
        for name in self.expert_names:
            h = hidden_dict[name]
            proj_h = self.projectors[name](h, x_raw=x_input)
            all_tokens.append(proj_h)
        
        # 2. Concat and Fuse
        packed_tokens = torch.cat(all_tokens, dim=1)
        fused_tokens = self.soft_moe(packed_tokens)
        fused_tokens = self.norm(packed_tokens + fused_tokens)
        
        # 3. Predict and Aggregate
        out = self.output_head(fused_tokens) 
        if out.dim() == 4:
            n_heads, B, Q, P = out.shape
            out_flat = out.view(n_heads * B, Q, P)
            output = self.aggregate(out_flat)
            output = output.view(n_heads, B, self.n_features, P)
            return output.transpose(2, 3)
        else:
            output = self.aggregate(out)
            return output.transpose(1, 2)
