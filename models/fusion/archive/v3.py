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
    def __init__(self, models_dict, seq_len, pred_len, n_features, 
                 queries_per_expert=8, d_fusion=512, num_experts=6, device='cuda'):
        super().__init__()
        self.models_dict = nn.ModuleDict(models_dict)
        self.device = device
        self.pred_len = pred_len
        self.n_features = n_features
        self.d_fusion = d_fusion
        self.queries_per_expert = queries_per_expert

        # 1. 为每个专家定义投影层（保持序列维度，不使用大的 Flatten Linear）
        self.projectors = nn.ModuleDict()
        self.expert_queries = nn.ParameterDict()
        self.expert_attns = nn.ModuleDict()

        for name, model in self.models_dict.items():
            # 冻结专家模型
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

            # 根据模型输出维度定义投影
            # m1: (B, 2, 512), m2: (B, 5, 9, 256), m3: (B, 164, 384)
            if name == 'm1': d_in = 512
            elif name == 'm2': d_in = 256
            elif name == 'm3': d_in = 384
            elif name == 'm4': d_in = 512 # 假设 m4 维度
            else: d_in = d_fusion

            self.projectors[name] = nn.Linear(d_in, d_fusion)
            
            # 每个专家专用的可学习 Queries
            self.expert_queries[name] = nn.Parameter(torch.randn(1, queries_per_expert, d_fusion) * 0.02)
            
            # 每个专家专用的提炼 Cross-Attention
            self.expert_attns[name] = nn.MultiheadAttention(d_fusion, num_heads=8, batch_first=True)

        # 2. 全局融合层
        self.soft_moe = SoftMoELayer(d_fusion, num_experts=num_experts, slots_per_expert=4)
        self.norm = nn.LayerNorm(d_fusion)
        
        # 3. 输出头
        self.output_head = nn.Linear(d_fusion * queries_per_expert * len(models_dict), n_features * pred_len)
        self.to(device)

    def forward(self, batch):
        B = batch['observe_power'].shape[0]
        all_expert_summaries = []

        for name, model in self.models_dict.items():
            # 1. 提取原始特征
            with torch.no_grad():
                h = model.forward_hidden(batch)
            
            # 2. 维度重塑与投影
            if name == 'm2': # (B, 5, 9, 256) -> (B, 45, 256)
                h = h.view(B, -1, h.size(-1))
            elif h.dim() == 2: # (B, D) -> (B, 1, D)
                h = h.unsqueeze(1)
            
            h_proj = self.projectors[name](h)
            
            # 3. 专家内部提炼 (Cross-Attention)
            q = self.expert_queries[name].expand(B, -1, -1)
            summary, _ = self.expert_attns[name](q, h_proj, h_proj)
            all_expert_summaries.append(summary)

        # 4. 拼接所有专家的提炼 Token (B, num_experts * queries_per_expert, d_fusion)
        combined_tokens = torch.cat(all_expert_summaries, dim=1)
        
        # 5. 全局 SoftMoE 交互
        fused_tokens = self.soft_moe(combined_tokens)
        fused_tokens = self.norm(combined_tokens + fused_tokens)
        
        # 6. 最终预测
        # 展平所有 Token 进行输出映射
        out = self.output_head(fused_tokens.flatten(1))
        return out.view(B, self.n_features, self.pred_len)
