import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSoftMoE(nn.Module):
    """
    Vectorized Soft MoE for interactions among aligned hidden tokens.
    """
    def __init__(self, d_model, num_experts, slots_per_expert):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.num_slots = num_experts * slots_per_expert

        self.phi = nn.Parameter(torch.randn(d_model, self.num_slots) * 0.02)
        self.w1 = nn.Parameter(torch.randn(num_experts, d_model, d_model * 2) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, d_model * 2, d_model) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(num_experts, 1, d_model * 2))
        self.b2 = nn.Parameter(torch.zeros(num_experts, 1, d_model))

    def forward(self, x):
        B, N, D = x.shape

        logits = torch.matmul(x, self.phi)
        dispatch_weights = F.softmax(logits, dim=1)
        combine_weights = F.softmax(logits, dim=2)

        slots_input = torch.matmul(dispatch_weights.transpose(1, 2), x)
        slots_input = slots_input.view(B, self.num_experts, self.slots_per_expert, D)

        h = torch.einsum('besd,edh->besh', slots_input, self.w1) + self.b1
        h = F.gelu(h)
        slots_output = torch.einsum('besh,ehd->besd', h, self.w2) + self.b2

        slots_output = slots_output.reshape(B, self.num_slots, D)
        return torch.matmul(combine_weights, slots_output)


class HiddenStatsGate(nn.Module):
    """
    Predicts expert reliability from hidden summaries only.
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        stat_dim = d_model * 5
        self.scorer = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )

    def forward(self, expert_tokens):
        stats = []
        for tokens in expert_tokens:
            mean = tokens.mean(dim=1)
            std = tokens.std(dim=1, unbiased=False)
            max_val = tokens.max(dim=1).values
            min_val = tokens.min(dim=1).values
            norm = tokens.norm(dim=-1).mean(dim=1, keepdim=True).expand_as(mean)
            stats.append(torch.cat([mean, std, max_val, min_val, norm], dim=-1))

        logits = torch.cat([self.scorer(s) for s in stats], dim=-1)
        return F.softmax(logits, dim=-1), logits


class QueryWeightedForecastHead(nn.Module):
    """
    Each token emits a forecast; a dynamic token gate aggregates forecasts.
    """
    def __init__(self, d_model, pred_len, n_features, dropout=0.1):
        super().__init__()
        hidden_dim = max(d_model // 2, 1)
        self.pred_len = pred_len
        self.n_features = n_features

        self.token_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_features * pred_len),
        )
        self.token_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens):
        B, T, _ = tokens.shape
        pred = self.token_head(tokens).view(B, T, self.n_features, self.pred_len)
        weight = F.softmax(self.token_gate(tokens), dim=1).view(B, T, 1, 1)
        return (pred * weight).sum(dim=1), weight.squeeze(-1).squeeze(-1)


class FusionModelV5(nn.Module):
    """
    Hidden-only fusion V5:
    - keeps V4's per-expert hidden alignment and hidden-stat reliability gate;
    - adds internal auxiliary decoders for each expert hidden summary;
    - mixes the fused decoder with the hidden-auxiliary decoder ensemble;
    - supervises the reliability gate with aux-error pseudo targets;
    - adds a guard loss so final output does not degrade below aux hidden mean.

    This model does not use expert prediction values. Auxiliary predictions are
    decoded inside the fusion model from expert hidden vectors.
    """
    def __init__(
        self,
        models_dict,
        seq_len,
        pred_len,
        n_features,
        queries_per_expert=8,
        d_fusion=512,
        num_experts=6,
        slots_per_expert=4,
        num_fusion_layers=1,
        num_heads=8,
        dropout=0.1,
        aux_loss_weight=0.1,
        gate_loss_weight=0.1,
        guard_loss_weight=0.2,
        gate_temperature=0.5,
        expert_dims=None,
        target_key='target_power',
        device='cuda',
    ):
        super().__init__()
        self.device = device
        self.pred_len = pred_len
        self.n_features = n_features
        self.d_fusion = d_fusion
        self.queries_per_expert = queries_per_expert
        self.aux_loss_weight = aux_loss_weight
        self.gate_loss_weight = gate_loss_weight
        self.guard_loss_weight = guard_loss_weight
        self.gate_temperature = gate_temperature
        self.target_key = target_key
        self.expert_names = list(models_dict.keys())
        self.total_tokens = queries_per_expert * len(self.expert_names)

        default_dims = {'m1': 512, 'm2': 256, 'm3': 384, 'm4': 512}
        if expert_dims is not None:
            default_dims.update(expert_dims)

        self.projectors = nn.ModuleDict()
        self.expert_queries = nn.ParameterDict()
        self.expert_attns = nn.ModuleDict()
        self.expert_norms = nn.ModuleDict()
        self.aux_heads = nn.ModuleDict()

        for name in self.expert_names:
            d_in = default_dims.get(name, d_fusion)
            self.projectors[name] = nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, d_fusion),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_fusion, d_fusion),
            )
            self.expert_queries[name] = nn.Parameter(
                torch.randn(1, queries_per_expert, d_fusion) * 0.02
            )
            self.expert_attns[name] = nn.MultiheadAttention(
                d_fusion, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.expert_norms[name] = nn.LayerNorm(d_fusion)
            self.aux_heads[name] = QueryWeightedForecastHead(
                d_fusion, pred_len, n_features, dropout=dropout
            )

        self.hidden_gate = HiddenStatsGate(d_fusion, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_fusion,
            nhead=num_heads,
            dim_feedforward=d_fusion * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.cross_expert_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_fusion_layers
        )

        self.soft_moe = FastSoftMoE(
            d_fusion, num_experts=num_experts, slots_per_expert=slots_per_expert
        )
        self.fusion_norm = nn.LayerNorm(d_fusion)
        self.dropout = nn.Dropout(dropout)
        self.final_head = QueryWeightedForecastHead(
            d_fusion, pred_len, n_features, dropout=dropout
        )
        self.mix_gate = nn.Sequential(
            nn.LayerNorm(d_fusion),
            nn.Linear(d_fusion, max(d_fusion // 2, 1)),
            nn.GELU(),
            nn.Linear(max(d_fusion // 2, 1), 1),
        )

        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _as_tokens(self, h):
        if h.dim() == 2:
            return h.unsqueeze(1)
        if h.dim() == 3:
            return h

        B = h.shape[0]
        return h.reshape(B, -1, h.shape[-1])

    def _get_target(self, batch):
        if self.target_key in batch:
            target = batch[self.target_key]
        elif 'observe_power_future' in batch:
            target = batch['observe_power_future']
        else:
            raise KeyError(
                f"Cannot find target key '{self.target_key}' or 'observe_power_future' in batch."
            )

        if target.dim() == 2:
            target = target.unsqueeze(1)
        elif target.dim() == 3 and target.shape[1] == self.pred_len:
            target = target.transpose(1, 2)
        return target

    def _forecast_loss(self, pred, target):
        huber = F.huber_loss(pred, target, delta=1.0, reduction='none')
        peak_weight = 1.0 + 0.5 * target.detach().clamp(min=0.0)
        loss_val = (huber * peak_weight).mean()

        if pred.shape[-1] > 1:
            diff_pred = pred[:, :, 1:] - pred[:, :, :-1]
            diff_target = target[:, :, 1:] - target[:, :, :-1]
            loss_trend = F.mse_loss(diff_pred, diff_target)
            return loss_val + 0.5 * loss_trend

        return loss_val

    def _sample_mae(self, pred, target):
        return (pred - target).abs().mean(dim=(1, 2))

    def forward(self, batch_tensor, batch, flag='test', mode='full', return_info=False):
        B = batch['observe_power'].shape[0]

        expert_summaries = []
        aux_preds = []
        for name in self.expert_names:
            h = self._as_tokens(batch_tensor[name])
            h = self.projectors[name](h)

            q = self.expert_queries[name].expand(B, -1, -1)
            summary, _ = self.expert_attns[name](q, h, h)
            summary = self.expert_norms[name](q + self.dropout(summary))
            expert_summaries.append(summary)

            aux_pred, _ = self.aux_heads[name](summary)
            aux_preds.append(aux_pred)

        expert_weight, gate_logits = self.hidden_gate(expert_summaries)
        aux_pred_stack = torch.stack(aux_preds, dim=1)
        aux_mean = aux_pred_stack.mean(dim=1)
        aux_mix = (aux_pred_stack * expert_weight.view(B, -1, 1, 1)).sum(dim=1)

        weighted_summaries = [
            tokens * expert_weight[:, i].view(B, 1, 1)
            for i, tokens in enumerate(expert_summaries)
        ]
        combined_tokens = torch.cat(weighted_summaries, dim=1)

        encoded_tokens = self.cross_expert_encoder(combined_tokens)
        moe_tokens = self.soft_moe(encoded_tokens)
        fused_tokens = self.fusion_norm(encoded_tokens + self.dropout(moe_tokens))

        final_pred, query_weight = self.final_head(fused_tokens)
        mix_weight = torch.sigmoid(self.mix_gate(fused_tokens.mean(dim=1))).view(B, 1, 1)
        output = mix_weight * final_pred + (1.0 - mix_weight) * aux_mix

        info = {
            'expert_weight': expert_weight,
            'gate_logits': gate_logits,
            'query_weight': query_weight,
            'mix_weight': mix_weight.squeeze(-1).squeeze(-1),
            'aux_mean': aux_mean,
            'aux_mix': aux_mix,
            'final_pred': final_pred,
            'aux_pred_stack': aux_pred_stack,
        }

        if flag == 'test':
            if return_info:
                return output, info
            return output

        target = self._get_target(batch)
        aux_losses = torch.stack([
            self._forecast_loss(aux_pred, target) for aux_pred in aux_preds
        ])

        if mode == 'aux':
            loss = aux_losses.mean()
            if return_info:
                return aux_mean, loss, info
            return aux_mean, loss

        main_loss = self._forecast_loss(output, target)
        loss = main_loss + self.aux_loss_weight * aux_losses.mean()

        if self.gate_loss_weight > 0:
            aux_err = torch.stack([
                self._sample_mae(aux_pred, target) for aux_pred in aux_preds
            ], dim=1)
            pseudo_weight = F.softmax(
                -aux_err.detach() / max(self.gate_temperature, 1e-6), dim=-1
            )
            gate_loss = F.kl_div(
                (expert_weight + 1e-8).log(),
                pseudo_weight,
                reduction='batchmean',
            )
            loss = loss + self.gate_loss_weight * gate_loss
            info['gate_loss'] = gate_loss.detach()
            info['pseudo_weight'] = pseudo_weight

        if self.guard_loss_weight > 0:
            err_output = self._sample_mae(output, target)
            err_aux_mean = self._sample_mae(aux_mean.detach(), target)
            guard_loss = F.relu(err_output - err_aux_mean).mean()
            loss = loss + self.guard_loss_weight * guard_loss
            info['guard_loss'] = guard_loss.detach()

        info['main_loss'] = main_loss.detach()
        info['aux_loss'] = aux_losses.mean().detach()

        if return_info:
            return output, loss, info
        return output, loss

    def loss_func(self, pred, target):
        if target.dim() == 2:
            target = target.unsqueeze(1)
        elif target.dim() == 3 and target.shape[1] == self.pred_len:
            target = target.transpose(1, 2)
        return self._forecast_loss(pred, target)


FusionModel = FusionModelV5
