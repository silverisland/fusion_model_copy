import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.revin import RevIN


def orthogonal_linear(in_features, out_features):
    layer = nn.Linear(in_features, out_features, bias=False)
    nn.init.orthogonal_(layer.weight)
    return layer


class FlattenTokenAdapter(nn.Module):
    """
    Flatten-first adapter for experts whose native heads flatten hidden states.

    Each branch projects the last hidden dimension, flattens all tokens, and
    reads out one compact token. K branches produce (B, K, d_model).
    """

    def __init__(
        self,
        input_dim,
        input_tokens,
        token_count,
        d_model,
        dropout=0.0,
        hidden_multiplier=2,
    ):
        super().__init__()
        self.input_tokens = input_tokens
        self.token_count = token_count
        self.d_model = d_model

        self.branch_projectors = nn.ModuleList()
        self.branch_readouts = nn.ModuleList()
        hidden_size = max(d_model, d_model * hidden_multiplier)
        for _ in range(token_count):
            self.branch_projectors.append(
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    orthogonal_linear(input_dim, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            self.branch_readouts.append(
                nn.Sequential(
                    nn.LayerNorm(input_tokens * d_model),
                    nn.Linear(input_tokens * d_model, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )

    def _as_tokens(self, hidden):
        if hidden.dim() == 2:
            return hidden.unsqueeze(1)
        if hidden.dim() == 3:
            return hidden

        batch_size = hidden.shape[0]
        return hidden.reshape(batch_size, -1, hidden.shape[-1])

    def forward(self, hidden):
        tokens = self._as_tokens(hidden)
        if tokens.shape[1] != self.input_tokens:
            raise ValueError(
                f"Expected {self.input_tokens} input tokens, got {tokens.shape[1]}."
            )

        compact_tokens = []
        for projector, readout in zip(self.branch_projectors, self.branch_readouts):
            projected = projector(tokens)
            compact_tokens.append(readout(projected.flatten(start_dim=1)))
        return torch.stack(compact_tokens, dim=1)


class ChannelWiseTokenAdapter(nn.Module):
    """
    M2-style adapter that preserves channel-wise processing before token mixing.
    """

    def __init__(
        self,
        input_dim,
        input_tokens,
        token_count,
        d_model,
        dropout=0.0,
    ):
        super().__init__()
        self.input_tokens = input_tokens
        self.token_count = token_count
        self.d_model = d_model

        self.channel_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, d_model * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, d_model),
                )
                for _ in range(input_tokens)
            ]
        )
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(input_tokens * d_model),
            nn.Linear(input_tokens * d_model, token_count * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, hidden):
        if hidden.dim() != 3:
            raise ValueError(f"M2 adapter expects 3D hidden, got {hidden.dim()}D.")
        if hidden.shape[1] != self.input_tokens:
            raise ValueError(
                f"Expected {self.input_tokens} input tokens, got {hidden.shape[1]}."
            )

        projected = [
            self.channel_projectors[i](hidden[:, i, :])
            for i in range(self.input_tokens)
        ]
        flattened = torch.stack(projected, dim=1).flatten(start_dim=1)
        mixed = self.token_mixer(flattened)
        return mixed.view(hidden.shape[0], self.token_count, self.d_model)


class ExpertSpecificAdapterBank(nn.Module):
    def __init__(
        self,
        expert_names,
        expert_dims,
        input_tokens,
        token_count,
        d_model,
        dropout=0.0,
    ):
        super().__init__()
        self.adapters = nn.ModuleDict()
        for name in expert_names:
            if name == "m2":
                self.adapters[name] = ChannelWiseTokenAdapter(
                    input_dim=expert_dims[name],
                    input_tokens=input_tokens[name],
                    token_count=token_count,
                    d_model=d_model,
                    dropout=dropout,
                )
            else:
                multiplier = 4 if name == "m4" else 2
                self.adapters[name] = FlattenTokenAdapter(
                    input_dim=expert_dims[name],
                    input_tokens=input_tokens[name],
                    token_count=token_count,
                    d_model=d_model,
                    dropout=dropout,
                    hidden_multiplier=multiplier,
                )

    def forward(self, batch_tensor, expert_names):
        return {
            name: self.adapters[name](batch_tensor[name])
            for name in expert_names
        }


class TransformerTokenMixer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm(tokens)


class ConditionalQueryEncoder(nn.Module):
    def __init__(self, seq_len, query_tokens, d_model, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.query_tokens = query_tokens
        self.d_model = d_model
        in_features = 2 * seq_len
        out_features = query_tokens * d_model
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
        )

    def _flatten_signal(self, signal, name):
        if signal.dim() == 3 and signal.shape[1] == 1:
            signal = signal.squeeze(1)
        if signal.dim() != 2:
            raise ValueError(f"{name} must be shaped (B, seq_len), got {tuple(signal.shape)}.")
        if signal.shape[1] != self.seq_len:
            raise ValueError(
                f"{name} length must be {self.seq_len}, got {signal.shape[1]}."
            )
        return signal

    def forward(self, batch):
        observe_power = self._flatten_signal(batch["observe_power"], "observe_power")
        chronos = self._flatten_signal(batch["chronos"], "chronos")
        query_input = torch.cat([observe_power, chronos], dim=-1)
        query = self.encoder(query_input)
        return query.view(query.shape[0], self.query_tokens, self.d_model)


class CrossAttentionForecastDecoder(nn.Module):
    def __init__(self, d_model, pred_len, n_heads, n_layers, query_tokens, dropout=0.0):
        super().__init__()
        self.query_encoder = None
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "query_norm": nn.LayerNorm(d_model),
                        "context_norm": nn.LayerNorm(d_model),
                        "attention": nn.MultiheadAttention(
                            d_model,
                            n_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "dropout": nn.Dropout(dropout),
                        "ffn_norm": nn.LayerNorm(d_model),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_model * 4, d_model),
                            nn.Dropout(dropout),
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(query_tokens * d_model),
            nn.Linear(query_tokens * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len),
        )

    def forward(self, query, context):
        attention_weights = []
        for block in self.blocks:
            norm_query = block["query_norm"](query)
            norm_context = block["context_norm"](context)
            attn_output, attn_weights = block["attention"](
                norm_query,
                norm_context,
                norm_context,
                need_weights=True,
                average_attn_weights=False,
            )
            query = query + block["dropout"](attn_output)
            query = query + block["ffn"](block["ffn_norm"](query))
            attention_weights.append(attn_weights)

        forecast = self.output_head(query.flatten(start_dim=1))
        return forecast, {
            "query_tokens": query,
            "attention_weights": attention_weights,
        }


class SharedAuxiliaryForecastHead(nn.Module):
    def __init__(self, token_count, d_model, pred_len, dropout=0.0):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(token_count * d_model),
            nn.Linear(token_count * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len),
        )

    def forward(self, tokens):
        return self.head(tokens.flatten(start_dim=1))


class ExpertSpecificAttentionFusion(nn.Module):
    """
    V6 hidden-level fusion.

    Expert hidden states are converted by expert-specific adapters to a shared
    token format. Tokens are mixed by self-attention, read by sample-conditioned
    forecast queries, and decoded directly to the final prediction. Auxiliary
    expert supervision is applied on each expert's compact tokens.
    """

    DEFAULT_EXPERT_DIMS = {"m1": 128, "m2": 512, "m3": 384, "m4": 256}
    DEFAULT_INPUT_TOKENS = {"m1": 9, "m2": 2, "m3": 162, "m4": 45}
    SUPPORTED_LOSSES = {"mse", "mae", "huber", "rmse"}

    def __init__(
        self,
        models_dict=None,
        seq_len=96,
        pred_len=192,
        n_features=1,
        expert_dims=None,
        expert_names=None,
        aligned_token_count=6,
        d_fusion=None,
        dropout=0.0,
        target_key="observe_power_future",
        loss_type="mse",
        aux_loss_weight=0.1,
        attention_heads=4,
        attention_layers=1,
        attention_query_tokens=2,
        device="cuda",
    ):
        super().__init__()
        self.seq_len = 96 if seq_len is None else seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.d_fusion = 128 if d_fusion is None else d_fusion
        self.aligned_token_count = aligned_token_count
        self.target_key = target_key
        self.loss_type = loss_type
        self.aux_loss_weight = aux_loss_weight
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.attention_query_tokens = attention_query_tokens
        self.expert_names = self._resolve_expert_names(models_dict, expert_names)

        self._validate_loss_type(loss_type)
        if self.d_fusion % attention_heads != 0:
            raise ValueError(
                f"d_fusion={self.d_fusion} must be divisible by attention_heads={attention_heads}."
            )

        resolved_dims = self._resolve_expert_dims(expert_dims)
        resolved_input_tokens = self._resolve_input_tokens()

        self.pv_revin_layer = RevIN(1, affine=1, subtract_last=0)
        self.adapters = ExpertSpecificAdapterBank(
            expert_names=self.expert_names,
            expert_dims=resolved_dims,
            input_tokens=resolved_input_tokens,
            token_count=aligned_token_count,
            d_model=self.d_fusion,
            dropout=dropout,
        )
        self.expert_embedding = nn.Parameter(
            torch.zeros(len(self.expert_names), 1, self.d_fusion)
        )
        self.token_embedding = nn.Parameter(
            torch.zeros(1, aligned_token_count, self.d_fusion)
        )
        self.token_mixer = TransformerTokenMixer(
            d_model=self.d_fusion,
            n_heads=attention_heads,
            n_layers=attention_layers,
            dropout=dropout,
        )
        self.query_encoder = ConditionalQueryEncoder(
            seq_len=self.seq_len,
            query_tokens=attention_query_tokens,
            d_model=self.d_fusion,
            dropout=dropout,
        )
        self.forecast_decoder = CrossAttentionForecastDecoder(
            d_model=self.d_fusion,
            pred_len=pred_len,
            n_heads=attention_heads,
            n_layers=attention_layers,
            query_tokens=attention_query_tokens,
            dropout=dropout,
        )
        self.auxiliary_head = SharedAuxiliaryForecastHead(
            token_count=aligned_token_count,
            d_model=self.d_fusion,
            pred_len=pred_len,
            dropout=dropout,
        )

        self.to(device)

    @classmethod
    def _validate_loss_type(cls, loss_type):
        if loss_type not in cls.SUPPORTED_LOSSES:
            valid = ", ".join(sorted(cls.SUPPORTED_LOSSES))
            raise ValueError(f"Unknown loss_type={loss_type!r}. Valid: {valid}.")

    def _resolve_expert_names(self, models_dict, expert_names):
        if expert_names is not None:
            missing = (
                []
                if models_dict is None
                else [name for name in expert_names if name not in models_dict]
            )
            if missing:
                raise ValueError(
                    "fusion_expert_names contains experts not in models_dict: "
                    + ", ".join(missing)
                )
            return list(expert_names)

        if models_dict is None:
            return list(self.DEFAULT_EXPERT_DIMS.keys())
        return list(models_dict.keys())

    def _resolve_expert_dims(self, expert_dims):
        resolved = dict(self.DEFAULT_EXPERT_DIMS)
        if expert_dims is not None:
            resolved.update(expert_dims)

        missing = [name for name in self.expert_names if name not in resolved]
        if missing:
            raise ValueError(
                "Missing expert_dims for: "
                + ", ".join(missing)
                + ". ExpertSpecificAttentionFusion needs each expert hidden dimension."
            )
        return resolved

    def _resolve_input_tokens(self):
        missing = [
            name for name in self.expert_names if name not in self.DEFAULT_INPUT_TOKENS
        ]
        if missing:
            raise ValueError(
                "Missing default input token counts for: " + ", ".join(missing)
            )
        return dict(self.DEFAULT_INPUT_TOKENS)

    def _format_output(self, output):
        if output.dim() == 2:
            output = output.unsqueeze(1)
        elif output.dim() == 3 and output.shape[1] == self.pred_len:
            output = output.transpose(1, 2)

        expected_shape = (output.shape[0], self.n_features, self.pred_len)
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                f"Prediction head output must be {expected_shape}, "
                f"got {tuple(output.shape)}."
            )
        return output

    def _set_revin_statistics(self, batch):
        if batch is None:
            raise ValueError("batch is required for RevIN normalization.")

        pv_his = batch["observe_power"].unsqueeze(1)
        tsfm = batch["chronos"].unsqueeze(1)
        pv = torch.cat([pv_his, tsfm], dim=2)
        pv = pv.permute(0, 2, 1)
        self.pv_revin_layer(pv, "norm")

    def _denorm_output(self, output):
        output = output.permute(0, 2, 1)
        output = self.pv_revin_layer(output, "denorm")
        if output.shape[-1] != self.n_features:
            output = output[..., : self.n_features]
        return output.permute(0, 2, 1)

    def _get_target(self, batch):
        if batch is None:
            raise ValueError("batch is required when flag is not 'test'.")

        if self.target_key in batch:
            target = batch[self.target_key]
        elif "target_power" in batch:
            target = batch["target_power"]
        else:
            raise KeyError(
                f"Cannot find target key '{self.target_key}' or 'target_power' in batch."
            )

        if target.dim() == 2:
            target = target.unsqueeze(1)
        elif target.dim() == 3 and target.shape[1] == self.pred_len:
            target = target.transpose(1, 2)

        expected_shape = (target.shape[0], self.n_features, self.pred_len)
        if tuple(target.shape) != expected_shape:
            raise ValueError(
                f"Target shape must be {expected_shape}, got {tuple(target.shape)}."
            )
        return target

    def loss_func(self, pred, target):
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        if self.loss_type == "rmse":
            return torch.sqrt(F.mse_loss(pred, target) + 1e-8)
        if self.loss_type == "mae":
            return F.l1_loss(pred, target)
        if self.loss_type == "huber":
            return F.huber_loss(pred, target, delta=1.0)
        raise ValueError(f"Unknown loss_type={self.loss_type!r}")

    def _build_context(self, tokens_by_expert):
        context_parts = []
        for index, name in enumerate(self.expert_names):
            tokens = tokens_by_expert[name]
            tokens = tokens + self.expert_embedding[index] + self.token_embedding
            context_parts.append(tokens)
        return torch.cat(context_parts, dim=1)

    def forward(self, batch_tensor, batch=None, flag="test", return_info=False):
        missing = [name for name in self.expert_names if name not in batch_tensor]
        if missing:
            raise KeyError("Missing hidden tensors for experts: " + ", ".join(missing))

        self._set_revin_statistics(batch)

        tokens_by_expert = self.adapters(batch_tensor, self.expert_names)
        context = self._build_context(tokens_by_expert)
        mixed_context = self.token_mixer(context)
        query = self.query_encoder(batch)

        output_norm, decoder_info = self.forecast_decoder(query, mixed_context)
        output_norm = self._format_output(output_norm)
        output = self._denorm_output(output_norm)

        info = {
            "expert_names": self.expert_names,
            "tokens_by_expert": tokens_by_expert,
            "context_tokens": context,
            "mixed_context_tokens": mixed_context,
            "query_tokens": query,
            "output_norm": output_norm,
            "decoder_info": decoder_info,
        }

        if flag == "test":
            if return_info:
                return output, info
            return output.squeeze(1)

        target = self._get_target(batch)
        main_loss = self.loss_func(output, target)
        aux_preds = {
            name: self._denorm_output(
                self._format_output(self.auxiliary_head(tokens_by_expert[name]))
            )
            for name in self.expert_names
        }
        aux_losses = torch.stack(
            [self.loss_func(aux_preds[name], target) for name in self.expert_names]
        )
        aux_loss = aux_losses.mean()
        loss = main_loss + self.aux_loss_weight * aux_loss

        info.update(
            {
                "main_loss": main_loss.detach(),
                "aux_loss": aux_loss.detach(),
                "total_loss": loss.detach(),
                "aux_preds": aux_preds,
                "aux_losses": {
                    name: aux_losses[i].detach()
                    for i, name in enumerate(self.expert_names)
                },
            }
        )

        if return_info:
            return output, loss, info
        return output, loss


FusionModel = ExpertSpecificAttentionFusion
