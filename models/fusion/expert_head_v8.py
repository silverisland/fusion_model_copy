import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.revin import RevIN


def orthogonal_linear(in_features, out_features):
    """
    Linear layer with orthogonal initialization.

    This keeps the flatten-first adapter fast while giving each projection
    branch an initially orthogonal basis. The weights are not constrained to
    remain orthogonal during training.
    """
    layer = nn.Linear(in_features, out_features, bias=False)
    nn.init.orthogonal_(layer.weight)
    return layer


def init_rsqrt_uniform_(weight, d):
    bound = d ** -0.5
    nn.init.uniform_(weight, -bound, bound)


class LinearEnsemble(nn.Module):
    """K independent linear output heads applied to (B, K, D)."""

    def __init__(self, in_features, out_features, k, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(k, out_features)) if bias else None
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weight, self.in_features)
        if self.bias is not None:
            init_rsqrt_uniform_(self.bias, self.in_features)

    def forward(self, x):
        if x.dim() != 3 or x.shape[1] != self.k:
            raise ValueError(
                f"LinearEnsemble expects (B, {self.k}, D), got {tuple(x.shape)}."
            )
        x = torch.einsum("bki,kio->bko", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x


class LinearEnsembleForecastMLP(nn.Module):
    """Forecast decoder with K fully independent linear MLP branches."""

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        k,
        dropout=0.0,
    ):
        super().__init__()
        self.k = k
        self.first = LinearEnsemble(in_features, hidden_features, k=k)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = LinearEnsemble(hidden_features, out_features, k=k)

    def forward(self, x):
        if x.dim() != 3 or x.shape[1] != self.k:
            raise ValueError(
                f"LinearEnsembleForecastMLP expects (B, {self.k}, D), "
                f"got {tuple(x.shape)}."
            )
        x = self.first(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.output(x)


class FlattenOrthogonalAdapter(nn.Module):
    """
    Builds compact expert tokens with a flatten-first prediction-head bias.

    For each expert, the adapter creates K projection branches:
        hidden -> flatten -> Linear -> one compact token

    Output shape:
        (B, token_count, d_model)
    """

    def __init__(
        self,
        input_dim,
        input_tokens,
        token_count=3,
        d_model=128,
        dropout=0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_tokens = input_tokens
        self.token_count = token_count
        self.d_model = d_model

        self.branch_readouts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(input_tokens * input_dim),
                    orthogonal_linear(input_tokens * input_dim, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(token_count)
            ]
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
        flattened = tokens.flatten(start_dim=1)
        for readout in self.branch_readouts:
            compact_tokens.append(readout(flattened))
        return torch.stack(compact_tokens, dim=1)

    def projection_weight_orthogonal_loss(self):
        if self.token_count < 2:
            first_weight = self.branch_readouts[0][1].weight
            return first_weight.new_tensor(0.0)

        losses = []
        weights = [branch[1].weight for branch in self.branch_readouts]
        for left, right in itertools.combinations(weights, 2):
            left_norm = F.normalize(left, dim=-1, eps=1e-6)
            right_norm = F.normalize(right, dim=-1, eps=1e-6)
            cross_gram = left_norm.matmul(right_norm.transpose(0, 1))
            losses.append(cross_gram.pow(2).mean())
        return torch.stack(losses).mean()


class AttentionBlock(nn.Module):
    """Cross-attends learned forecast tokens to compact expert/weather tokens."""

    def __init__(self, d_model=128, n_heads=4, dropout=0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query, context):
        norm_context = self.context_norm(context)
        attn_output, attn_weights = self.attention(
            self.query_norm(query),
            norm_context,
            norm_context,
            need_weights=True,
            average_attn_weights=False,
        )
        query = query + self.dropout(attn_output)
        query = query + self.ffn(self.ffn_norm(query))
        return query, attn_weights


class WeatherForecastEncoder(nn.Module):
    """
    Encode future weather forecasts into context tokens.

    Each configured weather key is expected to be shaped (B, pred_len), or a
    single-channel equivalent. Each weather variable becomes one context token.
    """

    def __init__(self, weather_keys, pred_len, d_model, dropout=0.0):
        super().__init__()
        self.weather_keys = list(weather_keys or [])
        self.pred_len = pred_len
        self.d_model = d_model
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(pred_len),
                    nn.Linear(pred_len, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in self.weather_keys
            ]
        )
        self.weather_embedding = nn.Parameter(
            torch.zeros(len(self.weather_keys), 1, d_model)
        )

    def _flatten_weather(self, value, key):
        if value.dim() == 3 and value.shape[1] == 1:
            value = value.squeeze(1)
        elif value.dim() == 3 and value.shape[-1] == 1:
            value = value.squeeze(-1)

        if value.dim() != 2:
            raise ValueError(
                f"Weather key {key!r} must be shaped (B, pred_len), "
                f"got {tuple(value.shape)}."
            )
        if value.shape[1] != self.pred_len:
            raise ValueError(
                f"Weather key {key!r} length must be {self.pred_len}, "
                f"got {value.shape[1]}."
            )
        return value

    def forward(self, batch):
        if not self.weather_keys:
            return None
        if batch is None:
            raise ValueError("batch is required when weather_keys are configured.")

        tokens = []
        for index, key in enumerate(self.weather_keys):
            if key not in batch:
                raise KeyError(f"Missing weather key {key!r} in batch.")
            weather = self._flatten_weather(batch[key], key)
            token = self.encoders[index](weather).unsqueeze(1)
            token = token + self.weather_embedding[index]
            tokens.append(token)
        return torch.cat(tokens, dim=1)


class WeatherAwareCrossAttentionForecastHead(nn.Module):
    """
    V5-style cross-attention head with extra weather forecast context tokens.
    """

    def __init__(
        self,
        expert_names,
        d_model=128,
        pred_len=192,
        n_heads=4,
        n_layers=1,
        query_tokens=None,
        dropout=0.0,
        ensemble_size=4,
        expert_drop_prob=0.0,
        weather_keys=None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}."
            )
        if ensemble_size < 1:
            raise ValueError(f"ensemble_size must be positive, got {ensemble_size}.")
        if query_tokens is None:
            query_tokens = ensemble_size
        if query_tokens != ensemble_size:
            raise ValueError(
                "For expert_head_v8, query_tokens must equal ensemble_size: "
                f"got query_tokens={query_tokens}, ensemble_size={ensemble_size}."
            )
        if not 0.0 <= expert_drop_prob < 1.0:
            raise ValueError(
                f"expert_drop_prob must be in [0, 1), got {expert_drop_prob}."
            )

        self.expert_names = list(expert_names)
        self.d_model = d_model
        self.query_tokens = query_tokens
        self.ensemble_size = ensemble_size
        self.expert_drop_prob = expert_drop_prob
        self.weather_keys = list(weather_keys or [])
        self.expert_embedding = nn.Parameter(
            torch.zeros(len(self.expert_names), 1, d_model)
        )
        self.query = nn.Parameter(torch.randn(1, query_tokens, d_model) * 0.02)
        self.weather_encoder = WeatherForecastEncoder(
            weather_keys=self.weather_keys,
            pred_len=pred_len,
            d_model=d_model,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.forecast_head = LinearEnsembleForecastMLP(
            in_features=d_model,
            hidden_features=d_model * 2,
            out_features=pred_len,
            k=ensemble_size,
            dropout=dropout,
        )

    def _expert_keep_mask(self, batch_size, device):
        if not self.training or self.expert_drop_prob <= 0.0:
            return None

        keep = torch.rand(
            batch_size,
            len(self.expert_names),
            device=device,
        ) >= self.expert_drop_prob

        all_dropped = ~keep.any(dim=1)
        if all_dropped.any():
            fallback = torch.randint(
                0,
                len(self.expert_names),
                (int(all_dropped.sum().item()),),
                device=device,
            )
            keep[all_dropped] = False
            keep[all_dropped, fallback] = True
        return keep

    def forward(self, tokens_by_expert, batch=None):
        first_tokens = tokens_by_expert[self.expert_names[0]]
        keep_mask = self._expert_keep_mask(
            batch_size=first_tokens.shape[0],
            device=first_tokens.device,
        )

        context_parts = []
        for index, name in enumerate(self.expert_names):
            tokens = tokens_by_expert[name] + self.expert_embedding[index]
            if keep_mask is not None:
                mask = keep_mask[:, index].to(tokens.dtype).view(-1, 1, 1)
                tokens = tokens * mask / (1.0 - self.expert_drop_prob)
            context_parts.append(tokens)

        weather_tokens = self.weather_encoder(batch)
        if weather_tokens is not None:
            context_parts.append(weather_tokens)

        context = torch.cat(context_parts, dim=1)
        batch_size = context.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        attention_weights = []
        for block in self.blocks:
            query, weights = block(query, context)
            attention_weights.append(weights)

        forecast_features = self.output_norm(query)
        ensemble_forecast = self.forecast_head(forecast_features)
        forecast = ensemble_forecast.mean(dim=1)
        return forecast, {
            "query_tokens": query,
            "context_tokens": context,
            "weather_tokens": weather_tokens,
            "weather_keys": self.weather_keys,
            "forecast_features": forecast_features,
            "ensemble_forecast": ensemble_forecast,
            "attention_weights": attention_weights,
            "expert_keep_mask": keep_mask,
        }


class WeatherAwareExpertHeadFusion(nn.Module):
    """
    V8 expert hidden fusion.

    This version keeps the v5 expert-token and LinearEnsemble decoder path, but
    appends future weather forecast tokens to the cross-attention context.
    """

    def __init__(
        self,
        models_dict=None,
        seq_len=None,
        pred_len=192,
        n_features=1,
        expert_dims=None,
        expert_names=None,
        aligned_token_count=3,
        d_fusion=None,
        dropout=0.0,
        target_key="observe_power_future",
        loss_type="mse",
        orth_loss_weight=1e-4,
        attention_heads=4,
        attention_layers=1,
        attention_query_tokens=None,
        ensemble_size=4,
        ensemble_scaling_init="normal",
        expert_drop_prob=0.0,
        weather_keys=None,
        focus_loss_start=59,
        focus_loss_end=152,
        focus_loss_weight=0.0,
        full_loss_weight=1.0,
        device="cuda",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.d_fusion = 128 if d_fusion is None else d_fusion
        self.aligned_token_count = aligned_token_count
        self.target_key = target_key
        self.loss_type = loss_type
        self.orth_loss_weight = orth_loss_weight
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        if attention_query_tokens is None:
            attention_query_tokens = ensemble_size
        if attention_query_tokens != ensemble_size:
            raise ValueError(
                "For expert_head_v8, attention_query_tokens must equal "
                f"ensemble_size: got {attention_query_tokens} and {ensemble_size}."
            )

        self.attention_query_tokens = attention_query_tokens
        self.ensemble_size = ensemble_size
        self.ensemble_scaling_init = ensemble_scaling_init
        self.expert_drop_prob = expert_drop_prob
        self.expert_names = self._resolve_expert_names(models_dict, expert_names)
        self.weather_keys = list(weather_keys or [])
        self.focus_loss_start = focus_loss_start
        self.focus_loss_end = focus_loss_end
        self.focus_loss_weight = focus_loss_weight
        self.full_loss_weight = full_loss_weight

        self._validate_loss_type(loss_type)
        self._validate_focus_loss_config()
        resolved_dims = self._resolve_expert_dims(expert_dims)
        resolved_input_tokens = self._resolve_input_tokens()

        self.pv_revin_layer = RevIN(1, affine=1, subtract_last=0)
        self.adapters = nn.ModuleDict()
        for name in self.expert_names:
            self.adapters[name] = FlattenOrthogonalAdapter(
                input_dim=resolved_dims[name],
                input_tokens=resolved_input_tokens[name],
                token_count=aligned_token_count,
                d_model=self.d_fusion,
                dropout=dropout,
            )

        self.weather_keys = list(weather_keys or [])
        self.forecast_head = WeatherAwareCrossAttentionForecastHead(
            expert_names=self.expert_names,
            d_model=self.d_fusion,
            pred_len=pred_len,
            n_heads=attention_heads,
            n_layers=attention_layers,
            query_tokens=self.attention_query_tokens,
            dropout=dropout,
            ensemble_size=ensemble_size,
            expert_drop_prob=expert_drop_prob,
            weather_keys=self.weather_keys,
        )

        self.to(device)

    DEFAULT_EXPERT_DIMS = {"m1": 128, "m2": 512, "m3": 384, "m4": 256}
    DEFAULT_INPUT_TOKENS = {"m1": 9, "m2": 2, "m3": 162, "m4": 45}
    SUPPORTED_LOSSES = {"mse", "mae", "huber", "rmse"}

    @classmethod
    def _validate_loss_type(cls, loss_type):
        if loss_type not in cls.SUPPORTED_LOSSES:
            valid = ", ".join(sorted(cls.SUPPORTED_LOSSES))
            raise ValueError(f"Unknown loss_type={loss_type!r}. Valid: {valid}.")

    def _validate_focus_loss_config(self):
        if self.full_loss_weight < 0:
            raise ValueError(
                f"full_loss_weight must be non-negative, got {self.full_loss_weight}."
            )
        if self.focus_loss_weight < 0:
            raise ValueError(
                f"focus_loss_weight must be non-negative, got {self.focus_loss_weight}."
            )
        if self.focus_loss_weight <= 0:
            return

        if self.focus_loss_start is None or self.focus_loss_end is None:
            raise ValueError(
                "focus_loss_start and focus_loss_end are required when "
                "focus_loss_weight > 0."
            )
        if not 0 <= self.focus_loss_start < self.focus_loss_end <= self.pred_len:
            raise ValueError(
                "Focus loss window must satisfy "
                f"0 <= start < end <= pred_len, got start={self.focus_loss_start}, "
                f"end={self.focus_loss_end}, pred_len={self.pred_len}."
            )

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
                + ". WeatherAwareExpertHeadFusion needs each expert hidden dimension."
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

    def _format_ensemble_output(self, output):
        if output.dim() == 3:
            output = output.unsqueeze(2)
        elif output.dim() == 4 and output.shape[2] == self.pred_len:
            output = output.transpose(2, 3)

        expected_shape = (
            output.shape[0],
            self.ensemble_size,
            self.n_features,
            self.pred_len,
        )
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                f"Ensemble prediction output must be {expected_shape}, "
                f"got {tuple(output.shape)}."
            )
        return output

    def _denorm_ensemble_output(self, output):
        members = [
            self._denorm_output(output[:, index])
            for index in range(output.shape[1])
        ]
        return torch.stack(members, dim=1)

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

    def prediction_loss_components(self, pred, target):
        full_loss = self.loss_func(pred, target)
        if self.focus_loss_weight <= 0:
            zero = full_loss.new_tensor(0.0)
            return full_loss, full_loss, zero

        focus_pred = pred[..., self.focus_loss_start : self.focus_loss_end]
        focus_target = target[..., self.focus_loss_start : self.focus_loss_end]
        focus_loss = self.loss_func(focus_pred, focus_target)
        total_loss = (
            self.full_loss_weight * full_loss
            + self.focus_loss_weight * focus_loss
        )
        return total_loss, full_loss, focus_loss

    def ensemble_loss_func(self, pred, target):
        losses = []
        full_losses = []
        focus_losses = []
        for index in range(pred.shape[1]):
            total_loss, full_loss, focus_loss = self.prediction_loss_components(
                pred[:, index],
                target,
            )
            losses.append(total_loss)
            full_losses.append(full_loss)
            focus_losses.append(focus_loss)
        return (
            torch.stack(losses).mean(),
            torch.stack(full_losses).mean(),
            torch.stack(focus_losses).mean(),
        )

    def _within_expert_token_loss(self, compact_tokens):
        if compact_tokens.shape[1] < 2:
            return compact_tokens.new_tensor(0.0)

        tokens = compact_tokens - compact_tokens.mean(dim=-1, keepdim=True)
        tokens = F.normalize(tokens, dim=-1, eps=1e-6)
        gram = tokens.matmul(tokens.transpose(1, 2))
        identity = torch.eye(
            gram.shape[-1],
            device=gram.device,
            dtype=gram.dtype,
        ).unsqueeze(0)
        return ((gram - identity) ** 2).mean()

    def _cross_expert_token_loss(self, tokens_by_expert):
        if len(self.expert_names) < 2:
            first = tokens_by_expert[self.expert_names[0]]
            return first.new_tensor(0.0)

        losses = []
        for left, right in itertools.combinations(self.expert_names, 2):
            left_tokens = tokens_by_expert[left].reshape(-1, self.d_fusion)
            right_tokens = tokens_by_expert[right].reshape(-1, self.d_fusion)

            left_tokens = left_tokens - left_tokens.mean(dim=0, keepdim=True)
            right_tokens = right_tokens - right_tokens.mean(dim=0, keepdim=True)
            left_norm = F.normalize(left_tokens, dim=-1, eps=1e-6)
            right_norm = F.normalize(right_tokens, dim=-1, eps=1e-6)

            pair_cosine = (left_norm * right_norm).sum(dim=-1).pow(2).mean()
            cross_corr = left_norm.transpose(0, 1).matmul(right_norm)
            cross_corr = cross_corr / max(left_norm.shape[0], 1)
            losses.append(pair_cosine + cross_corr.pow(2).mean())
        return torch.stack(losses).mean()

    def orthogonal_loss(self, tokens_by_expert):
        token_losses = [
            self._within_expert_token_loss(tokens_by_expert[name])
            for name in self.expert_names
        ]
        adapter_losses = [
            self.adapters[name].projection_weight_orthogonal_loss()
            for name in self.expert_names
        ]
        return (
            torch.stack(token_losses).mean()
            + self._cross_expert_token_loss(tokens_by_expert)
            + torch.stack(adapter_losses).mean()
        )

    def forward(self, batch_tensor, batch=None, flag="test", return_info=False):
        missing = [name for name in self.expert_names if name not in batch_tensor]
        if missing:
            raise KeyError("Missing hidden tensors for experts: " + ", ".join(missing))

        self._set_revin_statistics(batch)

        tokens_by_expert = {}
        for name in self.expert_names:
            tokens_by_expert[name] = self.adapters[name](batch_tensor[name])

        output_norm, fusion_info = self.forecast_head(tokens_by_expert, batch=batch)
        output_norm = self._format_output(output_norm)
        output = self._denorm_output(output_norm)
        ensemble_output_norm = self._format_ensemble_output(
            fusion_info["ensemble_forecast"]
        )
        ensemble_output = self._denorm_ensemble_output(ensemble_output_norm)

        info = {
            "expert_names": self.expert_names,
            "weather_keys": self.weather_keys,
            "tokens_by_expert": tokens_by_expert,
            "output_norm": output_norm,
            "ensemble_output_norm": ensemble_output_norm,
            "ensemble_output": ensemble_output,
            "fusion_info": fusion_info,
        }

        if flag == "test":
            if return_info:
                return output, info
            return output.squeeze(1)

        target = self._get_target(batch)
        main_loss, full_loss, focus_loss = self.ensemble_loss_func(
            ensemble_output,
            target,
        )
        if self.orth_loss_weight > 0:
            orth_loss = self.orthogonal_loss(tokens_by_expert)
        else:
            orth_loss = output.new_tensor(0.0)
        loss = main_loss + self.orth_loss_weight * orth_loss

        info.update(
            {
                "main_loss": main_loss.detach(),
                "full_loss": full_loss.detach(),
                "focus_loss": focus_loss.detach(),
                "orth_loss": orth_loss.detach(),
                "total_loss": loss.detach(),
                "focus_loss_start": self.focus_loss_start,
                "focus_loss_end": self.focus_loss_end,
                "focus_loss_weight": self.focus_loss_weight,
                "full_loss_weight": self.full_loss_weight,
            }
        )

        if return_info:
            return output, loss, info
        return output, loss


FusionModel = WeatherAwareExpertHeadFusion
