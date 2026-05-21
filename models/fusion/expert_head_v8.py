import torch
import torch.nn as nn

from .expert_head_v5 import (
    AttentionBlock,
    FlattenOrthogonalAttentionExpertHeadFusion,
    LinearEnsembleForecastMLP,
)


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


class WeatherAwareExpertHeadFusion(FlattenOrthogonalAttentionExpertHeadFusion):
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
        device="cuda",
    ):
        super().__init__(
            models_dict=models_dict,
            seq_len=seq_len,
            pred_len=pred_len,
            n_features=n_features,
            expert_dims=expert_dims,
            expert_names=expert_names,
            aligned_token_count=aligned_token_count,
            d_fusion=d_fusion,
            dropout=dropout,
            target_key=target_key,
            loss_type=loss_type,
            orth_loss_weight=orth_loss_weight,
            attention_heads=attention_heads,
            attention_layers=attention_layers,
            attention_query_tokens=attention_query_tokens,
            ensemble_size=ensemble_size,
            ensemble_scaling_init=ensemble_scaling_init,
            expert_drop_prob=expert_drop_prob,
            device=device,
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
        ).to(device)

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
        main_loss = self.ensemble_loss_func(ensemble_output, target)
        if self.orth_loss_weight > 0:
            orth_loss = self.orthogonal_loss(tokens_by_expert)
        else:
            orth_loss = output.new_tensor(0.0)
        loss = main_loss + self.orth_loss_weight * orth_loss

        info.update(
            {
                "main_loss": main_loss.detach(),
                "orth_loss": orth_loss.detach(),
                "total_loss": loss.detach(),
            }
        )

        if return_info:
            return output, loss, info
        return output, loss


FusionModel = WeatherAwareExpertHeadFusion
