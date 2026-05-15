import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.revin import RevIN


class CompressedTokenAdapter(nn.Module):
    """
    Compresses each expert hidden tensor to a fixed token grid (B, K, d_model).

    The operation is intentionally simple for this experiment:
    hidden -> flatten non-feature dims as tokens -> Linear(D_i, d_model)
    -> Linear(T_i, K) over the token dimension.
    """

    def __init__(self, input_dim, input_tokens, token_count=3, d_model=128, dropout=0.0):
        super().__init__()
        self.input_tokens = input_tokens
        self.token_count = token_count
        self.d_model = d_model

        self.feature_projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.Dropout(dropout),
        )
        self.token_projector = (
            nn.Identity()
            if input_tokens == token_count
            else nn.Linear(input_tokens, token_count)
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

        tokens = self.feature_projector(tokens)
        tokens = tokens.transpose(1, 2)
        tokens = self.token_projector(tokens)
        return tokens.transpose(1, 2)


class M1CompressedForecastHead(nn.Module):
    """M1-style second half: flatten compressed tokens then linear."""

    def __init__(self, token_count, d_model=128, pred_len=192, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(token_count * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        tokens = self.flatten(tokens)
        tokens = self.linear(tokens)
        return self.dropout(tokens)


class M2CompressedForecastHead(nn.Module):
    """M2-style second half: independent token regressors then mean."""

    def __init__(self, token_count, d_model=128, pred_len=192, dropout=0.0):
        super().__init__()
        self.token_count = token_count
        self.regression_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, pred_len),
                )
                for _ in range(token_count)
            ]
        )

    def forward(self, tokens):
        if tokens.shape[1] != self.token_count:
            raise ValueError(
                f"M2 compressed token count must be {self.token_count}, "
                f"got {tokens.shape[1]}."
            )
        preds = [
            self.regression_heads[i](tokens[:, i, :])
            for i in range(self.token_count)
        ]
        return torch.stack(preds, dim=1).mean(dim=1)


class M3CompressedForecastHead(M1CompressedForecastHead):
    """M3-style flattened linear readout after token compression."""

    def __init__(self, token_count, d_model=128, pred_len=192, dropout=0.3):
        super().__init__(
            token_count=token_count,
            d_model=d_model,
            pred_len=pred_len,
            dropout=dropout,
        )


class M4CompressedForecastHead(nn.Module):
    """M4-style MLP readout after token compression."""

    def __init__(self, token_count, d_model=128, pred_len=192, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        layers = []
        hidden_sizes = [1024, 256, 64]
        prev_size = token_count * d_model
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, pred_len))
        self.model = nn.Sequential(*layers)

    def forward(self, tokens):
        tokens = self.flatten(tokens)
        return self.model(tokens)


COMPRESSED_HEAD_REGISTRY = {
    "m1": M1CompressedForecastHead,
    "m2": M2CompressedForecastHead,
    "m3": M3CompressedForecastHead,
    "m4": M4CompressedForecastHead,
}


class CompressedExpertHeadFusion(nn.Module):
    """
    V3 expert-head fusion.

    All selected experts are compressed to the same latent shape
    (B, aligned_token_count, d_fusion), then decoded by expert-specific
    second-stage heads. Final fusion is a fixed mean over expert predictions.
    """

    DEFAULT_EXPERT_DIMS = {"m1": 128, "m2": 512, "m3": 384, "m4": 256}
    DEFAULT_INPUT_TOKENS = {"m1": 9, "m2": 2, "m3": 162, "m4": 45}
    SUPPORTED_LOSSES = {"mse", "mae", "huber"}

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
        aux_loss_weight=1.0,
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
        self.aux_loss_weight = aux_loss_weight
        self.expert_names = self._resolve_expert_names(models_dict, expert_names)

        self._validate_loss_type(loss_type)
        resolved_dims = self._resolve_expert_dims(expert_dims)
        resolved_input_tokens = self._resolve_input_tokens()

        self.pv_revin_layer = RevIN(1, affine=1, subtract_last=0)
        self.adapters = nn.ModuleDict()
        self.prediction_heads = nn.ModuleDict()
        for name in self.expert_names:
            self.adapters[name] = CompressedTokenAdapter(
                input_dim=resolved_dims[name],
                input_tokens=resolved_input_tokens[name],
                token_count=aligned_token_count,
                d_model=self.d_fusion,
                dropout=dropout,
            )
            head_cls = COMPRESSED_HEAD_REGISTRY[name]
            self.prediction_heads[name] = head_cls(
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
                + ". CompressedExpertHeadFusion needs each expert hidden dimension."
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
        pv = torch.cat([pv_his, tsfm], dim=1)
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
        if self.loss_type == "mae":
            return F.l1_loss(pred, target)
        if self.loss_type == "huber":
            return F.huber_loss(pred, target, delta=1.0)
        raise ValueError(f"Unknown loss_type={self.loss_type!r}")

    def forward(self, batch_tensor, batch=None, flag="test", return_info=False):
        missing = [name for name in self.expert_names if name not in batch_tensor]
        if missing:
            raise KeyError("Missing hidden tensors for experts: " + ", ".join(missing))

        self._set_revin_statistics(batch)

        compressed_by_expert = {}
        pred_by_expert = {}
        preds = []
        for name in self.expert_names:
            compressed = self.adapters[name](batch_tensor[name])
            pred = self._format_output(self.prediction_heads[name](compressed))
            pred = self._denorm_output(pred)
            compressed_by_expert[name] = compressed
            pred_by_expert[name] = pred
            preds.append(pred)

        pred_stack = torch.stack(preds, dim=1)
        output = pred_stack.mean(dim=1)

        info = {
            "expert_names": self.expert_names,
            "compressed_by_expert": compressed_by_expert,
            "pred_by_expert": pred_by_expert,
            "pred_stack": pred_stack,
        }

        if flag == "test":
            if return_info:
                return output, info
            return output.squeeze(1)

        if flag != "train":
            raise ValueError("flag must be either 'train' or 'test'.")

        target = self._get_target(batch)
        main_loss = self.loss_func(output, target)
        aux_losses = torch.stack(
            [self.loss_func(pred_by_expert[name], target) for name in self.expert_names]
        )
        aux_loss = aux_losses.mean()
        loss = main_loss + self.aux_loss_weight * aux_loss

        info.update(
            {
                "main_loss": main_loss.detach(),
                "aux_loss": aux_loss.detach(),
                "aux_losses": {
                    name: aux_losses[i].detach()
                    for i, name in enumerate(self.expert_names)
                },
            }
        )

        if return_info:
            return output, loss, info
        return output, loss


FusionModel = CompressedExpertHeadFusion
