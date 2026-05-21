import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.revin import RevIN
from .expert_head import EXPERT_HEAD_REGISTRY


class HiddenSummaryEncoder(nn.Module):
    """Summarize variable expert hidden tokens into one compact gate feature."""

    def __init__(self, input_dim, output_dim, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.Linear(input_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
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
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected hidden dim {self.input_dim}, got {tokens.shape[-1]}."
            )

        mean = tokens.mean(dim=1)
        std = tokens.std(dim=1, unbiased=False)
        return self.encoder(torch.cat([mean, std], dim=-1))


class ConstrainedExpertHeadFusion(nn.Module):
    """
    V7 prediction-level constrained fusion.

    Each expert hidden state is decoded by its native-style reconstructed head.
    The stable base forecast is the mean of all expert predictions. A small gate,
    initialized to uniform weights, can only reweight those already reconstructed
    predictions. This keeps the ensemble mean as the baseline instead of replacing
    it with a direct hidden-to-final decoder.
    """

    DEFAULT_EXPERT_DIMS = {"m1": 128, "m2": 512, "m3": 384, "m4": 256}
    SUPPORTED_LOSSES = {"mse", "mae", "huber", "rmse"}

    def __init__(
        self,
        models_dict=None,
        seq_len=None,
        pred_len=192,
        n_features=1,
        expert_dims=None,
        expert_names=None,
        target_key="observe_power_future",
        loss_type="mse",
        base_loss_weight=1.0,
        aux_loss_weight=0.5,
        gate_reg_weight=0.1,
        gate_temperature=5.0,
        d_fusion=None,
        dropout=0.0,
        device="cuda",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.target_key = target_key
        self.loss_type = loss_type
        self.base_loss_weight = base_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.gate_reg_weight = gate_reg_weight
        self.gate_temperature = gate_temperature
        self.gate_hidden_dim = 128 if d_fusion is None else d_fusion
        self.expert_names = self._resolve_expert_names(models_dict, expert_names)

        self._validate_loss_type(loss_type)
        if base_loss_weight < 0:
            raise ValueError(
                f"base_loss_weight must be non-negative, got {base_loss_weight}."
            )
        if gate_temperature <= 0:
            raise ValueError(f"gate_temperature must be positive, got {gate_temperature}.")
        if gate_reg_weight < 0:
            raise ValueError(f"gate_reg_weight must be non-negative, got {gate_reg_weight}.")

        resolved_dims = self._resolve_hidden_dims(expert_dims)
        self.pv_revin_layer = RevIN(1, affine=1, subtract_last=0)

        self.prediction_heads = nn.ModuleDict()
        self.hidden_encoders = nn.ModuleDict()
        for name in self.expert_names:
            head_cls = EXPERT_HEAD_REGISTRY[name]
            self.prediction_heads[name] = head_cls(
                hidden_dim=resolved_dims[name],
                seq_len=seq_len,
                pred_len=pred_len,
                n_features=n_features,
                head_dropout=dropout,
            )
            self.hidden_encoders[name] = HiddenSummaryEncoder(
                input_dim=resolved_dims[name],
                output_dim=self.gate_hidden_dim,
                dropout=dropout,
            )

        gate_input_dim = len(self.expert_names) * (self.gate_hidden_dim + 2)
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, self.gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.gate_hidden_dim, len(self.expert_names)),
        )
        self._init_gate_to_uniform()
        self.to(device)

    def _init_gate_to_uniform(self):
        final = self.gate[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @classmethod
    def _validate_loss_type(cls, loss_type):
        if loss_type not in cls.SUPPORTED_LOSSES:
            valid = ", ".join(sorted(cls.SUPPORTED_LOSSES))
            raise ValueError(f"Unknown loss_type={loss_type!r}. Valid: {valid}.")

    def _resolve_expert_names(self, models_dict, expert_names):
        if expert_names is not None:
            names = list(expert_names)
        elif models_dict is None:
            names = list(EXPERT_HEAD_REGISTRY.keys())
        else:
            names = list(models_dict.keys())

        unsupported = [name for name in names if name not in EXPERT_HEAD_REGISTRY]
        if unsupported:
            valid = ", ".join(sorted(EXPERT_HEAD_REGISTRY))
            raise ValueError(
                "Unsupported experts for expert_head_v7: "
                + ", ".join(unsupported)
                + f". Valid: {valid}."
            )
        if models_dict is not None:
            missing = [name for name in names if name not in models_dict]
            if missing:
                raise ValueError(
                    "fusion_expert_names contains experts not in models_dict: "
                    + ", ".join(missing)
                )
        return names

    def _resolve_hidden_dims(self, expert_dims):
        resolved_dims = dict(self.DEFAULT_EXPERT_DIMS)
        if expert_dims is not None:
            resolved_dims.update(expert_dims)

        missing = [name for name in self.expert_names if name not in resolved_dims]
        if missing:
            raise ValueError(
                "Missing expert_dims for: "
                + ", ".join(missing)
                + ". ConstrainedExpertHeadFusion needs each expert hidden dimension."
            )
        return resolved_dims

    def _format_output(self, output):
        if output.dim() == 2:
            output = output.unsqueeze(1)
        elif output.dim() == 3 and output.shape[1] == self.pred_len:
            output = output.transpose(1, 2)

        expected_shape = (output.shape[0], self.n_features, self.pred_len)
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                f"Prediction head output must be {expected_shape}, got {tuple(output.shape)}."
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
            raise ValueError(f"Target shape must be {expected_shape}, got {tuple(target.shape)}.")
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

    def _prediction_stats(self, pred):
        flattened = pred.flatten(start_dim=1)
        return torch.stack(
            [
                flattened.mean(dim=1),
                flattened.std(dim=1, unbiased=False),
            ],
            dim=1,
        )

    def _build_gate_input(self, batch_tensor, pred_by_expert):
        features = []
        for name in self.expert_names:
            hidden_feature = self.hidden_encoders[name](batch_tensor[name])
            pred_feature = self._prediction_stats(pred_by_expert[name])
            features.append(torch.cat([hidden_feature, pred_feature], dim=-1))
        return torch.cat(features, dim=-1)

    def _gate_weights(self, gate_input):
        logits = self.gate(gate_input)
        weights = F.softmax(logits / self.gate_temperature, dim=-1)
        return weights, logits

    def _uniform_weight_penalty(self, weights):
        uniform = weights.new_full(weights.shape, 1.0 / weights.shape[1])
        return (weights - uniform).pow(2).mean()

    def forward(self, batch_tensor, batch=None, flag="test", return_info=False):
        missing = [name for name in self.expert_names if name not in batch_tensor]
        if missing:
            raise KeyError("Missing hidden tensors for experts: " + ", ".join(missing))

        self._set_revin_statistics(batch)

        pred_by_expert = {}
        preds = []
        for name in self.expert_names:
            pred_norm = self._format_output(self.prediction_heads[name](batch_tensor[name]))
            pred = self._denorm_output(pred_norm)
            pred_by_expert[name] = pred
            preds.append(pred)

        pred_stack = torch.stack(preds, dim=1)
        base_output = pred_stack.mean(dim=1)

        gate_input = self._build_gate_input(batch_tensor, pred_by_expert)
        gate_weight, gate_logits = self._gate_weights(gate_input)
        output = (pred_stack * gate_weight.view(gate_weight.shape[0], -1, 1, 1)).sum(dim=1)

        info = {
            "expert_names": self.expert_names,
            "pred_by_expert": pred_by_expert,
            "pred_stack": pred_stack,
            "base_output": base_output,
            "gate_weight": gate_weight,
            "gate_logits": gate_logits,
        }

        if flag == "test":
            if return_info:
                return output, info
            return output.squeeze(1)

        target = self._get_target(batch)
        main_loss = self.loss_func(output, target)
        base_loss = self.loss_func(base_output, target)
        aux_losses = torch.stack(
            [self.loss_func(pred_by_expert[name], target) for name in self.expert_names]
        )
        aux_loss = aux_losses.mean()
        gate_reg_loss = self._uniform_weight_penalty(gate_weight)
        loss = (
            main_loss
            + self.base_loss_weight * base_loss
            + self.aux_loss_weight * aux_loss
            + self.gate_reg_weight * gate_reg_loss
        )

        info.update(
            {
                "main_loss": main_loss.detach(),
                "base_loss": base_loss.detach(),
                "base_loss_weight": output.new_tensor(self.base_loss_weight),
                "aux_loss": aux_loss.detach(),
                "gate_reg_loss": gate_reg_loss.detach(),
                "total_loss": loss.detach(),
                "aux_losses": {
                    name: aux_losses[i].detach()
                    for i, name in enumerate(self.expert_names)
                },
            }
        )

        if return_info:
            return output, loss, info
        return output, loss


FusionModel = ConstrainedExpertHeadFusion
