import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.revin import RevIN


class M1JointPredictionHead(nn.Module):
    # M1 hidden shape: (B, 9, 128)
    def __init__(
        self,
        nf=9 * 128,
        target_window=None,
        pred_len=192,
        head_dropout=0,
        **_,
    ):
        super().__init__()
        target_window = pred_len if target_window is None else target_window
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, hidden):
        hidden = self.flatten(hidden)
        hidden = self.linear(hidden)
        hidden = self.dropout(hidden)
        return hidden


class M2JointPredictionHead(nn.Module):
    # M2 hidden shape: (B, 2, 512)
    def __init__(
        self,
        hidden_channels=2,
        hidden_dim=512,
        dropout_rate=0,
        pred_len=192,
        head_dropout=None,
        **_,
    ):
        super().__init__()
        if head_dropout is not None:
            dropout_rate = head_dropout
        self.channel = hidden_channels
        self.targetseq_len = pred_len
        self.regression_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim * 2, self.targetseq_len),
                )
                for _ in range(self.channel)
            ]
        )

    def forward(self, hidden):
        if hidden.shape[1] != self.channel:
            raise ValueError(
                f"M2 hidden channel must be {self.channel}, got {hidden.shape[1]}."
            )
        preds = [self.regression_heads[i](hidden[:, i, :]) for i in range(self.channel)]
        return torch.stack(preds, dim=1).mean(dim=1)


class M3JointPredictionHead(nn.Module):
    # M3 hidden shape: (B, 162, 384)
    def __init__(
        self,
        in_dim=162 * 384,
        out_dim=None,
        pred_len=192,
        head_dropout=0.3,
        **_,
    ):
        super().__init__()
        out_dim = pred_len if out_dim is None else out_dim
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, hidden):
        hidden = self.flatten(hidden)
        hidden = self.dropout(hidden)
        hidden = self.linear(hidden)
        return hidden


class M4JointPredictionHead(nn.Module):
    # M4 hidden shape: (B, 5, 9, 256)
    def __init__(
        self,
        nf=5 * 9 * 256,
        target_window=None,
        pred_len=192,
        head_dropout=0,
        **_,
    ):
        super().__init__()
        target_window = pred_len if target_window is None else target_window
        self.flatten = nn.Flatten(start_dim=-3)
        layers = []
        hidden_sizes = [1024, 256, 64]
        prev_size = nf
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(head_dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, target_window))
        self.model = nn.Sequential(*layers)

    def forward(self, hidden):
        hidden = self.flatten(hidden)
        return self.model(hidden)


JOINT_EXPERT_HEAD_REGISTRY = {
    "m1": M1JointPredictionHead,
    "m2": M2JointPredictionHead,
    "m3": M3JointPredictionHead,
    "m4": M4JointPredictionHead,
}


class JointExpertPredictionHeads(nn.Module):
    """
    Joint expert-head reconstruction.

    This module is independent from `models/fusion/expert_head.py`. It trains
    all configured expert prediction heads on every batch and returns the mean
    prediction as the default forecast. It has no active-expert routing and no
    learned fusion gate.
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
        dropout=0.0,
        device="cuda",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.target_key = target_key
        self.loss_type = loss_type
        self.expert_names = self._resolve_expert_names(models_dict, expert_names)

        self._validate_loss_type(loss_type)
        resolved_dims = self._resolve_hidden_dims(expert_dims)

        self.pv_revin_layer = RevIN(1, affine=1, subtract_last=0)
        self.prediction_heads = nn.ModuleDict()
        for name in self.expert_names:
            head_cls = JOINT_EXPERT_HEAD_REGISTRY[name]
            self.prediction_heads[name] = head_cls(
                hidden_dim=resolved_dims[name],
                seq_len=seq_len,
                pred_len=pred_len,
                n_features=n_features,
                head_dropout=dropout,
            )

        self.to(device)

    @classmethod
    def _validate_loss_type(cls, loss_type):
        if loss_type not in cls.SUPPORTED_LOSSES:
            valid = ", ".join(sorted(cls.SUPPORTED_LOSSES))
            raise ValueError(f"Unknown loss_type={loss_type!r}. Valid: {valid}.")

    def _resolve_expert_names(self, models_dict, expert_names):
        if expert_names is not None:
            names = list(expert_names)
        elif models_dict is None:
            names = list(JOINT_EXPERT_HEAD_REGISTRY.keys())
        else:
            names = list(models_dict.keys())

        unsupported = [
            name for name in names if name not in JOINT_EXPERT_HEAD_REGISTRY
        ]
        if unsupported:
            valid = ", ".join(sorted(JOINT_EXPERT_HEAD_REGISTRY))
            raise ValueError(
                "Unsupported experts for expert_head_joint: "
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
                + ". JointExpertPredictionHeads needs each expert hidden dimension."
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
                f"Prediction head output must be {expected_shape}, "
                f"got {tuple(output.shape)}."
            )
        return output

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

    def loss_func_per_sample(self, pred, target):
        loss_dims = tuple(range(1, pred.ndim))
        if self.loss_type == "mse":
            return (pred - target).pow(2).mean(dim=loss_dims)
        if self.loss_type == "rmse":
            return torch.sqrt((pred - target).pow(2).mean(dim=loss_dims) + 1e-8)
        if self.loss_type == "mae":
            return (pred - target).abs().mean(dim=loss_dims)
        if self.loss_type == "huber":
            return F.huber_loss(pred, target, delta=1.0, reduction="none").mean(
                dim=loss_dims
            )
        raise ValueError(f"Unknown loss_type={self.loss_type!r}")

    def _get_expert_mask(self, batch, batch_size, device):
        if batch is None or "expert_mask" not in batch:
            return None

        mask = batch["expert_mask"]
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, device=device)
        else:
            mask = mask.to(device=device)
        mask = mask.float()

        expert_count = len(self.expert_names)
        if mask.dim() == 1:
            if mask.shape[0] != expert_count:
                raise ValueError(
                    f"expert_mask length must be {expert_count}, got {mask.shape[0]}."
                )
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        elif mask.dim() == 2:
            if tuple(mask.shape) != (batch_size, expert_count):
                raise ValueError(
                    f"expert_mask must be shaped {(batch_size, expert_count)}, "
                    f"got {tuple(mask.shape)}."
                )
        else:
            raise ValueError(
                f"expert_mask must be shaped ({expert_count},) or "
                f"({batch_size}, {expert_count}), got {tuple(mask.shape)}."
            )

        mask = (mask > 0).float()
        all_masked = mask.sum(dim=1) == 0
        if all_masked.any():
            mask = mask.clone()
            mask[all_masked] = 1.0
        return mask

    def _set_revin_statistics(self, batch):
        if batch is None:
            raise ValueError("batch is required for RevIN normalization.")
        pv_his = batch["observe_power"].unsqueeze(1)
        if "chronos" in batch:
            chronos = batch["chronos"].unsqueeze(1)
            pv = torch.cat([pv_his, chronos], dim=2)
        else:
            pv = pv_his
        pv = pv.permute(0, 2, 1)
        self.pv_revin_layer(pv, "norm")

    def _denorm_output(self, output):
        output = output.permute(0, 2, 1)
        output = self.pv_revin_layer(output, "denorm")
        if output.shape[-1] != self.n_features:
            output = output[..., : self.n_features]
        return output.permute(0, 2, 1)

    def _predict_one(self, name, hidden):
        output = self._format_output(self.prediction_heads[name](hidden))
        return self._denorm_output(output)

    def forward(
        self,
        batch_tensor,
        batch=None,
        flag="test",
        return_info=False,
        active_expert_name=None,
    ):
        missing = [name for name in self.expert_names if name not in batch_tensor]
        if missing:
            raise KeyError("Missing hidden tensors for experts: " + ", ".join(missing))

        self._set_revin_statistics(batch)

        pred_by_expert = {}
        preds = []
        for name in self.expert_names:
            pred = self._predict_one(name, batch_tensor[name])
            pred_by_expert[name] = pred
            preds.append(pred)

        pred_stack = torch.stack(preds, dim=1)
        expert_mask = self._get_expert_mask(
            batch,
            batch_size=pred_stack.shape[0],
            device=pred_stack.device,
        )
        if expert_mask is None:
            output = pred_stack.mean(dim=1)
            normalized_mask = None
        else:
            normalized_mask = expert_mask / expert_mask.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)
            output = (pred_stack * normalized_mask[:, :, None, None]).sum(dim=1)

        info = {
            "expert_names": self.expert_names,
            "pred_by_expert": pred_by_expert,
            "pred_stack": pred_stack,
        }
        if expert_mask is not None:
            info["expert_mask"] = expert_mask
            info["normalized_expert_mask"] = normalized_mask

        if flag == "test":
            if return_info:
                return output, info
            return output.squeeze(1)

        if flag != "train":
            raise ValueError("flag must be either 'train' or 'test'.")

        target = self._get_target(batch)
        per_sample_head_losses = torch.stack(
            [
                self.loss_func_per_sample(pred_by_expert[name], target)
                for name in self.expert_names
            ],
            dim=1,
        )
        if expert_mask is None:
            loss = per_sample_head_losses.mean()
        else:
            loss = (per_sample_head_losses * expert_mask).sum() / expert_mask.sum().clamp_min(1.0)
        head_losses = per_sample_head_losses.mean(dim=0)
        info.update(
            {
                "head_loss": loss.detach(),
                "head_losses": {
                    name: head_losses[i].detach()
                    for i, name in enumerate(self.expert_names)
                },
            }
        )

        if return_info:
            return output, loss, info
        return output, loss


FusionModel = JointExpertPredictionHeads
