import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.revin import RevIN    


class M1PredictionHead(nn.Module):
    # M1 模型输出的形状是(batchsize, 9, 128)
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


class M2PredictionHead(nn.Module):
    # M2 模型输出的形状是(batchsize, 2, 512)
    def __init__(
        self,
        hidden_channels=2,
        hidden_dim=512,
        dropout_rate=0,
        pred_len=192,
        head_dropout=None,
        **kwargs,
    ):
        super().__init__()
        if head_dropout is not None:
            dropout_rate = head_dropout
        self.channel = hidden_channels
        self.targetseq_len = pred_len
        
        # 有几个channel 就有几个独立的 regression_head (如果不共享权重)
        self.regression_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim * 2, self.targetseq_len),
            ) for _ in range(self.channel)
        ])

    def forward(self, hidden):
        # hidden shape: (batch_size, channel, hidden_dim)
        if hidden.shape[1] != self.channel:
            raise ValueError(
                f"M2 hidden channel must be {self.channel}, got {hidden.shape[1]}."
            )
        # 使用列表推导式和 torch.stack 避免 pre-allocate zeros 带来的开销和潜在的 device 匹配问题
        preds = [self.regression_heads[i](hidden[:, i, :]) for i in range(self.channel)]
        
        # 沿着 channel 维度 (dim=1) 堆叠，输出 shape: (batch_size, channel, targetseq_len)
        return torch.stack(preds, dim=1).mean(dim=1)



class M3PredictionHead(nn.Module):
    # M3 模型输出的形状是(batchsize, 162, 384)
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
        self.flatten = nn.Flatten(start_dim = 1)
        self.linear = nn.Linear(in_dim, out_dim) 
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, hidden):
        hidden = self.flatten(hidden)
        hidden = self.dropout(hidden)
        hidden = self.linear(hidden)
        return hidden



class M4PredictionHead(nn.Module):
    # M4 模型输出的形状是(batchsize, 5, 9, 256)
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
        self.flatten = nn.Flatten(start_dim = -3)
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
        x = self.flatten(hidden)       # (B, nf)
        x = self.model(x)           # (B, target_window)
        return x


EXPERT_HEAD_REGISTRY = {
    "m1": M1PredictionHead,
    "m2": M2PredictionHead,
    "m3": M3PredictionHead,
    "m4": M4PredictionHead,
}


class ExpertHeadReconstruction(nn.Module):
    """
    Single-expert head reconstruction experiment.

    This module does not fuse multiple experts. It selects one frozen expert's
    hidden state from batch_tensor, feeds it into a newly initialized prediction
    head with the same intended structure as that expert's original head, and
    trains only this new head.

    Expected input:
        batch_tensor[expert_name]: hidden state from the selected expert.
        batch[target_key]: forecast target during training.

    Output shape:
        (B, n_features, pred_len)
    """

    DEFAULT_EXPERT_DIMS = {"m1": 128, "m2": 512, "m3": 384, "m4": 256}
    SUPPORTED_LOSSES = {"mse", "mae", "huber"}

    def __init__(
        self,
        models_dict=None,
        seq_len=None,
        pred_len=192,
        n_features=1,
        expert_dims=None,
        expert_name="m1",
        target_key="observe_power_future",
        loss_type="mse",
        device="cuda",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.expert_name = expert_name
        self.target_key = target_key
        self.loss_type = loss_type

        self._validate_loss_type(loss_type)
        self._validate_expert_name(models_dict)
        hidden_dim = self._resolve_hidden_dim(expert_dims)

        head_cls = EXPERT_HEAD_REGISTRY[expert_name]
        self.prediction_head = head_cls(
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            n_features=n_features,
        )
        
        self.pv_revin_layer = RevIN(1, affine=1, subtract_last=0)
        self.to(device)

    @classmethod
    def _validate_loss_type(cls, loss_type):
        if loss_type not in cls.SUPPORTED_LOSSES:
            valid = ", ".join(sorted(cls.SUPPORTED_LOSSES))
            raise ValueError(f"Unknown loss_type={loss_type!r}. Valid: {valid}.")

    def _validate_expert_name(self, models_dict):
        if self.expert_name not in EXPERT_HEAD_REGISTRY:
            valid = ", ".join(sorted(EXPERT_HEAD_REGISTRY))
            raise ValueError(
                f"Unknown expert_name={self.expert_name!r}. Valid: {valid}."
            )

        if models_dict is not None and self.expert_name not in models_dict:
            available = ", ".join(models_dict.keys())
            raise ValueError(
                f"expert_name={self.expert_name!r} is not in models_dict. "
                f"Available experts: {available}."
            )

    def _resolve_hidden_dim(self, expert_dims):
        resolved_dims = dict(self.DEFAULT_EXPERT_DIMS)
        if expert_dims is not None:
            resolved_dims.update(expert_dims)

        if self.expert_name not in resolved_dims:
            raise ValueError(
                f"Missing expert_dims for {self.expert_name!r}. "
                "ExpertHeadReconstruction needs the selected expert hidden dimension."
            )
        return resolved_dims[self.expert_name]

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
        if self.loss_type == "mae":
            return F.l1_loss(pred, target)
        if self.loss_type == "huber":
            return F.huber_loss(pred, target, delta=1.0)
        raise ValueError(f"Unknown loss_type={self.loss_type!r}")

    def forward(self, batch_tensor, batch=None, flag="test", return_info=False):
        if self.expert_name not in batch_tensor:
            available = ", ".join(batch_tensor.keys())
            raise KeyError(
                f"Missing hidden tensor for expert {self.expert_name!r}. "
                f"Available hidden tensors: {available}."
            )

        pv_his = batch['observe_power'].unsqueeze(1)
        tsfm = batch['chronos'].unsqueeze(1)
        pv = torch.cat([pv_his, tsfm], dim=2)
        pv = pv.permute(0, 2, 1)
        pv = self.pv_revin_layer(pv, 'norm')
        pv = pv.permute(0, 2, 1)
        
        hidden = batch_tensor[self.expert_name]
        
        output = self._format_output(self.prediction_head(hidden))

        output = output.permute(0, 2, 1)
        output = self.pv_revin_layer(output, 'denorm')
        output = output.permute(0, 2, 1)
        
        if flag == "test":
            if return_info:
                return output, {"expert_name": self.expert_name, "hidden": hidden}
            return output.squeeze(1)

        if flag != "train":
            raise ValueError("flag must be either 'train' or 'test'.")

        target = self._get_target(batch)
        loss = self.loss_func(output, target)
        if return_info:
            return output, loss, {"expert_name": self.expert_name, "hidden": hidden}
        return output, loss


class MultiExpertHeadFusion(nn.Module):
    """
    Jointly trains all reconstructed expert heads with auxiliary supervision.

    Each frozen expert hidden state is decoded by its own prediction head. The
    final forecast is a learned weighted average of the four head predictions,
    while each individual head is still supervised against the target.
    """

    DEFAULT_EXPERT_DIMS = ExpertHeadReconstruction.DEFAULT_EXPERT_DIMS
    SUPPORTED_LOSSES = ExpertHeadReconstruction.SUPPORTED_LOSSES

    def __init__(
        self,
        models_dict=None,
        seq_len=None,
        pred_len=192,
        n_features=1,
        expert_dims=None,
        target_key="observe_power_future",
        loss_type="mse",
        aux_loss_weight=1.0,
        dropout=0.0,
        device="cuda",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.target_key = target_key
        self.loss_type = loss_type
        self.aux_loss_weight = aux_loss_weight
        self.expert_names = self._resolve_expert_names(models_dict)

        self._validate_loss_type(loss_type)
        resolved_dims = self._resolve_hidden_dims(expert_dims)

        self.pv_revin_layer = RevIN(1, affine=1, subtract_last=0)
        
        self.prediction_heads = nn.ModuleDict()
        for name in self.expert_names:
            head_cls = EXPERT_HEAD_REGISTRY[name]
            self.prediction_heads[name] = head_cls(
                hidden_dim=resolved_dims[name],
                seq_len=seq_len,
                pred_len=pred_len,
                n_features=n_features,
                head_dropout=dropout,
            )

        self.expert_logits = nn.Parameter(torch.zeros(len(self.expert_names)))
        self.to(device)

    def set_active_expert(self, active_expert_name=None):
        if active_expert_name is not None and active_expert_name not in self.expert_names:
            available = ", ".join(self.expert_names)
            raise KeyError(
                f"active_expert_name={active_expert_name!r} is not in "
                f"expert_names. Available experts: {available}."
            )

        for name, head in self.prediction_heads.items():
            trainable = active_expert_name is None or name == active_expert_name
            for param in head.parameters():
                param.requires_grad = trainable

        self.expert_logits.requires_grad = active_expert_name is None

    @classmethod
    def _validate_loss_type(cls, loss_type):
        if loss_type not in cls.SUPPORTED_LOSSES:
            valid = ", ".join(sorted(cls.SUPPORTED_LOSSES))
            raise ValueError(f"Unknown loss_type={loss_type!r}. Valid: {valid}.")

    def _resolve_expert_names(self, models_dict):
        if models_dict is None:
            return list(EXPERT_HEAD_REGISTRY.keys())

        expert_names = list(models_dict.keys())
        unsupported = [name for name in expert_names if name not in EXPERT_HEAD_REGISTRY]
        if unsupported:
            valid = ", ".join(sorted(EXPERT_HEAD_REGISTRY))
            raise ValueError(
                "Unsupported experts for multi_expert_head: "
                + ", ".join(unsupported)
                + f". Valid: {valid}."
            )
        return expert_names

    def _resolve_hidden_dims(self, expert_dims):
        resolved_dims = dict(self.DEFAULT_EXPERT_DIMS)
        if expert_dims is not None:
            resolved_dims.update(expert_dims)

        missing_dims = [name for name in self.expert_names if name not in resolved_dims]
        if missing_dims:
            raise ValueError(
                "Missing expert_dims for: "
                + ", ".join(missing_dims)
                + ". MultiExpertHeadFusion needs each expert hidden dimension."
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
        if self.loss_type == "mae":
            return F.l1_loss(pred, target)
        if self.loss_type == "huber":
            return F.huber_loss(pred, target, delta=1.0)
        raise ValueError(f"Unknown loss_type={self.loss_type!r}")

    def _predict_one(self, name, hidden):
        pred = self._format_output(self.prediction_heads[name](hidden))
        pred = pred.permute(0, 2, 1)
        pred = self.pv_revin_layer(pred, 'denorm')
        pred = pred.permute(0, 2, 1)
        return pred

    def forward(
        self,
        batch_tensor,
        batch=None,
        flag="test",
        return_info=False,
        active_expert_name=None,
    ):
        if active_expert_name is not None and active_expert_name not in self.expert_names:
            available = ", ".join(self.expert_names)
            raise KeyError(
                f"active_expert_name={active_expert_name!r} is not in "
                f"expert_names. Available experts: {available}."
            )

        missing = [name for name in self.expert_names if name not in batch_tensor]
        if flag == "train" and active_expert_name is not None:
            missing = [active_expert_name] if active_expert_name not in batch_tensor else []
        if missing:
            raise KeyError("Missing hidden tensors for experts: " + ", ".join(missing))

        pv_his = batch['observe_power'].unsqueeze(1)
        tsfm = batch['chronos'].unsqueeze(1)
        pv = torch.cat([pv_his, tsfm], dim=2)
        pv = pv.permute(0, 2, 1)
        pv = self.pv_revin_layer(pv, 'norm')
        pv = pv.permute(0, 2, 1)

        if flag == "train" and active_expert_name is not None:
            pred = self._predict_one(
                active_expert_name,
                batch_tensor[active_expert_name],
            )
            target = self._get_target(batch)
            loss = self.loss_func(pred, target)
            info = {
                "active_expert_name": active_expert_name,
                "pred_by_expert": {active_expert_name: pred},
                "main_loss": loss.detach(),
            }
            if return_info:
                return pred, loss, info
            return pred, loss

        pred_by_expert = {}
        preds = []
        for name in self.expert_names:
            pred = self._predict_one(name, batch_tensor[name])
            pred_by_expert[name] = pred
            preds.append(pred)

        pred_stack = torch.stack(preds, dim=1)
        expert_weight = F.softmax(self.expert_logits, dim=0)
        output = (pred_stack * expert_weight.view(1, -1, 1, 1)).sum(dim=1)

        info = {
            "expert_names": self.expert_names,
            "expert_weight": expert_weight,
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


FusionModel = ExpertHeadReconstruction
