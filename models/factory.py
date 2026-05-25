import inspect

import torch
import torch.nn as nn

from .fusion.expert_head import ExpertPredictionHeads
from .fusion.expert_head_joint import JointExpertPredictionHeads


FUSION_REGISTRY = {
    "expert_head": ExpertPredictionHeads,
    "expert_head_joint": JointExpertPredictionHeads,
}

DEFAULT_FUSION_VERSION = "expert_head"
HIDDEN_ONLY_FUSION_VERSIONS = {
    "expert_head",
    "expert_head_joint",
}


class FusionModelWithExperts(nn.Module):
    """
    Adapter that lets hidden-only fusion models use the standard experiment
    interface: model(batch, flag='train'|'test').
    """

    def __init__(self, expert_models, fusion_model, freeze_experts=True):
        super().__init__()
        self.expert_models = nn.ModuleDict(expert_models)
        self.fusion_model = fusion_model
        self.experts_frozen = None

        self.set_experts_trainable(not freeze_experts)

    def set_experts_trainable(self, trainable):
        self.experts_frozen = not trainable
        for model in self.expert_models.values():
            for param in model.parameters():
                param.requires_grad = trainable
            if trainable:
                model.train()
            else:
                model.eval()

    def set_active_expert(self, active_expert_name=None):
        if hasattr(self.fusion_model, "set_active_expert"):
            self.fusion_model.set_active_expert(active_expert_name)

        if self.experts_frozen:
            for model in self.expert_models.values():
                for param in model.parameters():
                    param.requires_grad = False
                model.eval()
            return

        for name, model in self.expert_models.items():
            trainable = active_expert_name is None or name == active_expert_name
            for param in model.parameters():
                param.requires_grad = trainable
            if trainable:
                model.train()
            else:
                model.eval()

    def forward(self, batch, flag="test", active_expert_name=None, **kwargs):
        batch_tensor = {}
        if flag == "train" and active_expert_name is not None:
            if active_expert_name not in self.expert_models:
                available = ", ".join(self.expert_models.keys())
                raise KeyError(
                    f"active_expert_name={active_expert_name!r} is not in "
                    f"expert_models. Available experts: {available}."
                )
            expert_items = [(active_expert_name, self.expert_models[active_expert_name])]
        else:
            expert_items = self.expert_models.items()

        for name, model in expert_items:
            if flag == "train" and not self.experts_frozen:
                model.train()
            else:
                model.eval()

            if self.experts_frozen:
                with torch.no_grad():
                    batch_tensor[name] = model.forward_hidden(batch)
            else:
                batch_tensor[name] = model.forward_hidden(batch)

        if active_expert_name is not None:
            kwargs["active_expert_name"] = active_expert_name

        return self.fusion_model(batch_tensor, batch, flag=flag, **kwargs)


def fusion_version_choices():
    return tuple(FUSION_REGISTRY.keys())


def get_fusion_model_class(version):
    try:
        return FUSION_REGISTRY[version]
    except KeyError as exc:
        valid = ", ".join(fusion_version_choices())
        raise ValueError(
            f"Unknown fusion_version={version!r}. Valid versions: {valid}"
        ) from exc


def parse_expert_dims(raw_value):
    if raw_value is None or raw_value == "":
        return None

    dims = {}
    for item in raw_value.split(","):
        name, sep, value = item.partition(":")
        if not sep:
            raise ValueError(
                "fusion_expert_dims must look like 'm1:512,m2:256,m3:384'."
            )
        dims[name.strip()] = int(value.strip())
    return dims


def parse_expert_names(raw_value):
    if raw_value is None or raw_value == "":
        return None
    return [name.strip() for name in raw_value.split(",") if name.strip()]


def _filter_constructor_kwargs(model_cls, kwargs):
    signature = inspect.signature(model_cls.__init__)
    parameters = signature.parameters
    if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        return kwargs

    return {
        key: value
        for key, value in kwargs.items()
        if key in parameters and value is not None
    }


def _can_wrap_experts(base_models):
    return bool(base_models) and all(
        isinstance(model, nn.Module) for model in base_models.values()
    )


def build_fusion_model(args, base_models=None, device=None):
    version = getattr(args, "fusion_version", DEFAULT_FUSION_VERSION)
    model_cls = get_fusion_model_class(version)

    expert_dims = parse_expert_dims(getattr(args, "fusion_expert_dims", None))
    expert_names = parse_expert_names(getattr(args, "fusion_expert_names", None))
    dropout = getattr(args, "fusion_dropout", None)
    target_key = getattr(args, "target_key", None)
    loss_type = getattr(args, "fusion_loss", None)

    constructor_kwargs = {
        "models_dict": base_models,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "n_features": args.enc_in,
        "expert_dims": expert_dims,
        "expert_names": expert_names,
        "dropout": dropout,
        "target_key": target_key,
        "loss_type": loss_type,
        "device": device,
    }
    constructor_kwargs = _filter_constructor_kwargs(model_cls, constructor_kwargs)

    fusion_model = model_cls(**constructor_kwargs).float()

    if version in HIDDEN_ONLY_FUSION_VERSIONS and _can_wrap_experts(base_models):
        return FusionModelWithExperts(base_models, fusion_model).float()

    return fusion_model
