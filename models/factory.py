import inspect

import torch
import torch.nn as nn

from .fusion.base import FusionBase
from .fusion.expert_head import ExpertHeadReconstruction, MultiExpertHeadFusion
from .fusion.expert_head_v2 import AlignedExpertHeadFusion
from .fusion.expert_head_v3 import CompressedExpertHeadFusion
from .fusion.expert_head_v4 import OrthogonalAttentionExpertHeadFusion
from .fusion.expert_head_v5 import FlattenOrthogonalAttentionExpertHeadFusion
from .fusion.expert_head_v6 import ExpertSpecificAttentionFusion
from .fusion.legacy import FusionModel as FusionLegacy
from .fusion.tensor_v3 import FusionModelV3 as FusionTensorV3
from .fusion.v2 import FusionModel as FusionV2
from .fusion.v3 import FusionModelV3
from .fusion.v4 import FusionModelV4
from .fusion.v5 import FusionModelV5


FUSION_REGISTRY = {
    "base": FusionBase,
    "expert_head": ExpertHeadReconstruction,
    "multi_expert_head": MultiExpertHeadFusion,
    "expert_head_v2": AlignedExpertHeadFusion,
    "expert_head_v3": CompressedExpertHeadFusion,
    "expert_head_v4": OrthogonalAttentionExpertHeadFusion,
    "expert_head_v5": FlattenOrthogonalAttentionExpertHeadFusion,
    "expert_head_v6": ExpertSpecificAttentionFusion,
    "legacy": FusionLegacy,
    "v2": FusionV2,
    "v3": FusionModelV3,
    "v4": FusionModelV4,
    "v5": FusionModelV5,
    "tensor_v3": FusionTensorV3,
}

DEFAULT_FUSION_VERSION = "base"
HIDDEN_ONLY_FUSION_VERSIONS = {
    "base",
    "expert_head",
    "multi_expert_head",
    "expert_head_v2",
    "expert_head_v3",
    "expert_head_v4",
    "expert_head_v5",
    "expert_head_v6",
    "v4",
    "v5",
    "tensor_v3",
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

        if freeze_experts:
            for model in self.expert_models.values():
                for param in model.parameters():
                    param.requires_grad = False
                model.eval()

    def forward(self, batch, flag="test", **kwargs):
        batch_tensor = {}
        for name, model in self.expert_models.items():
            model.eval()
            with torch.no_grad():
                batch_tensor[name] = model.forward_hidden(batch)

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
    aligned_tokens = parse_expert_dims(getattr(args, "fusion_aligned_tokens", None))
    expert_names = parse_expert_names(getattr(args, "fusion_expert_names", None))
    aligned_token_count = getattr(args, "fusion_aligned_token_count", None)
    adapter_type = getattr(args, "fusion_adapter_type", None)
    d_fusion = getattr(args, "fusion_d_model", None)
    dropout = getattr(args, "fusion_dropout", None)
    target_key = getattr(args, "target_key", None)
    loss_type = getattr(args, "fusion_loss", None)
    expert_name = getattr(args, "fusion_expert_name", None)
    aux_loss_weight = getattr(args, "fusion_aux_loss_weight", None)
    orth_loss_weight = getattr(args, "fusion_orth_loss_weight", None)
    attention_heads = getattr(args, "fusion_attention_heads", None)
    attention_layers = getattr(args, "fusion_attention_layers", None)
    attention_query_tokens = getattr(args, "fusion_attention_query_tokens", None)
    ensemble_size = getattr(args, "fusion_ensemble_size", None)
    ensemble_scaling_init = getattr(args, "fusion_ensemble_scaling_init", None)
    expert_drop_prob = getattr(args, "fusion_expert_drop_prob", None)

    constructor_kwargs = {
        "models_dict": base_models,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "n_features": args.enc_in,
        "expert_dims": expert_dims,
        "expert_names": expert_names,
        "aligned_tokens": aligned_tokens,
        "aligned_token_count": aligned_token_count,
        "adapter_type": adapter_type,
        "d_fusion": d_fusion,
        "dropout": dropout,
        "target_key": target_key,
        "loss_type": loss_type,
        "expert_name": expert_name,
        "aux_loss_weight": aux_loss_weight,
        "orth_loss_weight": orth_loss_weight,
        "attention_heads": attention_heads,
        "attention_layers": attention_layers,
        "attention_query_tokens": attention_query_tokens,
        "ensemble_size": ensemble_size,
        "ensemble_scaling_init": ensemble_scaling_init,
        "expert_drop_prob": expert_drop_prob,
        "device": device,
    }
    constructor_kwargs = _filter_constructor_kwargs(model_cls, constructor_kwargs)

    fusion_model = model_cls(**constructor_kwargs).float()

    if version in HIDDEN_ONLY_FUSION_VERSIONS and _can_wrap_experts(base_models):
        return FusionModelWithExperts(base_models, fusion_model).float()

    return fusion_model
