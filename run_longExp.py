import argparse
import os
import torch
import pandas as pd
import random
import numpy as np


FUSION_EXPERIMENT_ARGS = [
    {
        "name": "--fusion_dropout",
        "type": float,
        "default": None,
        "help": "prediction-head dropout override",
    },
    {
        "name": "--fusion_expert_dims",
        "type": str,
        "default": None,
        "help": "expert hidden dims, e.g. 'm1:512,m2:256,m3:384,m4:512'",
    },
    {
        "name": "--fusion_loss",
        "type": str,
        "default": None,
        "choices": ["mse", "mae", "huber", "rmse", "quantile"],
        "help": "prediction-head loss type",
    },
    {
        "name": "--fusion_head_train_mode",
        "type": str,
        "default": "round_robin",
        "choices": ["round_robin", "joint"],
        "help": "prediction-head training mode: round_robin trains one head per batch; joint trains all heads per batch",
    },
    {
        "name": "--fusion_unfreeze_epoch",
        "type": int,
        "default": -1,
        "help": "freeze expert models for this many epochs, then unfreeze; -1 disables",
    },
    {
        "name": "--fusion_expert_lr_scale",
        "type": float,
        "default": 0.1,
        "help": "expert-model learning-rate scale after unfreezing",
    },
]


SETTING_EXPERIMENT_COMPONENTS = [
    {"template": "_loss{value}", "attr": "fusion_loss"},
    {"template": "_drop{value}", "attr": "fusion_dropout"},
    {"template": "_headmode{value}", "attr": "fusion_head_train_mode"},
    {
        "template": "_unfreeze{0}x{1}",
        "attrs": ["fusion_unfreeze_epoch", "fusion_expert_lr_scale"],
    },
]


def add_argument_specs(parser, specs):
    for spec in specs:
        spec = dict(spec)
        name = spec.pop("name")
        parser.add_argument(name, **spec)


def arg_value(args, name, default="default"):
    value = getattr(args, name)
    return value if value is not None else default


def build_setting(args):
    fusion_expert_names = args.fusion_expert_names or "all"
    experiment_parts = []
    for component in SETTING_EXPERIMENT_COMPONENTS:
        if "attr" in component:
            experiment_parts.append(
                component["template"].format(
                    value=arg_value(args, component["attr"]),
                )
            )
        else:
            values = [arg_value(args, attr) for attr in component["attrs"]]
            experiment_parts.append(component["template"].format(*values))

    return (
        f'{args.model_id}_{args.model}_{args.fusion_version}_{args.data}'
        f'_sl{args.seq_len}_pl{args.pred_len}_bs{args.batch_size}'
        f'_opt{args.optimizer}_lr{args.learning_rate}_wd{args.weight_decay}'
        f'_lradj{args.lradj}'
        f'_mom{args.muon_momentum}_ns{args.muon_ns_steps}'
        + ''.join(experiment_parts)
        + f'_expert{fusion_expert_names}_{args.des}'
    )


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='FusionModel',
                        help='model name, options: [FusionModel, DLinear, PatchTST, iTransformer, TimesNet]')
    parser.add_argument('--fusion_version', type=str, default='expert_head',
                        choices=['expert_head', 'expert_head_joint'],
                        help='fusion model version selected by models/factory.py')
    parser.add_argument('--fusion_expert_names', type=str, default=None,
                        help="comma-separated experts to train, e.g. 'm1,m2,m4'; default uses all four")
    add_argument_specs(parser, FUSION_EXPERIMENT_ARGS)
    parser.add_argument('--target_key', type=str, default='observe_power_future',
                        help='target tensor key used by fusion models')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size (n_features)')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'muon', 'moun'],
                        help='optimizer type')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
    parser.add_argument('--muon_momentum', type=float, default=0.95, help='Muon momentum')
    parser.add_argument('--muon_ns_steps', type=int, default=5,
                        help='Muon Newton-Schulz iteration steps')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='none',
                        choices=['none', 'constant', 'type1', 'type2', 'type3', 'OneCycleLR', 'cosine'],
                        help='learning rate schedule')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct start')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    from exp.exp_main import Exp_Main

    Exp = Exp_Main

    setting = build_setting(args)

    train_df = pd.read_parquet('xxx')
    valid_df = pd.read_parquet('xxx')
    test_df = pd.read_parquet('xxx')

    for col in ['observe_power', 'observe_power_future', 'chronos']:
        train_df[col] /= 500 
        valid_df[col] /= 500 
        test_df[col] /= 500 

    exp = Exp(args)  # set experiments

    if args.is_training:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting, train_df, valid_df)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test_df)

        torch.cuda.empty_cache()
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test_df, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
