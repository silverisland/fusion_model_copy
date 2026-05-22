from data_provider.fusion_dataset import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.optimizers import Muon

from models.factory import build_fusion_model
    
# M1
from individual_models.fourier_moba_transformer.model import FourierMoBAPatchTST_V2
# M2
from individual_models.ylj_patchreg.src.experiments.ghi_reg import ExpFlexGHIReg
# M3
from individual_models.pv_forecast_moirai.src.grinder import create_model_wrapper
from individual_models.pv_forecast_moirai.src.utils.datatype import DDPConfig
# M4
from individual_models.m4_full_version.exp.exp_main import Exp_Main as m4_exp_main

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import os
import time
import math
import yaml
import warnings

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _move_to_device(self, batch):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _build_model(self):
        model_dict = {
            'M1': FourierMoBAPatchTST_V2,
            'M2': ExpFlexGHIReg,
            'M3': create_model_wrapper,
            'M4': m4_exp_main,
        }
        
        base_models = {}

        with open('./configs/m1config.yaml', 'r', encoding = 'utf-8') as f:
            config1 = yaml.safe_load(f)
        with open('./configs/m2config.yaml', 'r', encoding = 'utf-8') as f:
            config2 = yaml.safe_load(f)
        with open('./configs/m3config.yaml', 'r', encoding = 'utf-8') as f:
            config3 = yaml.safe_load(f)
        with open('./configs/m4config.yaml', 'r', encoding = 'utf-8') as f:
            config4 = yaml.safe_load(f)
            
        if 'M1' in model_dict.keys():
            base_models['m1'] = model_dict['M1'](config1)
        if 'M2' in model_dict.keys():
            base_models['m2'] = model_dict['M2'](config2).network
        if 'M3' in model_dict.keys():
            ddp_config = DDPConfig()
            base_models['m3'] = model_dict['M3'](config3, ddp_config)
        if 'M4' in model_dict.keys():
            base_models['m4'] = model_dict['M4'](config4).model

        model = build_fusion_model(self.args, base_models, self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, df):
        data_set, data_loader = data_provider(self.args, df, flag)
        return data_set, data_loader

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _trainable_param_groups(self):
        model = self._unwrap_model()
        expert_lr_scale = getattr(self.args, "fusion_expert_lr_scale", 1.0)
        if not hasattr(model, "expert_models") or not hasattr(model, "fusion_model"):
            return [{"params": self._trainable_parameters()}]

        fusion_params = [
            p for p in model.fusion_model.parameters()
            if p.requires_grad
        ]
        expert_params = [
            p for p in model.expert_models.parameters()
            if p.requires_grad
        ]

        param_groups = []
        if fusion_params:
            param_groups.append({
                "params": fusion_params,
                "lr": self.args.learning_rate,
            })
        if expert_params:
            param_groups.append({
                "params": expert_params,
                "lr": self.args.learning_rate * expert_lr_scale,
            })
        return param_groups

    def _muon_param_groups(self):
        param_groups = []
        for group in self._trainable_param_groups():
            params = group["params"]
            lr = group.get("lr", self.args.learning_rate)
            muon_params = [p for p in params if p.ndim >= 2]
            adamw_params = [p for p in params if p.ndim < 2]
            if muon_params:
                param_groups.append({
                    "params": muon_params,
                    "lr": lr,
                    "use_muon": True,
                })
            if adamw_params:
                param_groups.append({
                    "params": adamw_params,
                    "lr": lr,
                    "use_muon": False,
                })
        return param_groups

    def _select_optimizer(self):
        trainable_param_groups = self._trainable_param_groups()
        optimizer_name = self.args.optimizer.lower()
        optimizer_kwargs = {
            "lr": self.args.learning_rate,
            "weight_decay": self.args.weight_decay,
        }

        if optimizer_name == "adam":
            return optim.Adam(trainable_param_groups, **optimizer_kwargs)
        if optimizer_name == "adamw":
            return optim.AdamW(trainable_param_groups, **optimizer_kwargs)
        if optimizer_name in {"muon", "moun"}:
            return Muon(
                self._muon_param_groups(),
                momentum=self.args.muon_momentum,
                ns_steps=self.args.muon_ns_steps,
                **optimizer_kwargs,
            )

        raise ValueError(f"Unknown optimizer={self.args.optimizer!r}.")

    def _select_scheduler(self, optimizer, train_steps, train_epochs=None):
        lradj = str(self.args.lradj).lower()
        if lradj in {"none", "constant", "type1", "type2", "type3"}:
            return None

        if train_epochs is None:
            train_epochs = self.args.train_epochs
        total_steps = max(1, train_steps * train_epochs)
        if lradj == "onecyclelr":
            return optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=[group["lr"] for group in optimizer.param_groups],
                steps_per_epoch=train_steps,
                epochs=train_epochs,
                pct_start=self.args.pct_start,
            )

        if lradj == "cosine":
            warmup_steps = max(1, int(total_steps * self.args.pct_start))

            def lr_lambda(step):
                step = step + 1
                if step <= warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
                return 0.1 + 0.9 * cosine

            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        raise ValueError(
            f"Unknown lradj={self.args.lradj!r}. "
            "Valid values: none, constant, type1, type2, type3, OneCycleLR, cosine."
        )

    def _maybe_unfreeze_experts(self, epoch, train_steps):
        unfreeze_epoch = getattr(self.args, "fusion_unfreeze_epoch", -1)
        if unfreeze_epoch is None or unfreeze_epoch < 0 or epoch != unfreeze_epoch:
            return None, None

        model = self._unwrap_model()
        if not hasattr(model, "set_experts_trainable"):
            return None, None

        expert_lr = self.args.learning_rate * getattr(
            self.args,
            "fusion_expert_lr_scale",
            1.0,
        )
        print(
            "\n[staged-unfreeze] Epoch {} begins: unfreezing expert models. "
            "Frozen epochs: {}, fusion lr: {:.6g}, expert lr: {:.6g}".format(
                epoch + 1,
                unfreeze_epoch,
                self.args.learning_rate,
                expert_lr,
            )
        )
        model.set_experts_trainable(True)
        remaining_epochs = max(1, self.args.train_epochs - epoch)
        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(
            model_optim,
            train_steps,
            train_epochs=remaining_epochs,
        )
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            "[staged-unfreeze] Optimizer and scheduler rebuilt. "
            "Trainable parameters: {:,}\n".format(
                trainable_params,
            )
        )
        return model_optim, scheduler

    def vali(self, vali_data, vali_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch = self._move_to_device(batch)
                outputs, loss = self.model(batch, flag = 'train')
                
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, train_df, valid_df):
        train_data, train_loader = self._get_data(flag='train', df = train_df)
        vali_data, vali_loader = self._get_data(flag='val', df = valid_df)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(model_optim, train_steps)
        lradj = str(self.args.lradj).lower()

        for epoch in range(self.args.train_epochs):
            new_optim, new_scheduler = self._maybe_unfreeze_experts(
                epoch,
                train_steps,
            )
            if new_optim is not None:
                model_optim = new_optim
                scheduler = new_scheduler

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch = self._move_to_device(batch)

                outputs, loss = self.model(batch, flag = 'train')
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if lradj in {"onecyclelr", "cosine"} and scheduler is not None:
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if lradj in {"onecyclelr", "cosine"} and scheduler is not None:
                print("Learning rate: {}".format(scheduler.get_last_lr()[0]))
            elif lradj not in {"none", "constant"}:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test_df, test=0):
        test_data, test_loader = self._get_data(flag='test', df = test_df)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = self._move_to_device(batch)
                outputs = self.model(batch, flag = 'test')

                pred = outputs.detach().cpu().numpy()
                true = batch['observe_power_future'].detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mae:{}, mse:{}, rmse:{}'.format(mae, mse, rmse))

        result_df = pd.DataFrame([])
        result_df['timestamp_win'] = test_data.timestamp 
        result_df['observe_power_predict'] = [x for x in preds]
        result_df['observe_power_future'] = [x for x in trues]
        result_df.to_parquet(f'results_{setting}.parquet', index = False)

        return
