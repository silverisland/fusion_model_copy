from data_provider.fusion_dataset import data_provider
from exp.exp_basic import Exp_Basic
from models import FusionModel, DLinear, PatchTST, iTransformer, TimesNet
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import os
import time
import yaml
import warnings

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'M1': DLinear,
            'M2': PatchTST,
            'M3': iTransformer,
            'M4': TimesNet,
        }
        
        if self.args.model == 'FusionModel':
            # Load base models
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
                base_models['m2'] = model_dict['M2'](config2) 
            if 'M3' in model_dict.keys():
                base_models['m3'] = model_dict['M3'](config3)
            if 'M4' in model_dict.keys():
                base_models['m4'] = model_dict['M4'](config4)

            model = FusionModel(base_models, self.args.seq_len, self.args.pred_len, self.args.enc_in, device=self.device).float()
        else:
            model = model_dict[self.args.model](self.args.seq_len, self.args.pred_len).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, df):
        data_set, data_loader = data_provider(self.args, df, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.model == 'FusionModel':
            # Optimize all trainable parameters (fusion layers)
            model_optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr=self.args.learning_rate
                )
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # Composite Loss for PV Power: Huber + Trend (1st-order difference)
        # Huber is robust to outliers (cloud cover spikes)
        # Trend loss handles distribution shifts by focusing on the change rate
        huber = nn.HuberLoss(delta=1.0)
        mse = nn.MSELoss()
        
        def composite_loss(pred, target):
            # 1. Base robust regression loss
            loss_val = huber(pred, target)
            
            # 2. Trend (Ramp) loss: focuses on the shape/change rate
            # Handles (B, P, C) or (n_heads, B, P, C)
            if pred.shape[1] > 1:
                diff_pred = pred[:, 1:] - pred[:, :-1]
                diff_target = target[:, 1:] - target[:, :-1]
                loss_trend = mse(diff_pred, diff_target)
                return loss_val + 0.5 * loss_trend # lambda=0.5
            
            return loss_val
            
        return composite_loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                outputs = self.model(batch)
                
                loss = criterion(outputs, batch['observe_power_future'])
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
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                outputs = self.model(batch)

                loss = criterion(outputs, batch['observe_power_future'])
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

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
                batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                outputs = self.model(batch)

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
        result_df.to_parquet('result_df.parquet', index = False)

        return