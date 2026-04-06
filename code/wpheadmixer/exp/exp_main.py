import os
import time
import warnings

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.model import WPMixer
from models.model_kan import WPMixerKAN
from utils.logger import ExperimentLogger
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')

try:
    from thop import profile
except ImportError:
    profile = None


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.best_val_loss = np.inf
        self.best_val_mae = np.inf
        self.best_test_loss_at_best_val = np.inf
        self.best_test_mae_at_best_val = np.inf
        self.best_epoch = -1
        self.logger = None

    def _build_model(self):
        model_dict = {'WPMixer': WPMixer, 'WPMixerKAN': WPMixerKAN}
        model = model_dict[self.args.model](
            self.args.c_in,
            self.args.c_out,
            self.args.seq_len,
            self.args.pred_len,
            self.args.d_model,
            self.args.dropout,
            self.args.embedding_dropout,
            self.device,
            self.args.batch_size,
            self.args.tfactor,
            self.args.dfactor,
            self.args.wavelet,
            self.args.level,
            self.args.patch_len,
            self.args.stride,
            self.args.no_decomposition,
            self.args.use_amp,
            getattr(self.args, 'head_type', 'linear'),
            getattr(self.args, 'match_head_params', False),
            getattr(self.args, 'head_param_budget', None),
            getattr(self.args, 'mlp_hidden_dim', None),
            getattr(self.args, 'kan_grid_size', 5),
            getattr(self.args, 'kan_spline_order', 3),
            getattr(self.args, 'hybrid_linear_ratio', 0.5),
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _unwrap_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    def _select_criterion(self):
        criterion = {'mse': torch.nn.MSELoss(), 'smoothL1': torch.nn.SmoothL1Loss()}
        try:
            return criterion[self.args.loss]
        except KeyError as exc:
            raise ValueError(f"Invalid argument: {exc} (loss: {self.args.loss})")

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        preds_mean, trues = [], []

        with torch.no_grad():
            for batch_x, batch_y, *rest in vali_loader:
                pred_mean, true = self._process_one_batch(vali_data, batch_x, batch_y, 'vali')
                preds_mean.append(pred_mean)
                trues.append(true)

        preds_mean = torch.cat(preds_mean).cpu()
        trues = torch.cat(trues).cpu()

        preds_mean = preds_mean.reshape(-1, preds_mean.shape[-2], preds_mean.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds_mean.numpy(), trues.numpy())
        self.model.train()
        return mse, mae

    def train(self, setting, optunaTrialReport=None):
        self.logger = ExperimentLogger(log_dir="./logs", experiment_name=setting)
        self.logger.log_experiment_config(self.args)
        self.logger.log_model_info(self._unwrap_model(), input_shape=(self.args.batch_size, self.args.seq_len, self.args.c_in))

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = None
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler(init_scale=1024)

        for epoch in range(self.args.train_epochs):
            self.logger.log_epoch_start(epoch, self.args.train_epochs)
            train_loss_values = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, *rest) in enumerate(train_loader):
                model_optim.zero_grad(set_to_none=True)
                pred_mean, true = self._process_one_batch(train_data, batch_x, batch_y, 'train')
                loss = criterion(pred_mean, true)

                if getattr(self.args, 'kan_reg_weight', 0.0) > 0:
                    regularization_loss = self._unwrap_model().regularization_loss(
                        regularize_activation=self.args.kan_reg_weight,
                        regularize_entropy=getattr(self.args, 'kan_entropy_weight', 0.0),
                    )
                    loss = loss + regularization_loss

                train_loss_values.append(loss.item())

                if self.args.use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            epoch_duration = time.time() - epoch_time
            print("Epoch {}: cost time: {:.2f} sec".format(epoch + 1, epoch_duration))

            train_loss = float(np.mean(train_loss_values)) if train_loss_values else float('nan')
            vali_loss, vali_mae = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae = self.vali(test_data, test_loader, criterion)
            current_lr = model_optim.param_groups[0]['lr']

            self.logger.log_epoch_metrics(
                epoch,
                train_loss,
                vali_loss,
                vali_mae,
                test_loss,
                test_mae,
                epoch_duration,
                current_lr,
            )

            if vali_loss < self.best_val_loss:
                self.best_val_loss = float(vali_loss)
                self.best_val_mae = float(vali_mae)
                self.best_test_loss_at_best_val = float(test_loss)
                self.best_test_mae_at_best_val = float(test_mae)
                self.best_epoch = epoch

            if optunaTrialReport is not None:
                optunaTrialReport.report(vali_loss, epoch)
                if optunaTrialReport.should_prune():
                    raise optuna.exceptions.TrialPruned()

            print(
                "\tEpoch {0}: Steps- {1} | Train Loss: {2:.5f} Vali.MSE: {3:.5f} Vali.MAE: {4:.5f} Test.MSE: {5:.5f} Test.MAE: {6:.5f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    vali_mae,
                    test_loss,
                    test_mae,
                )
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("\tEarly stopping")
                self.logger.log_early_stopping(epoch, early_stopping.val_loss_min)
                break

            if np.isnan(train_loss):
                print("stopping: train-loss-nan")
                break

            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        total_training_time = time.time() - time_now
        self.logger.log_training_complete(
            self.best_epoch,
            self.best_val_loss,
            self.best_test_loss_at_best_val,
            total_training_time,
        )
        self.logger.save_metrics()

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting):
        if self.logger is None:
            self.logger = ExperimentLogger(log_dir="./logs", experiment_name=setting + "_test_only")
            self.logger.log_experiment_config(self.args)

        self.logger.log_test_start()
        test_data, test_loader = self._get_data(flag='test')
        self._select_criterion()
        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for i, (batch_x, batch_y, *rest) in enumerate(test_loader):
                pred, true = self._process_one_batch(test_data, batch_x, batch_y, 'test')
                preds.append(pred)
                trues.append(true)

        preds = torch.cat(preds).cpu()
        trues = torch.cat(trues).cpu()

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds.numpy(), trues.numpy())
        self.logger.log_test_results(mse, mae, rmse, mape, mspe)
        self.logger.save_metrics()

        print('mse: {}, mae: {}'.format(mse, mae))
        return mse, mae

    def get_gflops(self):
        if profile is None:
            raise ImportError('thop is not installed. Install it to use get_gflops().')

        batch = self.args.batch_size
        seq = self.args.seq_len
        channel = self.args.c_in
        input_tensor = torch.randn(batch, seq, channel).to(self.device)

        self.model.eval()
        macs, params = profile(self.model, inputs=(input_tensor,), verbose=True)
        gflops = 2 * macs / 1e9
        print(f"Total GFLOPs: {gflops:.4f}")
        return gflops

    def predict(self, setting, load=False):
        raise NotImplementedError("not implemented for uncertainty")

    def _process_one_batch(self, dataset_object, batch_x, target, function):
        batch_x = batch_x.to(dtype=torch.float, device=self.device)
        target = target.to(dtype=torch.float, device=self.device)
        target = target[:, -self.args.pred_len:, -self.args.c_out:]

        scaler = None
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                pred = self.model(batch_x)
        else:
            pred = self.model(batch_x)

        pred = pred[:, -self.args.pred_len:, -self.args.c_out:]
        return pred, target
