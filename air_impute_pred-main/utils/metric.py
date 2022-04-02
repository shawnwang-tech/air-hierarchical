import os.path
from typing import Any, Callable, Optional
import torch
import numpy as np
from torchmetrics import Metric
from sklearn.metrics import mean_squared_error


class LightningMetric(Metric):

    def __init__(
        self,
        dir_t_now,
        mask,
        pm25_mean,
        pm25_std,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.dir_t_now = dir_t_now
        self.mask = mask
        self.pm25_mean = pm25_mean,
        self.pm25_std = pm25_std,
        self.add_state("y_true", default=[], dist_reduce_fx=None)
        self.add_state("y_pred", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor, t: torch.Tensor):
        self.y_pred.append(preds)
        self.y_true.append(target)
        self.t.append(t)

    def compute(self):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        t = torch.cat(self.t, dim=0)

        y_pred_denorm = (y_pred * self.pm25_std[0] + self.pm25_mean[0]).cpu().data.numpy()
        y_true_denorm = (y_true * self.pm25_std[0] + self.pm25_mean[0]).cpu().data.numpy()
        t = t.cpu().data.numpy()

        mask = self.mask.cpu().data.numpy()

        y_pred_sta = y_pred_denorm[:, :, mask != 0]
        y_true_sta = y_true_denorm[:, :, mask != 0]

        metric_dict = {}

        rmse = mean_squared_error(y_pred_sta.flatten(), y_true_sta.flatten(), squared=False)

        metric_dict.update({'rmse': rmse})

        np.save(os.path.join(self.dir_t_now, 'y_pred.npy'), y_pred_denorm)
        np.save(os.path.join(self.dir_t_now, 'y_true.npy'), y_true_denorm)
        np.save(os.path.join(self.dir_t_now, 't.npy'), t)

        print(self.dir_t_now)

        return metric_dict
