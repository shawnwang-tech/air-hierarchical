import os.path
from typing import Any, Callable, Optional
import torch
import numpy as np
from torchmetrics import Metric
from sklearn.metrics import mean_squared_error
import os.path
import os
import numpy as np
import matplotlib.pyplot as plt
import moviepy
import pandas as pd
from tqdm import tqdm


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


class LightningMetric(Metric):

    def __init__(
        self,
        dir_t_now,
        air_mean,
        air_std,
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
        self.air_mean = air_mean
        self.air_std = air_std
        self.add_state("y_true", default=[], dist_reduce_fx=None)
        self.add_state("y_pred", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor, t: torch.Tensor):
        self.y_pred.append(preds)
        self.y_true.append(target)
        self.t.append(t)

    @staticmethod
    def save_img(y_pred_denorm, y_true_denorm, t_arr, save_dir):
        vmin = 0
        vmax = 300

        t_offset = pd.to_datetime('2016-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S').tz_localize('UTC')

        batch = y_true_denorm.shape[0]
        h = y_true_denorm.shape[1]

        print('Save image...')
        for b in range(batch):

            # TODO
            h = 1

            fig, axs = plt.subplots(nrows=2, ncols=h, figsize=(3 * h, 6 * h))

            for t in range(h):
                if h == 1:
                    axs[0].imshow(y_pred_denorm[b, t], vmin=vmin, vmax=vmax)
                else:
                    axs[0, t].imshow(y_pred_denorm[b, t], vmin=vmin, vmax=vmax)
            for t in range(h):
                if h == 1:
                    axs[1].imshow(y_true_denorm[b, t], vmin=vmin, vmax=vmax)
                else:
                    axs[1, t].imshow(y_true_denorm[b, t], vmin=vmin, vmax=vmax)

            t_hour = t_offset + pd.Timedelta(hours=t_arr[b, t])

            img_fp = os.path.join(save_dir, 'img', '%s.png' % b)
            os.makedirs(os.path.dirname(img_fp), exist_ok=True)
            plt.title(t_hour)
            plt.savefig(img_fp)
            plt.close()

    def compute(self):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        t = torch.cat(self.t, dim=0)

        y_pred_denorm = y_pred.cpu().data.numpy() * self.air_std[0] + self.air_mean[0]
        y_true_denorm = y_true.cpu().data.numpy() * self.air_std[0] + self.air_mean[0]
        y_pred_denorm = y_pred_denorm[..., 0]
        y_true_denorm = y_true_denorm[..., 0]
        t = t.cpu().data.numpy()

        metric_dict = {}

        # TODO: add PM10...

        rmse = mean_squared_error(y_pred_denorm.flatten(), y_true_denorm.flatten(), squared=False)
        rmse = np.round(rmse, 2)

        metric_dict.update({'rmse': rmse})

        np.save(os.path.join(self.dir_t_now, 'y_pred.npy'), y_pred_denorm)
        np.save(os.path.join(self.dir_t_now, 'y_true.npy'), y_true_denorm)
        np.save(os.path.join(self.dir_t_now, 't.npy'), t)

        # LightningMetric.save_img(y_pred_denorm, y_true_denorm, t, self.dir_t_now)

        return metric_dict
