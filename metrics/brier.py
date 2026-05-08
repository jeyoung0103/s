# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by TSG author, 2026.

from typing import Optional

import torch
from torchmetrics import Metric

from metrics.utils import topk
from metrics.utils import valid_filter


class Brier(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(Brier, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               min_criterion: str = 'FDE') -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, prob_topk = topk(self.max_guesses, pred, prob)
        if min_criterion == 'FDE':
            # Find last valid timestep for each sample
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
            # Calculate FDE for each mode
            fde_per_mode = torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                                      target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                                      p=2, dim=-1)  # [N, K]
            # Find best mode (minimum FDE)
            inds_best = fde_per_mode.argmin(dim=-1)  # [N]
            # Get FDE of best mode
            fde_best = fde_per_mode[torch.arange(pred.size(0)), inds_best]  # [N]
            # Get probability of best mode
            prob_best = prob_topk[torch.arange(pred.size(0)), inds_best]  # [N]
            # brier-minFDE = FDE + (1-p)^2 (according to Argoverse 2 leaderboard)
            # "we add (1.0 - p)^2 to the endpoint L2 distance"
            brier_minfde = fde_best + (1.0 - prob_best).pow(2)  # [N]
            self.sum += brier_minfde.sum()
        elif min_criterion == 'ADE':
            # Calculate ADE for each mode
            ade_per_mode = (torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1) *
                           valid_mask.unsqueeze(1)).sum(dim=-1) / valid_mask.sum(dim=-1, keepdim=True)  # [N, K]
            # Find best mode (minimum ADE)
            inds_best = ade_per_mode.argmin(dim=-1)  # [N]
            # Get ADE of best mode
            ade_best = ade_per_mode[torch.arange(pred.size(0)), inds_best]  # [N]
            # Get probability of best mode
            prob_best = prob_topk[torch.arange(pred.size(0)), inds_best]  # [N]
            # brier-minADE = ADE + (1-p)^2
            brier_minade = ade_best + (1.0 - prob_best).pow(2)  # [N]
            self.sum += brier_minade.sum()
        else:
            raise ValueError('{} is not a valid criterion'.format(min_criterion))
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
