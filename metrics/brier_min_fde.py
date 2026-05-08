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


class BrierMinFDE(Metric):
    """
    Brier Minimum Final Displacement Error (bminFDE)
    
    This is similar to minFDE. The only difference is we add (1.0 - p)^2 
    to the endpoint L2 distance, where p corresponds to the probability 
    of the best forecasted trajectory.
    """

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(BrierMinFDE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, prob_topk = topk(self.max_guesses, pred, prob)
        inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
        
        # Calculate endpoint errors for all trajectories
        endpoint_errors = torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                                     target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                                     p=2, dim=-1)
        
        # Find the best trajectory (minimum endpoint error)
        min_errors, inds_best = endpoint_errors.min(dim=-1)
        
        # Get probability of best trajectory
        prob_best = prob_topk[torch.arange(pred.size(0)), inds_best]
        
        # bminFDE = endpoint_error + (1.0 - p)^2
        brier_penalty = (1.0 - prob_best).pow(2)
        bminFDE_values = min_errors + brier_penalty
        
        self.sum += bminFDE_values.sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count

