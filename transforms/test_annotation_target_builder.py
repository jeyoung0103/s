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

import os
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from utils import wrap_angle


class TestAnnotationTargetBuilder(BaseTransform):
    """
    Test annotation 파일에서 ground truth trajectory를 로드하여 target 생성.
    
    기존 TargetBuilder와 달리 annotation 파일에서 future trajectory를 로드하고,
    heading을 계산하여 target을 생성합니다.
    """

    def __init__(self,
                 annotation_path: str,
                 num_historical_steps: int,
                 num_future_steps: int,
                 focal_only: bool = True) -> None:
        """
        Args:
            annotation_path: Test annotation parquet 파일 경로
            num_historical_steps: Historical time steps 수
            num_future_steps: Future time steps 수
            focal_only: True인 경우 focal tracks만 처리 (기본값: True)
        """
        self.annotation_path = annotation_path
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.focal_only = focal_only
        
        # Load annotation file
        print(f"Loading test annotation from {annotation_path}...")
        self.annotations_df = pd.read_parquet(annotation_path)
        
        # Filter focal tracks if needed
        if self.focal_only:
            self.annotations_df = self.annotations_df[self.annotations_df['is_focal_track'] == True]
            print(f"Filtered to {len(self.annotations_df)} focal tracks")
        
        # Create lookup dictionary: (scenario_id, track_id) -> annotation row
        self.gt_dict = {}
        for _, row in self.annotations_df.iterrows():
            key = (str(row['scenario_id']), str(row['track_id']))
            self.gt_dict[key] = {
                'gt_x': row['gt_trajectory_x'],
                'gt_y': row['gt_trajectory_y'],
                'is_focal': row['is_focal_track']
            }
        
        print(f"Loaded {len(self.gt_dict)} ground truth trajectories")

    def __call__(self, data: HeteroData) -> HeteroData:
        """
        Annotation에서 GT를 로드하여 target 생성.
        
        Args:
            data: HeteroData 객체 (scenario_id와 agent 정보 포함)
        
        Returns:
            target 필드가 추가된 HeteroData
        """
        scenario_id = data['scenario_id']
        agent_ids = data['agent']['id']  # List of track IDs
        num_agents = data['agent']['num_nodes']
        
        # Get origin and heading from last historical step
        origin = data['agent']['position'][:, self.num_historical_steps - 1]
        theta = data['agent']['heading'][:, self.num_historical_steps - 1]
        
        # Initialize target tensor
        target = origin.new_zeros(num_agents, self.num_future_steps, 4)
        
        # Rotation matrix for each agent
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(num_agents, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        
        # Process each agent
        for agent_idx in range(num_agents):
            track_id = str(agent_ids[agent_idx])
            key = (str(scenario_id), track_id)
            
            if key in self.gt_dict:
                # Get GT trajectory from annotation
                gt_x = self.gt_dict[key]['gt_x']  # numpy array [60]
                gt_y = self.gt_dict[key]['gt_y']  # numpy array [60]
                
                # Convert to torch tensor
                gt_positions = torch.from_numpy(
                    np.stack([gt_x, gt_y], axis=-1)
                ).float().to(origin.device)  # [60, 2]
                
                # Compute heading from trajectory
                # Use motion vectors between consecutive positions
                motion_vectors = gt_positions[1:] - gt_positions[:-1]  # [59, 2]
                headings = torch.atan2(motion_vectors[:, 1], motion_vectors[:, 0])  # [59]
                # Extend last heading for the final step
                headings = torch.cat([headings, headings[-1:]], dim=0)  # [60]
                
                # Relative positions (before rotation)
                relative_pos = gt_positions - origin[agent_idx, :2].unsqueeze(0)  # [60, 2]
                
                # Apply rotation to get target coordinates
                target[agent_idx, :, :2] = torch.mm(relative_pos, rot_mat[agent_idx])  # [60, 2]
                
                # Relative heading
                target[agent_idx, :, 3] = wrap_angle(headings - theta[agent_idx])
                
                # Z coordinate if 3D
                if data['agent']['position'].size(2) == 3:
                    # For 3D, we need z coordinate from annotation
                    # Since annotation only has x, y, we use 0 or interpolate from historical
                    # For now, use 0 (assuming 2D evaluation)
                    target[agent_idx, :, 2] = torch.zeros(self.num_future_steps, device=origin.device)
            else:
                # No annotation available for this agent
                # Set target to zeros (will be masked out in evaluation)
                target[agent_idx] = 0.0
        
        data['agent']['target'] = target
        return data

