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
import math
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from urllib import request
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import extract_tar
from tqdm import tqdm

try:
    from av2.geometry.interpolate import compute_midpoint_line
    from av2.map.map_api import ArgoverseStaticMap
    from av2.map.map_primitives import Polyline
    from av2.utils.io import read_json_file
except ImportError:
    print("Error: av2 package not found. Please install argoverse2 package.")
    exit(1)

from utils import safe_list_index
from utils import side_to_directed_lineseg


class ArgoverseV2Preprocessor:
    """Preprocessor for Argoverse 2 Motion Forecasting Dataset."""

    def __init__(self,
                 data_root: str,
                 split: str,
                 dim: int = 3,
                 num_historical_steps: int = 50,
                 num_future_steps: int = 60,
                 predict_unseen_agents: bool = False,
                 vector_repr: bool = True) -> None:
        """
        Args:
            data_root: Root directory of the dataset (e.g., agroverse2)
            split: Dataset split ('train', 'val', or 'test')
            dim: 2D or 3D data
            num_historical_steps: Number of historical time steps
            num_future_steps: Number of future time steps
            predict_unseen_agents: If False, filter out agents that are unseen during historical time steps
            vector_repr: If True, a time step t is valid only when both t and t-1 are valid
        """
        self.data_root = os.path.expanduser(os.path.normpath(data_root))
        self.split = split
        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.predict_unseen_agents = predict_unseen_agents
        self.vector_repr = vector_repr
        
        # Set up directories
        self.raw_dir = os.path.join(self.data_root, split, split)
        self.processed_dir = os.path.join(self.data_root, split, 'processed')
        
        # Dataset URLs and sample counts
        self._url = f'https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/{split}.tar'
        self._num_samples = {
            'train': 199908,
            'val': 24988,
            'test': 24984,
        }[split]

        # Dataset constants
        self._agent_types = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background',
                             'construction', 'riderless_bicycle', 'unknown']
        self._agent_categories = ['TRACK_FRAGMENT', 'UNSCORED_TRACK', 'SCORED_TRACK', 'FOCAL_TRACK']
        self._polygon_types = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
        self._polygon_is_intersections = [True, False, None]
        self._point_types = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
                             'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
                             'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'SOLID_BLUE',
                             'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']
        self._point_sides = ['LEFT', 'RIGHT', 'CENTER']
        self._polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']

        # Get list of scenarios to process
        self.raw_file_names = []
        self._check_and_download_data()

    def process(self) -> None:
        """Process all scenarios and save to processed directory."""
        # Skip if no scenarios to process
        if len(self.raw_file_names) == 0:
            print(f"No scenarios to process for {self.split} split (already processed)")
            return
        
        # Create processed directory if it doesn't exist
        if not os.path.isdir(self.processed_dir):
            os.makedirs(self.processed_dir)
            print(f"Created processed directory: {self.processed_dir}")

        print(f"Processing {self.split} split...")
        for raw_file_name in tqdm(self.raw_file_names):
            try:
                df = pd.read_parquet(os.path.join(self.raw_dir, raw_file_name, f'scenario_{raw_file_name}.parquet'))
                map_dir = Path(self.raw_dir) / raw_file_name
                map_files = list(map_dir.glob('log_map_archive_*.json'))
                if not map_files:
                    print(f"Warning: No map file found for {raw_file_name}")
                    continue
                map_path = map_files[0]
                map_data = read_json_file(map_path)
                centerlines = {lane_segment['id']: Polyline.from_json_data(lane_segment['centerline'])
                               for lane_segment in map_data['lane_segments'].values()}
                map_api = ArgoverseStaticMap.from_json(map_path)
                
                data = dict()
                data['scenario_id'] = self.get_scenario_id(df)
                data['city'] = self.get_city(df)
                data['agent'] = self.get_agent_features(df)
                data.update(self.get_map_features(map_api, centerlines))
                
                with open(os.path.join(self.processed_dir, f'{raw_file_name}.pkl'), 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"\nError processing {raw_file_name}: {e}")
                continue

        print(f"Processing complete! Processed files saved to {self.processed_dir}")

    def _check_and_download_data(self) -> None:
        """Check if data exists and download if necessary."""
        # Check if processed data already exists
        if os.path.isdir(self.processed_dir):
            processed_files = [name for name in os.listdir(self.processed_dir) if
                              os.path.isfile(os.path.join(self.processed_dir, name)) and
                              name.endswith(('pkl', 'pickle'))]
            if len(processed_files) == self._num_samples:
                print(f"Processed data already exists for {self.split} split ({len(processed_files)} files)")
                self.raw_file_names = []
                return
        
        # Check if raw data exists
        if os.path.isdir(self.raw_dir):
            self.raw_file_names = [name for name in os.listdir(self.raw_dir) if
                                  os.path.isdir(os.path.join(self.raw_dir, name))]
            if len(self.raw_file_names) == self._num_samples:
                print(f"Raw data already exists for {self.split} split ({len(self.raw_file_names)} scenarios)")
                return
        
        # Download and extract data
        print(f"Downloading and extracting {self.split} split...")
        print(f"URL: {self._url}")
        print(f"Expected samples: {self._num_samples}")
        self._download_and_extract()
        
        # Get list of scenarios after download
        if os.path.isdir(self.raw_dir):
            self.raw_file_names = [name for name in os.listdir(self.raw_dir) if
                                  os.path.isdir(os.path.join(self.raw_dir, name))]
            print(f"Found {len(self.raw_file_names)} scenarios to process in {self.raw_dir}")
            print(f"Expected: {self._num_samples} scenarios")
            if len(self.raw_file_names) != self._num_samples:
                print(f"Warning: Expected {self._num_samples} scenarios but found {len(self.raw_file_names)}")
        else:
            raise ValueError(f"Failed to create raw directory: {self.raw_dir}")

    def _download_and_extract(self) -> None:
        """Download and extract the dataset."""
        # Create data root directory if it doesn't exist
        if not os.path.isdir(self.data_root):
            os.makedirs(self.data_root)
        
        # Download tar file if it doesn't exist
        tar_path = os.path.join(self.data_root, f'{self.split}.tar')
        if not os.path.isfile(tar_path):
            print(f'Downloading {self._url}')
            self._download_with_progress(self._url, tar_path)
        
        # Remove existing split directory if it exists
        split_dir = os.path.join(self.data_root, self.split)
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)
        
        # Remove existing raw directory if it exists
        if os.path.isdir(self.raw_dir):
            shutil.rmtree(self.raw_dir)
        
        # Create raw directory
        os.makedirs(self.raw_dir)
        
        # Extract tar file
        print(f'Extracting {tar_path} to {self.raw_dir}')
        print('This may take several minutes for large files...')
        extract_tar(path=tar_path, folder=self.raw_dir, mode='r')
        
        # Move files from {raw_dir}/{split}/* to {raw_dir}/*
        extracted_split_dir = os.path.join(self.raw_dir, self.split)
        if os.path.isdir(extracted_split_dir):
            for item in os.listdir(extracted_split_dir):
                src = os.path.join(extracted_split_dir, item)
                dst = os.path.join(self.raw_dir, item)
                shutil.move(src, dst)
            os.rmdir(extracted_split_dir)
        
        print(f'Extraction complete for {self.split} split')

    def _download_with_progress(self, url: str, filepath: str) -> None:
        """Download file with progress bar using tqdm."""
        try:
            # Get file size first
            with urlopen(url) as response:
                total_size = int(response.headers.get('Content-Length', 0))
            
            if total_size == 0:
                print("Warning: Could not determine file size, downloading without progress bar...")
                request.urlretrieve(url, filepath)
                return
            
            # Download with progress bar
            with urlopen(url) as response:
                with open(filepath, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, 
                            desc=f"Downloading {self.split}.tar") as pbar:
                        while True:
                            chunk = response.read(8192)  # 8KB chunks
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"Download complete: {filepath}")
            
        except Exception as e:
            print(f'\nDownload failed: {e}')
            # Clean up partial file
            if os.path.exists(filepath):
                os.remove(filepath)
            raise

    @staticmethod
    def get_scenario_id(df: pd.DataFrame) -> str:
        return df['scenario_id'].values[0]

    @staticmethod
    def get_city(df: pd.DataFrame) -> str:
        return df['city'].values[0]

    def get_agent_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.predict_unseen_agents:
            historical_df = df[df['timestep'] < self.num_historical_steps]
            agent_ids = list(historical_df['track_id'].unique())
            df = df[df['track_id'].isin(agent_ids)]
        else:
            agent_ids = list(df['track_id'].unique())

        num_agents = len(agent_ids)
        av_idx = agent_ids.index('AV')

        # initialization
        valid_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        predict_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        agent_id: List[Optional[str]] = [None] * num_agents
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        position = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)
        heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        velocity = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)

        for track_id, track_df in df.groupby('track_id'):
            agent_idx = agent_ids.index(track_id)
            agent_steps = track_df['timestep'].values

            valid_mask[agent_idx, agent_steps] = True
            current_valid_mask[agent_idx] = valid_mask[agent_idx, self.num_historical_steps - 1]
            predict_mask[agent_idx, agent_steps] = True
            if self.vector_repr:
                valid_mask[agent_idx, 1: self.num_historical_steps] = (
                        valid_mask[agent_idx, :self.num_historical_steps - 1] &
                        valid_mask[agent_idx, 1: self.num_historical_steps])
                valid_mask[agent_idx, 0] = False
            predict_mask[agent_idx, :self.num_historical_steps] = False
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, self.num_historical_steps:] = False

            agent_id[agent_idx] = track_id
            agent_type[agent_idx] = self._agent_types.index(track_df['object_type'].values[0])
            agent_category[agent_idx] = track_df['object_category'].values[0]
            position[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['position_x'].values,
                                                                              track_df['position_y'].values],
                                                                             axis=-1)).float()
            heading[agent_idx, agent_steps] = torch.from_numpy(track_df['heading'].values).float()
            velocity[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['velocity_x'].values,
                                                                              track_df['velocity_y'].values],
                                                                             axis=-1)).float()

        if self.split == 'test':
            predict_mask[current_valid_mask
                         | (agent_category == 2)
                         | (agent_category == 3), self.num_historical_steps:] = True

        return {
            'num_nodes': num_agents,
            'av_index': av_idx,
            'valid_mask': valid_mask,
            'predict_mask': predict_mask,
            'id': agent_id,
            'type': agent_type,
            'category': agent_category,
            'position': position,
            'heading': heading,
            'velocity': velocity,
        }

    def get_map_features(self,
                         map_api: ArgoverseStaticMap,
                         centerlines: Mapping[str, Polyline]) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        lane_segment_ids = map_api.get_scenario_lane_segment_ids()
        cross_walk_ids = list(map_api.vector_pedestrian_crossings.keys())
        polygon_ids = lane_segment_ids + cross_walk_ids
        num_polygons = len(lane_segment_ids) + len(cross_walk_ids) * 2

        # initialization
        polygon_position = torch.zeros(num_polygons, self.dim, dtype=torch.float)
        polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
        polygon_height = torch.zeros(num_polygons, dtype=torch.float)
        polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
        point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            centerline = torch.from_numpy(centerlines[lane_segment.id].xyz).float()
            polygon_position[lane_segment_idx] = centerline[0, :self.dim]
            polygon_orientation[lane_segment_idx] = torch.atan2(centerline[1, 1] - centerline[0, 1],
                                                                centerline[1, 0] - centerline[0, 0])
            polygon_height[lane_segment_idx] = centerline[1, 2] - centerline[0, 2]
            polygon_type[lane_segment_idx] = self._polygon_types.index(lane_segment.lane_type.value)
            polygon_is_intersection[lane_segment_idx] = self._polygon_is_intersections.index(
                lane_segment.is_intersection)

            left_boundary = torch.from_numpy(lane_segment.left_lane_boundary.xyz).float()
            right_boundary = torch.from_numpy(lane_segment.right_lane_boundary.xyz).float()
            point_position[lane_segment_idx] = torch.cat([left_boundary[:-1, :self.dim],
                                                          right_boundary[:-1, :self.dim],
                                                          centerline[:-1, :self.dim]], dim=0)
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[lane_segment_idx] = torch.cat([torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                                                             torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                                                             torch.atan2(center_vectors[:, 1], center_vectors[:, 0])],
                                                            dim=0)
            point_magnitude[lane_segment_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                      right_vectors[:, :2],
                                                                      center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_height[lane_segment_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                                                       dim=0)
            left_type = self._point_types.index(lane_segment.left_mark_type.value)
            right_type = self._point_types.index(lane_segment.right_mark_type.value)
            center_type = self._point_types.index('CENTERLINE')
            point_type[lane_segment_idx] = torch.cat(
                [torch.full((len(left_vectors),), left_type, dtype=torch.uint8),
                 torch.full((len(right_vectors),), right_type, dtype=torch.uint8),
                 torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_side[lane_segment_idx] = torch.cat(
                [torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                 torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        for crosswalk in map_api.get_scenario_ped_crossings():
            crosswalk_idx = polygon_ids.index(crosswalk.id)
            edge1 = torch.from_numpy(crosswalk.edge1.xyz).float()
            edge2 = torch.from_numpy(crosswalk.edge2.xyz).float()
            start_position = (edge1[0] + edge2[0]) / 2
            end_position = (edge1[-1] + edge2[-1]) / 2
            polygon_position[crosswalk_idx] = start_position[:self.dim]
            polygon_position[crosswalk_idx + len(cross_walk_ids)] = end_position[:self.dim]
            polygon_orientation[crosswalk_idx] = torch.atan2((end_position - start_position)[1],
                                                             (end_position - start_position)[0])
            polygon_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.atan2((start_position - end_position)[1],
                                                                                   (start_position - end_position)[0])
            polygon_height[crosswalk_idx] = end_position[2] - start_position[2]
            polygon_height[crosswalk_idx + len(cross_walk_ids)] = start_position[2] - end_position[2]
            polygon_type[crosswalk_idx] = self._polygon_types.index('PEDESTRIAN')
            polygon_type[crosswalk_idx + len(cross_walk_ids)] = self._polygon_types.index('PEDESTRIAN')
            polygon_is_intersection[crosswalk_idx] = self._polygon_is_intersections.index(None)
            polygon_is_intersection[crosswalk_idx + len(cross_walk_ids)] = self._polygon_is_intersections.index(None)

            if side_to_directed_lineseg((edge1[0] + edge1[-1]) / 2, start_position, end_position) == 'LEFT':
                left_boundary = edge1
                right_boundary = edge2
            else:
                left_boundary = edge2
                right_boundary = edge1
            num_centerline_points = math.ceil(torch.norm(end_position - start_position, p=2, dim=-1).item() / 2.0) + 1
            centerline = torch.from_numpy(
                compute_midpoint_line(left_ln_boundary=left_boundary.numpy(),
                                      right_ln_boundary=right_boundary.numpy(),
                                      num_interp_pts=int(num_centerline_points))[0]).float()

            point_position[crosswalk_idx] = torch.cat([left_boundary[:-1, :self.dim],
                                                       right_boundary[:-1, :self.dim],
                                                       centerline[:-1, :self.dim]], dim=0)
            point_position[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [right_boundary.flip(dims=[0])[:-1, :self.dim],
                 left_boundary.flip(dims=[0])[:-1, :self.dim],
                 centerline.flip(dims=[0])[:-1, :self.dim]], dim=0)
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[crosswalk_idx] = torch.cat(
                [torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                 torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                 torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
            point_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.atan2(-right_vectors.flip(dims=[0])[:, 1], -right_vectors.flip(dims=[0])[:, 0]),
                 torch.atan2(-left_vectors.flip(dims=[0])[:, 1], -left_vectors.flip(dims=[0])[:, 0]),
                 torch.atan2(-center_vectors.flip(dims=[0])[:, 1], -center_vectors.flip(dims=[0])[:, 0])], dim=0)
            point_magnitude[crosswalk_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                   right_vectors[:, :2],
                                                                   center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_magnitude[crosswalk_idx + len(cross_walk_ids)] = torch.norm(
                torch.cat([-right_vectors.flip(dims=[0])[:, :2],
                           -left_vectors.flip(dims=[0])[:, :2],
                           -center_vectors.flip(dims=[0])[:, :2]], dim=0), p=2, dim=-1)
            point_height[crosswalk_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                                                    dim=0)
            point_height[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [-right_vectors.flip(dims=[0])[:, 2],
                 -left_vectors.flip(dims=[0])[:, 2],
                 -center_vectors.flip(dims=[0])[:, 2]], dim=0)
            crosswalk_type = self._point_types.index('CROSSWALK')
            center_type = self._point_types.index('CENTERLINE')
            point_type[crosswalk_idx] = torch.cat([
                torch.full((len(left_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(right_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_type[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.full((len(right_vectors),), crosswalk_type, dtype=torch.uint8),
                 torch.full((len(left_vectors),), crosswalk_type, dtype=torch.uint8),
                 torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_side[crosswalk_idx] = torch.cat(
                [torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                 torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
            point_side[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.full((len(right_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                 torch.full((len(left_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
        point_to_polygon_edge_index = torch.stack(
            [torch.arange(num_points.sum(), dtype=torch.long),
             torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
        polygon_to_polygon_edge_index = []
        polygon_to_polygon_type = []
        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            pred_inds = []
            for pred in lane_segment.predecessors:
                pred_idx = safe_list_index(polygon_ids, pred)
                if pred_idx is not None:
                    pred_inds.append(pred_idx)
            if len(pred_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(pred_inds, dtype=torch.long),
                                 torch.full((len(pred_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(pred_inds),), self._polygon_to_polygon_types.index('PRED'), dtype=torch.uint8))
            succ_inds = []
            for succ in lane_segment.successors:
                succ_idx = safe_list_index(polygon_ids, succ)
                if succ_idx is not None:
                    succ_inds.append(succ_idx)
            if len(succ_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(succ_inds, dtype=torch.long),
                                 torch.full((len(succ_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(succ_inds),), self._polygon_to_polygon_types.index('SUCC'), dtype=torch.uint8))
            if lane_segment.left_neighbor_id is not None:
                left_idx = safe_list_index(polygon_ids, lane_segment.left_neighbor_id)
                if left_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor([[left_idx], [lane_segment_idx]], dtype=torch.long))
                    polygon_to_polygon_type.append(
                        torch.tensor([self._polygon_to_polygon_types.index('LEFT')], dtype=torch.uint8))
            if lane_segment.right_neighbor_id is not None:
                right_idx = safe_list_index(polygon_ids, lane_segment.right_neighbor_id)
                if right_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor([[right_idx], [lane_segment_idx]], dtype=torch.long))
                    polygon_to_polygon_type.append(
                        torch.tensor([self._polygon_to_polygon_types.index('RIGHT')], dtype=torch.uint8))
        if len(polygon_to_polygon_edge_index) != 0:
            polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
            polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
        else:
            polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
            polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

        map_data = {
            'map_polygon': {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
        }
        map_data['map_polygon']['num_nodes'] = num_polygons
        map_data['map_polygon']['position'] = polygon_position
        map_data['map_polygon']['orientation'] = polygon_orientation
        if self.dim == 3:
            map_data['map_polygon']['height'] = polygon_height
        map_data['map_polygon']['type'] = polygon_type
        map_data['map_polygon']['is_intersection'] = polygon_is_intersection
        if len(num_points) == 0:
            map_data['map_point']['num_nodes'] = 0
            map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['type'] = torch.tensor([], dtype=torch.uint8)
            map_data['map_point']['side'] = torch.tensor([], dtype=torch.uint8)
        else:
            map_data['map_point']['num_nodes'] = num_points.sum().item()
            map_data['map_point']['position'] = torch.cat(point_position, dim=0)
            map_data['map_point']['orientation'] = torch.cat(point_orientation, dim=0)
            map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.cat(point_height, dim=0)
            map_data['map_point']['type'] = torch.cat(point_type, dim=0)
            map_data['map_point']['side'] = torch.cat(point_side, dim=0)
        map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

        return map_data


if __name__ == '__main__':
    parser = ArgumentParser(description='Preprocess Argoverse 2 Motion Forecasting Dataset')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset (e.g., agroverse2)')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='Splits to preprocess (default: train val test)')
    parser.add_argument('--dim', type=int, default=3,
                        help='2D or 3D data (default: 3)')
    parser.add_argument('--num_historical_steps', type=int, default=50,
                        help='Number of historical time steps (default: 50)')
    parser.add_argument('--num_future_steps', type=int, default=60,
                        help='Number of future time steps (default: 60)')
    parser.add_argument('--predict_unseen_agents', action='store_true',
                        help='If set, do not filter out agents that are unseen during historical time steps')
    parser.add_argument('--no_vector_repr', action='store_true',
                        help='If set, do not require both t and t-1 to be valid for time step t')
    
    args = parser.parse_args()

    for split in args.splits:
        print(f"\n{'='*80}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*80}")
        
        preprocessor = ArgoverseV2Preprocessor(
            data_root=args.data_root,
            split=split,
            dim=args.dim,
            num_historical_steps=args.num_historical_steps,
            num_future_steps=args.num_future_steps,
            predict_unseen_agents=args.predict_unseen_agents,
            vector_repr=not args.no_vector_repr
        )
        
        preprocessor.process()
    
    print(f"\n{'='*80}")
    print("All preprocessing complete!")
    print(f"{'='*80}")

