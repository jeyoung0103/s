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
"""
Validation split evaluation for Argoverse 2 Motion Forecasting (leaderboard-style metrics).

Uses processed val .pkl files and scenario parquet ground truth (TargetBuilder), same metrics
as the public leaderboard:
- minFDE (K=6, K=1)
- minADE (K=6, K=1)
- MR (K=6, K=1)
- brier-minFDE (K=6)

Example (from this directory, PYTHONPATH includes the repo):

  python test_leaderboard1.py --ckpt_path /home/kgh/waymo/2026/sgnet2_1_0126_m/wandb_logs/QCNet/0223/checkpoints/epoch=68-step=861672.ckpt
  python test_leaderboard1.py --ckpt_path /home/kgh/waymo/sgnet2_1_0126/epoch=62-step=786744.ckpt
  python test_leaderboard1.py --ckpt_path /home/kgh/waymo/QCNet/wandb_logs/1208/checkpoints/epoch=61-step=774256.ckpt
"""
from argparse import ArgumentParser
import os

import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from predictors import QCNetWithTSGFormer
from metrics import minADE, minFDE, MR, Brier
try:
    from metrics import minADE1, minFDE1
except ImportError:
    # Fallback: use minADE/minFDE with max_guesses=1
    minADE1 = None
    minFDE1 = None
from transforms import TargetBuilder

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. gFLOPs calculation will be skipped.")


if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser(description='Evaluate model on val split with leaderboard metrics')
    default_root = '/home/kgh/waymo/QCNet/agroverse2'
    parser.add_argument('--root', type=str, default=default_root,
                        help='Dataset root (default: %(default)s)')
    parser.add_argument('--val_processed_dir', type=str, default=None,
                        help='Path to val processed .pkl directory (default: {root}/val/processed)')
    parser.add_argument('--val_raw_dir', type=str, default=None,
                        help='Path to val raw scenario folders. If omitted, uses {root}/val/val '
                             'when it exists, else {root}/val/raw, else dataset default.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--calculate_flops', action='store_true',
                        help='Calculate gFLOPs (requires thop)')
    args = parser.parse_args()

    if args.val_processed_dir is None:
        args.val_processed_dir = os.path.join(args.root, 'val', 'processed')
    if args.val_raw_dir is None:
        nested_val = os.path.join(args.root, 'val', 'val')
        standard_raw = os.path.join(args.root, 'val', 'raw')
        if os.path.isdir(nested_val):
            args.val_raw_dir = nested_val
        elif os.path.isdir(standard_raw):
            args.val_raw_dir = standard_raw
        else:
            args.val_raw_dir = None

    # Load model from checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    hyperparams = checkpoint.get('hyper_parameters', {})

    num_agent_types = hyperparams.get('num_agent_types', 6)
    num_edge_types = hyperparams.get('num_edge_types', 21)
    num_speed_bins = hyperparams.get('num_speed_bins', 50)
    num_spatial_bins = hyperparams.get('num_spatial_bins', 200)
    num_ttc_bins = hyperparams.get('num_ttc_bins', 100)

    print(f"\nLoaded checkpoint hyperparameters:")
    print(f"  num_agent_types: {num_agent_types}")
    print(f"  num_edge_types: {num_edge_types}")
    print(f"  num_speed_bins: {num_speed_bins}")
    print(f"  num_spatial_bins: {num_spatial_bins}")
    print(f"  num_ttc_bins: {num_ttc_bins}")

    model = QCNetWithTSGFormer.load_from_checkpoint(checkpoint_path=args.ckpt_path, strict=False)
    model.eval()

    # Get model hyperparameters
    num_historical_steps = model.num_historical_steps
    num_future_steps = model.num_future_steps
    
    target_builder = TargetBuilder(num_historical_steps, num_future_steps)

    val_dataset = ArgoverseV2Dataset(
        root=args.root,
        split='val',
        raw_dir=args.val_raw_dir,
        processed_dir=args.val_processed_dir,
        transform=target_builder
    )

    dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers
    )

    # Initialize all leaderboard metrics
    if minADE1 is not None:
        minADE1_metric = minADE1(max_guesses=1)
    else:
        minADE1_metric = minADE(max_guesses=1)
    minADE6_metric = minADE(max_guesses=6)
    if minFDE1 is not None:
        minFDE1_metric = minFDE1(max_guesses=1)
    else:
        minFDE1_metric = minFDE(max_guesses=1)
    minFDE6_metric = minFDE(max_guesses=6)
    MR1_metric = MR(max_guesses=1, miss_threshold=2.0)
    MR6_metric = MR(max_guesses=6, miss_threshold=2.0)
    brier_minFDE6_metric = Brier(max_guesses=6)

    # Move metrics to device
    device = next(model.parameters()).device
    minADE1_metric = minADE1_metric.to(device)
    minADE6_metric = minADE6_metric.to(device)
    minFDE1_metric = minFDE1_metric.to(device)
    minFDE6_metric = minFDE6_metric.to(device)
    MR1_metric = MR1_metric.to(device)
    MR6_metric = MR6_metric.to(device)
    brier_minFDE6_metric = brier_minFDE6_metric.to(device)

    # Evaluation loop
    print("\n" + "="*80)
    print(f"Val raw: {args.val_raw_dir or '(default under root)'}")
    print(f"Val processed: {args.val_processed_dir}")
    print("Starting evaluation on val split with leaderboard metrics...")
    print("="*80)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # Move data to device
            data = data.to(device)
            
            if isinstance(data, Batch):
                data['agent']['av_index'] += data['agent']['ptr'][:-1]
            
            # Get model predictions
            pred = model(data)
            
            # Prepare trajectory predictions
            if model.output_head:
                traj_refine = torch.cat([
                    pred['loc_refine_pos'][..., :model.output_dim],
                    pred['loc_refine_head'],
                    pred['scale_refine_pos'][..., :model.output_dim],
                    pred['conc_refine_head']
                ], dim=-1)
            else:
                traj_refine = torch.cat([
                    pred['loc_refine_pos'][..., :model.output_dim],
                    pred['scale_refine_pos'][..., :model.output_dim]
                ], dim=-1)
            pi = pred['pi']

            # Get ground truth and masks
            reg_mask = data['agent']['predict_mask'][:, model.num_historical_steps:]
            
            # For Argoverse 2, evaluate focal tracks (category == 3)
            # Also filter by annotation availability (target != 0)
            if model.dataset == 'argoverse_v2':
                eval_mask = data['agent']['category'] == 3
            else:
                raise ValueError(f'{model.dataset} is not a valid dataset')
            
            valid_mask_eval = reg_mask[eval_mask]
            traj_eval = traj_refine[eval_mask, :, :, :model.output_dim + model.output_head]
            
            # Add heading if not output_head
            if not model.output_head:
                traj_2d_with_start_pos_eval = torch.cat([
                    traj_eval.new_zeros((traj_eval.size(0), model.num_modes, 1, 2)),
                    traj_eval[..., :2]
                ], dim=-2)
                motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
                head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
                traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
            
            pi_eval = torch.nn.functional.softmax(pi[eval_mask], dim=-1)
            gt_eval = torch.cat([
                data['agent']['target'][eval_mask, ..., :model.output_dim],
                data['agent']['target'][eval_mask, ..., -1:]
            ], dim=-1)

            # Update all metrics
            minADE1_metric.update(
                pred=traj_eval[..., :model.output_dim],
                target=gt_eval[..., :model.output_dim],
                prob=pi_eval,
                valid_mask=valid_mask_eval
            )
            minADE6_metric.update(
                pred=traj_eval[..., :model.output_dim],
                target=gt_eval[..., :model.output_dim],
                prob=pi_eval,
                valid_mask=valid_mask_eval
            )
            minFDE1_metric.update(
                pred=traj_eval[..., :model.output_dim],
                target=gt_eval[..., :model.output_dim],
                prob=pi_eval,
                valid_mask=valid_mask_eval
            )
            minFDE6_metric.update(
                pred=traj_eval[..., :model.output_dim],
                target=gt_eval[..., :model.output_dim],
                prob=pi_eval,
                valid_mask=valid_mask_eval
            )
            MR1_metric.update(
                pred=traj_eval[..., :model.output_dim],
                target=gt_eval[..., :model.output_dim],
                prob=pi_eval,
                valid_mask=valid_mask_eval,
                miss_criterion='FDE'
            )
            MR6_metric.update(
                pred=traj_eval[..., :model.output_dim],
                target=gt_eval[..., :model.output_dim],
                prob=pi_eval,
                valid_mask=valid_mask_eval,
                miss_criterion='FDE'
            )
            brier_minFDE6_metric.update(
                pred=traj_eval[..., :model.output_dim],
                target=gt_eval[..., :model.output_dim],
                prob=pi_eval,
                valid_mask=valid_mask_eval,
                min_criterion='FDE'
            )

            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1} batches...")

    # Compute final metrics
    minADE1_value = minADE1_metric.compute().item()
    minADE6_value = minADE6_metric.compute().item()
    minFDE1_value = minFDE1_metric.compute().item()
    minFDE6_value = minFDE6_metric.compute().item()
    MR1_value = MR1_metric.compute().item()
    MR6_value = MR6_metric.compute().item()
    brier_minFDE6_value = brier_minFDE6_metric.compute().item()

    # Print results in leaderboard format
    print("\n" + "="*80)
    print("LEADERBOARD EVALUATION RESULTS")
    print("="*80)
    print(f"{'Metric':<30} {'Value':<15} {'Direction':<10}")
    print("-"*80)
    print(f"{'minFDE (K=6)':<30} {minFDE6_value:<15.6f} {'↑':<10}")
    print(f"{'minFDE (K=1)':<30} {minFDE1_value:<15.6f} {'↑':<10}")
    print(f"{'minADE (K=6)':<30} {minADE6_value:<15.6f} {'↑':<10}")
    print(f"{'minADE (K=1)':<30} {minADE1_value:<15.6f} {'↑':<10}")
    print(f"{'MR (K=6)':<30} {MR6_value:<15.6f} {'↑':<10}")
    print(f"{'MR (K=1)':<30} {MR1_value:<15.6f} {'↑':<10}")
    print(f"{'brier-minFDE (K=6)':<30} {brier_minFDE6_value:<15.6f} {'↓':<10}")
    print("="*80)

    # Calculate gFLOPs if requested
    if args.calculate_flops and THOP_AVAILABLE:
        print("\nCalculating gFLOPs...")
        model.eval()
        GFLOPs_list = []
        
        sample_batches = []
        for i, data in enumerate(dataloader):
            data = data.to(device)
            if isinstance(data, Batch):
                data['agent']['av_index'] += data['agent']['ptr'][:-1]
            sample_batches.append(data)
            if i >= 4:
                break
        
        for data in sample_batches:
            try:
                macs, params = profile(model, inputs=(data,), verbose=False)
                GFLOPs = 2 * macs / 1e9
                GFLOPs_list.append(GFLOPs)
            except Exception as e:
                print(f"Warning: FLOPs calculation failed for a batch: {e}")
                continue
        
        if GFLOPs_list:
            avg_GFLOPs = sum(GFLOPs_list) / len(GFLOPs_list)
            print(f"\nAverage gFLOPs: {avg_GFLOPs:.2f} G")
            print("="*80)
    elif args.calculate_flops and not THOP_AVAILABLE:
        print("\nWarning: thop is not available. Skipping gFLOPs calculation.")

