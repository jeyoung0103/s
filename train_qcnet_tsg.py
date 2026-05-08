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

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from predictors import QCNetWithTSGFormer

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--auto_prepare_data', action='store_true')
    parser.add_argument('--limit_large_samples', action='store_true')
    parser.add_argument('--large_sample_threshold_kb', type=int, default=300)
    parser.add_argument('--max_large_per_batch', type=int, default=1)
    parser.add_argument('--sampler_seed', type=int, default=2023)
    parser.add_argument('--sampler_drop_last', action='store_true')
    parser.add_argument('--max_file_size_kb', type=int, default=500)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=60)
    QCNetWithTSGFormer.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.limit_large_samples and args.max_file_size_kb is None:
        args.max_file_size_kb = 500

    model = QCNetWithTSGFormer(**vars(args))
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[args.dataset](**vars(args))
    logger = None
    if args.use_wandb:
        logger = WandbLogger(project=args.wandb_project,
                             entity=args.wandb_entity,
                             name=args.wandb_run_name,
                             save_dir=args.wandb_save_dir,
                             offline=args.wandb_offline,
                             log_model=False)
        logger.log_hyperparams(vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         strategy=strategy,
                         callbacks=[model_checkpoint, lr_monitor], logger=logger, max_epochs=args.max_epochs)
    trainer.fit(model, datamodule)
