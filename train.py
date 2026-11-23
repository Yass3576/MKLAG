#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger

from configs.utils import setup_cfg, cfg2flatdict, data_kwargs
from DataLoading.sequencedatamodule.SequenceLoader import SequenceDataModule
from models.MKLAG import MKLag


def main(cfg, exp_name):
    pl.seed_everything(cfg.seed)

    # Data & model
    dm = SequenceDataModule(**data_kwargs(cfg))
    model = MKLag(cfg)

    # Logging
    base_dir = os.environ.get("PWD", "./")
    save_dir = os.path.join(base_dir, "results/")

    logger = WandbLogger(
        project=cfg.project_name,
        name=exp_name,
        save_dir=save_dir,
        log_model=False,
        config=cfg2flatdict(cfg),
    )

    model.hparams.update({
        "project_name": cfg.project_name,
        "experiment_id": f"{exp_name}_{logger.version}",
        "experiment_name": exp_name,
    })

    # Checkpoints
    ckpt_dir = os.path.join(save_dir, exp_name, str(logger.version), "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val/accs_cat",
            filename="epoch_{epoch}_val_acc_{val/accs_cat:.3f}",
            save_last=True,
            auto_insert_metric_name=False,
            mode="max",
            save_top_k=3,
        )
    ]
    if cfg.opti.swa:
        callbacks.append(StochasticWeightAveraging(swa_lrs=cfg.opti.swa_lrs,
                                                   swa_epoch_start=cfg.opti.swa_epoch_start))

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    print("Callbacks: ", trainer.callbacks)

    resume = cfg.trainer.resume_from_checkpoint or None
    trainer.fit(model, dm, ckpt_path=resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_file", nargs="*", help="optional config file", default=[])
    parser.add_argument("--set", dest="set_cfgs", help="set config keys", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = setup_cfg(args.cfg_file, args.set_cfgs)

    exp_name = (f"MKLag_{Path(cfg.data.data_name).stem}_seed_{cfg.seed}"
                f"_maxframe{cfg.data.max_frame}_{cfg.n_tclasses}_classes_sampling_ordinal_l2_rn50_supcon_tm")

    print('============')
    print(cfg)
    print('============')

    main(cfg, exp_name)
