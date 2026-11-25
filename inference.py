#!/usr/bin/env python3
import argparse
import pandas as pd
import torch
import pytorch_lightning as pl

from configs.utils import setup_cfg, data_kwargs
from DataLoading.sequencedatamodule.SequenceLoader import SequenceDataModule
from models.MKLAG import MKLag
from utils.utils import load_weights, head_classes



def main(cfg, out):
    test_loader = SequenceDataModule(**data_kwargs(cfg)).test_dataloader()

    if cfg.ckpt:
        cfg.head_tclasses = head_classes(cfg.ckpt, ordinal=(cfg.Loss.mk_type == "ordinal"))

    model = MKLag(cfg)
    if cfg.ckpt:
        load_weights(model, cfg.ckpt)

    trainer = pl.Trainer(accelerator='auto', devices=1, logger=False)
    results = trainer.predict(model=model, dataloaders=test_loader)

    pd.DataFrame(results).set_index('video_name').to_pickle(out)
    print(f"Predictions saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_file", nargs="*", default=[], help="yaml config file(s)")
    parser.add_argument("--out", default="predictions.pkl", help="output pickle")
    parser.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER, help="key value ...")
    args = parser.parse_args()

    cfg = setup_cfg(args.cfg_file, args.set_cfgs)
    main(cfg, args.out)
