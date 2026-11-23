# MKLag: code guide

This document is for whoever takes over the code. It explains how the pieces fit together, where to change things, and which traps to avoid. To use the code without changing it, the [main README](README.md) is enough.

---

## 1. Big picture

```
video ──(frame encoder, run beforehand)──►  embeddings (T, in_dim)  ──►  MKLag  ──►  developmental stage of each frame (T,)
                                            <video>.npy                             LAG: 3 × [presence score, start, end]
                                                                                    category (n_tclasses)
```

- **Inputs.** The training code never reads videos. It reads one `<video>.npy` of shape `(T, in_dim)` per video from `data.embedding_path`. These embeddings come from different frame encoders. In our case, this is a ResNet-50 (`in_dim` = 2048, see `features/backbone.py`). The encoder is trained and runned separately, before the training of MKLag.

### The model (`models/blocksLag.py`)

The model is a stack of blocks given by `FACT.block`, for example `iuu` = one input block followed by two update blocks. The last block must be an update block, because its output are the predictions.

| Part | What it does | Config |
|---|---|---|
| Frame branch | MS-TCN++ (dilated temporal convolutions). The frame branch provides the frame-wise developmental stage prediction.| `Bi.f = m2`, `Bi.f_dim`, `Bi.f_layers` |
| Stage-token branch | `FACT.ntoken` learned queries, each standing for one stage segment. Input block: self- and cross-attention to the frames (`sca`). Update blocks: self-attention (`sa`), plus frame/token cross-attention. Tokens predict a stage class, or "null". | `Bi.a*`, `Bu.a*`, `FACT.ntoken` |
| LAG head | 3 learned queries (one for each sub-segment candidate) with cross-attention to the stage tokens. Each output `[a, u, v]`, decoded into an ordinal existence logit (P(n ≥ k) computed from a), a start = sigmoid(u) and a duration = softplus(v). | `models/classificationHead.py: LagLocalizationHead`, `models/loss.py: decode_lag` |
| Morphokinetic head | A transformer with a CLS token aggregates the merged frame and stage representation, then a sigmoid gate driven by the LAG prediction (`lag_embed`, `lag_gate`), then an MLP. | `Bi.s_head_*` |

**Final stage prediction** (`Block._inference`). For each frame, the class probabilities of the frame branch (weight `FACT.mwt`) are mixed with those of the token that attends most to that frame.

### Losses (`models/loss.py`, `Block.compute_loss`)

| Term | Weight | Notes |
|---|---|---|
| Frame cross-entropy | `Loss.frame` | per frame, 11 classes = developmental stages |
| Token classification | `Loss.action` | after Hungarian matching of tokens to ground-truth segments (`Loss.match`: `o2o` / `o2m` / `seq`; costs `Loss.pc` for class probability and `Loss.a2fc` for attention IoU) |
| Morphokinetic | `Loss.mk` | cross-entropy, 5 or 8 morphokinetic categories |
| LAG | `Loss.lag_ord_w`, `lag_reg_w`, `lag_regu_w` | ordinal BCE on the number of segments; L2 on start / end; order and non-overlap penalties |
| Temporal smoothing | `Loss.sw` | truncated MSE between consecutive log-probabilities (MS-TCN). Reduces over-segmentation, but can erase very short cytokinesis segments. |
| Cross-attention alignment | `Loss.ca` | update blocks only |

---

## 2. Code map

```
train.py                    Lightning training, wandb logging
inference.py                predictions on the test split, outputs a pandas pickle
processResults.py           computes metrics from the pickle
configs/default.py          every parameter and its default value (yacs)
configs/bovine.yaml         seting of the parameters for one experiment
configs/utils.py            setup_cfg: defaults <- yaml files <- --set key value
DataLoading/sequencedatamodule/
    SequenceLoader.py       Dataset (embeddings + labels), DataModule (class-balanced train sampler)
    utils.py                label construction (getlabel_cda, getlabel_ciray, get_LAG_phase),
                            video reading (load_frames, normalize) shared with the tool
models/
    MKLAG.py                LightningModule: batch -> model -> losses, logging, optimiser
    blocksLag.py            the model: mklag (stack of blocks), InputBlock, UpdateBlock
    basic.py                FACT building blocks (MS-TCN, attention layers, positional encoding)
    classificationHead.py   SceneAggregator, GlobalVideoHead_MLP, LagLocalizationHead
    loss.py                 MatchCriterionLAG (matching + all losses), decode_lag, smooth_loss
    optim.py                optimisers and LR schedulers
Metrics/metrics.py          accuracy, confusion matrices (single / mean ± std), LAG metrics
utils/utils.py              load_weights, to_numpy
features/backbone.py        ResNet-50 frame encoder (prediction tool)
MKLag_tool/                 end-user prediction tool (see its README)
```

**Where to change things:**

| To… | Edit |
|---|---|
| add or change a hyper-parameter | `configs/default.py` (yaml files can only set keys that exist there) |
| change the architecture | `models/blocksLag.py` (`InputBlock` / `UpdateBlock`); heads in `classificationHead.py` |
| change a loss | `models/loss.py` (`MatchCriterionLAG`); combination in `Block.compute_loss` |
| change the labels | `DataLoading/sequencedatamodule/utils.py` (`getlabel_cda`, `get_LAG_phase`); category mapping in `SequenceLoader.CATEGORY_MAPPINGS` |
| log a new metric | `models/MKLAG.py` (`evaluations`, `logging`) |
| add an evaluation | `Metrics/metrics.py` + a `--tasks` entry in `processResults.py` |

---

## 3. Data

### Labels built by the dataset

- **Stages** (`n_actions` = 11): 0 = CDT, 1–8 = number of cells, 9 = ≥ 9 cells, 10 = blocked. The CDA scheme is used with `data.version: cda`, `cdt_annotation: true`, `blocked: true`. From the first invalid cycle on, frames are labelled 10.
- **Category** (`n_tclasses`, `SequenceLoader.CATEGORY_MAPPINGS`):
  - `8`: DE = 0 … EHB = 7;
  - `5`: SSE–DE = 0 (non-transferable), ADD = 1, SSB = 2, HB = 3, EHB = 4;
  - `2`: SSE–DE = 0, EHB–ADD = 1.
- **LAG**: `[has_seg, start, end] × 3` + `n_segment`, normalised by 300 frames.

### Sampling

The training batches are drawn with a `WeightedRandomSampler` that is balanced on the MORPHOKINETIC category. Validation and test use one video per batch.

---

## 4. Configuration

A configuration is built in three layers, each one overriding the previous: defaults (`configs/default.py`) ← yaml (`--cfg`) ← command line (`--set key value …`).

```bash
python train.py --cfg configs/bovine.yaml --set seed 1 Loss.sw 0.0 data.batch_size 16
```

Keys that must match the data or a checkpoint:

| Key | Must match |
|---|---|
| `in_dim` | dimension of the embeddings in `data.embdedding_path` |
| `n_tclasses` | the morphokinetic head of the checkpoint you load (generally 8, 5 or 2) |
| `Bi.*`, `Bu.*`, `FACT.*` | the architecture of the checkpoint (otherwise `load_state_dict` fails) |
| `data.version`, `cdt_annotation`, `blocked` | the annotation scheme; only `v4 / true / true` (CDA) and `cs` (cell stages) are valid |

---

## 5. Running

```bash

# training
WANDB_MODE=offline python train.py --cfg configs/bovine.yaml --set data.embedding_path /path/to/embeddings
#   -> results/<exp_name>/<run_id>/checkpoints/{epoch_X_val_acc_Y.ckpt (top 3), last.ckpt}
#   resume: --set trainer.resume_from_checkpoint results/.../last.ckpt

# test set predictions
python inference.py --cfg configs/bovine.yaml --out preds/seed93.pkl --set ckpt results/.../last.ckpt

# metrics (the output folder must exist)
mkdir -p res
python processResults.py --path_to_pickle preds/seed93.pkl --out res/seed93 --tasks cat dev lag --n_tclasses 5
python processResults.py --mean_cm preds/s1.pkl preds/s76.pkl preds/s93.pkl --out res/mean --n_tclasses 5
```

**Prediction pickle.** It has one row per test video, indexed by `video_name`. The useful columns:

| Column | Content |
|---|---|
| `pred_cat`, `ycat`, `scene_logits` | predicted and true morphokinetic category, category logits |
| `pred_dev`, `ydev` | predicted and true stage of every frame |
| `pred_lag` | `[z_k, s_k, d_k] × 3`. sigmoid(z_k) = P(n ≥ k), start s_k, end s_k + d_k (fraction of 300 frames) |
| `has_segk`, `lag_sk`, `lag_ek` | true LAG phase |

**processResults outputs.** For `--out res/seed93`:

| Task | Files |
|---|---|
| `cat` | `seed93_cat.csv`, `seed93_cat.png` (confusion matrix) |
| `dev` | `seed93_dev.csv`, `seed93_dev.png` |
| `lag` | folder `seed93_lag/`: `global_temporal_metrics.csv`, `per_segment_temporal_metrics.csv`, `temporal_localization_{analysis,summary}.csv`, figures |
| `--mean_cm` | `mean_morphokinetic.png`, `mean_morphokinetic_detailed.txt` |

---

## 7. Prediction tool and frame encoder

`MKLag_tool/predict.py` chains two steps:
1. read the video, resize it to 224 × 224, grayscale replicated to 3 channels, ImageNet normalisation (`DataLoading/.../utils.py: load_frames, normalize`);
2. ResNet-50 without its classification layer (`features/backbone.py`), then MKLag.

Its settings are in `MKLag_tool/predict_config.yaml`, and its user guide is [`MKLag_tool/README.md`](../MKLag_tool/README.md).

**When you release a new model for the tool,** copy three files into `MKLag_tool/`:
- the training config to `MKLag_setup/config.yaml`;
- the MKLag checkpoint to `MKLag_setup/mklag.ckpt`;
- the encoder weights to `Backbone_setup/backbone.ckpt`.

The encoder must be the one that produced the training embeddings: same weights, same preprocessing, and an output dimension equal to `in_dim`.
