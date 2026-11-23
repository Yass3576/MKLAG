# Code modified from https://github.com/ZijiaLewisLu/CVPR2024-FACT/tree/main

import torch
import torch.nn as nn

from configs.utils import update_from
from models import basic
from models.basic import time_mask
from models.loss import (MatchCriterionLAG, smooth_loss, decode_lag, pack_lag_prediction, N_LAG_SEGMENTS,
                         decode_ordinal, ordinal_class_logprob)
from models.classificationHead import (SceneAggregator, GlobalVideoHead_MLP,
                                       LagLocalizationHead, CLASS_GROUPS, merge_class_logits)


#########################################################
# MKLAG MODEL WITH MORPHOKINETIC + LAG HEADS
#########################################################

class mklag(nn.Module):
    """
    @brief mklag with:
            - frame branch      -> developmental stage of each frame
            - stage branch      -> stage tokens (actions in the original FACT)
            - LAG head          -> localisation the LAG phase up to 3 sub-segments
            - scene head        -> video-level morphokinetic category
    """

    def __init__(self, cfg, in_dim, n_stages, n_tclasses):
        super().__init__()
        self.cfg = cfg
        self.n_stages = n_stages
        self.n_tclasses = n_tclasses

        base_cfg = cfg.Bi

        self.frame_pe = basic.PositionalEncoding(base_cfg.hid_dim, max_len=300, empty=(not cfg.FACT.fpos))
        self.channel_masking_dropout = nn.Dropout2d(p=cfg.FACT.cmr or 0.0)

        if not cfg.FACT.trans:
            self.stage_query = nn.Parameter(torch.randn([cfg.FACT.ntoken, 1, base_cfg.a_dim]))
        else:
            self.stage_pe = basic.PositionalEncoding(base_cfg.a_dim, max_len=60)
            self.stage_embed = nn.Embedding(n_stages, base_cfg.a_dim)

        block_list = []
        for t in cfg.FACT.block:
            if t == 'i':
                block = InputBlock(cfg, in_dim, n_stages, n_tclasses)
            elif t == 'u':
                update_from(cfg.Bu, base_cfg, inplace=True)
                base_cfg = cfg.Bu
                block = UpdateBlock(cfg, n_stages, n_tclasses)
            else:
                raise ValueError(f"Unknown block type: {t}")
            block_list.append(block)

        assert isinstance(block_list[-1], UpdateBlock), "The last block must be an UpdateBlock ('u')"

        self.block_list = nn.ModuleList(block_list)
        self.mcriterion = MatchCriterionLAG(cfg, n_stages, n_tclasses)

    def _forward_one_video(self, seq, transcript=None):
        """
        seq: T, 1, D
        """
        frame_feature = seq
        frame_pe = self.frame_pe(seq)

        # Regularisation: channel masking (drops whole feature channels)
        if self.cfg.FACT.cmr:
            frame_feature = frame_feature.permute([1, 2, 0]).unsqueeze(-1)  # 1, D, T, 1
            frame_feature = self.channel_masking_dropout(frame_feature)
            frame_feature = frame_feature.squeeze(-1).permute([2, 0, 1])   # T, 1, D

        # Regularisation: temporal masking
        if self.cfg.TM.use and self.training:
            frame_feature = time_mask(
                frame_feature, self.cfg.TM.t, self.cfg.TM.m, self.cfg.TM.p,
                replace_with_zero=True, clone=True,
            )

        if not self.cfg.FACT.trans:
            stage_pe = self.stage_query
            stage_feature = torch.zeros_like(stage_pe)
        else:
            stage_pe = self.stage_pe(transcript)
            stage_feature = self.stage_embed(transcript).unsqueeze(1) + stage_pe
            stage_pe = torch.zeros_like(stage_pe)

        for block in self.block_list:
            frame_feature, stage_feature, scene_feature, scene_logits, lag_pred = block(
                frame_feature, stage_feature, frame_pe, stage_pe
            )

        return scene_feature, scene_logits, lag_pred

    def _loss_one_video(self, frame_label, scene_label, lag_label):
        mcriterion = self.mcriterion
        mcriterion.set_label(frame_label, scene_label, lag_label)

        last_block = self.block_list[-1]
        cprob = basic.logit2prob(last_block.stage_clogit, dim=-1)
        match = mcriterion.match(cprob, last_block.a2f_attn)

        losses = [block.compute_loss(mcriterion, match) for block in self.block_list]
        n = len(losses)
        total_loss = sum(l[0] for l in losses) / n
        scene_loss = sum(l[1] for l in losses) / n
        lag_loss = sum(l[2] for l in losses) / n
        return total_loss, scene_loss, lag_loss

    def forward(self, seq_list, frame_label_list=None, scene_label_list=None, lag_labels=None,
                compute_loss=False):
        """
        @param seq_list: [B, T, D] (or list of [T, D])
        @param frame_label_list: [B, T]  developmental stage of each frame
        @param scene_label_list: [B]     morphokinetic category
        @param lag_labels: dict of [B] tensors with keys
                    's1', 'e1', 's2', 'e2', 's3', 'e3', 'n_segment'
        """
        frame_preds, scene_preds, scene_logits_list, scene_features, lag_preds = [], [], [], [], []
        total_losses, scene_losses, lag_losses = [], [], []

        for i, seq in enumerate(seq_list):
            transcript = None

            scene_feature, scene_logits, lag_pred = self._forward_one_video(seq.unsqueeze(1), transcript=transcript)

            last_block = self.block_list[-1]
            frame_preds.append(last_block.inference(transcript))       # (T,)
            scene_preds.append(last_block.scene_clogit.argmax(-1))     # (1,)
            scene_logits_list.append(scene_logits.squeeze(0))          # (n_tclasses,)
            scene_features.append(scene_feature.squeeze(0))            # (D,)
            lag_preds.append(lag_pred)                                 # (9,)

            if compute_loss:
                lag_label = {k: v[i] for k, v in lag_labels.items()}
                loss, scene_loss, lag_loss = self._loss_one_video(
                    frame_label_list[i], scene_label_list[i], lag_label
                )
                total_losses.append(loss)
                scene_losses.append(scene_loss)
                lag_losses.append(lag_loss)

        res = {
            'frame': torch.stack(frame_preds, dim=0),
            'scene': torch.stack(scene_preds, dim=0),
            'scene_features': torch.stack(scene_features, dim=0),
            'scene_logits': torch.stack(scene_logits_list, dim=0),
            'lag_probs': torch.stack(lag_preds, dim=0),
        }

        if compute_loss:
            res['loss'] = sum(total_losses) / len(total_losses)
            res['scene_loss'] = sum(scene_losses) / len(scene_losses)
            res['lag_loss'] = sum(lag_losses) / len(lag_losses)

        return res

    def save_model(self, fname):
        torch.save(self.state_dict(), fname)


#########################################################
# BRANCH BLOCKS
#########################################################

class Block(nn.Module):
    """Base class for the FACT blocks (frame / stage branches + LAG and scene heads)"""

    def __str__(self):
        return (f"{type(self).__name__}(\n  f:{self.frame_branch},\n  a:{self.stage_branch},\n"
                f"  a2f:{getattr(self, 'a2f_layer', None)},\n  f2a:{getattr(self, 'f2a_layer', None)}\n)")

    def __repr__(self):
        return str(self)

    @staticmethod
    def process_feature(feature, nclass):
        """Split the last `nclass` channels as logits and replace them by probabilities."""
        clogit = feature[:, :, -nclass:]
        feature = feature[:, :, :-nclass]
        cprob = basic.logit2prob(clogit, dim=-1)
        feature = torch.cat([feature, cprob], dim=-1)
        return feature, clogit

    # ---------------------------------------------------
    # Builders
    # ---------------------------------------------------

    # Frame Branch
    def create_fbranch(self, cfg, in_dim=None, f_inmap=False):
        if in_dim is None:
            in_dim = cfg.f_dim

        if cfg.f == 'm':     # MSTCN
            return basic.MSTCN(in_dim, cfg.f_dim, cfg.hid_dim, cfg.f_layers,
                               dropout=cfg.dropout, ln=cfg.f_ln, ngroup=cfg.f_ngp, in_map=f_inmap)
        if cfg.f == 'm2':    # MSTCN++
            return basic.MSTCN2(in_dim, cfg.f_dim, cfg.hid_dim, cfg.f_layers,
                                dropout=cfg.dropout, ln=cfg.f_ln, ngroup=cfg.f_ngp, in_map=f_inmap)
        raise ValueError(cfg.f)

    # Stage Branch (Action branch in the original FACT)
    def create_abranch(self, cfg):
        if cfg.a == 'sa':    # self-attention layers, for update blocks
            layer = basic.SALayer(cfg.a_dim, cfg.a_nhead, dim_feedforward=cfg.a_ffdim,
                                  dropout=cfg.dropout, attn_dropout=cfg.dropout)
            return basic.SADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, layer, cfg.a_layers, in_map=False)
        if cfg.a == 'sca':   # self+cross-attention layers, for input blocks without transcript
            layer = basic.SCALayer(cfg.a_dim, cfg.hid_dim, cfg.a_nhead, cfg.a_ffdim,
                                   dropout=cfg.dropout, attn_dropout=cfg.dropout)
            norm = torch.nn.LayerNorm(cfg.a_dim)
            return basic.SCADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, layer, cfg.a_layers, norm=norm, in_map=False)
        if cfg.a in ['gru', 'gru_om']:  # GRU, for input blocks when transcripts are available
            assert self.cfg.FACT.trans
            out_map = (cfg.a == 'gru_om')
            return basic.ActionUpdate_GRU(cfg.a_dim, cfg.a_dim, cfg.hid_dim, cfg.a_layers,
                                          dropout=cfg.dropout, out_map=out_map)
        raise ValueError(cfg.a)

    # Scene / morphokinetic head
    def create_shead(self, cfg, n_tclasses):
        return GlobalVideoHead_MLP(cfg.hid_dim, n_tclasses)
        

    def create_cross_attention(self, cfg, outdim, kq_pos=True):
        # one layer of cross-attention for cross-branch communication
        return basic.X2Y_map(cfg.hid_dim, cfg.hid_dim, outdim, head_dim=cfg.hid_dim,
                             dropout=cfg.dropout, kq_pos=kq_pos)

    def create_lag_and_scene_modules(self, cfg, n_tclasses):
        """Modules shared by InputBlock and UpdateBlock (LAG head + scene head with LAG gating)."""
        self.ordinal = self.cfg.Loss.mk_type == "ordinal"
        n_head = self.cfg.head_tclasses or n_tclasses       # 8 for the old 8-class checkpoints
        self.scene_head = self.create_shead(cfg, n_head - 1 if self.ordinal else n_head)
        self.class_groups = None if n_head == n_tclasses else CLASS_GROUPS[(n_head, n_tclasses)]
        self.scene_aggregator = SceneAggregator(cfg)

        self.lag_head = LagLocalizationHead(cfg.hid_dim, n_queries=N_LAG_SEGMENTS)
        self.stage_pos_enc_bis = basic.PositionalEncoding(cfg.hid_dim, max_len=60)

        self.lag_embed = nn.Sequential(
            nn.Linear(3 * N_LAG_SEGMENTS, cfg.hid_dim),
            nn.ReLU()
        )
        self.scene_norm = nn.LayerNorm(cfg.hid_dim)
        self.lag_norm = nn.LayerNorm(cfg.hid_dim)
        self.lag_gate = nn.Linear(cfg.hid_dim, cfg.hid_dim)

    def lag_forward(self, stage_feature):
        """
        @brief LAG localisation from the stage tokens
        @return raw head output (9,) and decoded prediction (9,) [z_k, s_k, d_k] x 3
        """
        stage_pos = self.stage_pos_enc_bis(stage_feature)
        lag_out, _, _ = self.lag_head(stage_feature, stage_pos)
        lag_out = lag_out.reshape(-1)

        cum_logits, start, duration, _ = decode_lag(lag_out, ordinal=self.cfg.Loss.lag_type == "ordinal")
        lag_pred = pack_lag_prediction(cum_logits, start, duration)
        return lag_out, lag_pred

    def scene_forward(self, scene_feature, frame_pos, lag_out):
        """
        @brief Morphokinetci prediction from the aggregated scene feature modulated by the LAG prediction.
        """
        scene_agg = self.scene_norm(self.scene_aggregator(scene_feature, frame_pos))
        lag_feat = self.lag_norm(self.lag_embed(lag_out.detach()).unsqueeze(0))

        gate = torch.sigmoid(self.lag_gate(scene_agg))
        scene_input = scene_agg * (1 + gate * lag_feat)

        scene_clogit = self.scene_head(scene_input)
        if self.ordinal:       # K-1 cumulative logits -> K class log-probabilities
            self.scene_cum = decode_ordinal(scene_clogit)
            scene_clogit = ordinal_class_logprob(self.scene_cum)
        if self.class_groups is not None:      # e.g. 8 head classes -> 5 classes
            scene_clogit = merge_class_logits(scene_clogit, self.class_groups)
        return scene_agg, scene_clogit

    @staticmethod
    def _inference(stage_clogit, a2f_attn, frame_clogit, weight):
        """
        @brief Inference of the frame-wise developmental stages
        """
        fbranch_prob = torch.softmax(frame_clogit.squeeze(1), dim=-1)

        stage_clogit = stage_clogit.squeeze(1)            
        null_cid = stage_clogit.shape[-1] - 1
        is_stage = stage_clogit.argmax(1) != null_cid       

        if not bool(is_stage.any()):
            return fbranch_prob.argmax(1)

        qtk_prob = torch.softmax(stage_clogit[:, :-1], dim=1)  

        attn = a2f_attn.masked_fill(~is_stage.unsqueeze(0), float('-inf'))  
        stage_pred = attn.argmax(-1)                                        
        abranch_prob = qtk_prob[stage_pred]

        prob = (1 - weight) * abranch_prob + weight * fbranch_prob
        return prob.argmax(1)

    @staticmethod
    def _eval_w_transcript(transcript, a2f_attn, frame_clogit, weight):
        fbranch_prob = torch.softmax(frame_clogit.squeeze(1), dim=-1)
        fbranch_prob = fbranch_prob[:, transcript]

        N = len(transcript)
        abranch_prob = torch.softmax(a2f_attn[:, :N], dim=-1)  # f, s'

        prob = (1 - weight) * abranch_prob + weight * fbranch_prob
        pred = prob.argmax(1)
        return transcript[pred]

    def inference(self, transcript=None):
        if not self.cfg.FACT.trans:
            return self._inference(self.stage_clogit, self.a2f_attn, self.frame_clogit, self.cfg.FACT.mwt)
        return self._eval_w_transcript(transcript, self.a2f_attn, self.frame_clogit, self.cfg.FACT.mwt)


    def common_losses(self, criterion: MatchCriterionLAG, match):
        """
        @brief Frame, stage token, scene and LAG losses.
        """
        w = self.cfg.Loss
        frame_loss = w.frame * criterion.frame_loss(self.frame_clogit.squeeze(1))     # developmental stages
        atk_loss = w.action * criterion.action_token_loss(match, self.stage_clogit)   # stage tokens
        if self.ordinal:                                                       # morphokinetic
            scene_loss = criterion.video_loss_ordinal(self.scene_cum)
        else:
            scene_loss = criterion.video_loss(self.scene_clogit)
        lag_ord, lag_regr, lag_regu = criterion.lag_loss_se(self.lag_out)      # LAG localisation
        return frame_loss, atk_loss, scene_loss, lag_ord, lag_regr, lag_regu


class InputBlock(Block):
    """Initial block"""

    def __init__(self, cfg, in_dim, n_stages, n_tclasses):
        super().__init__()
        self.cfg = cfg
        self.n_stages = n_stages
        cfg = cfg.Bi

        self.frame_branch = self.create_fbranch(cfg, in_dim, f_inmap=True)
        self.stage_branch = self.create_abranch(cfg)
        self.create_lag_and_scene_modules(cfg, n_tclasses)

    def forward(self, frame_feature, stage_feature, frame_pos, stage_pos):
        # Frame branc
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.n_stages)

        # Stage branch
        stage_feature = self.stage_branch(stage_feature, frame_feature, pos=frame_pos, query_pos=stage_pos)
        stage_feature, stage_clogit = self.process_feature(stage_feature, self.n_stages + 1)

        # LAG from the stage tokens
        lag_out, lag_pred = self.lag_forward(stage_feature)

        # Morphokinetic from the frame features (+ LAG gating)
        scene_agg, scene_clogit = self.scene_forward(frame_feature, frame_pos, lag_out)

        self.frame_clogit = frame_clogit
        self.stage_clogit = stage_clogit
        self.scene_clogit = scene_clogit
        self.lag_out = lag_out

        return frame_feature, stage_feature, scene_agg, scene_clogit, lag_pred

    def compute_loss(self, criterion: MatchCriterionLAG, match=None):
        frame_loss, atk_loss, scene_loss, lag_ord, lag_regr, lag_regu = self.common_losses(criterion, match)

        smooth = smooth_loss(self.frame_clogit.transpose(0, 1))
        reg_loss = self.cfg.Loss.sw * smooth

        lag_loss = lag_ord + lag_regr + lag_regu
        total_loss = frame_loss + atk_loss + self.cfg.Loss.mk * scene_loss + lag_loss + reg_loss
        return total_loss, scene_loss, lag_loss


class UpdateBlock(Block):
    """Update block"""

    def __init__(self, cfg, n_stages, n_tclasses):
        super().__init__()
        self.cfg = cfg
        self.n_stages = n_stages
        cfg = cfg.Bu
        self.w = 0.5  # weight of the stage context vs frame context in the scene feature

        # Branches
        self.frame_branch = self.create_fbranch(cfg)
        self.stage_branch = self.create_abranch(cfg)
        self.create_lag_and_scene_modules(cfg, n_tclasses)

        self.stage_norm = nn.LayerNorm(cfg.hid_dim)
        self.frame_norm = nn.LayerNorm(cfg.hid_dim)

        # Cross-attention layers
        self.f2a_layer = self.create_cross_attention(cfg, cfg.a_dim)
        self.a2f_layer = self.create_cross_attention(cfg, cfg.f_dim)

    def forward(self, frame_feature, stage_feature, frame_pos, stage_pos):
        # Frame -> Stage communication
        stage_feature, f2a_attn = self.f2a_layer(frame_feature, stage_feature, X_pos=frame_pos, Y_pos=stage_pos)

        # Update stage features
        stage_feature = self.stage_branch(stage_feature, stage_pos)
        stage_feature, stage_clogit = self.process_feature(stage_feature, self.n_stages + 1)

        # Stage -> Frame communication
        frame_feature, a2f_attn = self.a2f_layer(stage_feature, frame_feature, X_pos=stage_pos, Y_pos=frame_pos)

        # Update frame features
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.n_stages)

        # Per-frame representation mixing stage and frame features.
        #frame2stage = f2a_attn.T
        #frame2stage = frame2stage / (frame2stage.sum(dim=-1, keepdim=True) + 1e-8)  # T, n_token
        frame2stage = torch.softmax(f2a_attn.T, dim=-1)
        
        stage_context = self.stage_norm(torch.mm(frame2stage, stage_feature[:, 0]))
        frame_context = self.frame_norm(frame_feature[:, 0])
        stageframe_feature = (self.w * stage_context + (1 - self.w) * frame_context).unsqueeze(1)

        # LAG from the stage tokens
        lag_out, lag_pred = self.lag_forward(stage_feature)

        # Morphokinetic (+ LAG gating)
        scene_agg, scene_clogit = self.scene_forward(stageframe_feature, frame_pos, lag_out)


        self.frame_clogit = frame_clogit
        self.stage_clogit = stage_clogit
        self.scene_clogit = scene_clogit
        self.lag_out = lag_out
        self.f2a_attn = f2a_attn
        self.a2f_attn = a2f_attn

        return frame_feature, stage_feature, scene_agg, scene_clogit, lag_pred

    def compute_loss(self, criterion: MatchCriterionLAG, match=None):
        """
        Total loss of the block:
          frame + stage-token + scene + LAG (ordinal + L2 + consistency)
          + cross-attention alignment + temporal smoothness
        """
        frame_loss, atk_loss, scene_loss, lag_ord, lag_regr, lag_regu = self.common_losses(criterion, match)

        # Cross-attention alignment losses
        f2a_loss = criterion.cross_attn_loss(match, torch.transpose(self.f2a_attn.unsqueeze(0), 1, 2), dim=1)
        a2f_loss = criterion.cross_attn_loss(match, self.a2f_attn.unsqueeze(0), dim=2)

        # Smoothing losses
        smooth = smooth_loss(torch.transpose(self.frame_clogit, 0, 1))
        smooth = smooth + smooth_loss(self.a2f_attn.unsqueeze(0))
        smooth = smooth + smooth_loss(torch.transpose(self.f2a_attn.unsqueeze(0), 1, 2))

        reg_loss = self.cfg.Loss.sw * smooth + self.cfg.Loss.ca * (f2a_loss + a2f_loss)

        lag_loss = lag_ord + lag_regr + lag_regu
        total_loss = frame_loss + atk_loss + self.cfg.Loss.mk * scene_loss + lag_loss + reg_loss
        return total_loss, scene_loss, lag_loss