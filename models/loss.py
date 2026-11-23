"""
@brief Losses for MKLag.

Code adapted from https://github.com/ZijiaLewisLu/CVPR2024-FACT (MatchCriterion),
extended with:
  - a video-level morphokinetic classification loss,
  - a LAG phase localisation loss (ordinal count of segments + L2 regression
    of normalised start/end times + temporal-consistency penalties).
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from models import basic
from utils import utils

N_LAG_SEGMENTS = 3


#########################################################
# Generic helpers
#########################################################

def smooth_loss(logit, is_logit=True):
    """
    Temporal smoothing loss (truncated MSE between consecutive log-probabilities).
    logit: B, T, C
    """
    logsoft = F.log_softmax(logit, dim=2) if is_logit else logit
    loss = torch.clamp((logsoft[:, 1:] - logsoft[:, :-1]) ** 2, min=0, max=16)
    return loss.mean()


def torch_class_label_to_segment_label(label):
    """Frame-wise class labels -> (transcript, frame-wise segment index)."""
    return basic.torch_class_label_to_segment_label(label)


def logit2prob(clogit, dim=-1, class_sep=None):
    return basic.logit2prob(clogit, dim=dim, class_sep=class_sep)


#########################################################
# LAG head decoding
#########################################################

def decode_ordinal(raw):
    """z_1 = r_1, z_k = z_{k-1} - softplus(r_k): decreasing cumulative logits, z_k = logit P(y >= k)."""
    return torch.cat([raw[..., :1], raw[..., :1] - torch.cumsum(F.softplus(raw[..., 1:]), dim=-1)], dim=-1)


def ordinal_class_logprob(cum_logits):
    """Cumulative logits (..., K-1) -> class log-probabilities (..., K), P(y = k) = P(y >= k) - P(y >= k+1)."""
    first = F.logsigmoid(-cum_logits[..., :1])
    last = F.logsigmoid(cum_logits[..., -1:])
    a, b = cum_logits[..., :-1], cum_logits[..., 1:]
    mid = F.logsigmoid(a) + F.logsigmoid(-b) + torch.log(-torch.expm1((b - a).clamp(max=-1e-6)))
    return torch.cat([first, mid, last], dim=-1)


def decode_lag(lag_out, n_segments=N_LAG_SEGMENTS, ordinal=True):
    """
    @brief Decode the raw output of the LAG head.

    @param lag_out: (3 * n_segments,) raw values ordered [a_1, u_1, v_1, a_2, u_2, v_2, ...]

   
    @return
    cum_logits : (n_segments,) ordinal cumulative logits, z_k = logit P(n_segment >= k).
                 They are built to be strictly decreasing:
                     z_1 = a_1,   z_k = z_{k-1} - softplus(a_k)
                 so that P(n >= 1) >= P(n >= 2) >= P(n >= 3) holds (rank consistency). 
                 This replaces the former "nesting" penalty relu(p_{k+1} - p_k).
    start      : (n_segments,) normalised start time in [0, 1]  (sigmoid)
    duration   : (n_segments,) normalised duration > 0          (softplus)
    end        : (n_segments,) start + duration
    """
    raw = lag_out.reshape(n_segments, 3)
    a = raw[:, 0]
    cum_logits = decode_ordinal(a) if ordinal else a   # not ordinal: independent presence logits

    start = torch.sigmoid(raw[:, 1])
    duration = F.softplus(raw[:, 2])
    end = start + duration
    return cum_logits, start, duration, end


def pack_lag_prediction(cum_logits, start, duration):
    """
    @brief Pack the decoded LAG prediction: [z_1, s_1, d_1, z_2, s_2, d_2, z_3, s_3, d_3]
    """
    return torch.stack([cum_logits, start, duration], dim=1).reshape(-1)


def ordinal_count_loss(cum_logits, n_segment):
    """
    @brief Ordinal loss on the number of LAG segments (n in {0, ..., K}).
    """
    K = cum_logits.shape[0]
    thresholds = torch.arange(1, K + 1, device=cum_logits.device, dtype=cum_logits.dtype)
    targets = (n_segment.to(cum_logits.dtype) >= thresholds).to(cum_logits.dtype)
    return F.binary_cross_entropy_with_logits(cum_logits, targets, reduction="sum")


#########################################################
# Criterion
#########################################################

class MatchCriterionLAG:
    def __init__(self, cfg, action_nclasses, n_tclasses, bg_ids=(), class_weight=None):
        self.cfg = cfg
        self.action_nclasses = action_nclasses
        self.n_tclasses = n_tclasses
        self.bg_ids = list(bg_ids)
        self._class_weight = class_weight

        # Loss weights of the LAG terms
        loss_cfg = cfg.Loss
        self.lag_ord_w = getattr(loss_cfg, "lag_ord_w", 1.0)
        self.lag_reg_w = getattr(loss_cfg, "lag_reg_w", 1.0)
        self.lag_regu_w = getattr(loss_cfg, "lag_regu_w", 1.0)

    def set_label(self, frame_label, video_label, lag_label):
        device = video_label.device

        self.video_label = video_label
        self.lag_label = lag_label

        self.class_label = frame_label
        self.transcript, self.seg_label = torch_class_label_to_segment_label(frame_label)
        self.onehot_class_label = self._label_to_onehot(self.class_label, self.action_nclasses)
        self.onehot_seg_label = self._label_to_onehot(self.seg_label, len(self.transcript))

        # class weights for developmental stages
        cweight = torch.ones(self.action_nclasses + 1, device=device)
        # weight of the null class (negative value = not set -> 1.0)
        cweight[-1] = self.cfg.Loss.nullw if self.cfg.Loss.nullw >= 0 else 1.0
        if self._class_weight is not None:
            for i in range(self.action_nclasses):
                cweight[i] = self._class_weight[i]
        else:
            for i in self.bg_ids:
                cweight[i] = self.cfg.Loss.bgw

        # weight of each groundtruth segment, derived from its class
        sweight = torch.ones_like(self.transcript, dtype=torch.float32)
        if self._class_weight is not None:
            for i, t in enumerate(self.transcript.tolist()):
                sweight[i] = self._class_weight[t]
        else:
            for i in self.bg_ids:
                sweight[self.transcript == i] = self.cfg.Loss.bgw

        self.cweight = cweight
        self.sweight = sweight

    @staticmethod
    def _label_to_onehot(label, nclass):
        onehot_label = torch.zeros(len(label), nclass, device=label.device)
        onehot_label[torch.arange(len(label)), label] = 1
        return onehot_label

    #########################################################
    # Hungarian matching between stage tokens and GT segments
    #########################################################

    @staticmethod
    def a2f_soft_iou(a2f_attn, onehot_seg_label):
        """
        a2f_attn: 1, f, a  (sum over a == 1)
        onehot_seg_label: f, s
        return: a, s
        """
        a2f_attn_np = utils.to_numpy(a2f_attn[0].unsqueeze(-1))           # f, a, 1
        onehot_seg_label_np = utils.to_numpy(onehot_seg_label.unsqueeze(1))  # f, 1, s
        overlap = np.einsum('tax,txs->as', a2f_attn_np, onehot_seg_label_np)
        union = np.minimum(a2f_attn_np + onehot_seg_label_np, 1.0).sum(0)  # a, s
        return np.nan_to_num(overlap / union, nan=0.0)

    def match(self, cprob, a2f_attn):
        """
        cprob: a, 1, c  (stage-token class probabilities)
        a2f_attn: f, a
        """
        assert cprob.shape[1] == 1  # batch_size == 1

        match_cfg = self.cfg.Loss
        transcript = self.transcript
        onehot_seg_label = self.onehot_seg_label

        # sequential matching between tokens and groundtruth segments
        if match_cfg.match == 'seq':
            A = cprob.shape[0]
            S = onehot_seg_label.shape[-1]
            assert A >= S, (A, S)
            ind = torch.arange(S, dtype=torch.int64)
            return ind, ind

        # matching cost
        cost = np.zeros((cprob.shape[0], len(transcript)))
        with torch.no_grad():
            if match_cfg.pc > 0:
                prob = torch.index_select(cprob.squeeze(1), 1, transcript)  # a, s
                cost -= match_cfg.pc * utils.to_numpy(prob)

            if match_cfg.a2fc > 0:
                a2f_iou = self.a2f_soft_iou(a2f_attn.unsqueeze(0), onehot_seg_label)
                cost -= match_cfg.a2fc * a2f_iou

        if match_cfg.match == 'o2o':    # one-to-one matching
            action_ind, seg_ind = linear_sum_assignment(cost)
        elif match_cfg.match == 'o2m':  # one-to-many matching
            action_ind, seg_ind = self._one_to_many_match(cost)
        else:
            raise ValueError(f"Unknown matching strategy: {match_cfg.match}")

        action_ind = torch.as_tensor(action_ind, dtype=torch.int64)  # id of stage token
        seg_ind = torch.as_tensor(seg_ind, dtype=torch.int64)        # id of groundtruth segment
        return action_ind, seg_ind

    def _one_to_many_match(self, cost):
        transcript_np = utils.to_numpy(self.transcript)
        actions = np.unique(transcript_np)
        token2action_cost = np.stack(
            [cost[:, transcript_np == a].sum(1) for a in actions], axis=1
        )

        _aid, _cid = linear_sum_assignment(token2action_cost)

        unassign_aid = [a for a in range(cost.shape[0]) if a not in _aid]
        unassign_cid = token2action_cost[unassign_aid].argmin(1)

        all_aid = np.array(_aid.tolist() + unassign_aid)
        all_cid = np.array([actions[i] for i in _cid.tolist() + unassign_cid.tolist()])

        atoken_cid = np.zeros(cost.shape[0])
        atoken_cid[all_aid] = all_cid

        match = {}
        for a in actions:
            seg_where = np.where(transcript_np == a)[0]
            token_where = np.where(atoken_cid == a)[0]
            assign = cost[token_where][:, seg_where].argmin(0)
            for s, t in zip(seg_where, assign):
                match[s] = token_where[t]

        sid_new = list(match.keys())
        aid_new = list(match.values())
        return aid_new, sid_new

    #########################################################
    # Frame / stage-token losses (FACT)
    #########################################################

    def action_token_loss(self, match, action_clogit, is_logit=True):
        aind, sind = match
        A, C = action_clogit.shape[0], action_clogit.shape[-1]

        # default target = null class
        clabel = torch.full((A,), C - 1, dtype=torch.long, device=action_clogit.device)
        clabel[aind] = self.transcript[sind]
        if is_logit:
            return F.cross_entropy(action_clogit.squeeze(1), clabel, weight=self.cweight)
        return F.nll_loss(action_clogit.squeeze(1), clabel, weight=self.cweight)

    def cross_attn_loss(self, match, attn, dim=None):
        assert dim >= 1
        aind, sind = match

        frame_tgt = self.onehot_seg_label[:, sind]   # f, s
        attn = attn[0, :, aind]                      # f, s
        attn_logp = torch.log_softmax(attn, dim=dim - 1)
        loss = - attn_logp * frame_tgt
        if self.sweight is not None:
            # sweight is indexed by segment: align it with the matched segments
            loss = loss * self.sweight[sind]
        return loss.sum() / self.onehot_seg_label.sum()

    def frame_loss(self, frame_clogit, is_logit=True):
        logp = torch.log_softmax(frame_clogit, dim=-1) if is_logit else frame_clogit
        cweight = self.cweight[:frame_clogit.shape[-1]]  # remove the weight of the null class
        loss = - logp * self.onehot_class_label * cweight
        return loss.sum() / self.onehot_class_label.sum()

    #########################################################
    # Video-level (morphokinetic) losses
    #########################################################

    def video_loss(self, scene_clogit, is_logit=True, label_smoothing=0.0, weight=None):
        """
        @brief Video-level morphokinetic classification loss.
        
        @param scene_clogit [1, n_classes]
        @param is_logit if False, scene_clogit is assumed to be log-probabilities
        """
        target = self.video_label.reshape(1)
        if is_logit:
            return F.cross_entropy(scene_clogit, target, weight=weight, label_smoothing=label_smoothing)
        return F.nll_loss(scene_clogit, target, weight=weight)

    def video_loss_ordinal(self, cum_logits):
        """@brief Ordinal morphokinetic loss: sum of the BCEs of the K-1 thresholds y >= k."""
        return ordinal_count_loss(cum_logits.reshape(-1), self.video_label.reshape(()))


    #########################################################
    # LAG localisation loss
    #########################################################

    def lag_loss_se(self, lag_out):
        """
        @brief Lag phase localisation loss.

        @param lag_out (9,) raw output of the LAG head

        @return (already weighted):
          loss_ord  : ordinal loss on the number of segments (replaces the 3 independent
                      presence BCEs + the "specify" penalty p_{k+1} <= p_k of the paper)
          loss_regr : L2 regression of normalised start / end of each GT segment
          loss_regu : temporal-consistency penalties between consecutive segments
                      (s_k <= s_{k+1} and no overlap e_k <= s_{k+1})
        """
        cum_logits, start, _, end = decode_lag(lag_out, ordinal=self.cfg.Loss.lag_type == "ordinal")


        n = self.lag_label["n_segment"]
        gt_start = torch.stack([self.lag_label[f"s{k}"] for k in range(1, N_LAG_SEGMENTS + 1)]).to(start.dtype)
        gt_end = torch.stack([self.lag_label[f"e{k}"] for k in range(1, N_LAG_SEGMENTS + 1)]).to(start.dtype)

        thresholds = torch.arange(1, N_LAG_SEGMENTS + 1, device=start.device, dtype=start.dtype)
        mask = (n.to(start.dtype) >= thresholds).to(start.dtype)  # segment k exists ?


        loss_ord = ordinal_count_loss(cum_logits, n)

        se_start = F.mse_loss(start, gt_start, reduction="none")
        se_end = F.mse_loss(end, gt_end, reduction="none")
        loss_regr = (mask * (se_start + se_end)).sum()

        next_mask = mask[1:]
        order_penalty = next_mask * torch.relu(start[:-1] - start[1:])
        overlap_penalty = next_mask * torch.relu(end[:-1] - start[1:])
        loss_regu = order_penalty.sum() + overlap_penalty.sum()

        return (self.lag_ord_w * loss_ord,
                self.lag_reg_w * loss_regr,
                self.lag_regu_w * loss_regu)