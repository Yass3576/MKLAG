import torch
import torch.nn as nn

# Head trained with more classes than the labels (e.g. the old 8-class models evaluated in 5 classes):
# head class i belongs to class CLASS_GROUPS[(n_head, n_tclasses)][i]
#   8: H, G, F, E, D, C, B, A  ->  5: NT, ADB, SSB, HB, EHB  ->  2: NT, T
CLASS_GROUPS = {
    (8, 5): [0, 0, 0, 0, 1, 2, 3, 4],
    (8, 2): [0, 0, 0, 0, 1, 1, 1, 1],
    (5, 2): [0, 1, 1, 1, 1],
}


def merge_class_logits(logits, groups):
    """
    @brief P(group) = sum of the probabilities of its classes, returned as logits
           (softmax of the output = summed probabilities, so CE / argmax work as usual).
    """
    groups = torch.as_tensor(groups, device=logits.device)
    return torch.stack([logits[..., groups == g].logsumexp(-1)
                        for g in range(int(groups.max()) + 1)], dim=-1)


class SceneAggregator(nn.Module):
    """
    @brief Create a global video embedding from all frames embeddings 
    """
    def __init__(self, cfg):
        super().__init__()
        D = cfg.hid_dim
      
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=cfg.s_head_num_heads,
            dim_feedforward=cfg.s_head_ffdim,
            dropout=cfg.dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.CLS = nn.Parameter(torch.randn(1, 1, D) * 0.02)  # [1,1,D]
    

    def forward(self, scene_feature, pos=None):
        """
        @param scene_feature: [T, B, D]
        @param pos [T, B, D] or [T, 1, D] -> optional positional encoding
        @return classification token [B, D]
        """

        T, B, D = scene_feature.shape

        if pos is not None:
            scene_feature = scene_feature + pos

        CLS = self.CLS.expand(-1, B, -1)          # [1,B,D]
        seq = torch.cat([CLS, scene_feature], 0)  # [1+T,B,D]

        seq = self.transformer(seq)

        return seq[0]  # [B,D]


class GlobalVideoHead_MLP(nn.Module):
    """
    @brief Classification head for prediction of the morphokinetic category
    """
    def __init__(self, in_dim, n_classes, hidden_dim=512, dropout=0.3):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, x):
        return self.mlp(x)


class LagLocalizationHead(nn.Module):
    """
    @brief Lag prediction and localisation head
    @details `n_queries` learned queries (one per candidate LAG segment)
            cross-attend to the stage tokens. Each query outputs 3 raw values
            (presence, start, duration), decoded by `models.loss.decode_lag`.
    """

    def __init__(self, token_dim, n_queries=3, n_heads=4, hidden_dim=128):
        super().__init__()
        self.n_queries = n_queries

        self.lag_queries = nn.Parameter(torch.randn(n_queries, 1, token_dim))
        self.query_pe = nn.Parameter(torch.randn(n_queries, 1, token_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=n_heads, batch_first=False)
        self.norm = nn.LayerNorm(token_dim)

        self.head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, stage_tokens, pos=None):
        """
        stage_tokens: (N_token, 1, D)
        return: pred (n_queries, 3), attention map, attended queries
        """
        queries = (self.lag_queries + self.query_pe).expand(-1, stage_tokens.shape[1], -1)  # (Q, 1, D)

        if pos is not None:
            stage_tokens = stage_tokens + pos

        attn_out, attn_map = self.cross_attn(queries, stage_tokens, stage_tokens)
        attn_out = self.norm(attn_out + queries)

        pred = self.head(attn_out).squeeze(1)  # (Q, 3)
        return pred, attn_map, attn_out
