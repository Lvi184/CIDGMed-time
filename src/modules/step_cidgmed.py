
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class StepCIDGMed(nn.Module):
    """
    Step-level CIDGMed 适配版:
    输入:
        x_clinical      当前临床/检验/人口学等特征
        x_prev_med      上一步药物多标签
        x_time_ctx      step, prev_step_time, cumulative_time_before_step
    输出:
        med_logits      当前 step 的药物多标签
        time_pred       当前 step 的时间
    """

    def __init__(
        self,
        clinical_dim,
        prev_med_dim,
        time_ctx_dim,
        num_drugs,
        hidden_dim=128,
        dropout=0.2,
        use_causal_bias=False,
        causal_effect_matrix=None,
        causal_weight=0.1,
    ):
        super().__init__()

        # 三路编码器
        self.clinical_encoder = MLPBlock(clinical_dim, hidden_dim, dropout=dropout)
        self.prev_med_encoder = MLPBlock(prev_med_dim, hidden_dim, dropout=dropout)
        self.time_encoder = MLPBlock(time_ctx_dim, hidden_dim // 2, dropout=dropout)

        # 融合层
        fusion_dim = hidden_dim + hidden_dim + hidden_dim // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 输出头
        self.med_head = nn.Linear(hidden_dim, num_drugs)
        self.time_head = nn.Linear(hidden_dim, 1)

        # 因果修正模块（预留接口）
        self.use_causal_bias = use_causal_bias
        self.causal_effect_matrix = None
        self.causal_weight = causal_weight

        if use_causal_bias and causal_effect_matrix is not None:
            self.register_buffer("causal_effect_matrix", torch.tensor(causal_effect_matrix, dtype=torch.float32))

    def forward(self, x_clinical, x_prev_med, x_time_ctx):
        # 编码各部分
        h_clinical = self.clinical_encoder(x_clinical)
        h_prev = self.prev_med_encoder(x_prev_med)
        h_time = self.time_encoder(x_time_ctx)

        # 融合
        h = torch.cat([h_clinical, h_prev, h_time], dim=-1)
        h = self.fusion(h)

        # 输出
        med_logits = self.med_head(h)
        time_pred = self.time_head(h).squeeze(-1)

        # 因果修正（如果启用）
        if self.use_causal_bias and self.causal_effect_matrix is not None:
            # 简化版：x_clinical 投影到因果效应矩阵上
            # 这里预留接口，后续可以接入真实的 CIDGMed causal effect
            causal_bias = x_clinical @ self.causal_effect_matrix
            med_logits = med_logits + self.causal_weight * causal_bias

        return med_logits, time_pred

