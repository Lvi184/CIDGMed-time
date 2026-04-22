
import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class StepCIDGMed(nn.Module):
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

        # ✅ 基础维度检查
        assert clinical_dim > 0
        assert prev_med_dim > 0
        assert time_ctx_dim > 0
        assert num_drugs > 0

        self.use_causal_bias = use_causal_bias
        self.causal_weight = causal_weight

        self.clinical_encoder = MLPBlock(clinical_dim, hidden_dim, dropout)
        self.prev_med_encoder = MLPBlock(prev_med_dim, hidden_dim, dropout)
        self.time_encoder = MLPBlock(time_ctx_dim, hidden_dim // 2, dropout)

        fusion_dim = hidden_dim + hidden_dim + hidden_dim // 2

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.med_head = nn.Linear(hidden_dim, num_drugs)
        self.time_head = nn.Linear(hidden_dim, 1)

        # ✅ 因果矩阵检查
        if use_causal_bias and causal_effect_matrix is not None:
            assert causal_effect_matrix.shape[0] == clinical_dim
            assert causal_effect_matrix.shape[1] == num_drugs

            self.register_buffer(
                "causal_effect_matrix",
                torch.tensor(causal_effect_matrix, dtype=torch.float32),
            )
        else:
            self.causal_effect_matrix = None

    def forward(self, x_clinical, x_prev_med, x_time_ctx):
        h1 = self.clinical_encoder(x_clinical)
        h2 = self.prev_med_encoder(x_prev_med)
        h3 = self.time_encoder(x_time_ctx)

        h = torch.cat([h1, h2, h3], dim=-1)
        h = self.fusion(h)

        med_logits = self.med_head(h)
        time_pred = self.time_head(h).squeeze(-1)

        if self.use_causal_bias and self.causal_effect_matrix is not None:
            causal_bias = x_clinical @ self.causal_effect_matrix
            med_logits = med_logits + self.causal_weight * causal_bias

        return med_logits, time_pred

