
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalityReview(nn.Module):
    """Adapted causal review for your relevance matrices"""
    def __init__(self, num_diag, num_proc, num_med, causal_weight=0.1):
        super().__init__()
        self.num_med = num_med
        self.causal_weight = causal_weight
        # Thresholds will be learned or set
        self.high_thresh = nn.Parameter(torch.tensor(0.8))
        self.low_thresh = nn.Parameter(torch.tensor(0.2))
        
    def forward(self, med_logits, diag_features, proc_features, diag_med_mat, proc_med_mat):
        # diag_features: batch x num_diag
        # proc_features: batch x num_proc
        # diag_med_mat: num_diag x num_med
        # proc_med_mat: num_proc x num_med
        
        # Compute relevance scores per med
        diag_bias = torch.matmul(diag_features, diag_med_mat)  # batch x num_med
        proc_bias = torch.matmul(proc_features, proc_med_mat)  # batch x num_med
        total_bias = (diag_bias + proc_bias) * self.causal_weight
        
        # Apply simple thresholding (optional, can be expanded)
        adjusted_logits = med_logits + total_bias
        
        return adjusted_logits


class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.layers(x)


class StepCIDGMedFull(nn.Module):
    def __init__(
        self,
        clinical_dim,
        prev_med_dim,
        time_ctx_dim,
        num_med,
        num_diag,
        num_proc,
        hidden_dim=128,
        dropout=0.2,
        use_causal_bias=True,
        causal_weight=0.1
    ):
        super().__init__()
        
        # Encoders
        self.clinical_encoder = MLPBlock(clinical_dim, hidden_dim, hidden_dim, dropout)
        self.prev_med_encoder = MLPBlock(prev_med_dim, hidden_dim, hidden_dim, dropout)
        self.time_encoder = MLPBlock(time_ctx_dim, hidden_dim // 2, hidden_dim // 2, dropout)
        
        # Fusion
        fusion_dim = hidden_dim + hidden_dim + (hidden_dim // 2)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Heads
        self.med_head = nn.Linear(hidden_dim, num_med)
        self.time_head = nn.Linear(hidden_dim, 1)
        
        # Causal review
        self.use_causal_bias = use_causal_bias
        if use_causal_bias:
            self.causality_review = CausalityReview(num_diag, num_proc, num_med, causal_weight)
        
    def forward(self, x_clinical, x_prev_med, x_time_ctx, diag_features, proc_features, diag_med_mat=None, proc_med_mat=None):
        # Encode each part
        h_clinical = self.clinical_encoder(x_clinical)
        h_prev_med = self.prev_med_encoder(x_prev_med)
        h_time_ctx = self.time_encoder(x_time_ctx)
        
        # Concat and fuse
        h = torch.cat([h_clinical, h_prev_med, h_time_ctx], dim=-1)
        h = self.fusion(h)
        
        # Predict heads
        med_logits = self.med_head(h)
        time_pred = self.time_head(h).squeeze(-1)
        
        # Apply causal review (optional)
        if self.use_causal_bias and diag_med_mat is not None and proc_med_mat is not None:
            med_logits = self.causality_review(med_logits, diag_features, proc_features, diag_med_mat, proc_med_mat)
        
        return med_logits, time_pred
