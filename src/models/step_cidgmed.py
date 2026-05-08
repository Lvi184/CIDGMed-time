import torch
import torch.nn as nn


class StepCIDGMedNet(nn.Module):
    """Multi-task baseline/history model.

    Outputs:
    - med_logits: multi-label medication regimen probabilities after sigmoid
    - time_pred: final outcome time, defaults to LOS
    - duration_pred: current/next step duration target built from drug.time
    """

    def __init__(self, input_dim: int, med_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.med_head = nn.Linear(hidden_dim, med_dim)
        self.time_head = nn.Linear(hidden_dim, 1)
        self.duration_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        z = self.backbone(x)
        med_logits = self.med_head(z)
        time_pred = self.time_head(z)
        duration_pred = self.duration_head(z)
        return med_logits, time_pred, duration_pred
