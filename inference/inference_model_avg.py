from pathlib import Path

import torch

from inference.abstract import SFDInference


class SFDInferenceModelAvg(SFDInference):
    def __init__(self, folder_path: str | Path, device: str = "cpu"):
        super().__init__(folder_path=folder_path, device=device)

    @property
    def avg_embeddings(self):
        return self.decoder.Y.weight

    @torch.no_grad()
    def get_x(self, idx):
        return self.decoder(batch_size=1, batch_idx=idx)
