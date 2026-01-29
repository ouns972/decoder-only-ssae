from pathlib import Path

import torch

from inference.abstract import SFDInference


class SFDInferenceModelTrainableInput(SFDInference):
    def __init__(self, folder_path: str | Path, device: str = "cpu"):
        super().__init__(folder_path=folder_path, device=device)

    @property
    def embeddings(self):
        return self.decoder.Y.weight

    def get_x(self, idx):
        return self.decoder(batch_size=1, batch_idx=idx)

    def switch_property(
        self, promt_idx_old: int, property_old: int, property_new: int
    ) -> torch.Tensor:
        y = self.decoder.Y.detach()[promt_idx_old, :]
        print("y = ", y)
