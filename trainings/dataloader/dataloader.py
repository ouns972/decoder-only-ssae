import json
import os
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from trainings.config.config import check_yaml_params
from trainings.dataloader.checks import run_checks_dataset
from trainings.dataloader.properties.properties import Properties
from trainings.dataloader.properties.same_id import SameId

MAX_MIN = "MAX_MIN"


class H5Dataset(Dataset):
    def __init__(
        self,
        folder_path,
        truncate_n_prompts=None,
        truncate_embds_topk=None,
        add_property_is_the_same=False,
        simulated=False,
        dim_clip_simulated=5000,
        logger=None,
        normalize=None,
    ):
        self.folder_path = folder_path
        self.truncate_n_prompts = truncate_n_prompts
        self.logger = logger
        self.log_print = print if self.logger is None else logger.print
        self.truncate_embds_topk = truncate_embds_topk
        self.indices_truncate_embds_topk = None
        self.normalize = None  # we init the dateset with self.normalize = None and overwrite it to its actual value (normalize) at the end of the init

        self.X = None

        # simulated mode
        # for the moment, simulated uses the same mask as the real data (i.e., same number of prompts)
        self.simulated = simulated
        self.dim_clip_simulated = dim_clip_simulated

        # extract files
        self.folder_embds = folder_path + "embds/"

        self.folders = [
            f
            for f in os.listdir(self.folder_embds)
            if os.path.isdir(os.path.join(self.folder_embds, f))
            and f.startswith("embds_")
        ]

        self.folders = sorted(self.folders, key=lambda x: int(x.split("_")[1]))

        # truncate number of prompts
        if self.truncate_n_prompts is not None:
            self.log_print(
                f"Truncating the number of prompts from {len(self.folders)} to {self.truncate_n_prompts}."
            )
            self.folders = self.folders[: self.truncate_n_prompts]

        self.files = [f + "/embds.h5" for f in self.folders]
        self.files_pooled = [f + "/embds_pooled.h5" for f in self.folders]
        self.files_prompts = [f + "/prompts.txt" for f in self.folders]

        # detect if dataset has pooled_embds files
        self.has_pooled_embds = False
        if os.path.exists(os.path.join(self.folder_embds, self.files_pooled[0])):
            self.log_print(
                "Has found /embds_pooled.h5 files, will only use /embds_pooled.h5"
            )
            self.log_print("has_pooled_embds set to True")
            self.has_pooled_embds = True

        # detect if dataset has embds files
        self.has_embds = False
        if os.path.exists(os.path.join(self.folder_embds, self.files[0])):
            self.log_print("Has found /embds.h5 files, will only use /embds_pooled.h5")
            self.log_print("has_embds set to True")
            self.has_embds = True

        # get properties
        self.properties = Properties(folder_path=folder_path, logger=logger)

        print("self.properties = ", self.properties.category_to_cid)

        # get same id
        self.same_id = SameId(
            folder_path=folder_path, properties=self.properties, logger=logger
        )

        self.mask_reduced = torch.tensor(
            np.array(list(self.properties.tid_to_rm.values())), dtype=torch.int16
        )

        if self.logger is not None:
            self.logger.log_matrix(
                "mask_reduced.txt", self.mask_reduced.cpu().detach().numpy(), fmt="%d"
            )

        self.log_print("self.mask_reduced = ", self.mask_reduced)
        self.log_print("self.mask_reduced shape = ", self.mask_reduced.shape)

        if self.simulated:
            self.X_simulated = torch.rand(
                self.mask_reduced.shape[0], self.dim_clip_simulated
            )
            self.log_print(
                f"Simulated dataset: created dummy clip values of shape {self.X_simulated.shape}"
            )

        self.log_print(
            f"Initialised dataset with {self.properties.n_properties} properties and {self.mask_reduced.shape[0]} prompts."
        )

        if not self.simulated:
            assert len(self.files) == self.mask_reduced.shape[0]

        # get dim_x
        vector_0, mask_0 = self.__getitem__(0)
        self.dim_x = int(vector_0.size()[0])
        self.log_print(f"dim_x = {self.dim_x}")

        # truncate_embds_topk
        self.truncate_embds()

        # normalization
        self.normalize = check_yaml_params(normalize, possible_values=[MAX_MIN])
        if self.normalize is not None:
            self.get_min_max_X()

        # run checks
        run_checks_dataset(
            properties=self.properties, same_id=self.same_id, logger=self.logger
        )

    def __len__(self):
        return self.mask_reduced.shape[0]

    def _get_item_real(self, idx):
        vector1 = torch.empty(0)
        vector2 = torch.empty(0)
        # pooled embd
        if self.has_pooled_embds:
            file_pooled_path = os.path.join(self.folder_embds, self.files_pooled[idx])

            with h5py.File(file_pooled_path, "r") as f:
                vector2 = f["vector"][:]  # Read dataset
                vector2 = torch.from_numpy(vector2).to(torch.float32).flatten()

        if self.has_embds:
            # embd
            file_path = os.path.join(self.folder_embds, self.files[idx])

            with h5py.File(file_path, "r") as f:
                vector1 = f["vector"][:]  # Read dataset
                vector1 = torch.from_numpy(vector1).to(torch.float32).flatten()

        embds = torch.cat((vector1, vector2))
        if self.indices_truncate_embds_topk is not None:
            embds = embds[self.indices_truncate_embds_topk]

        if self.normalize is not None and self.normalize == MAX_MIN:
            embds = (embds - self.embds_min) / (self.embds_max - self.embds_min + 1e-6)

        return embds, self.mask_reduced[idx, :]

    def __getitem__(self, idx):
        if self.simulated:
            return self._get_item_simulated(idx)
        else:
            return self._get_item_real(idx)

    def _get_item_simulated(self, idx):
        return self.X_simulated[idx, :], self.mask_reduced[idx]

    def _get_X_simulated(self):
        self.X = self.X_simulated
        return self.X

    def _get_X_real(self):
        self.X = torch.zeros((len(self), self.dim_x))
        for i in range(1, len(self)):
            y = self.__getitem__(i)[0][None, :]
            self.X[i] = y
        self.log_print(f"X stored in memory with a shape of {self.X.shape}")
        return self.X

    def get_X(self):
        normalize_saved = self.normalize
        self.normalize = None
        if self.simulated:
            X = self._get_X_simulated()
        else:
            X = self._get_X_real()
        self.normalize = normalize_saved
        return X

    def build_property_is_the_same(self):
        pass

    def truncate_embds(self):
        if self.truncate_embds_topk is not None:
            self.log_print(
                f"Truncating embeddings from {self.dim_x} to {self.truncate_embds_topk}..."
            )
            file = f"indices_top_{self.truncate_embds_topk}.json"
            if os.path.exists(os.path.join(self.folder_path, file)):
                self.log_print(f"Found {file}")
                with open(os.path.join(self.folder_path, file), "r") as f:
                    self.indices_truncate_embds_topk = json.load(f)
            else:
                self.log_print(f"Did not find {file}. Re-calculating it...")
                self.indices_truncate_embds_topk = (
                    self.get_indices_truncate_embds_topk()
                )
                self.log_print(f"Saving {file}.")
                with open(os.path.join(self.folder_path, file), "w") as f:
                    json.dump(self.indices_truncate_embds_topk, f)
            self.dim_x = self.truncate_embds_topk
            self.log_print(f"dim_x is now {self.truncate_embds_topk}.")

    def get_indices_truncate_embds_topk(self):
        if self.X is None:
            _ = self.get_X()

        diff = torch.max(self.X, dim=0)[0] - torch.min(self.X, dim=0)[0]

        diff_argsort = diff.argsort(descending=True)

        self.log_print("top 50 diff_argsort = ", diff_argsort[:50])
        self.log_print("top 50 diff = ", diff[diff_argsort[:50]])

        self.log_print(
            "Highest diff (max - min) in new dataset : ", diff[diff_argsort[0]]
        )
        self.log_print(
            "Lowest diff (max - min) in new dataset : ",
            diff[diff_argsort[self.truncate_embds_topk]],
        )

        return diff_argsort[: self.truncate_embds_topk].tolist()

    def get_min_max_X(self):
        if self.normalize == MAX_MIN:
            self.log_print(f"Setting up {MAX_MIN} normalization")
            file_max = (
                f"embds_max_top_{self.truncate_embds_topk}.json"
                if self.truncate_embds_topk is not None
                else "embds_max.json"
            )
            file_min = (
                f"embds_min_top_{self.truncate_embds_topk}.json"
                if self.truncate_embds_topk is not None
                else "embds_min.json"
            )
            if os.path.exists(os.path.join(self.folder_path, file_max)):
                self.log_print(f"Found {file_max}")
                with open(os.path.join(self.folder_path, file_max), "r") as f:
                    self.embds_max = json.load(f)
                with open(os.path.join(self.folder_path, file_min), "r") as f:
                    self.embds_min = json.load(f)

            else:
                self.log_print(f"Did not find {file_max}. Re-calculating it...")
                self.embds_max, self.embds_min = self._get_min_max_X()

                for file, embds_extreme in zip(
                    [file_max, file_min], [self.embds_max, self.embds_min]
                ):
                    with open(os.path.join(self.folder_path, file), "w") as f:
                        json.dump(embds_extreme.tolist(), f)

            self.embds_max = torch.tensor(self.embds_max, dtype=torch.float32)
            self.embds_min = torch.tensor(self.embds_min, dtype=torch.float32)

    def _get_min_max_X(self):
        if self.X is None:
            _ = self.get_X()

        if self.indices_truncate_embds_topk is not None:
            return (
                torch.max(self.X[:, self.indices_truncate_embds_topk], dim=0)[0],
                torch.min(self.X[:, self.indices_truncate_embds_topk], dim=0)[0],
            )

        return torch.max(self.X, dim=0)[0], torch.min(self.X, dim=0)[0]

    def denormalize(self, vector):
        if self.normalize == MAX_MIN:
            return vector * (self.embds_max - self.embds_min) + self.embds_min
        else:
            raise Exception("Wrong normalization")
