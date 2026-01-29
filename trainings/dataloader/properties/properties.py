import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from trainings.utils.logger import Logger


def get_mapping_pid_to_property(
    dict_categories_with_properties: Dict[str, str],
) -> tuple[Dict[str, int], Dict[int, str], int]:
    property_to_pid = {}
    pid_to_property = {}
    s = 0
    for _, value in dict_categories_with_properties.items():
        for v in value:
            property_to_pid[v] = s
            pid_to_property[s] = v
            s += 1
    n_properties = s
    return property_to_pid, pid_to_property, n_properties


def get_mapping_cid_to_category(
    dict_categories_with_properties: Dict[str, str],
) -> tuple[Dict[str, int], Dict[int, str], int]:
    category_to_cid = {}
    cid_to_category = {}
    s = 0
    for key, _ in dict_categories_with_properties.items():
        category_to_cid[key] = s
        cid_to_category[s] = key
        s += 1
    n_categories = s
    return category_to_cid, cid_to_category, n_categories


def get_tid_to_rm(
    prompts: List[Dict], property_to_pid: Dict[str, int], n_properties: int
) -> Dict[int, np.array]:
    # mapping between prompt (tid) and reduced mask (rm)

    tid_to_rm = {}
    for tid, prompt in enumerate(prompts):
        prompt = prompt["prompt"]

        mask_reduced = np.zeros(n_properties, dtype=np.int32)
        for f in prompt.split(", "):
            if f in property_to_pid.keys():
                mask_reduced[property_to_pid[f]] = 1

        tid_to_rm[tid] = mask_reduced

    return tid_to_rm


def get_tid_to_pids(
    prompts: List[Dict], property_to_pid: Dict[str, int], n_properties: int
) -> Dict[int, np.array]:
    # mapping between prompt (tid) to list of pids

    tid_to_pids = {}
    for tid, prompt in enumerate(prompts):
        prompt = prompt["prompt"]

        pids = []
        for f in prompt.split(", "):
            if f in property_to_pid.keys():
                pids.append(property_to_pid[f])

        tid_to_pids[tid] = sorted(pids)  # ordered by categories

    return tid_to_pids


def get_cid_to_pids(
    dict_categories_with_properties: Dict[str, str],
    property_to_pid: Dict[str, int],
    category_to_cid: Dict[str, int],
) -> tuple[Dict[int, np.array], Dict[int, int]]:
    # example cid_to_pids : {0: [0,1,2,3], 1: [4,5]} meaning that category 0 has for properties 0,1,2 and 3.
    # example pid_to_cid : {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2}

    cid_to_pids = {}
    pid_to_cid = {}
    for category, properties in dict_categories_with_properties.items():
        cid_to_pids[category_to_cid[category]] = [
            property_to_pid[property] for property in properties
        ]

        for property in properties:
            pid_to_cid[property_to_pid[property]] = category_to_cid[category]

    return cid_to_pids, pid_to_cid


class Properties:
    def __init__(self, folder_path: Path | str, logger: Logger = None):
        # tid : text (prompt) id
        # pid : property id
        # cid : category id
        # rm: reduced mask

        # cids and pids start at 0

        self.folder_path = folder_path
        self.filename_categories_with_properties = Path(
            self.folder_path, "properties.json"
        )
        self.filename_prompts = Path(self.folder_path, "prompts.json")
        self.logger = logger
        self.log_print = print if self.logger is None else logger.print

        self.log_print("Initializing Properties class...")

        self.reader()
        self.get_mappings()

        # log values
        self.log_print("property_to_pid : ", self.property_to_pid)
        self.log_print("category_to_cid : ", self.category_to_cid)
        self.log_print("cid_to_pids : ", self.cid_to_pids)
        self.log_print("n_properties : ", self.n_properties)
        self.log_print("n_categories : ", self.n_categories)

        self.log_print("Properties class initialized.")

    def reader(self) -> None:
        with open(self.filename_categories_with_properties, "r") as f:
            self.dict_categories_with_properties = json.load(f)

        with open(self.filename_prompts, "r") as f:
            self.prompts = json.load(f)

    def get_mappings(self) -> None:
        self.property_to_pid, self.pid_to_property, self.n_properties = (
            get_mapping_pid_to_property(self.dict_categories_with_properties)
        )

        self.category_to_cid, self.cid_to_category, self.n_categories = (
            get_mapping_cid_to_category(self.dict_categories_with_properties)
        )

        self.tid_to_rm = get_tid_to_rm(
            self.prompts, self.property_to_pid, self.n_properties
        )

        self.tid_to_pids = get_tid_to_pids(
            self.prompts, self.property_to_pid, self.n_properties
        )

        self.cid_to_pids, self.pid_to_cid = get_cid_to_pids(
            self.dict_categories_with_properties,
            self.property_to_pid,
            self.category_to_cid,
        )
