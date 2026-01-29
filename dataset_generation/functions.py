import itertools
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import List

from inference.utils import flatten_dict_of_list, is_in


class PromptsGenerator:
    def __init__(
        self,
        filename_categories_with_properties: Path | str,
        output_folder: Path | str,
        n_prompts: int = 1500,
        overwrite: bool = True,
    ) -> None:
        self.filename_categories_with_properties = filename_categories_with_properties
        self.output_folder = output_folder
        self.n_prompts = n_prompts
        self.overwrite = overwrite
        self.seed = 0

        self.create_output_folder()

        self.read_categories_with_properties()
        self.get_categories_with_properties_list()

        self.generate_all_permuations()
        self.shuffle()
        self.truncate()

        self.format_output()
        self.write()

        self.add_metadata()

    def create_output_folder(self) -> None:
        if os.path.exists(self.output_folder):
            if self.overwrite:
                shutil.rmtree(self.output_folder)
            else:
                raise Exception(f"Folder {self.output_folder} already exists.")

        os.mkdir(self.output_folder)

    def read_categories_with_properties(self) -> None:
        with open(self.filename_categories_with_properties, "r") as f:
            self.dict_categories_with_properties = json.load(f)

        self.properties = flatten_dict_of_list(self.dict_categories_with_properties)

    def get_categories_with_properties_list(self) -> List[List[str]]:
        self.categories_with_properties_list = []
        for k, v in self.dict_categories_with_properties.items():
            self.categories_with_properties_list.append(v)

    @property
    def n_properties(self) -> int:
        return len(self.properties)

    @property
    def n_categories(self) -> int:
        return len(list(self.dict_categories_with_properties.keys()))

    def generate_all_permuations(self) -> None:
        print(
            f"Generating prompts for {self.n_categories} categories and {self.n_properties} properties..."
        )
        combinations = list(itertools.product(*self.categories_with_properties_list))
        self.prompts_list = [", ".join(map(str, combo)) for combo in combinations]
        print(f"Done generated {len(self.prompts_list)} prompts")

    def shuffle(self) -> None:
        random.seed(0)
        random.shuffle(self.prompts_list)

    def truncate(self) -> None:
        self.n_prompts_not_truncated = len(self.prompts_list)
        self.prompts_list = self.prompts_list[: self.n_prompts]

    def format_output(self) -> None:
        self.prompts_formatted = []
        for i, prompt in enumerate(self.prompts_list):
            self.prompts_formatted.append(
                {"id": i, "prompt": prompt, "features": None}  # maybe for later
            )

    def write(self) -> None:
        with open(Path(self.output_folder, "prompts.json"), "w") as json_file:
            json.dump(self.prompts_formatted, json_file, indent=4)

    def add_metadata(self) -> None:
        metadata = {
            "n_promts": len(self.prompts_list),
            "n_promts_not_truncated": self.n_prompts_not_truncated,
            "seed": self.seed,
        }

        with open(Path(self.output_folder, "metadata.json"), "w") as json_file:
            json.dump(metadata, json_file, indent=4)
