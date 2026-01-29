import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Set

import yaml


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
    seen_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    if seen_keys is None:
        seen_keys = set()

    params = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        seen_keys.add(new_key)

        if isinstance(v, dict):
            params.update(flatten_dict(v, new_key, sep=sep, seen_keys=seen_keys))
        else:
            params[new_key] = v

    params = {k.split(".")[-1]: v for k, v in params.items()}
    return params


def read_training_params_from_yaml(
    path_yaml: Optional[str | Path] = None,
) -> tuple[Dict[str, Any], Path | str]:
    if path_yaml is None:
        path_yaml = Path("trainings/config/params_default.yaml")

    with open(path_yaml, "r") as f:
        yaml_params = yaml.safe_load(f)

    return flatten_dict(yaml_params), path_yaml


def initialise_instance(model, all_possible_args):
    """initialise a function or a class"""
    sig = inspect.signature(model)
    args_model = list(sig.parameters.keys())
    filtered_kwargs = {k: v for k, v in all_possible_args.items() if k in args_model}
    return model(**filtered_kwargs)


def check_yaml_params(value, possible_values):
    if value is not None and isinstance(value, str) and value not in possible_values:
        raise Exception(f"Wrong value {value}.Mmust be in {possible_values}")
    return value
