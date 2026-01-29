import numpy as np


def run_checks_dataset(properties, same_id, logger):
    log_print = print if logger is None else logger.print

    check_properties_mask_consistency(properties, log_print)


def check_properties_mask_consistency(properties, log_print):
    log_print("Checking that the reduced mask matches the prompts...")

    random_prompts_idx = [0, 10, 15]

    for idx in random_prompts_idx:
        # extract the properties text from the reduced mask
        rm = properties.tid_to_rm[idx]
        rm_nz_indices = np.where(rm)[0]
        properties_from_rm = [properties.pid_to_property[pid] for pid in rm_nz_indices]

        # extract the properties text from the tids
        pids = properties.tid_to_pids[idx]
        properties_from_tid = [properties.pid_to_property[pid] for pid in pids]

        assert properties_from_rm.sort() == properties_from_tid.sort()
