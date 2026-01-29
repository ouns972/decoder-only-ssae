import argparse

from trainable_inputs_all_clips import training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, help="output folder")
    parser.add_argument(
        "--path_yaml",
        type=str,
        help="path to the config file (yaml)",
        default="trainings/config/params_default.yaml",
    )
    parser.add_argument(
        "--overwrite_output",
        type=bool,
        help="to overwrite the output if already exists",
        default=True,
    )
    args = parser.parse_args()

    training(
        output_folder=args.output_folder,
        path_yaml=args.path_yaml,
        overwrite_output=args.overwrite_output,
    )
