from condition_prediction.run import ConditionPrediction
import wandb
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from tqdm import tqdm
import gc

api = wandb.Api()
wandb_entity = "ceb-sre"
wandb_project = "orderly"

# Loop through all relevant runs on wandb to get run_ids, datasets and random seeds
# For each rerun the conditionprediction with skip_training=True and resume=True
DATASETS = [
    "with_trust_with_map",
    "with_trust_no_map",
    "no_trust_no_map",
    "no_trust_with_map",
]
BASE_PATH = pathlib.Path("/project/studios/orderly-preprocessing/ORDerly/")
DATASETS_PATH = BASE_PATH / "data/orderly/datasets/"
MODEL_PATH = pathlib.Path("models/")
configs = []
# for random_seed in [98765]:
for dataset in DATASETS:
    filters = {
        "state": "finished",
        "config.output_folder_path": {
            "$in": [
                f"models/{dataset}",
                str(MODEL_PATH / dataset),
                f"/Users/Kobi/Documents/Research/phd_code/ORDerly/models/{dataset}",
            ],
        },
        # "config.random_seed": random_seed,
        # "config.train_fraction": 1.0,
        # Switching back to v3 for the paper
        "config.dataset_version": "v3",
        # "config.train_mode": 0,  # Teacher forcing
    }
    runs = api.runs(f"{wandb_entity}/{wandb_project}", filters=filters)
    # if not len(runs) == 5: # For 5 training fractions
    #     raise ValueError(f"Not 5 runs for {dataset} (found {len(runs)}, seed {random_seed})")

    for run in runs:
        config = dict(run.config)
        train_data_path = pathlib.Path(
            f"{DATASETS_PATH}/orderly_{dataset}_train.parquet"
        )
        test_data_path = pathlib.Path(f"{DATASETS_PATH}/orderly_{dataset}_test.parquet")
        fp_directory = train_data_path.parent / "fingerprints"
        train_fp_path = fp_directory / (train_data_path.stem + ".npy")
        test_fp_path = fp_directory / (test_data_path.stem + ".npy")
        output_folder_path = MODEL_PATH / dataset
        output_folder_path.mkdir(parents=True, exist_ok=True)
        tags = dataset.split("_")
        tags = [f"{tags[0]}_{tags[1]}", f"{tags[2]}_{tags[3]}"]
        config.update(
            {
                "train_data_path": train_data_path,
                "test_data_path": test_data_path,
                "train_fp_path": train_fp_path,
                "test_fp_path": test_fp_path,
                "output_folder_path": output_folder_path,
                "skip_training": True,
                "resume": True,
                "resume_from_best": True,
                "generate_fingerprints": False,
                "wandb_run_id": run.id,
                "wandb_tags": tags,
            }
        )
        configs.append(config)
        del config["n_val"]
        del config["n_test"]
        del config["n_train"]
        instance = ConditionPrediction(**config)
        instance.run_model_arguments()
        wandb.finish()
        gc.collect()
