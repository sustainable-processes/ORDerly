"""Hyperparameter tuning script"""
import json
import multiprocessing as mp
import random
import shutil
import string
import subprocess
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

from halton import generate_search
from yaml import dump, load

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


def id_generator(size=6, chars=string.ascii_lowercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def construct_search_space(parameters: Dict[str, Dict]):
    search_space = {}
    conditional_search_spaces = {}
    for parameter_name, parameter_config in parameters.items():
        # Continuous search space
        if "min" in parameter_config and "max" in parameter_config:
            search_space[parameter_name] = {
                "min": parameter_config["min"],
                "max": parameter_config["max"],
                "scaling": parameter_config.get("scaling", "linear"),
                "type": parameter_config.get("type", "float"),
            }
        # Discrete search space
        if "values" in parameter_config:
            if isinstance(parameter_config["values"], list):
                values = parameter_config["values"]
            elif isinstance(parameter_config["values"], dict):
                # Create a new conditional search space if needed
                for branch, branch_config in parameter_config["values"].items():
                    branch_search_space = branch_config["params"]
                    branch_parameter_name = parameter_name, branch
                    conditional_search_spaces[branch_parameter_name] = {
                        "key": branch_config.get("key"),
                        "search_space": construct_search_space(branch_search_space)[0],
                        "num_trials": branch_config.get("num_trials", 5),
                    }
                values = list(parameter_config["values"].keys())
            else:
                raise ValueError("Values must be a list or dict")
            search_space[parameter_name] = {"feasible_points": values}
    return search_space, conditional_search_spaces


def add_conditional_searches(search_space, trials):
    for (parameter_name, branch), conditional_search_space in search_space.items():
        conditional_trials = generate_search(
            conditional_search_space["search_space"],
            num_trials=conditional_search_space["num_trials"],
        )
        key = conditional_search_space["key"]
        for i in range(len(trials)):
            # If parameter has the same value as the branch, then
            # add n-1 trials with all parameters fixed
            # at this trial's values and the conditional
            # parameters set to the conditional trial values
            # (n is the number of conditional trials)
            if trials[i][parameter_name] == branch:
                for j, conditional_trial in enumerate(conditional_trials):
                    if key:
                        conditional_trial = nest_dict(
                            {}, key.split("."), conditional_trial
                        )
                    new_trial = {**trials[i], **conditional_trial}
                    if j == 0:
                        trials[i] = new_trial
                    else:
                        trials.append(new_trial)
    return trials


def nest_dict(d, keys, value):
    """Nest a dictionary"""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
    return d


def run_search(
    sweep_config_path: str,
    start_idx: int = 0,
    sweep_id: Optional[str] = None,
    sweep_filepath: Optional[str] = None,
    trials_dir: Optional[str] = None,
    resume: bool = False,
    dry_run: bool = False,
    shuffle: bool = True,
    max_parallel: int = 1,
    delete_save_dirs: bool = False,
):
    """Run hyperparameter search

    Parameters
    ----------
    sweep_config_path : str
        Path to sweep config file
    start_idx: int
        Starting trial index. Useful if resuming a sweep.
    sweep_id: str, optional
        The sweep id. Default randomly generated
    sweep_filepath: str, optional
        Path to save sweep results
    resume: bool, optional
        Resume a sweep from sweep_filepath starting at start_idx
    dry_run : bool, optional
        Preview what commands will be run in a sweep. Default False
    shuffle : bool, optional
        Shuffle trials. Default True
    max_parallel : int, optional
        Maximum number of parallel trials. Default 1
    delete_save_dirs : bool, optional
        Delete save directories after sweep. Default False

    Notes
    -----

    """
    with open(sweep_config_path, "r") as f:
        sweep_config = load(f, Loader=Loader)
    num_trials = sweep_config.get("num_trials", 10)

    with open(sweep_config["base_config_filepath"], "r") as f:
        base_config_all = load(f, Loader=Loader)
    key = sweep_config["key"]
    # base_config = base_config_all[key]
    if key in base_config_all:
        base_config = base_config_all[key]
    else:
        keys = key.split(".")
        base_config = base_config_all
        for k in keys:
            if k in base_config:
                base_config = base_config[k]
            else:
                raise ValueError(f"Key {key} not found in base config")

    # Halton quasi-random search
    if sweep_filepath and resume:
        with open(sweep_filepath, "r") as f:
            trials = json.load(f)
    elif "params" in sweep_config:
        search_space, conditional_search_spaces = construct_search_space(
            sweep_config["params"]
        )
        print("Generating search space")
        trials = generate_search(search_space, num_trials=num_trials)
        trials = add_conditional_searches(conditional_search_spaces, trials)
        for trial in trials:
            for param_name in sweep_config["params"]:
                if "key" in sweep_config["params"][param_name]:
                    param_key = sweep_config["params"][param_name]["key"]
                    new_param_dict = nest_dict(
                        {}, param_key.split("."), {param_name: trial[param_name]}
                    )
                    base_key = list(new_param_dict.keys())[0]
                    if base_key in trial:
                        trial[base_key].update(new_param_dict[base_key])
                    else:
                        trial.update(new_param_dict)
                    del trial[param_name]
        if sweep_filepath:
            with open(sweep_filepath, "w") as f:
                json.dump(trials, f)
    else:
        raise ValueError("No parameters found in config file")

    if shuffle:
        random.shuffle(trials)

    # Sweep ID for grouping in wandb interface
    if not sweep_id:
        sweep_id = id_generator()
        sweep_id = f"sweep_{sweep_id}"
        print("Starting sweep ", sweep_id)
    else:
        print("Resuming sweep ", sweep_id)

    # Create config files and run
    if not trials_dir:
        trials_dir = f"conf/local/sweeps/{sweep_id}"
    config_dir = Path(trials_dir)
    config_dir.mkdir(exist_ok=True, parents=True)
    results = []
    if not dry_run:
        pool = mp.Pool(max_parallel)
    else:
        pool = None
    for i, trial in enumerate(trials[start_idx:]):
        print("Running trial", i)

        # Override base config parameters
        trial_config = {}
        trial_config["pipeline"] = sweep_config["pipeline"]
        params = deepcopy(base_config)
        for param_name, value in trial.items():
            params[param_name] = value

        # Add wandb kwargs
        if "wandb_kwargs" in params:
            params["wandb_kwargs"]["group"] = sweep_id
            # params["wandb_kwargs"]["trial"] = i
        else:
            params["wandb_kwargs"] = {"group": sweep_id}
        params["save_dir"] = f"data/07_model_output/{sweep_id}/trial_{i}"

        # Create kedro config
        trial_config["params"] = {key: params}
        run_config = {"run": trial_config}
        kedro_config_path = config_dir / f"trial_{i}.yml"
        with open(kedro_config_path, "w") as f:
            f.write(dump(run_config, Dumper=Dumper))

        # Run trial
        cmd = ["kedro", "run", "--config", str(kedro_config_path)]
        if dry_run:
            print(cmd)
        if pool:
            results.append(
                pool.apply_async(
                    run_cmd,
                    args=(cmd, params["save_dir"]),
                    kwds={"delete_save_dirs": delete_save_dirs},
                ),
            )

    if pool:
        for result in results:
            result.wait()


def run_cmd(cmd: str, save_dir, delete_save_dirs=False):
    # cmd = " ".join(cmd)
    print(cmd)
    print(Path.cwd())
    subprocess.run(cmd, shell=False)
    if delete_save_dirs:
        delete_dirs = [save_dir, "wandb", "artifacts", "lightning_logs"]
        for delete_dir in delete_dirs:
            delete_dir = Path(delete_dir)
            if delete_dir.exists():
                shutil.rmtree(delete_dir)


if __name__ == "__main__":
    parser = ArgumentParser(prog="Hyperparameter search")
    parser.add_argument(
        "--sweep_config_path",
        type=str,
        default="conf/base/sweeps/pyg.yml",
        help="Path to sweep config file",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Index of first trial to run",
    )
    parser.add_argument(
        "--sweep_filepath",
        type=str,
        default=None,
        help="Path to save sweep results",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="Sweep ID for grouping in wandb interface",
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=2,
        help="Maximum number of parallel trials",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands instead of running",
    )
    parser.add_argument(
        "--delete_save_dirs",
        action="store_true",
        help="Delete save directories after each trial. Only run if max_parallel=1",
        default=False,
    )
    run_search(**vars(parser.parse_args()))
