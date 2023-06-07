"""Hyperparameter tuning script"""
import json
import multiprocessing as mp
import random
import shutil
import string
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Union

import click
from sweep.halton import generate_search
from yaml import Dumper, Loader, dump, load


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


def run_sweep(
    sweep_config_path: Union[str, Path],
    start_idx: int = 0,
    sweep_id: Optional[str] = None,
    commands_filepath: Optional[Union[str, Path]] = None,
    resume: bool = False,
    dry_run: bool = False,
    shuffle: bool = True,
    max_parallel: int = 1,
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
    commands_filepath: str, optional
        Path to to save commands for each trial. Defaults to
        the sweep config filename with "_commands.txt" appended
    resume: bool, optional
        Resume a sweep from sweep_filepath starting at start_idx
    dry_run : bool, optional
        Preview what commands will be run in a sweep. Default False
    shuffle : bool, optional
        Shuffle trials. Default True
    max_parallel : int, optional
        Maximum number of parallel trials. Default 1


    Notes
    -----

    """
    with open(sweep_config_path, "r") as f:
        sweep_config = load(f, Loader=Loader)
    num_trials = sweep_config.get("num_trials", 10)
    base_command = sweep_config["base_command"]

    if commands_filepath and resume:
        with open(commands_filepath, "r") as f:
            lines = f.readlines()
        sweep_id = lines[0].strip()
        cmds = [line.strip() for line in lines[1:]]
        print(f"Resuming sweep {sweep_id} from trial {start_idx}")
        run_commands(
            cmds, dry_run=dry_run, max_parallel=max_parallel, start_idx=start_idx
        )
        return
    elif resume and not commands_filepath:
        raise ValueError("Must provide commands_filepath to resume a sweep")

    # Halton quasi-random search
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

    if shuffle:
        random.shuffle(trials)

    # Sweep ID for grouping in wandb interface
    if not sweep_id:
        sweep_id = id_generator()
        sweep_id = f"sweep_{sweep_id}"

    # Create commands
    cmds = []
    for i, trial in enumerate(trials[start_idx:]):
        # Create command line
        cmd = deepcopy(base_command)
        for param_name, value in trial.items():
            cmd += f" --{param_name}={value}"

        # Add wandb group
        cmd += f" --wandb_group={sweep_id}"

        # Run trial
        cmds.append(cmd)

    if commands_filepath is None:
        commands_filepath_pathlib = Path("sweeps")
        commands_filepath_pathlib.mkdir(exist_ok=True)
        sweep_config_pathlib = Path(sweep_config_path)
        commands_filepath_pathlib = (
            commands_filepath_pathlib / f"{sweep_config_pathlib.stem}_commands.txt"
        )
        commands_filepath = commands_filepath_pathlib
    with open(commands_filepath, "w") as f:
        f.writelines([sweep_id + "\n"] + [cmd + "\n" for cmd in cmds])

    print("Starting sweep ", sweep_id)
    run_commands(cmds, dry_run=dry_run, max_parallel=max_parallel)


def run_commands(cmds, dry_run: bool = False, max_parallel: int = 1, start_idx=0):
    if not dry_run:
        pool = mp.Pool(max_parallel)
    else:
        pool = None

    results = []
    idx = start_idx
    for cmd in cmds[start_idx:]:
        if dry_run:
            print(f"Running trial {idx}")
            print(cmd)
        if pool:
            results.append(
                pool.apply_async(
                    run_cmd,
                    args=(cmd, idx),
                )
            )
        idx += 1

    if pool:
        for result in results:
            result.get()


def run_cmd(cmd: str, trial_idx: int):
    print(f"Running trial {trial_idx}")
    print(cmd)
    subprocess.run(cmd, shell=True)


@click.command()
@click.argument(
    "sweep_config_path",
    type=str,
    # help="Path to sweep config file",
)
@click.option(
    "--start_idx",
    type=int,
    default=0,
    help="Index of first trial to run",
)
@click.option(
    "--commands_filepath",
    type=str,
    default=None,
    help="Path to save sweep results",
)
@click.option(
    "--sweep_id",
    type=str,
    default=None,
    help="Sweep ID for grouping in wandb interface",
)
@click.option(
    "--max_parallel",
    type=int,
    default=2,
    help="Maximum number of parallel trials",
)
@click.option(
    "--dry_run",
    is_flag=True,
    help="Print commands instead of running",
)
@click.option(
    "--resume",
    default=False,
    is_flag=True,
    help="Resume a sweep from sweep_filepath starting at start_idx",
)
def run_sweep_click(
    sweep_config_path: str,
    start_idx: int,
    commands_filepath: str,
    sweep_id: str,
    max_parallel: int,
    dry_run: bool,
    resume: bool,
):
    run_sweep(
        sweep_config_path=sweep_config_path,
        start_idx=start_idx,
        commands_filepath=commands_filepath,
        sweep_id=sweep_id,
        max_parallel=max_parallel,
        dry_run=dry_run,
        resume=resume,
    )
