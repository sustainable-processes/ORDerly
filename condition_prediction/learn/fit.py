import typing
import numpy as np
import torch
import torchmetrics
from tqdm import trange

import param_sharing.learn.util
from torch.utils.tensorboard import SummaryWriter


def get_batch_size(size, length):
    if isinstance(size, int):
        return size / length
    elif isinstance(size, float):
        assert (0.0 < size) & (size <= 1.0)
        return size


def train_loop(
    model,
    train_data,
    *,
    epochs,
    batch_size: typing.Union[float, int],
    loss_fn,
    optimizer,
    targets,
    val_data=None,
    train_kwargs: dict = {},
    val_kwargs: dict = {},
    train_eval: bool = True,
    write_summary: bool = True,
):
    if write_summary:
        writer = SummaryWriter(
            log_dir=param_sharing.learn.util.log_dir(prefix="TORCH_")
        )

    train_size = train_data["product_fp"].shape[0]
    batch_size = get_batch_size(batch_size, length=train_size)

    acc_metrics = {}
    losses = {"sum": {"train": [], "val": [], "train_eval": []}}
    for target in targets:
        losses[target] = {"train": [], "val": [], "train_eval": []}
        if target == "temperature":
            continue
        num_classes = train_data[target].shape[1]
        acc_metrics[target] = {
            "top1": {
                "metric": torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=1
                ),
                "train": [],
                "val": [],
                "train_eval": [],
                "train_batch": [],
            },
            "top3": {
                "metric": torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=3
                ),
                "train": [],
                "val": [],
                "train_eval": [],
                "train_batch": [],
            },
            "top5": {
                "metric": torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=5
                ),
                "train": [],
                "val": [],
                "train_eval": [],
                "train_batch": [],
            },
        }

    for e in range(epochs):
        output_str = f"{e+1}/{epochs} | "

        idxes = np.arange(train_size)
        np.random.shuffle(idxes)

        prev_idx = 0
        interval = int(idxes.shape[0] * batch_size)

        # storage for use during mini-batching, reset every epoch
        epoch_losses = {"sum": {"train": []}}
        for target in targets:
            epoch_losses[target] = {"train": []}
        for target in targets:
            if target == "temperature":
                continue
            for top in ["top1", "top3", "top5"]:
                acc_metrics[target][top]["train_batch"] = []

        # run across training
        for idx in (
            t := trange(interval, idxes.shape[0] + 1, interval, desc="", leave=True)
        ):
            if batch_size < 1.0:
                batch_idxes = idxes[prev_idx:idx]
            else:
                batch_idxes = idxes
            prev_idx = idx

            pred = model.forward_dict(
                data=train_data,
                indexes=batch_idxes,
                training=True,
                **train_kwargs,
            )

            loss = 0
            for (
                target
            ) in (
                targets
            ):  # we can change targets to be loss functions in the future if the loss function changes
                target_batch_loss = loss_fn(
                    pred[target], train_data[target][batch_idxes]
                )
                factor = 1e-4 if target == "temperature" else 1.0
                loss += factor * target_batch_loss
                epoch_losses[target]["train"].append(
                    target_batch_loss.detach().numpy().item()
                )
            epoch_losses["sum"]["train"].append(loss.detach().numpy().item())

            for target in targets:
                if target == "temperature":
                    continue
                for top in ["top1", "top3", "top5"]:
                    acc_metrics[target][top]["train_batch"].append(
                        acc_metrics[target][top]["metric"](
                            pred[target], train_data[target][batch_idxes].argmax(axis=1)
                        )
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calc the average train loss
        losses["sum"]["train"].append(np.mean(epoch_losses["sum"]["train"]))
        if write_summary:
            writer.add_scalar("Loss/train/sum", losses["sum"]["train"][-1], e)
        for target in targets:
            losses[target]["train"].append(np.mean(epoch_losses[target]["train"]))
            if write_summary:
                writer.add_scalar(
                    f"Loss/train/{target}", losses[target]["train"][-1], e
                )

        # calc the average train acc
        for target in targets:
            if target == "temperature":
                continue
            for top in ["top1", "top3", "top5"]:
                acc_metrics[target][top]["train"].append(
                    np.mean(acc_metrics[target][top]["train_batch"])
                )
                if write_summary:
                    writer.add_scalar(
                        f"{top}_acc/train/{target}",
                        acc_metrics[target][top]["train"][-1],
                        e,
                    )

        output_str += f'Train loss: {losses["sum"]["train"][-1]:.3f}'

        # evaluate with train data
        if train_eval:
            with torch.no_grad():
                pred = model.forward_dict(
                    data=train_data,
                    indexes=slice(None),
                    training=False,
                    **val_kwargs,
                )

                loss = 0
                for (
                    target
                ) in (
                    targets
                ):  # we can change targets to be loss functions in the future if the loss function changes
                    target_batch_loss = loss_fn(pred[target], train_data[target])
                    factor = 1e-4 if target == "temperature" else 1.0
                    loss += factor * target_batch_loss
                    losses[target]["train_eval"].append(
                        target_batch_loss.detach().numpy().item()
                    )
                    if write_summary:
                        writer.add_scalar(
                            f"Loss/train_eval/{target}",
                            losses[target]["train_eval"][-1],
                            e,
                        )
                losses["sum"]["train_eval"].append(loss.detach().numpy().item())
                if write_summary:
                    writer.add_scalar(
                        "Loss/train_eval/sum", losses["sum"]["train_eval"][-1], e
                    )

                for target in targets:
                    if target == "temperature":
                        continue
                    for top in ["top1", "top3", "top5"]:
                        acc_metrics[target][top]["train_eval"].append(
                            acc_metrics[target][top]["metric"](
                                pred[target], train_data[target].argmax(axis=1)
                            )
                        )
                        if write_summary:
                            writer.add_scalar(
                                f"{top}_acc/train_eval/{target}",
                                acc_metrics[target][top]["train_eval"][-1],
                                e,
                            )
                output_str += (
                    f' | Train eval loss: {losses["sum"]["train_eval"][-1]:.3f} '
                )

        # evaluate with validation data
        if val_data is not None:
            with torch.no_grad():
                pred = model.forward_dict(
                    data=val_data,
                    indexes=slice(None),
                    training=False,
                    **val_kwargs,
                )

                loss = 0
                for (
                    target
                ) in (
                    targets
                ):  # we can change targets to be loss functions in the future if the loss function changes
                    target_batch_loss = loss_fn(pred[target], val_data[target])
                    factor = 1e-4 if target == "temperature" else 1.0
                    loss += factor * target_batch_loss
                    losses[target]["val"].append(
                        target_batch_loss.detach().numpy().item()
                    )
                    if write_summary:
                        writer.add_scalar(
                            f"Loss/val/{target}", losses[target]["val"][-1], e
                        )
                losses["sum"]["val"].append(loss.detach().numpy().item())
                if write_summary:
                    writer.add_scalar("Loss/val/sum", losses["sum"]["val"][-1], e)

                for target in targets:
                    if target == "temperature":
                        continue
                    for top in ["top1", "top3", "top5"]:
                        acc_metrics[target][top]["val"].append(
                            acc_metrics[target][top]["metric"](
                                pred[target], val_data[target].argmax(axis=1)
                            )
                        )
                        if write_summary:
                            writer.add_scalar(
                                f"{top}_acc/val/{target}",
                                acc_metrics[target][top]["val"][-1],
                                e,
                            )
                output_str += f' | Val loss: {losses["sum"]["val"][-1]:.3f} '
        print(output_str)
    return losses, acc_metrics
