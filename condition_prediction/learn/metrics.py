import typing
import torchmetrics


def get_topk_acc(pred, true, k: typing.Union[typing.List[int], int] = 1):
    # we can either pass in a single k and get the acc or pass in an
    # iterable containing k and get a dictionary of accs
    if isinstance(k, int):
        metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=pred.shape[1], top_k=k
        )
        acc = metric(pred, true.argmax(axis=1))
        return acc
    else:
        return {_k: get_topk_acc(pred=pred, true=true, k=_k) for _k in k}
