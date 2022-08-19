import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Union, Tuple
from tqdm import tqdm
from torchmetrics import (
    F1Score,
    Precision,
    Recall,
    Accuracy,
    Dice,
    MatthewsCorrCoef,
    LabelRankingAveragePrecision,
    LabelRankingLoss,
)


class Metrics:
    def __init__(
        self,
        threshold=0.5,
        num_classes: int = None,
        _criterion=nn.BCELoss(),
        _device: torch.device = torch.device("cuda"),
    ):
        self.acc = 0.0  # accuracy
        self.prec = 0.0  # precision
        self.rec = 0.0  # recall
        self.f1 = 0.0  # f1 score
        self.dice = 0.0  # dice coefficient
        self.loss = 0.0  # loss
        self.mcc = 0.0  # matthews correlation coefficient
        self.thresh = threshold
        self.update_count = 0
        self.sample_count = 0
        self.num_classes = num_classes
        self.device = _device
        self.criterion = _criterion
        self.lrloss = 0.0  # LabelRankingLoss
        self.lrap = 0.0  # LabelRankingAveragePrecision

        # average=micro says the function to compute f1 by considering total true positives, false negatives
        #   and false positives (no matter of the prediction for each label in the dataset)
        #
        # average=macro says the function to compute f1 for each label, and returns the average without
        #   considering the proportion for each label in the dataset.
        #
        # average=weighted says the function to compute f1 for each label, and returns the average
        #   considering the proportion for each label in the dataset.
        #
        # average=samples says the function to compute f1 for each instance,
        #   and returns the average. Use it for multilabel classification.
        #
        # Source: https://stackoverflow.com/a/55759038/216953

        self._f1 = F1Score(
            num_classes=self.num_classes,
            multiclass=False,
            average="samples",
            threshold=0.5,
        ).to(self.device)

        self._mcc = MatthewsCorrCoef(
            num_classes=self.num_classes,
            threshold=0.5,
        ).to(self.device)

        self._dice = Dice(
            num_classes=self.num_classes,
            multiclass=False,
            average="samples",
            threshold=0.5,
        ).to(self.device)

        self._precision = Precision(
            num_classes=self.num_classes,
            multiclass=False,
            average="samples",
            threshold=0.5,
        ).to(self.device)

        self._recall = Recall(
            num_classes=self.num_classes,
            multiclass=False,
            average="samples",
            threshold=0.5,
        ).to(self.device)

        self._accuracy = Accuracy(
            num_classes=self.num_classes,
            multiclass=False,
            average="samples",
            threshold=0.5,
        ).to(self.device)

        self._lrloss = LabelRankingLoss(
            num_classes=self.num_classes,
            threshold=0.5,
        ).to(self.device)

        self._lrap = LabelRankingAveragePrecision(
            num_classes=self.num_classes,
            threshold=0.5,
        ).to(self.device)

        def label_error(y_pred, y_true):
            """
            Compute the label error for each class.
            When a prediction is incorrect, it accumulates the error as positive value.
            """
            return torch.logical_xor(y_pred.int(), y_true.int()).sum(dim=0)

        self.label_error = label_error

        def label_frequency(y_true: torch.Tensor):
            """
            Compute the label frequency for each class.
            when class is absent, add -1 from the frequency.
            when class is present, add 1 to the frequency.
            """
            y_true[y_true == 0] = -1  # replace 0 with -1
            return y_true.sum(dim=0)  # sum over the batch dimension and return

        self.label_frequency = label_frequency

        self.lf = torch.zeros(self.num_classes).int().to(self.device)

    def update(self, preds: torch.Tensor, _target: torch.Tensor, fed='gen'):
        sigmoid = nn.Sigmoid()
        if (fed == 'gen'):
            _preds = (preds['output'] > self.thresh).int()
            self.loss += self.criterion(preds['output'], _target.float()).item()
        else:
            _preds = (sigmoid(preds) > self.thresh).int()
            self.loss += self.criterion(preds.sigmoid(), _target.float()).item()
        self.f1 += self._f1(_preds, _target)
        self.mcc += self._mcc(_preds, _target)
        self.dice += self._dice(_preds, _target)
        self.lrap += self._lrap(_preds, _target)
        self.rec += self._recall(_preds, _target)
        self.acc += self._accuracy(_preds, _target)
        self.lrloss += self._lrloss(_preds, _target)
        self.prec += self._precision(_preds, _target)

        self.lf += self.label_frequency(_target)

        self.update_count += 1
        self.sample_count += _target.size(dim=0)  # number of samples

    def __str__(self) -> str:
        return (
            f"\tF1: {self.f1 / self.update_count:.8f}\n"
            f"\tMCC: {self.mcc / self.update_count:.8f}\n"
            f"\tDice: {self.dice / self.update_count:.8f}\n"
            f"\tPrecision: {self.prec / self.update_count:.8f}\n"
            f"\tRecall: {self.rec / self.update_count:.8f}\n"
            f"\tAccuracy: {self.acc / self.update_count:.8f}\n"
            f"\tValLoss: {self.loss / self.update_count:.8f}\n"
            f"\tLabelRankingLoss: {self.lrloss / self.update_count:.8f}\n"
            f"\tLabelRankingAveragePrecision: {self.lrap / self.update_count:.8f}\n"
            # f"\tLabelFrequency: {add_class_names(self.lf / self.sample_count)}\n"
        )

    def get(self) -> Dict[str, float]:
        return {
            "f1": self.f1 / self.update_count,
            "mcc": self.mcc / self.update_count,
            "dice": self.dice / self.update_count,
            "precision": self.prec / self.update_count,
            "recall": self.rec / self.update_count,
            "accuracy": self.acc / self.update_count,
            "val_loss": self.loss / self.update_count,
            "lrloss": self.lrloss / self.update_count,
            "lrap": self.lrap / self.update_count,
            # "label_frequency": add_class_names(self.lf / self.sample_count),
        }

# %% Evaluator

def evaluator(_model, _loader, device):
    _model.eval()  # set model to evaluation mode
    _metrics = Metrics(num_classes=19, _device=device)
    for _data, _target in tqdm(_loader, total=len(_loader), desc="Evaluator", unit="Batch", dynamic_ncols=True):
        _data = _data.type(torch.float32).to(device)
        _target = _target.type(torch.int8).to(device)
        _preds = _model(_data)
        _metrics.update(_preds, _target)
    return _metrics

