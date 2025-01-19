import torch
import torch.nn as nn
from tabpfn.utils import load_model_criterion_config
from typing import Literal

class EasyTabPFNV2(nn.Module):
    def __init__(self, task: Literal["cls", "reg"], seed: int = 42):
        super().__init__()
        self.task = task
        if task == "cls":
            self.model, _, _ = load_model_criterion_config(
                model_path="/home/jeremy/.cache/tabpfn/tabpfn-v2-classifier.ckpt",
                check_bar_distribution_criterion=False,
                cache_trainset_representation=False,
                which="classifier",
                version="v2",
                download=True,
                model_seed=42,
            )
            self.bar_distribution = None
        else:
            self.model, bardist, _ = load_model_criterion_config(
                model_path="/home/jeremy/.cache/tabpfn/tabpfn-v2-regressor.ckpt",
                check_bar_distribution_criterion=True,
                cache_trainset_representation=False,
                which="regressor",
                version="v2",
                download=True,
                model_seed=seed,
            )
            self.bar_distribution = bardist

    def forward(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        task: Literal["cls", "reg"],  # classification or regression
    ) -> torch.Tensor:
        eval_pos = y_src.shape[1]
        
        # switch from batch first to batch second
        x_src, y_src = x_src.transpose(0, 1), y_src.transpose(0, 1)
        output = self.model(train_x=x_src[:eval_pos], train_y=y_src, test_x=x_src[eval_pos:])
        output = output.transpose(0, 1)

        return output


if __name__ == '__main__':
    from sklearn.datasets import load_iris, load_breast_cancer
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    # test binary classification
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = EasyTabPFNV2("cls")
    clf.to('cuda:0')

    X_train, X_test, y_train = torch.Tensor(X_train).unsqueeze(0), torch.Tensor(X_test).unsqueeze(0), torch.Tensor(y_train).unsqueeze(0)
    X_train, X_test, y_train = X_train.cuda(), X_test.cuda(), y_train.cuda()

    prediction_logits = clf(torch.cat([X_train, X_test], dim=1), y_train, task="cls")
    prediction_probabilities = torch.nn.functional.softmax(prediction_logits[:, :, :2] / 0.9, dim=-1).squeeze(0)
    prediction_probabilities = prediction_probabilities.detach().cpu().numpy()

    print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))
    predictions = prediction_probabilities.argmax(1)
    print("Accuracy", accuracy_score(y_test, predictions))

    # test multi-class classification
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
    )
    X_train, X_test, y_train = torch.Tensor(X_train).unsqueeze(0), torch.Tensor(X_test).unsqueeze(0), torch.Tensor(y_train).unsqueeze(0)
    X_train, X_test, y_train = X_train.cuda(), X_test.cuda(), y_train.cuda()

    prediction_logits = clf(torch.cat([X_train, X_test], dim=1), y_train, task="cls")
    prediction_probabilities = torch.nn.functional.softmax(prediction_logits[:, :, :len(torch.unique(y_train))] / 0.9, dim=-1).squeeze(0)
    prediction_probabilities = prediction_probabilities.detach().cpu().numpy()

    print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities, multi_class="ovr"))
    predictions = prediction_probabilities.argmax(1)
    print("Accuracy", accuracy_score(y_test, predictions))