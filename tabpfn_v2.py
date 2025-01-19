import torch
import torch.nn as nn
from tabpfn.utils import load_model_criterion_config
from typing import Literal
from tabpfn.model.bar_distribution import FullSupportBarDistribution

class EasyBarDist(nn.Module):
    def __init__(self, borders: torch.Tensor):
        super().__init__()
        self.borders = borders
    
    @property
    def bucket_widths(self) -> torch.Tensor:
        return self.borders[1:] - self.borders[:-1]

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        p = torch.softmax(logits, -1)
        side_normals = (
            FullSupportBarDistribution.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            FullSupportBarDistribution.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        bucket_means[0] = -side_normals[0].mean + self.borders[1]
        bucket_means[-1] = side_normals[1].mean + self.borders[-2]

        return torch.einsum("lbc,cb->lb", p, bucket_means.to(logits.device).type(logits.dtype))

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
        else:
            self.model, self.bardist_, _ = load_model_criterion_config(
                model_path="/home/jeremy/.cache/tabpfn/tabpfn-v2-regressor.ckpt",
                check_bar_distribution_criterion=True,
                cache_trainset_representation=False,
                which="regressor",
                version="v2",
                download=True,
                model_seed=seed,
            )

    def forward(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        task: Literal["cls", "reg"],  # classification or regression
    ) -> torch.Tensor:
        eval_pos = y_src.shape[1]
        
        # switch from batch first to batch second
        x_src, y_src = x_src.transpose(0, 1), y_src.transpose(0, 1)
        if task == "reg":
            # Standardize y
            mean = torch.mean(y_src, dim=0, keepdim=True)
            std = torch.std(y_src, dim=0, keepdim=True)
            y_src = (y_src - mean) / std
            self.renormalized_criterion_ = EasyBarDist(
                self.bardist_.borders[:, None] * std + mean,
            )
        
        output = self.model(train_x=x_src[:eval_pos], train_y=y_src, test_x=x_src[eval_pos:])

        if task == "reg":
            output = self.renormalized_criterion_.mean(output)
        
        output = output.transpose(0, 1)
        return output


if __name__ == '__main__':
    from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

    # regression
    # Load data
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
    )
    X_train, X_test, y_train = torch.Tensor(X_train).unsqueeze(0), torch.Tensor(X_test).unsqueeze(0), torch.Tensor(y_train).unsqueeze(0)
    X_train, X_test, y_train = X_train.cuda(), X_test.cuda(), y_train.cuda()

    X_train, X_test, y_train = X_train.repeat(16, 1, 1), X_test.repeat(16, 1, 1), y_train.repeat(16, 1)

    # Initialize a regressor
    reg = EasyTabPFNV2("reg")
    reg.to('cuda:0')

    # Predict a point estimate (using the mean)
    predictions = reg(torch.cat([X_train, X_test], dim=1), y_train, task="reg").detach().cpu().numpy().squeeze()[0]

    print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
    print("R-squared (R^2):", r2_score(y_test, predictions))

