import torch
import torch.nn as nn
from tabpfn.utils import load_model_criterion_config
from typing import Literal
from tabpfn.model.bar_distribution import FullSupportBarDistribution
from tabpfn.model.multi_head_attention import MultiHeadAttention

def reset_weights_recursively(module):
    if len(list(module.children())) == 0:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        elif isinstance(module, MultiHeadAttention):
            # manually reset the parameters of the MultiHeadAttention module
            module = MultiHeadAttention(
                input_size=module._input_size,
                output_size=module._output_size,
                d_k=module._d_k,
                d_v=module._d_v,
                nhead=module._nhead,
                device=module._device,
                dtype=module._dtype,
                dropout_p=module.dropout_p,
                softmax_scale=module.softmax_scale,
                recompute=module.recompute,
                init_gain=module.init_gain,
                two_sets_of_queries=module.two_sets_of_queries,
            )
        return

    for child in module.children():
        reset_weights_recursively(child)

class EasyTabPFNV2(nn.Module):
    def __init__(self, pretrained=False, seed: int = 42):
        super().__init__()
        self.cls_model, _, _ = load_model_criterion_config(
            model_path=".cache/tabpfn/tabpfn-v2-classifier.ckpt",
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2",
            download=True,
            model_seed=seed,
        )
        self.reg_model, self.bardist_, _ = load_model_criterion_config(
            model_path=".cache/tabpfn/tabpfn-v2-regressor.ckpt",
            check_bar_distribution_criterion=True,
            cache_trainset_representation=False,
            which="regressor",
            version="v2",
            download=True,
            model_seed=seed,
        )
        
        if not pretrained:
            reset_weights_recursively(self.cls_model)
            reset_weights_recursively(self.reg_model)

    def forward(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        task: Literal["cls", "reg"],  # classification or regression
        return_logits: bool = False,
    ) -> torch.Tensor:
        eval_pos = y_src.shape[1]
        
        # switch from batch first to batch second
        x_src, y_src = x_src.transpose(0, 1), y_src.transpose(0, 1).squeeze(2)
        
        if task == "reg":
            self.bar_dists_criteria = []
            for b in range(x_src.shape[1]):
                y_train_mean_ = torch.mean(y_src[:, b])
                y_train_std_ = torch.std(y_src[:, b])
                y_src[:, b] = (y_src[:, b] - y_train_mean_) / y_train_std_
                self.bar_dists_criteria.append(FullSupportBarDistribution(
                    self.bardist_.borders * y_train_std_ + y_train_mean_,
                ).float())
        
        logits = self._forward(x_src, y_src, task, eval_pos)
        
        if task == "reg":
            output = []
            for b in range(x_src.shape[1]):
                output.append(self.bar_dists_criteria[b].mean(logits[:, b]))
            output = torch.stack(output, dim=1)
        else:
            output = logits
        
        output = output.transpose(0, 1)
        if return_logits:
            return output, logits.transpose(0, 1)
        return output
    
    def _forward(
        self, 
        x_src: torch.Tensor, 
        y_src: torch.Tensor, 
        task: Literal["cls", "reg"],
        eval_pos: int,
    ) -> torch.Tensor:
        model = self.cls_model if task == "cls" else self.reg_model
        output = model(train_x=x_src[:eval_pos], train_y=y_src, test_x=x_src[eval_pos:])
        return output

    def get_reg_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        result = []
        for b in range(logits.shape[0]):
            result.append(self.bar_dists_criteria[b](logits[b], y[b]))
        return torch.stack(result, dim=0)

if __name__ == '__main__':
    from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    # test binary classification
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = EasyTabPFNV2(pretrained=False)
    model.to('cuda:0')

    X_train, X_test, y_train = torch.Tensor(X_train).unsqueeze(0), torch.Tensor(X_test).unsqueeze(0), torch.Tensor(y_train).unsqueeze(0)
    X_train, X_test, y_train = X_train.cuda(), X_test.cuda(), y_train.cuda()

    prediction_logits = model(torch.cat([X_train, X_test], dim=1), y_train, task="cls")
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

    prediction_logits = model(torch.cat([X_train, X_test], dim=1), y_train, task="cls")
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

    # Predict a point estimate (using the mean)
    predictions, logits = model(torch.cat([X_train, X_test], dim=1), y_train, task="reg", return_logits=True)
    predictions = predictions.detach().cpu().numpy().squeeze()[0]
    
    loss = model.get_reg_loss(logits, torch.Tensor(y_test).unsqueeze(0).repeat(16, 1).cuda())
    print(f'Mean Loss: {loss.mean().item()}')

    print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
    print("R-squared (R^2):", r2_score(y_test, predictions))

