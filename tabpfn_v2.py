from typing import Literal

import torch
import torch.nn as nn
from tabpfn.model.bar_distribution import FullSupportBarDistribution
from tabpfn.model.multi_head_attention import MultiHeadAttention
from tabpfn.utils import load_model_criterion_config
from tabpfn.model.bar_distribution import get_bucket_limits


def reset_weights_recursively(module: nn.Module):
    if len(list(module.children())) == 0:
        if hasattr(module, "reset_parameters"):
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
    def __init__(self, pretrained=False, seed: int = 42, num_buckets: int = 5000):
        super().__init__()
        model_dir = ".cache/tabpfn/"
        # cls_path = model_dir + "tabpfn-v2-classifier.ckpt"
        reg_path = model_dir + "tabpfn-v2-regressor.ckpt"
        # self.cls_model, _, _ = load_model_criterion_config(
        #     model_path=cls_path,
        #     check_bar_distribution_criterion=False,
        #     cache_trainset_representation=False,
        #     which="classifier",
        #     version="v2",
        #     download=True,
        #     model_seed=seed,
        # )
        self.reg_model, self.bardist_, _ = load_model_criterion_config(
            model_path=reg_path,
            check_bar_distribution_criterion=True,
            cache_trainset_representation=False,
            which="regressor",
            version="v2",
            download=True,
            model_seed=seed,
        )
        self.num_features = None
        self.num_buckets = num_buckets

        if not pretrained:
            # reset_weights_recursively(self.cls_model)
            reset_weights_recursively(self.reg_model)

            self.reg_model.decoder_dict.standard[2] = torch.nn.Linear(
                self.reg_model.decoder_dict.standard[2].in_features, num_buckets
            )

        custom_borders = get_bucket_limits(num_outputs=self.num_buckets, ys=torch.empty(1000 * self.num_buckets).uniform_(-6, 6))
        self.bar_dists_criteria = FullSupportBarDistribution(custom_borders,).float()
        

    def forward(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        task: Literal["cls", "reg"],  # classification or regression
        return_logits: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        eval_pos = y_src.shape[1]

        # switch from batch first to batch second
        x_src, y_src = x_src.transpose(0, 1), y_src.transpose(0, 1).squeeze(2)

        if task == "reg":
            y_train_mean_ = torch.mean(y_src)
            y_train_std_ = torch.std(y_src)
            y_src = (y_src - y_train_mean_) / y_train_std_

        logits = self._forward(x_src, y_src, task, eval_pos)

        if task == "reg":
            output = self.bar_dists_criteria.mean(logits[:, 0], temperature=temperature).unsqueeze(0)
            output = output * y_train_std_ + y_train_mean_
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
        assert task == 'reg', "Only regression is supported"
        model = self.cls_model if task == "cls" else self.reg_model
        output = model(train_x=x_src[:eval_pos], train_y=y_src, test_x=x_src[eval_pos:])
        return output

    def get_reg_loss(self, logits: torch.Tensor, y: torch.Tensor, temperature: float) -> torch.Tensor:
        return self.bar_dists_criteria(logits[0], y[0], temperature=temperature).unsqueeze(0)

if __name__ == '__main__':
    
    from sklearn.datasets import make_moons, make_regression
    import torch
    from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    model = EasyTabPFNV2(pretrained=True)
    model.cuda()

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
    )
    X_train, X_test, y_train = torch.Tensor(X_train).unsqueeze(0), torch.Tensor(X_test).unsqueeze(0), torch.Tensor(y_train).unsqueeze(0)
    X_train, X_test, y_train = X_train.cuda(), X_test.cuda(), y_train.cuda()

    X_train, X_test, y_train = X_train.repeat(1, 1, 1), X_test.repeat(1, 1, 1), y_train.repeat(1, 1).unsqueeze(-1)

    
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    for _ in range(2):
        predictions, logits = model(torch.cat([X_train, X_test], dim=1), y_train, task="reg", return_logits=True)
        predictions = predictions.detach().cpu().numpy().squeeze()
        
        loss = model.get_reg_loss(logits, torch.Tensor(y_test).unsqueeze(0).repeat(16, 1).cuda(), temperature=1.0)
        print(f'Mean Loss: {loss.mean().item()}')

        print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
        print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
        print("R-squared (R^2):", r2_score(y_test, predictions))
        
        with torch.no_grad():
            predictions, logits = model(torch.cat([X_train, X_test], dim=1), (10 * y_train**3 - 5), task="reg", return_logits=True)
            predictions = predictions.detach().cpu().numpy().squeeze()
            print("TEST R-squared (R^2):", r2_score((10 * y_test**3 - 5), predictions))

        optim.zero_grad()
        loss.mean().backward()
        optim.step()
        
        print('-' * 100)