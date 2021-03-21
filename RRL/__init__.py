import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class RRLModel(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.neuron = nn.Linear(m+1, 1, bias=True)
        nn.init.uniform_(self.neuron.weight, -.2, .2)
        nn.init.constant_(self.neuron.bias, 0)

    def forward(self, features):
        return torch.tanh(
            self.neuron(features)
        )


def sharpe_ratio(returns: torch.Tensor, eps: float = 1e-6):
    expected_return = torch.mean(returns, dim=-1)
    # The reference writeup used the biased STD estimator
    expected_squared_return = torch.mean(returns ** 2, dim=-1)
    sharpe = expected_return / (torch.sqrt(
        expected_squared_return - expected_return ** 2
    ) + eps)
    return sharpe


def reward_function(asset_returns: torch.Tensor, miu: float, delta: float, Ft: torch.Tensor, m: int):
    n = Ft.shape[-1] - 1
    returns = miu * (
        Ft[:n] * asset_returns[m:m+n]
    ) - (
        delta * torch.abs(Ft[1:] - Ft[:n])
    )
    sharpe = sharpe_ratio(returns)
    return returns, sharpe


def update_Ft(normalized_asset_returns: torch.Tensor, model: RRLModel):
    m = model.m
    t = normalized_asset_returns.shape[-1] - m
    Ft = torch.zeros(t + 1).to(normalized_asset_returns.device)
    for i in range(1, t):
        features = torch.cat([
            normalized_asset_returns[i-1:i+m-1], Ft[i-1:i]
        ])
        Ft[i] = model(features)
    return Ft[1:]


def gradient_accent(
        asset_returns: torch.Tensor,
        normalized_asset_returns: torch.Tensor,
        model: RRLModel,
        max_iter: int, lr: float):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    rewards = []
    for i in range(max_iter):
        optimizer.zero_grad()
        Ft = update_Ft(normalized_asset_returns, model)
        returns, reward = reward_function(asset_returns, miu=1., delta=0, Ft=Ft, m=model.m)
        (-1 * reward).backward()
        optimizer.step()
        rewards.append(reward.detach().cpu())
    return rewards, returns, Ft


def train(prices: torch.Tensor, m: int, t: int, delta: float = 0, max_iter: int = 100, lr: float = 0.1):
    assert len(prices.size()) == 1
    # asset returns are the ratio of the amount of change to the previous price
    asset_returns = (
        prices[1:] - prices[:-1]
    ).float() / prices[:-1]
    # to_be_predicted = prices.shape[0] - t - m
    scaler = StandardScaler()
    normalized_asset_returns = torch.tensor(scaler.fit_transform(
        asset_returns[:m+t][:, None].numpy()
    )[:, 0]).float()

    model = RRLModel(m)
    train_rewards, train_returns, train_Ft = gradient_accent(
        asset_returns, normalized_asset_returns, model, max_iter, lr
    )

    normalized_asset_returns = torch.tensor(
        scaler.transform(asset_returns[t:][:, None].numpy())[:, 0]
    ).float()
    Ft_ahead = update_Ft(normalized_asset_returns, model)
    returns_ahead, reward_ahead = reward_function(asset_returns[t:], 1., delta, Ft_ahead, model.m)
    percentage_returns = (torch.exp(
        torch.log(1 + returns_ahead).cumsum(dim=-1)
    ) - 1) * 100
    return {
        "valid_reward": reward_ahead,
        "valid_Ft": Ft_ahead,
        "valid_asset_returns": asset_returns[m+t:],
        "valid_asset_percentage_returns": (torch.exp(
            torch.log(1 + asset_returns[m+t:]).cumsum(dim=-1)
        ) - 1) * 100,
        "valid_percentage_returns": percentage_returns,
        "rewards_iter": train_rewards,
        "train_percentage_returns": (torch.exp(
            torch.log(1 + train_returns).cumsum(dim=-1)
        ) - 1) * 100,
        "train_Ft": train_Ft
    }
