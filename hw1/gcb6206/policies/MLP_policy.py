"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    1. get_action (line 111)
    2. forward (line 126)
    3. update (line 141)
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from gcb6206.infrastructure import pytorch_util as ptu
from gcb6206.policies.base_policy import BasePolicy


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to actions

    Attributes
    ----------
    logits_na: nn.Sequential
        A neural network that outputs dicrete actions
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    get_action:
        Calls the actor forward function
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """

    def __init__(
        self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        discrete=False,
        learning_rate=1e-4,
        training=True,
        nn_baseline=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(), self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate,
            )

    ##################################

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        :param obs: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        # HINT 1: DO NOT forget to change the type of observation (to torch tensor).
        # Take a close look at `infrastructure/pytorch_util.py`.
        # HINT 2: We would use self.forward function to get the distribution,
        # And we will sample actions from the distribution.
        # HINT 3: Return a numpy action, not torch tensor

        # numpy array를 PyTorch tensor로 변환하고 GPU로 이동
        # ptu.from_numpy(): numpy -> torch.FloatTensor로 변환하고 device(GPU/CPU)로 이동
        observation = ptu.from_numpy(observation)

        # forward()를 호출하여 행동 분포 얻기
        # 반환: torch.distributions.Normal 객체 (평균=mean_net(obs), 표준편차=exp(logstd))
        action_distribution = self.forward(observation)

        # 분포에서 행동 샘플링
        # .sample(): 정규분포에서 무작위로 행동 샘플링 (확률적 정책)
        # 매번 다른 행동 샘플링 -> exploration
        # shape: (batch_size, ac_dim) 또는 (ac_dim,)
        action = action_distribution.sample()

        # PyTorch tensor를 numpy array로 변환
        # ptu.to_numpy(): tensor -> numpy로 변환 (GPU -> CPU, gradient detach)
        # 환경(env)은 numpy array를 기대하므로 변환 필요
        action = ptu.to_numpy(action)

        # single observation 입력 시 배치 차원 제거
        # obs가 1D (single observation)인 경우 action도 1D로 반환
        # env.step()은 1D action을 기대하므로 (1, ac_dim) -> (ac_dim,) 변환
        if len(obs.shape) == 1:
            action = action[0]

        return action

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it.
        # We are only considering continuous action cases. (we do not need to consider the case where self.discrete is True)
        # So, we would like to return a normal distirbution from which we can sample actions.
        # HINT 1: Search up documentation `torch.distributions.Distribution` object
        # And design the function to return such a distribution object.
        # HINT 2: In self.get_action and self.update, we will sample from this distribution.
        # HINT 3: Think about how to convert logstd to regular std.

        # mean_net을 통해 observation을 입력받아 행동의 평균(mean) 계산
        # mean_net: MLP 신경망, (batch_size, ob_dim) -> (batch_size, ac_dim)
        mean = self.mean_net(observation)

        # logstd를 표준편차(std)로 변환
        # torch.exp(): e^logstd 계산
        # logstd는 unbounded(-∞~+∞)지만, std는 항상 양수여야 하므로 exp 사용
        # 예: logstd=-2 -> std=e^(-2)≈0.135, logstd=0 -> std=1, logstd=2 -> std≈7.39
        # log space에서 학습하면 수치적 안정성↑, gradient flow 좋음
        std = torch.exp(self.logstd)

        # torch.distributions.Normal: 정규분포 객체 생성
        # loc=mean: 평균 (batch_size, ac_dim)
        # scale=std: 표준편차 (ac_dim,) - broadcasting으로 모든 배치에 적용
        # 각 행동 차원이 독립적인 정규분포를 따른다고 가정
        # 반환된 분포 객체로 sample(), log_prob() 등 호출 가능
        return torch.distributions.Normal(mean, std)

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss
        # HINT 1: DO NOT forget to call zero_grad to clear gradients from the previous update.
        # HINT 2: DO NOT forget to change the type of observations and actions, just like get_action.
        # HINT 3: DO NOT forget to step the optimizer.

        # 이전 gradient 초기화 (PyTorch는 gradient를 누적하므로 매번 초기화 필요)
        # 안 하면 이전 step의 gradient가 더해져서 잘못된 업데이트 발생
        self.optimizer.zero_grad()

        # numpy array를 PyTorch tensor로 변환하고 GPU로 이동
        observations = ptu.from_numpy(observations)  # (batch_size, ob_dim)
        actions = ptu.from_numpy(actions)            # (batch_size, ac_dim)

        # forward()를 호출하여 행동 분포 얻기
        # 반환: Normal(mean=(batch_size, ac_dim), std=(ac_dim,))
        action_distribution = self.forward(observations)

        # 주어진 actions의 로그 확률 계산
        # log_prob(actions): 각 차원별 로그 확률
        # shape: (batch_size, ac_dim)
        log_probs = action_distribution.log_prob(actions)

        # 각 행동 차원의 로그 확률을 합산 (차원들이 독립이므로 곱 -> 로그 공간에서 합)
        # log P(a) = log P(a1) + log P(a2) + ... + log P(an)
        # sum(dim=-1): 마지막 차원(행동 차원)을 따라 합산
        # shape: (batch_size,)
        log_prob_sum = log_probs.sum(dim=-1)

        # Negative Log Likelihood (NLL) loss 계산
        # BC는 supervised learning: 전문가 행동의 확률을 최대화
        # = log P(expert_action|state) 최대화
        # = -log P(expert_action|state) 최소화 (optimizer는 최소화하므로)
        # mean(): 배치 평균으로 최종 loss
        loss = -log_prob_sum.mean()

        # Backpropagation: loss에서 모든 파라미터로 gradient 계산
        # computational graph를 역방향으로 탐색하며 chain rule 적용
        loss.backward()

        # Optimizer step: 계산된 gradient로 파라미터 업데이트
        # Adam: θ = θ - lr * m / (√v + ε)
        # mean_net과 logstd의 모든 파라미터 업데이트
        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            # loss를 numpy로 변환하여 로깅 (tensorboard 등에 기록)
            "Training Loss": ptu.to_numpy(loss),
        }
