from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import gcb6206.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        # Epsilon-greedy로 Action 선택
        if np.random.random() < epsilon:
            # Random exploration - int 직접 반환
            return np.random.randint(self.num_actions)
        else:
            # Greedy exploitation - tensor를 int로 변환하여 반환
            with torch.no_grad():
                q_values = self.critic(observation)
                action = torch.argmax(q_values, dim=1)
            return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(student): compute target values
            # Target network로 다음 Status의 Q-value 계산
            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                # Choose action with argmax of critic network
                # Double DQN: online network로 Action 선택
                next_action = torch.argmax(self.critic(next_obs), dim=1)
            else:
                # Choose action with argmax of target critic network
                # Vanilla DQN: target network로 Action 선택
                next_action = torch.argmax(next_qa_values, dim=1)

            # see torch.gather
            # 선택된 Action의 Q-value 추출
            next_q_values = torch.gather(next_qa_values, 1, next_action.unsqueeze(1)).squeeze(1)

            # TD target 계산: r + 할인율*(1-d)*Q(s',a')
            target_values = reward + self.discount * (1 - done) * next_q_values

        # TODO(student): train the critic with the target values
        # Use self.critic_loss for calculating the loss
        # Online network로 현재 Q-value 계산
        qa_values = self.critic(obs)
        # Compute from the data actions; see torch.gather
        # 실제 수행한 Action의 Q-value 추출
        q_values = torch.gather(qa_values, 1, action.unsqueeze(1)).squeeze(1)
        # 예측 Q-value와 target Q-value 간의 MSE loss
        loss = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed
        # HINT: Update the target network if step % self.target_update_period is 0
        # Critic network 업데이트
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        # Target network 주기적 업데이트
        if step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats
