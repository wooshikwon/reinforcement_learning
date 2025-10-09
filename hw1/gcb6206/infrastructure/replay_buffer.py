"""
A simple, generic replay buffer

Functions to edit:
    sample_random_data: line 103
"""

from gcb6206.infrastructure.utils import *


class ReplayBuffer:
    """
    Defines a replay buffer to store past trajectories

    Attributes
    ----------
    trajs: list
        A list of rollouts
    obs: np.array
        An array of observations
    acs: np.array
        An array of actions
    rews: np.array
        An array of rewards
    next_obs:
        An array of next observations
    terminals:
        An array of terminals

    Methods
    -------
    add_rollouts:
        Add rollouts and processes them into their separate components
    sample_random_data:
        Selects a random batch of data
    sample_recent_data:
        Selects the most recent batch of data
    """

    def __init__(self, max_size=1000000):
        self.max_size = max_size

        # store each rollout
        self.trajs = []

        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):
        if self.obs is not None:
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, trajs, concat_rew=True):
        """
        Adds trajs into the buffer and processes them into separate components

        :param trajs: a list of trajs to add
        :param concat_rew: whether rewards should be concatenated or appended
        """
        # add new rollouts into our list of rollouts
        for traj in trajs:
            self.trajs.append(traj)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, rewards, next_observations, terminals = (
            convert_listofrollouts(trajs, concat_rew)
        )

        if self.obs is None:
            self.obs = observations[-self.max_size :]
            self.acs = actions[-self.max_size :]
            self.rews = rewards[-self.max_size :]
            self.next_obs = next_observations[-self.max_size :]
            self.terminals = terminals[-self.max_size :]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size :]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size :]
            if concat_rew:
                self.rews = np.concatenate([self.rews, rewards])[-self.max_size :]
            else:
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size :]
            self.next_obs = np.concatenate([self.next_obs, next_observations])[
                -self.max_size :
            ]
            self.terminals = np.concatenate([self.terminals, terminals])[
                -self.max_size :
            ]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):
        """
        Samples a batch of random transitions

        :param batch_size: the number of transitions to sample
        :return:
            obs: a batch of observations
            acs: a batch of actions
            rews: a batch of rewards
            next_obs: a batch of next observations
            terminals: a batch of terminals
        """
        assert (
            self.obs.shape[0]
            == self.acs.shape[0]
            == self.rews.shape[0]
            == self.next_obs.shape[0]
            == self.terminals.shape[0]
        )

        ## TODO return batch_size number of random entries
        ## from each of the 5 component arrays above.
        ## HINT 1: use np.random.choice to sample random indices.
        ## Remember not to "replace" when sampling data
        ## HINT 2: return corresponding data points from each array
        ## (i.e., not different indices from each array)
        ## You would use same indices for all arrays.
        ## HINT 3: look at the sample_recent_data function below

        # 버퍼에서 무작위로 batch_size개의 인덱스 샘플링
        # np.random.choice(n, size, replace)
        # - n: 0부터 n-1까지 범위
        # - size: 샘플링할 개수
        # - replace=False: 중복 없이 샘플링 (without replacement)
        #   같은 데이터를 여러 번 뽑지 않음 -> 배치 내 다양성 보장
        indices = np.random.choice(
            len(self.obs),           # 버퍼 크기 (0 ~ len-1)
            size=batch_size,         # 샘플링할 개수
            replace=False            # 중복 없이
        )

        # 샘플링한 인덱스로 각 배열에서 데이터 추출
        # 모든 배열에 같은 인덱스 사용 -> 데이터 정합성 유지
        # 예: indices=[5, 100, 37]이면
        #     obs[5], acs[5], ... 와 obs[100], acs[100], ... 를 추출
        # 무작위 샘플링으로 시간적 상관관계 제거 (i.i.d. 가정 만족)
        return (
            self.obs[indices],        # (batch_size, ob_dim)
            self.acs[indices],        # (batch_size, ac_dim)
            self.rews[indices],       # (batch_size,)
            self.next_obs[indices],   # (batch_size, ob_dim)
            self.terminals[indices]   # (batch_size,)
        )

    def sample_recent_data(self, batch_size=1):
        """
        Samples a batch of the most recent transitions transitions

        :param batch_size: the number of transitions to sample
        :return:
            obs: a batch of observations
            acs: a batch of actions
            rews: a batch of rewards
            next_obs: a batch of next observations
            terminals: a batch of terminals
        """
        return (
            self.obs[-batch_size:],
            self.acs[-batch_size:],
            self.rews[-batch_size:],
            self.next_obs[-batch_size:],
            self.terminals[-batch_size:],
        )
