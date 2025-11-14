# GCB6206 Homework 4: Soft Actor-Critic - 완전 솔루션 가이드

## 목차
1. [소개](#소개)
2. [Section 3: Training Loop](#section-3-training-loop)
3. [Section 4: Bootstrapping](#section-4-bootstrapping)
4. [Section 5: Entropy Bonus와 SAC](#section-5-entropy-bonus와-sac)
5. [Section 6: REINFORCE를 사용한 Actor 업데이트](#section-6-reinforce를-사용한-actor-업데이트)
6. [Section 7: REPARAMETRIZE를 사용한 Actor 업데이트](#section-7-reparametrize를-사용한-actor-업데이트)
7. [Section 8: Target Value 안정화](#section-8-target-value-안정화)
8. [완전한 코드 솔루션](#완전한-코드-솔루션)
9. [실험 실행하기](#실험-실행하기)

---

## 소개

이 가이드는 continuous control 작업을 위한 최신 off-policy actor-critic 알고리즘인 Soft Actor-Critic (SAC)을 구현하기 위한 완전한 솔루션을 제공합니다.

**핵심 개념:**
- **Actor-Critic**: Value 기반(critic)과 policy 기반(actor) 방법론 결합
- **Off-policy**: 현재 policy뿐만 아니라 재생된 경험으로부터 학습
- **Soft**: 더 나은 탐색을 위해 entropy를 최대화
- **Continuous actions**: max Q-value 대신 policy gradient 방법 사용

---

## Section 3: Training Loop

### 문제
`gcb6206/scripts/run_hw4.py`에서 training loop를 구현하기 위한 TODO를 완성하세요.

### 솔루션

**위치**: `gcb6206/scripts/run_hw4.py`

#### TODO 1: action 선택 (Line 63-64)
```python
# TODO(student): Select an action
action = agent.get_action(observation)
```

**설명**: agent의 `get_action()` 메서드는 policy distribution에서 action을 샘플링합니다. 처음 `random_steps` 반복 동안에는 초기 탐색을 위해 무작위 action을 사용합니다.

#### TODO 2: replay buffer에서 batch 샘플링 (Line 85-87)
```python
# TODO(student): Sample a batch of config["batch_size"] transitions from the replay buffer
# Please refer to gcb6206/infrastructure/replay_buffer.py
batch = replay_buffer.sample(config["batch_size"])
```

**설명**: replay buffer의 `sample()` 메서드는 다음을 포함하는 딕셔너리를 반환합니다:
- `observations`: 현재 state들
- `actions`: 수행한 action들
- `rewards`: 받은 reward들
- `next_observations`: 다음 state들
- `dones`: Episode 종료 플래그

#### TODO 3: agent 학습 (Line 92-93)
```python
# TODO(student): Train the agent using `update` method. `batch` is a dictionary of torch tensors.
update_info = agent.update(
    observations=batch["observations"],
    actions=batch["actions"],
    rewards=batch["rewards"],
    next_observations=batch["next_observations"],
    dones=batch["dones"],
    step=step,
)
```

**설명**: `update()` 메서드는 actor와 critic network 모두에 대해 한 번의 gradient step을 수행합니다.

---

## Section 4: Bootstrapping

### 문제 1: Bootstrapping을 사용한 critic 업데이트 구현

**위치**: `gcb6206/agents/sac_agent.py`, 메서드 `update_critic` (lines 170-232)

### 솔루션

#### TODO 1: actor에서 샘플링 (Lines 186-188)
```python
# TODO(student): Sample from the actor
next_action_distribution: torch.distributions.Distribution = self.actor(next_obs)
next_action = next_action_distribution.sample()
```

**설명**:
- 다음 state들에 대한 policy distribution π(a|s')를 가져옴
- target value를 계산하기 위해 next action a_{t+1} ~ π(·|s_{t+1})를 샘플링
- `.sample()` 사용 (`.rsample()` 아님) - target 계산에서는 gradient가 필요 없기 때문

#### TODO 2: next Q-value 계산 (Lines 190-192)
```python
# TODO(student)
# Compute the next Q-values using `self.target_critic` for the sampled actions
next_qs = self.target_critic(next_obs, next_action)
```

**설명**:
- 안정성을 위해 target critic network 사용 (현재 critic 아님)
- `next_qs`의 shape은 `(num_critic_networks, batch_size)`
- Target network는 moving target 문제를 방지

#### TODO 3: Entropy bonus 추가 (Lines 204-207)
```python
if self.use_entropy_bonus and self.backup_entropy:
    # TODO(student): Add entropy bonus to the target values for SAC
    # NOTE: use `self.entropy()`
    next_action_entropy = self.entropy(next_action_distribution)
    next_qs += self.temperature * next_action_entropy
```

**설명**:
- Entropy는 무작위성에 reward를 줌으로써 탐색을 장려
- `self.temperature` (β)는 entropy와 reward 간의 tradeoff를 제어
- Entropy는 "soft" value function을 만들기 위해 Q-value에 추가됨

#### TODO 4: target Q-value 계산 (Lines 209-215)
```python
# TODO(student): Compute the target Q-value
# HINT: implement Equation (1) in Homework 4
target_values: torch.Tensor = reward + self.discount * (1 - done) * next_qs
```

**설명**:
- Bellman backup: y = r + γ(1 - d)Q'(s', a')
- `(1 - done)`은 episode가 끝날 때 미래 reward를 0으로 만듦
- `self.discount`는 discount factor γ

#### TODO 5: Q-value 예측 (Lines 217-218)
```python
# TODO(student): Predict Q-values using `self.critic`
q_values = self.critic(obs, action)
```

**설명**: batch에 있는 state-action 쌍에 대한 현재 Q-value입니다.

#### TODO 6: Loss 계산 (Lines 221-222)
```python
# TODO(student): Compute loss using `self.critic_loss`
loss: torch.Tensor = self.critic_loss(q_values, target_values)
```

**설명**: 예측된 Q-value와 target value 간의 MSE loss입니다.

---

### 문제 2: Critic을 여러 번 업데이트

**위치**: `gcb6206/agents/sac_agent.py`, 메서드 `update` (lines 339-382)

### 솔루션

#### TODO: num_critic_updates 스텝 동안 critic 업데이트 (Lines 352-353)
```python
# TODO(student): Update the critic for num_critic_upates steps, and add the output stats to critic_infos
critic_infos = []
for _ in range(self.num_critic_updates):
    critic_info = self.update_critic(
        observations, actions, rewards, next_observations, dones
    )
    critic_infos.append(critic_info)
```

**설명**: Actor 업데이트당 여러 번의 critic 업데이트는 value function의 정확도를 향상시킬 수 있습니다.

---

### 문제 3: Target network 업데이트 구현

**위치**: `gcb6206/agents/sac_agent.py`, 메서드 `update` (lines 339-382)

### 솔루션

#### TODO: Actor 업데이트 (Lines 355-356)
```python
# TODO(student): Update the actor
actor_info = self.update_actor(observations)
```

#### TODO: Hard 또는 soft target 업데이트 수행 (Lines 358-365)
```python
# TODO(student): Perform either hard or soft target updates.
# Relevant variables:
#  - step
#  - self.target_update_period (None when using soft updates)
#  - self.soft_target_update_rate (None when using hard updates)
# For hard target updates, you should do it every self.target_update_period step
# For soft target updates, you should do it every step
# HINT: use `self.update_target_critic` or `self.soft_update_target_critic`

if self.target_update_period is not None:
    # Hard update
    if step % self.target_update_period == 0:
        self.update_target_critic()
else:
    # Soft update
    self.soft_update_target_critic(self.soft_target_update_rate)
```

**설명**:
- **Hard update**: φ' ← φ, K 스텝마다 (완전한 복사)
- **Soft update**: φ' ← φ' + τ(φ - φ'), 매 스텝마다 (exponential moving average)
- Soft update가 일반적으로 더 안정적

---

### 문제 4: "아무것도 하지 않는" policy에 대한 기대 Q-value 계산

**질문**: Pendulum-v1의 "아무것도 하지 않는" reward는 스텝당 약 −10입니다. agent가 항상 "아무것도 하지 않고" 절대 종료되지 않는다고 가정할 때, discount factor γ = 0.99를 고려한 평균 Q-value는 얼마여야 할까요?

### 답변

상수 reward r = -10이고 discount γ = 0.99인 무한 horizon의 경우:

```
Q(s, a) = E[∑_{t=0}^∞ γ^t · r]
        = r · ∑_{t=0}^∞ γ^t
        = r / (1 - γ)
        = -10 / (1 - 0.99)
        = -10 / 0.01
        = -1000
```

**예상 Q-value**: 약 **-1000**

이는 과제에서 언급된 예상 안정화 지점(-700 정도)과 일치합니다. 차이는 샘플링 분산이나 pendulum이 가끔 넘어질 수 있다는 사실 때문일 수 있습니다.

---

## Section 5: Entropy Bonus와 SAC

### 문제: Entropy 계산 및 soft Q-value 구현

**위치**: `gcb6206/agents/sac_agent.py`

### 솔루션

#### TODO: Entropy 메서드 구현 (Lines 234-247)
```python
def entropy(self, action_distribution: torch.distributions.Distribution):
    """
    Compute the (approximate) entropy of the action distribution for each batch element.
    """

    # TODO(student): Compute the entropy of the action distribution
    # HINT: use one action sample for estimating the entropy.
    # HINT: use action_distribution.log_prob to get the log probability.
    # NOTE: think about whether to use .rsample() or .sample() here
    action = action_distribution.rsample()
    entropy = -action_distribution.log_prob(action)

    assert entropy.shape == action.shape[:-1]
    return entropy
```

**설명**:
- **Entropy**: H(π) = E_a~π[-log π(a|s)]
- Reparameterization trick을 위해 `.rsample()` 사용 (필요시 gradient 흐름 가능)
- 단일 샘플을 사용한 Monte Carlo 추정: H ≈ -log π(â|s) where â ~ π(·|s)
- 높은 entropy = 더 무작위적인 policy = 더 나은 탐색

**왜 rsample?** 여기서는 entropy를 통해 backprop하지 않지만, rsample을 사용하는 것이 reparameterization 접근법과 일관되며 actor 업데이트에서 필요시 gradient 흐름을 가능하게 합니다.

#### 참고: Entropy bonus는 update_critic에 이미 추가됨
Entropy 항은 Section 4, TODO 3에서 이미 target value에 추가되었습니다.

---

### 예상 결과

Reward 최대화 없이 entropy만 최대화하는 Pendulum-v1의 경우, entropy는 다음에 근접해야 합니다:

```
H(U[-1, 1]) = -log(1/2) = log(2) ≈ 0.69
```

이는 1D action space [-1, 1]에 대한 균등 분포의 최대 entropy입니다.

---

## Section 6: REINFORCE를 사용한 Actor 업데이트

### 문제: REINFORCE gradient estimator 구현

**위치**: `gcb6206/agents/sac_agent.py`, 메서드 `actor_loss_reinforce` (lines 249-286)

### 솔루션

```python
def actor_loss_reinforce(self, obs: torch.Tensor):
    batch_size = obs.shape[0]

    # TODO(student): Generate an action distribution
    action_distribution: torch.distributions.Distribution = self.actor(obs)

    with torch.no_grad():
        # TODO(student): Draw self.num_actor_samples samples from the action distribution for each batch element
        # NOTE: think about whether to use .rsample() or .sample() here
        action = action_distribution.sample((self.num_actor_samples,))
        assert action.shape == (
            self.num_actor_samples,
            batch_size,
            self.action_dim,
        ), action.shape

        # TODO(student): Compute Q-values for the current state-action pair
        # HINT: need to add one dimension with `self.num_actor_samples` at the beginning of `obs`
        # HINT: for this, you can use either `repeat` or `expand`
        q_values = self.critic(
            obs[None].expand(self.num_actor_samples, batch_size, -1),
            action
        )
        assert q_values.shape == (
            self.num_critic_networks,
            self.num_actor_samples,
            batch_size,
        ), q_values.shape

        # Our best guess of the Q-values is the mean of the ensemble
        q_values = torch.mean(q_values, axis=0)

    # Do REINFORCE (without baseline)
    # TODO(student): Calculate log-probs
    log_probs = action_distribution.log_prob(action)
    assert log_probs.shape == q_values.shape

    # TODO(student): Compute policy gradient using log-probs and Q-values
    loss = -(log_probs * q_values).mean()

    return loss, torch.mean(self.entropy(action_distribution))
```

**설명**:

1. **Action distribution 생성**: batch의 각 state에 대한 π_θ(·|s)
2. **Action 샘플링**: REINFORCE는 reparameterization을 사용하지 않으므로 `.sample()` 사용 (`.rsample()` 아님)
3. **Observation 확장**: action 샘플과 일치하도록 각 observation을 `num_actor_samples`번 반복
4. **Q-value 계산**: 각 샘플링된 action에 대한 Q(s, a)
5. **Log probability 계산**: log π_θ(a|s)
6. **REINFORCE gradient**: ∇_θ E[Q] ≈ E[∇_θ log π(a|s) · Q(s,a)]
7. **Loss**: 최대화하려고 하므로 음수 (gradient ascent)

**왜 sample()이지 rsample()이 아닌가?** REINFORCE는 reparameterization이 아닌 log-derivative trick (score function)을 사용합니다.

**다중 샘플**: 여러 action 샘플을 사용하면 gradient 추정의 분산이 줄어듭니다.

---

## Section 7: REPARAMETRIZE를 사용한 Actor 업데이트

### 문제: Reparameterized gradient estimator 구현

**위치**: `gcb6206/agents/sac_agent.py`, 메서드 `actor_loss_reparametrize` (lines 288-304)

### 솔루션

```python
def actor_loss_reparametrize(self, obs: torch.Tensor):
    batch_size = obs.shape[0]

    # Sample from the actor
    action_distribution: torch.distributions.Distribution = self.actor(obs)

    # TODO(student): Sample actions
    # Note: Think about whether to use .rsample() or .sample() here...
    action = action_distribution.rsample()

    # TODO(student): Compute Q-values for the sampled state-action pair
    q_values = self.critic(obs, action)

    # TODO(student): Compute the actor loss using Q-values
    loss = -q_values.mean()

    return loss, torch.mean(self.entropy(action_distribution))
```

**설명**:

1. **Action distribution 가져오기**: actor network로부터 π_θ(·|s)
2. **Reparameterization으로 샘플링**: 샘플링을 통한 gradient 흐름을 가능하게 하기 위해 `.rsample()` 사용
3. **Q-value 계산**: a = μ_θ(s) + σ_θ(s) · ε, ε ~ N(0,1)인 Q(s, a)
4. **Loss 계산**: Q-value 최대화 (음의 Q-value 최소화)

**Reparameterization Trick**:
- 대신에: a ~ π_θ(·|s), 샘플링을 통해 backprop 불가
- 사용: a = μ_θ(s) + σ_θ(s) · ε where ε ~ N(0,1)
- 이제 gradient가 흐름: ∇_θ Q(s, a) = ∇_θ Q(s, μ_θ(s) + σ_θ(s)·ε)

**REINFORCE에 비한 장점**:
- 훨씬 낮은 분산
- 단일 샘플로도 잘 작동
- 효율적인 최적화 가능

---

## Section 8: Target Value 안정화

### 문제: Double-Q와 Clipped Double-Q 구현

**위치**: `gcb6206/agents/sac_agent.py`, 메서드 `q_backup_strategy` (lines 127-168)

### 솔루션

```python
def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
    """
    Handle Q-values from multiple different target critic networks to produce target values.

    For example:
     - for "vanilla", we can just leave the Q-values as-is (we only have one critic).
     - for double-Q, swap the critics' predictions (so each uses the other as the target).
     - for clip-Q, clip to the minimum of the two critics' predictions.

    Parameters:
        next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size).
            Leading dimension corresponds to target values FROM the different critics.
    Returns:
        torch.Tensor: Target values of shape (num_critics, batch_size).
            Leading dimension corresponds to target values FOR the different critics.
    """

    assert (
        next_qs.ndim == 2
    ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
    num_critic_networks, batch_size = next_qs.shape
    assert num_critic_networks == self.num_critic_networks

    # TODO(student): Implement the different backup strategies.
    if self.target_critic_backup_type == "doubleq":
        # Double-Q: Swap critics - each uses the other as target
        if num_critic_networks == 2:
            next_qs = torch.stack([next_qs[1], next_qs[0]], dim=0)
        else:
            # For more than 2 critics, rotate them
            next_qs = torch.roll(next_qs, shifts=1, dims=0)

    elif self.target_critic_backup_type == "min":
        # Clipped Double-Q: Use minimum across all critics
        next_qs = next_qs.min(dim=0, keepdim=False).values

    else:
        # Default (mean or single critic), we don't need to do anything.
        pass

    # If our backup strategy removed a dimension, add it back in explicitly
    # (assume the target for each critic will be the same)
    if next_qs.shape == (batch_size,):
        next_qs = next_qs[None].expand((self.num_critic_networks, batch_size)).contiguous()

    assert next_qs.shape == (
        self.num_critic_networks,
        batch_size,
    ), next_qs.shape
    return next_qs
```

**설명**:

### Double-Q Learning
- **문제**: 단일 Q-network는 value를 과대평가하는 경향
- **해결책**: 두 개의 critic Q_A와 Q_B 사용
  - Q_A는 Q'_B를 target으로 사용: y_A = r + γQ'_B(s', a')
  - Q_B는 Q'_A를 target으로 사용: y_B = r + γQ'_A(s', a')
- **구현**: 두 target critic의 Q-value를 교환

### Clipped Double-Q (TD3 스타일)
- **문제**: Double-Q로도 여전히 약간의 과대평가
- **해결책**: 비관적 추정 사용
  - 두 critic 모두 사용: y = r + γ min(Q'_A(s', a'), Q'_B(s', a'))
- **구현**: 모든 critic에서 최소값 취함
- **결과**: 모든 critic에 대해 동일한 target, 더 보수적

### 왜 중요한가
- 과대평가 편향은 학습 중 누적됨
- 불안정한 학습이나 발산으로 이어질 수 있음
- Clipped Double-Q가 일반적으로 가장 안정적

---

## 완전한 코드 솔루션

### 모든 코드 변경사항 요약:

#### 파일: `gcb6206/scripts/run_hw4.py`

**Line 63-64**: Action 선택
```python
action = agent.get_action(observation)
```

**Line 85-87**: Batch 샘플링
```python
batch = replay_buffer.sample(config["batch_size"])
```

**Line 92-93**: Agent 학습
```python
update_info = agent.update(
    observations=batch["observations"],
    actions=batch["actions"],
    rewards=batch["rewards"],
    next_observations=batch["next_observations"],
    dones=batch["dones"],
    step=step,
)
```

#### 파일: `gcb6206/agents/sac_agent.py`

위에서 자세히 제공된 모든 솔루션:
- `update_critic()` 메서드
- `entropy()` 메서드
- `actor_loss_reinforce()` 메서드
- `actor_loss_reparametrize()` 메서드
- `q_backup_strategy()` 메서드
- `update()` 메서드

---

## 실험 실행하기

### Section 4.2: Bootstrapping 검증

```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/sanity_pendulum_1.yaml
```

**예상 결과**: Q-value가 -700에서 -1000 사이로 안정화

### Section 5.2: Entropy 검증

```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/sanity_pendulum_2.yaml
```

**예상 결과**: Entropy가 ~0.69 (log 2)까지 증가

### Section 6.2: REINFORCE 실험

**Inverted Pendulum**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/sanity_invertedpendulum_reinforce.yaml
```
**예상 결과**: Reward ~1000

**HalfCheetah REINFORCE-1**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/halfcheetah_reinforce1.yaml
```
**예상 결과**: 500K 스텝에서 positive reward

**HalfCheetah REINFORCE-10**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/halfcheetah_reinforce10.yaml
```
**예상 결과**: 200K 스텝에서 >500 evaluation return (REINFORCE-1보다 빠름)

### Section 7.2: REPARAMETRIZE 실험

**Inverted Pendulum**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/sanity_invertedpendulum_reparametrize.yaml
```
**예상 결과**: Reward ~1000

**HalfCheetah**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/halfcheetah_reparametrize.yaml
```
**예상 결과**: 모든 방법 중 최고 성능

### Section 8.2: 안정화 실험

**Single-Q Hopper**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/hopper.yaml --seed 48
```

**Double-Q Hopper**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/hopper_doubleq.yaml --seed 48
```

**Clipped Double-Q Hopper**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/hopper_clipq.yaml --seed 48
```

**예상 결과**: Clipped Double-Q가 일반적으로 가장 낮은 Q-value 과대평가로 최고 성능

---

## 핵심 요점

1. **SAC는 두 세계의 장점을 결합**: Value 기반 안정성 + policy gradient 유연성
2. **Entropy regularization**: 탐색을 개선하고 조기 수렴 방지
3. **Reparameterization trick**: REINFORCE보다 훨씬 낮은 분산
4. **Target network**: Off-policy 학습에서 안정성에 필수적
5. **Clipped Double-Q**: 과대평가 편향을 효과적으로 해결

---

## 일반적인 함정과 디버깅 팁

1. **Gradient가 흐르지 않음**: Reparameterization에는 `.rsample()`, REINFORCE에는 `.sample()` 사용 확인
2. **Shape 불일치**: 특히 여러 샘플/critic이 있을 때 batch 차원에 주의
3. **폭발/소멸하는 Q-value**: Target network 업데이트가 작동하는지 확인
4. **학습 안 됨**: 학습이 시작되기 전에 replay buffer에 충분한 샘플이 있는지 확인
5. **Entropy가 너무 높거나 낮음**: Temperature 계수와 entropy 구현 확인

---

## 추가 자료

- **SAC Paper**: https://arxiv.org/abs/1801.01290
- **TD3 Paper** (Clipped Double-Q): https://arxiv.org/abs/1802.09477
- **OpenAI Spinning Up**: https://spinningup.openai.com/en/latest/algorithms/sac.html
