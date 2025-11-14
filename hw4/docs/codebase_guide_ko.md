# GCB6206 Homework 4 코드베이스 구조 가이드
## Soft Actor-Critic 구현 이해를 위한 초보자 가이드

---

## 목차

1. [소개](#소개)
2. [강화학습 기초](#강화학습-기초)
3. [Soft Actor-Critic (SAC) 개요](#soft-actor-critic-sac-개요)
4. [프로젝트 구조](#프로젝트-구조)
5. [핵심 컴포넌트 Deep Dive](#핵심-컴포넌트-deep-dive)
6. [데이터 흐름과 Training Loop](#데이터-흐름과-training-loop)
7. [핵심 개념과 디자인 패턴](#핵심-개념과-디자인-패턴)
8. [코드베이스 탐색 방법](#코드베이스-탐색-방법)

---

## 소개

### 이 프로젝트는 무엇인가요?

이 코드베이스는 continuous control 작업을 위한 가장 성공적인 딥 강화학습 알고리즘 중 하나인 **Soft Actor-Critic (SAC)**을 구현합니다. 로봇 팔이 목표에 도달하도록 훈련하거나, 휴머노이드가 걷도록 가르치거나, 시뮬레이션된 치타가 달리도록 제어하는 등의 작업을 위해 agent를 훈련하도록 설계되었습니다.

### 이 가이드는 누구를 위한 것인가요?

- 강화학습 코드를 이해하고 싶은 **Python 초보자**
- 이론적 개념이 코드로 어떻게 변환되는지 보고 싶은 **RL 초보자**
- 과제를 수행하는 **학생들**
- 현대 RL 알고리즘이 어떻게 구현되는지 궁금한 **모든 사람**

### 무엇을 배우게 되나요:

1. RL 개념(state, action, reward)이 코드로 매핑되는 방법
2. RL 코드베이스의 구조와 조직
3. Actor-critic 방법에서 신경망이 사용되는 방법
4. RL 구현을 위한 모범 사례

---

## 강화학습 기초

### 큰 그림

로봇 개를 걷도록 훈련한다고 상상해 보세요:

```
┌─────────────┐
│ Environment │ ← 시뮬레이션된 세계 (물리, 중력 등)
└─────────────┘
      ↓ observation (state)
      ↑ action
┌─────────────┐
│    Agent    │ ← 무엇을 할지 결정하는 "두뇌"
└─────────────┘
```

**학습 루프:**
1. Agent는 현재 **state**를 관찰 (관절 각도, 속도)
2. Agent는 **action**을 선택 (모터 토크)
3. Environment는 새로운 **state**와 **reward**로 응답
4. Agent는 시간에 따른 총 reward를 최대화하도록 학습

### 핵심 RL 개념

#### 1. Markov Decision Process (MDP)

MDP는 RL을 위한 수학적 프레임워크입니다:

- **State (s)**: Environment의 완전한 설명
  - 예: [position, velocity, angle, angular_velocity]
- **Action (a)**: Agent가 할 수 있는 것
  - 이산: {left, right, jump}
  - 연속: [motor_1_torque, motor_2_torque, ...]
- **Reward (r)**: 스칼라 피드백 신호
  - 예: 똑바로 서 있으면 +1, 넘어지면 -100
- **Transition**: p(s'|s,a) - 다음 state의 확률
- **Policy (π)**: π(a|s) - agent의 전략 (state → action)

#### 2. Value Function

**Q-function (Action-Value)**: Q(s, a) = state s에서 시작하여 action a를 취한 후 policy π를 따를 때 예상되는 총 reward

```
Q(s, a) = E[r_0 + γr_1 + γ²r_2 + γ³r_3 + ...]
```

여기서 γ (gamma)는 discount factor입니다 (0 < γ < 1):
- γ = 0.99: 장기 reward를 고려
- γ = 0.0: 즉각적인 reward만 고려

**Q-function을 사용하는 이유?**
모든 action에 대한 Q(s, a)를 알면 최선의 action을 선택할 수 있습니다:
```
π*(s) = argmax_a Q(s, a)
```

#### 3. Policy Gradient 방법

Q-value를 학습하는 대신, policy π_θ(a|s)를 직접 학습:

**아이디어**: 예상 reward를 증가시키도록 policy 파라미터 θ 조정

```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) · Q(s,a)]
```

**직관**: Action a가 좋은 Q-value로 이어졌다면, 더 가능성 있게 만들기!

#### 4. Actor-Critic 방법

두 접근법 결합:

- **Actor (Policy)**: π_θ(a|s) - 어떤 action을 취할지 결정
- **Critic (Value)**: Q_φ(s, a) - action이 얼마나 좋은지 평가

**장점:**
- 순수 policy gradient보다 낮은 분산
- 순수 value 방법보다 효율적
- Continuous action에 잘 작동

---

## Soft Actor-Critic (SAC) 개요

### SAC를 "Soft"하게 만드는 것은?

**표준 RL 목표**: 총 reward 최대화
```
max E[∑ γ^t r_t]
```

**SAC 목표**: Reward + entropy 최대화
```
max E[∑ γ^t (r_t + α·H(π(·|s_t)))]
```

여기서 H(π)는 entropy입니다: H(π) = -E[log π(a|s)]

**왜 entropy?**
- **탐색**: 다양한 action 시도를 장려
- **강건성**: 작업을 해결하는 여러 방법 학습
- **붕괴 방지**: 조기 수렴 방지

### SAC 알고리즘 구성요소

```
┌──────────────────────────────────────┐
│           SAC Agent                  │
├──────────────────────────────────────┤
│                                      │
│  ┌────────────┐    ┌──────────────┐ │
│  │   Actor    │    │   Critics    │ │
│  │  π_θ(a|s)  │    │  Q_φ(s,a)    │ │
│  └────────────┘    └──────────────┘ │
│        │                  │          │
│        ↓                  ↓          │
│   Sample action      Evaluate Q     │
│                                      │
│  ┌──────────────────────────────┐   │
│  │   Target Critics (천천히    │   │
│  │   업데이트되는 critic 복사본) │   │
│  └──────────────────────────────┘   │
└──────────────────────────────────────┘
```

**핵심 특징:**
1. **Off-policy**: 과거 경험에서 학습 (replay buffer)
2. **Maximum entropy**: Entropy bonus 추가
3. **Twin critics**: 과대평가 편향 감소
4. **Continuous action**: 실수값 action 처리

### SAC 훈련 알고리즘

```
반복:
    1. Action 샘플링: a ~ π_θ(·|s)
    2. Action 실행, 관찰 (s, a, r, s', done)
    3. Replay buffer에 저장

    4. Replay buffer에서 batch 샘플링

    5. Critic 업데이트:
       - Target 계산: y = r + γ(Q'(s',a') + α·H(π(·|s')))
       - 최소화: (Q_φ(s,a) - y)²

    6. Actor 업데이트:
       - 최대화: Q_φ(s, π_θ(s)) + α·H(π_θ(·|s))

    7. Target critic 업데이트 (천천히)
```

---

## 프로젝트 구조

### 디렉토리 레이아웃

```
hw4/
├── gcb6206/                      # 메인 패키지
│   ├── agents/                   # Agent 구현체
│   │   └── sac_agent.py         # SAC 알고리즘
│   ├── networks/                 # 신경망 아키텍처
│   │   ├── mlp_policy.py        # Actor network (policy)
│   │   └── state_action_value_critic.py  # Critic network
│   ├── infrastructure/           # 유틸리티와 헬퍼
│   │   ├── replay_buffer.py     # Experience replay
│   │   ├── logger.py            # TensorBoard 로깅
│   │   ├── pytorch_util.py      # PyTorch 헬퍼
│   │   ├── distributions.py     # 커스텀 분포
│   │   └── utils.py             # 일반 유틸리티
│   ├── env_configs/             # Environment 설정
│   │   ├── sac_config.py        # SAC 하이퍼파라미터
│   │   └── schedule.py          # Learning rate 스케줄
│   └── scripts/                 # 진입점
│       ├── run_hw4.py           # 메인 훈련 스크립트
│       └── scripting_utils.py   # Config 로딩
├── experiments/                  # 실험 설정
│   └── sac/                     # SAC 실험 YAML 파일
│       ├── sanity_pendulum_1.yaml
│       ├── halfcheetah_reinforce1.yaml
│       └── ...
├── docs/                        # 문서
├── requirements.txt             # Python 의존성
└── setup.py                     # 패키지 설치
```

### 설계 철학

**관심사의 분리 (Separation of Concerns):**
- **agents/**: 알고리즘 로직 (무엇을 학습할지)
- **networks/**: 신경망 아키텍처 (함수 근사기)
- **infrastructure/**: 재사용 가능한 컴포넌트 (replay buffer, 로깅)
- **env_configs/**: 하이퍼파라미터 (어떻게 학습할지)
- **scripts/**: 실행 로직 (언제 무엇을 할지)

---

## 핵심 컴포넌트 Deep Dive

### 1. Agent: `gcb6206/agents/sac_agent.py`

**목적**: SAC 학습 알고리즘 구현

**클래스**: `SoftActorCritic`

#### 핵심 속성

```python
class SoftActorCritic(nn.Module):
    def __init__(...):
        # Actor: π_θ(a|s) - policy network
        self.actor = make_actor(observation_shape, action_dim)

        # Critics: Q_φ(s,a) - value network들 (여러 개 가능)
        self.critics = nn.ModuleList([
            make_critic(observation_shape, action_dim)
            for _ in range(num_critic_networks)
        ])

        # Target critics: Q_φ'(s,a) - 천천히 업데이트되는 복사본
        self.target_critics = nn.ModuleList([...])

        # Optimizer
        self.actor_optimizer = make_actor_optimizer(...)
        self.critic_optimizer = make_critic_optimizer(...)

        # 하이퍼파라미터
        self.discount = discount  # γ (gamma)
        self.temperature = temperature  # α (alpha) for entropy
```

#### 핵심 메서드

**`get_action(observation)`**
```python
def get_action(self, observation: np.ndarray) -> np.ndarray:
    """Environment에서 실행할 action 선택"""
    # 1. numpy → torch 변환
    # 2. Policy distribution π(·|s) 가져오기
    # 3. Action 샘플링 a ~ π(·|s)
    # 4. torch → numpy 변환
```

**`update_critic(obs, action, reward, next_obs, done)`**
```python
def update_critic(...):
    """Critic network에 대한 한 번의 gradient step"""
    # 1. 다음 action 샘플링: a' ~ π(·|s')
    # 2. Target 계산: y = r + γ(Q'(s',a') + α·H(π(·|s')))
    # 3. Q-value 계산: Q(s,a)
    # 4. Loss 계산: MSE(Q, y)
    # 5. Backprop과 업데이트
```

**`update_actor(obs)`**
```python
def update_actor(obs):
    """Actor network에 대한 한 번의 gradient step"""
    # 두 가지 변형:
    # REINFORCE: ∇ log π(a|s) · Q(s,a)
    # REPARAMETRIZE: ∇ Q(s, π(s))
```

**`update(observations, actions, rewards, next_observations, dones, step)`**
```python
def update(...):
    """메인 업데이트: critic + actor + target network"""
    # 1. Critic 업데이트 (여러 번)
    # 2. Actor 업데이트 (한 번)
    # 3. Target network 업데이트
```

---

### 2. Actor Network: `gcb6206/networks/mlp_policy.py`

**목적**: Action에 대한 확률 분포를 출력하는 신경망

**클래스**: `MLPPolicy`

#### 아키텍처

```
입력: state (observation)
    ↓
[Linear Layer → Activation] × n_layers
    ↓
Output Layer
    ↓
Continuous action의 경우:
    - Mean: μ(s)
    - Std: σ(s) (선택적으로 state 의존적)
    ↓
Distribution: π(a|s) = N(μ(s), σ(s))  또는  Tanh(N(μ(s), σ(s)))
```

#### 핵심 코드

```python
def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
    """
    입력: observation batch [batch_size, obs_dim]
    출력: 각 observation에 대한 action distribution
    """
    if self.state_dependent_std:
        # Mean과 std 모두 state에 의존
        mean, std = torch.chunk(self.net(obs), 2, dim=-1)
        std = F.softplus(std) + 1e-2  # 양수 보장
    else:
        # Mean만 state에 의존
        mean = self.net(obs)
        std = F.softplus(self.std) + 1e-2  # 학습 가능한 파라미터

    if self.use_tanh:
        # Action을 [-1, 1]로 압축
        return make_tanh_transformed(mean, std)
    else:
        return make_multi_normal(mean, std)
```

**왜 Tanh?**
- 많은 environment가 [-1, 1] 범위의 action을 기대
- Tanh 변환: a = tanh(ã) where ã ~ N(μ, σ)
- 극단적인 action 방지

---

### 3. Critic Network: `gcb6206/networks/state_action_value_critic.py`

**목적**: Q(s, a)를 추정하는 신경망

**클래스**: `StateActionCritic`

#### 아키텍처

```
입력: 연결된 [state, action]
    ↓
[Linear Layer → Activation] × n_layers
    ↓
Output Layer (1 값)
    ↓
출력: Q(s, a) - 스칼라 값
```

#### 핵심 코드

```python
class StateActionCritic(nn.Module):
    def __init__(self, ob_dim, ac_dim, n_layers, size):
        super().__init__()
        # MLP: (obs_dim + action_dim) → hidden → ... → 1
        self.net = ptu.build_mlp(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        )

    def forward(self, obs, acs):
        # State와 action 연결
        input = torch.cat([obs, acs], dim=-1)
        # Q-value 출력
        return self.net(input).squeeze(-1)
```

**설계 참고사항**: 모든 action에 대한 Q(s, a)를 출력하는 DQN과 달리, 이것은 특정 (s, a) 쌍에 대한 Q-value를 출력합니다. 이는 continuous action space에 필요합니다.

---

### 4. Replay Buffer: `gcb6206/infrastructure/replay_buffer.py`

**목적**: Off-policy 학습을 위해 과거 경험을 저장하고 샘플링

**클래스**: `ReplayBuffer`

#### 왜 Replay Buffer?

**문제**: RL 데이터는 높은 상관관계를 가짐
- 연속적인 state는 유사함
- 과적합과 불안정성으로 이어짐

**해결책**: 경험을 저장하고 무작위로 샘플링
- 상관관계 제거
- 데이터 효율적 재사용
- Off-policy 학습 가능

#### 핵심 연산

```python
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.max_size = capacity
        self.observations = None  # 지연 할당
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dones = None

    def insert(self, observation, action, reward, next_observation, done):
        """하나의 transition (s, a, r, s', done) 추가"""
        # Circular buffer: 가득 차면 가장 오래된 것 덮어씀
        idx = self.size % self.max_size
        self.observations[idx] = observation
        # ... 다른 필드 저장
        self.size += 1

    def sample(self, batch_size):
        """무작위 transition batch 샘플링"""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }
```

**메모리 효율성**: Numpy 배열 사용, 한 번만 할당

---

### 5. Configuration: `gcb6206/env_configs/sac_config.py`

**목적**: 모든 하이퍼파라미터를 한 곳에 정의

**함수**: `sac_config(env_name, **kwargs)`

#### 핵심 하이퍼파라미터

```python
def sac_config(
    env_name: str,

    # Network 아키텍처
    hidden_size: int = 128,
    num_layers: int = 3,

    # Learning rate
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,

    # 훈련
    total_steps: int = 300000,
    batch_size: int = 128,
    discount: float = 0.99,

    # 탐색
    random_steps: int = 5000,  # 시작 시 무작위 action
    training_starts: int = 10000,  # 이후 훈련 시작

    # Target network
    use_soft_target_update: bool = False,
    target_update_period: int = None,  # Hard update
    soft_target_update_rate: float = None,  # Soft update τ

    # Actor-critic
    actor_gradient_type: str = "reinforce",  # 또는 "reparametrize"
    num_actor_samples: int = 1,
    num_critic_updates: int = 1,
    num_critic_networks: int = 1,  # Double-Q의 경우 2

    # Entropy
    use_entropy_bonus: bool = True,
    temperature: float = 0.1,  # entropy에 대한 α
):
    # 모든 config가 포함된 dict 반환
```

**Configuration 패턴**:
- Python의 기본 configuration
- YAML 파일로 재정의 (`experiments/sac/*.yaml`)
- 쉬운 실험 가능

---

### 6. Training Script: `gcb6206/scripts/run_hw4.py`

**목적**: 메인 training loop, 모든 것을 연결

#### Training Loop 구조

```python
def run_training_loop(config, logger, args):
    # 1. 설정
    env = config["make_env"]()
    agent = SoftActorCritic(...)
    replay_buffer = ReplayBuffer(...)

    observation, _ = env.reset()

    # 2. 메인 루프
    for step in range(config["total_steps"]):

        # 3. 데이터 수집
        if step < config["random_steps"]:
            action = env.action_space.sample()  # 무작위
        else:
            action = agent.get_action(observation)  # Policy에서

        next_observation, reward, done, truncated, info = env.step(action)
        replay_buffer.insert(observation, action, reward, next_observation, done)

        # 4. Agent 훈련
        if step >= config["training_starts"]:
            batch = replay_buffer.sample(config["batch_size"])
            batch = ptu.from_numpy(batch)  # numpy → torch
            update_info = agent.update(**batch, step=step)

            # 훈련 통계 로깅
            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)

        # 5. 평가
        if step % args.eval_interval == 0:
            eval_returns = evaluate(agent, eval_env)
            logger.log_scalar(np.mean(eval_returns), "eval_return", step)

        # 6. 완료 시 리셋
        if done or truncated:
            observation, _ = env.reset()
        else:
            observation = next_observation
```

**핵심 단계:**
1. **무작위 탐색** (0 to random_steps): 초기 replay buffer 구축
2. **학습** (training_starts 이후): Network 업데이트
3. **평가** (주기적): 탐색 노이즈 없이 테스트

---

## 데이터 흐름과 Training Loop

### 완전한 데이터 흐름 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                         Training Loop                           │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌──────────────────┴──────────────────┐
        ↓                                      ↓
┌──────────────┐                     ┌──────────────┐
│ Environment  │                     │ Replay Buffer│
│              │                     │              │
│ observation  │────────────────────→│ (s,a,r,s',d) │
│   ↓          │                     │              │
│ Agent.get    │                     │              │
│   _action()  │                     │ Sample batch │
│   ↓          │                     │      ↓       │
│ action       │                     └──────┼───────┘
│   ↓          │                            │
│ env.step()   │                            ↓
│   ↓          │                   ┌────────────────┐
│(s',r,done)   │                   │ Agent.update() │
└──────────────┘                   │                │
                                   │  Update critic │
                                   │  Update actor  │
                                   │  Update target │
                                   └────────────────┘
```

### 단계별 실행

**Step 1: Environment 상호작용**
```python
observation = env.reset()  # 초기 state 가져오기
# observation shape: [obs_dim]
```

**Step 2: Action 선택**
```python
action = agent.get_action(observation)
# get_action() 내부:
#   obs_torch = ptu.from_numpy(observation)[None]  # [1, obs_dim]
#   dist = self.actor(obs_torch)  # π(·|s) 가져오기
#   action = dist.sample()  # a ~ π(·|s) 샘플링
#   return ptu.to_numpy(action).squeeze(0)  # [action_dim]
```

**Step 3: Environment Step**
```python
next_obs, reward, done, truncated, info = env.step(action)
# next_obs: [obs_dim]
# reward: 스칼라
# done: bool
```

**Step 4: Replay Buffer에 저장**
```python
replay_buffer.insert(observation, action, reward, next_obs, done)
```

**Step 5: 샘플링 및 훈련**
```python
batch = replay_buffer.sample(batch_size)
# batch = {
#   "observations": [batch_size, obs_dim],
#   "actions": [batch_size, action_dim],
#   "rewards": [batch_size],
#   "next_observations": [batch_size, obs_dim],
#   "dones": [batch_size],
# }

batch = ptu.from_numpy(batch)  # torch tensor로 변환

update_info = agent.update(**batch, step=step)
# 훈련 메트릭의 dict 반환
```

### `agent.update()` 내부

```python
def update(self, observations, actions, rewards, next_observations, dones, step):
    # 1. Critic 여러 번 업데이트
    for _ in range(self.num_critic_updates):
        critic_info = self.update_critic(
            observations, actions, rewards, next_observations, dones
        )

    # 2. Actor 한 번 업데이트
    actor_info = self.update_actor(observations)

    # 3. Target network 업데이트
    if step % self.target_update_period == 0:  # Hard update
        self.update_target_critic()
    # 또는
    self.soft_update_target_critic(tau=0.005)  # Soft update

    return {**actor_info, **critic_info}
```

### `update_critic()` 내부

```python
def update_critic(self, obs, action, reward, next_obs, done):
    # 1. Target 계산 (gradient 없음)
    with torch.no_grad():
        next_action_dist = self.actor(next_obs)
        next_action = next_action_dist.sample()
        next_q = self.target_critic(next_obs, next_action)

        if self.use_entropy_bonus:
            entropy = self.entropy(next_action_dist)
            next_q += self.temperature * entropy

        target = reward + self.discount * (1 - done) * next_q

    # 2. Q-value 예측
    q_values = self.critic(obs, action)

    # 3. Loss 계산 및 업데이트
    loss = self.critic_loss(q_values, target)  # MSE

    self.critic_optimizer.zero_grad()
    loss.backward()
    self.critic_optimizer.step()
```

### `update_actor()` 내부

**REINFORCE 버전:**
```python
def actor_loss_reinforce(self, obs):
    # 1. Policy distribution 가져오기
    action_dist = self.actor(obs)

    # 2. Action 샘플링 (action에 대한 gradient 없음)
    with torch.no_grad():
        actions = action_dist.sample((num_samples,))
        q_values = self.critic(obs, actions)

    # 3. REINFORCE gradient 계산
    log_probs = action_dist.log_prob(actions)
    loss = -(log_probs * q_values).mean()

    return loss
```

**REPARAMETRIZE 버전:**
```python
def actor_loss_reparametrize(self, obs):
    # 1. Policy distribution 가져오기
    action_dist = self.actor(obs)

    # 2. Reparameterization으로 샘플링 (gradient 흐름!)
    action = action_dist.rsample()

    # 3. Q-value 계산 (action을 통한 gradient 흐름!)
    q_values = self.critic(obs, action)

    # 4. Loss (최대화하므로 음수)
    loss = -q_values.mean()

    return loss
```

---

## 핵심 개념과 디자인 패턴

### 1. 관심사의 분리

**신경망 (networks/)**: 순수 함수 근사
```python
class MLPPolicy:
    def forward(self, obs):
        # obs → distribution
        # RL 로직 없음, 단지 신경망
```

**Agent (agents/)**: RL 알고리즘 로직
```python
class SoftActorCritic:
    def update_critic(self, ...):
        # Bootstrapping, target network, loss 계산
        # 순수 RL 로직, NN은 networks/에 위임
```

**Infrastructure (infrastructure/)**: 재사용 가능한 유틸리티
```python
class ReplayBuffer:
    # 일반적인 experience replay
    # 모든 off-policy 알고리즘에 사용 가능
```

### 2. Configuration 관리

**계층 구조:**
1. **기본 기본값**: `sac_config.py`에
2. **실험별**: YAML 파일에
3. **명령줄**: argparse를 통해

**예시:**
```yaml
# experiments/sac/my_experiment.yaml
base_config: sac
env_name: HalfCheetah-v4
temperature: 0.2  # 기본값 재정의
```

```bash
python run_hw4.py -cfg experiments/sac/my_experiment.yaml --seed 42
```

### 3. 로깅과 모니터링

**TensorBoard 통합:**
```python
logger.log_scalar(value, name, step)
logger.log_scalar(q_values.mean().item(), "q_values", step)
```

**결과 보기:**
```bash
tensorboard --logdir data/
```

### 4. 모듈화된 Network 생성

**Factory 패턴:**
```python
def sac_config(...):
    def make_actor(obs_shape, action_dim):
        return MLPPolicy(
            ob_dim=obs_shape[0],
            ac_dim=action_dim,
            n_layers=num_layers,
            layer_size=hidden_size,
        )

    return {
        "agent_kwargs": {
            "make_actor": make_actor,
            # ...
        }
    }
```

**장점:**
- 아키텍처 교체 용이
- 지연된 구성
- Configuration 유연성

### 5. PyTorch 유틸리티

**디바이스 관리:**
```python
# pytorch_util.py
device = None  # 전역 device

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
    else:
        device = torch.device("cpu")

def from_numpy(data):
    """numpy → torch 변환, device로 이동"""
    return torch.from_numpy(data).float().to(device)

def to_numpy(tensor):
    """torch → numpy 변환"""
    return tensor.to("cpu").detach().numpy()
```

**MLP 빌더:**
```python
def build_mlp(input_size, output_size, n_layers, size, activation="tanh"):
    layers = []
    for _ in range(n_layers):
        layers += [nn.Linear(in_size, size), activation]
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    return nn.Sequential(*layers)
```

### 6. Target Network 업데이트

**Hard update (주기적 복사):**
```python
def update_target_critic(self):
    """가중치 완전 복사"""
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)
```

**Soft update (exponential moving average):**
```python
def soft_update_target_critic(self, tau=0.005):
    """Polyak averaging: θ' ← θ' + τ(θ - θ')"""
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
```

---

## 코드베이스 탐색 방법

### 시작점: 새로운 기능 이해하기

**질문: "SAC는 어떻게 action을 선택하나요?"**

1. **진입점에서 시작**: `scripts/run_hw4.py`
   - Action 선택 찾기: `action = agent.get_action(observation)`

2. **Agent로 따라가기**: `agents/sac_agent.py`
   - `get_action()` 메서드 찾기
   - `self.actor(observation)` 호출 확인

3. **Network로 따라가기**: `networks/mlp_policy.py`
   - `forward()` 메서드 찾기
   - Distribution을 반환하는 것 이해

4. **유틸리티 확인**: `infrastructure/distributions.py`
   - 커스텀 distribution 구현 확인

### 코드베이스 읽기: 추천 순서

**SAC 이해를 위해:**

1. 시작: `scripts/run_hw4.py`
   - 전체 training loop
   - 언제 무슨 일이 일어나는지

2. 그 다음: `agents/sac_agent.py`
   - `__init__`: 어떤 컴포넌트가 존재하는지
   - `get_action`: Action이 어떻게 선택되는지
   - `update`: 메인 학습 로직

3. 그 다음: `agents/sac_agent.py` (상세)
   - `update_critic`: Value가 어떻게 학습되는지
   - `update_actor`: Policy가 어떻게 학습되는지

4. 그 다음: Network 아키텍처
   - `networks/mlp_policy.py`: Actor
   - `networks/state_action_value_critic.py`: Critic

5. 마지막: Infrastructure
   - `infrastructure/replay_buffer.py`: 데이터 저장
   - `infrastructure/utils.py`: 헬퍼 함수

### 디버깅 팁

**Q-value가 폭발/소멸하나요?**
- 확인: `update_critic()` - Target이 올바르게 계산되는가?
- 확인: Target network 업데이트 - 발생하고 있는가?
- 로그: Q-value, target value, critic loss

**Policy가 개선되지 않나요?**
- 확인: `update_actor()` - Loss가 감소하는가?
- 확인: Entropy - 너무 높거나 낮은가?
- 로그: Actor loss, entropy, policy std

**전혀 학습이 안 되나요?**
- 확인: Replay buffer에 충분한 샘플이 있는가
- 확인: `training_starts` 파라미터
- 확인: Learning rate
- 검증: Gradient가 흐르는가 (grad norm 출력)

### 일반적인 수정 패턴

**Network 아키텍처 변경:**
```python
# env_configs/sac_config.py에서
def make_actor(obs_shape, action_dim):
    return MLPPolicy(
        ...,
        n_layers=5,  # 3에서 변경
        layer_size=256,  # 128에서 변경
    )
```

**새로운 하이퍼파라미터 추가:**
```python
# sac_config.py에서
def sac_config(..., my_new_param=default_value):
    return {
        "agent_kwargs": {
            ...,
            "my_new_param": my_new_param,
        }
    }

# sac_agent.py에서
class SoftActorCritic:
    def __init__(self, ..., my_new_param):
        self.my_new_param = my_new_param
```

**새로운 로깅 추가:**
```python
# sac_agent.py에서
def update_critic(...):
    ...
    return {
        "critic_loss": loss.item(),
        "q_values": q_values.mean().item(),
        "my_new_metric": new_value.item(),  # 이것 추가
    }
```

---

## 고급 주제

### 1. Reparameterization Trick

**문제**: 샘플링을 통한 backpropagation 불가
```python
a ~ N(μ(s), σ(s))  # ∇_θ를 어떻게 얻나?
```

**해결책**: Reparameterize
```python
ε ~ N(0, 1)  # 표준 정규분포
a = μ(s) + σ(s) · ε  # 이제 μ와 σ를 통해 gradient 흐름!
```

**코드에서:**
```python
# .sample(): Gradient 없음
action = distribution.sample()

# .rsample(): Reparameterized, gradient 흐름
action = distribution.rsample()
```

### 2. Bounded Action을 위한 Tanh 변환

**문제**: Environment는 [-1, 1]의 action을 기대하지만, Gaussian은 무한

**해결책**: Tanh를 통해 압축
```python
a_unbounded ~ N(μ, σ)
a = tanh(a_unbounded)  # 이제 a ∈ (-1, 1)
```

**확률에 대한 보정:**
```python
log π(a|s) = log π(a_unbounded|s) - log|da/da_unbounded|
            = log π(a_unbounded|s) - log(1 - tanh²(a_unbounded))
```

### 3. 다중 Critic Network

**왜?**
- Q-learning에서 과대평가 편향
- 단일 critic은 낙관적인 경향

**해결책:**
- **Double-Q**: 두 critic, 각각 다른 것을 target으로 사용
- **Clipped Double-Q**: Target에 min(Q1, Q2) 사용
- **Mean**: 여러 critic의 평균

**구현:**
```python
self.critics = nn.ModuleList([
    make_critic(...) for _ in range(num_critic_networks)
])

def critic(self, obs, action):
    # 반환: [num_critics, batch_size]
    return torch.stack([critic(obs, action) for critic in self.critics])
```

### 4. Entropy-Regularized RL

**목표:**
```
J(π) = E[∑ γ^t (r_t + α H(π(·|s_t)))]
```

**효과:**
- 조기 수렴 방지
- 강건한 policy 학습
- 자동 탐색

**Temperature (α):**
- 높은 α: 더 무작위 (더 많은 탐색)
- 낮은 α: 더 결정적 (더 많은 활용)
- 자동으로 학습 가능 (이 과제에서는 아님)

---

## 용어집

**Actor**: Action distribution을 출력하는 policy network π_θ(a|s)

**Critic**: 예상 return을 추정하는 value network Q_φ(s,a)

**Bellman Equation**: 재귀 관계: Q(s,a) = r + γE[Q(s',a')]

**Bootstrapping**: 학습 target에서 미래 value의 추정치 사용

**Discount Factor (γ)**: 미래 대 즉각 reward를 얼마나 중시할지

**Entropy**: 무작위성의 측정: H(π) = -E[log π(a|s)]

**Episode**: 시작 state에서 종료 state까지의 완전한 시퀀스

**Off-policy**: 다른 policy가 생성한 데이터로부터 학습

**On-policy**: 현재 policy가 생성한 데이터로부터 학습

**Policy**: State에서 action으로의 매핑: π(a|s)

**Replay Buffer**: 과거 경험을 위한 저장소 (s, a, r, s', done)

**Reparameterization**: Gradient 흐름을 위한 a = μ + σ·ε 기법

**Return**: Discounted reward의 합: G_t = ∑_{k=0}^∞ γ^k r_{t+k}

**Reward**: Environment로부터의 스칼라 피드백 신호

**State**: 시간 t에서 environment의 완전한 설명

**Target Network**: 안정성을 위해 천천히 업데이트되는 value network의 복사본

**Temperature (α)**: SAC에서 entropy bonus의 가중치

**Trajectory/Rollout**: (s, a, r) 튜플의 시퀀스

**Value Function**: 예상 return: V(s) = E[G_t | s_t = s]

---

## 다음 단계

### 이해를 깊게 하려면:

1. **SAC 논문 읽기**: https://arxiv.org/abs/1801.01290
2. **변형 구현하기**: 다양한 아키텍처, 하이퍼파라미터 시도
3. **실험하기**: 새로운 environment 시도, 학습된 policy 시각화
4. **알고리즘 비교**: TD3, PPO 구현하고 비교

### 자료:

- **OpenAI Spinning Up**: https://spinningup.openai.com/
- **Berkeley CS 285**: http://rail.eecs.berkeley.edu/deeprlcourse/
- **PyTorch 튜토리얼**: https://pytorch.org/tutorials/
- **Gymnasium 문서**: https://gymnasium.farama.org/

---

## 요약

이 코드베이스는 완전하고 프로덕션 수준의 SAC agent를 구현합니다. 핵심 요점:

1. **모듈화 설계**: 관심사 분리 (알고리즘, network, infrastructure)
2. **Configuration 주도**: 하이퍼파라미터 실험 용이
3. **모범 사례**: Target network, replay buffer, entropy regularization
4. **확장 가능**: 수정 및 확장 용이

이 코드베이스를 이해하면 다음을 얻을 수 있습니다:
- **실용적인 RL 구현 기술**
- **딥러닝 엔지니어링 패턴**
- **연구 및 프로덕션 RL을 위한 기초**

즐거운 학습 되세요! 🚀
