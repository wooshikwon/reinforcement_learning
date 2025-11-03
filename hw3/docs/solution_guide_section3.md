# Homework 3 Solution Guide - Section 3

## 5. DQN Agent Implementation (DQN 에이전트 구현)

**파일 위치**: `gcb6206/agents/dqn_agent.py`

### 5.1 클래스 구조와 초기화

```python
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
```

#### 파라미터 설명

**`observation_shape: Sequence[int]`**
- 관찰 공간의 shape
- CartPole: `(4,)` - 4차원 벡터 [위치, 속도, 각도, 각속도]
- Atari: `(4, 84, 84)` - 4개 stacked 84×84 grayscale 이미지

**`num_actions: int`**
- 행동 공간의 크기
- CartPole: 2 (왼쪽, 오른쪽)
- BankHeist: 18 (조이스틱 조합)

**`make_critic: Callable`**
- **Factory function**으로 critic network 생성
- 왜 network 자체가 아니라 factory function을 받는가?
  - Target network도 같은 구조로 만들어야 하기 때문
  - Config에서 쉽게 다른 architecture로 교체 가능

**CartPole의 make_critic 예시:**

```python
def make_critic(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module:
    return ptu.build_mlp(
        input_size=np.prod(observation_shape),  # 4
        output_size=num_actions,                # 2
        n_layers=2,                              # hidden layers
        size=64,                                 # hidden size
    )
```

이것은 다음과 같은 MLP를 생성합니다:
```
Input (4) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(2) → Output
```

**Atari의 make_critic 예시:**

```python
def make_critic(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module:
    return nn.Sequential(
        PreprocessAtari(),  # uint8 → float32, /255
        nn.Conv2d(4, 32, kernel_size=8, stride=4),  # (4,84,84) → (32,20,20)
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2), # (32,20,20) → (64,9,9)
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1), # (64,9,9) → (64,7,7)
        nn.ReLU(),
        nn.Flatten(),                                # (64,7,7) → (3136,)
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear(512, num_actions),                # → (num_actions,)
    )
```

이것은 **Nature DQN** 논문의 architecture입니다.

**`make_optimizer: Callable`**
- Optimizer factory function
- 예: `lambda params: torch.optim.Adam(params, lr=0.001)`

**`make_lr_schedule: Callable`**
- Learning rate scheduler factory
- 예: `lambda optimizer: torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)`

**`discount: float`**
- Discount factor $\gamma$
- 보통 0.99 (CartPole, Atari 공통)
- 의미: 미래 보상의 중요도
  - $\gamma = 0$: 즉각적인 보상만 고려
  - $\gamma = 1$: 모든 미래 보상을 동등하게 고려
  - $\gamma = 0.99$: 100 step 후 보상은 $0.99^{100} \approx 0.37$배로 할인

**`target_update_period: int`**
- Target network 업데이트 주기
- CartPole: 1000 steps
- Atari: 2000 steps
- 너무 자주 업데이트: 불안정 (moving target)
- 너무 느리게 업데이트: 학습 느림

**`use_double_q: bool`**
- Double DQN 사용 여부
- Section 5에서 구현 예정

**`clip_grad_norm: Optional[float]`**
- Gradient clipping threshold
- Atari: 10.0 (큰 gradient 방지)
- CartPole: None (clipping 없음)

#### 초기화 코드

```python
super().__init__()

self.critic = make_critic(observation_shape, num_actions)
self.target_critic = make_critic(observation_shape, num_actions)
self.critic_optimizer = make_optimizer(self.critic.parameters())
self.lr_scheduler = make_lr_schedule(self.critic_optimizer)
```

**`super().__init__()`**
- `nn.Module`의 생성자 호출
- PyTorch의 모듈 시스템에 등록하기 위해 필요
- 이것이 있어야 `.parameters()`, `.to(device)` 등이 작동

**두 개의 Critic Network:**

1. **Online Network (`self.critic`)**:
   - 매 step 업데이트
   - 행동 선택에 사용
   - 현재 학습 중인 Q-function

2. **Target Network (`self.target_critic`)**:
   - 주기적으로만 업데이트 (복사)
   - 타겟 값 계산에만 사용
   - 학습 안정성을 위해 필요

**Loss Function:**

```python
self.critic_loss = nn.MSELoss()
```

Mean Squared Error Loss:
$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

DQN에서:
- $y_i$: target value = $r + \gamma \max_{a'} Q_{\theta^-}(s', a')$
- $\hat{y}_i$: predicted value = $Q_\theta(s, a)$

**Target Network 초기화:**

```python
self.update_target_critic()

def update_target_critic(self):
    self.target_critic.load_state_dict(self.critic.state_dict())
```

- `state_dict()`: 모델의 모든 파라미터를 dictionary로 반환
- `load_state_dict()`: dictionary에서 파라미터 로드
- 결과: target network의 가중치를 online network와 동일하게 설정

---

### 5.2 get_action() 구현 - Epsilon-Greedy

```python
def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
    """
    Used for evaluation.
    """
    observation = ptu.from_numpy(np.asarray(observation))[None]
    # TODO(student): get the action from the critic using an epsilon-greedy strategy
    action = ...
    return ptu.to_numpy(action).squeeze(0).item()
```

#### 입력 전처리

```python
observation = ptu.from_numpy(np.asarray(observation))[None]
```

이 한 줄은 세 가지 변환을 수행합니다:

1. **`np.asarray(observation)`**
   - 이미 numpy array일 수 있지만, 확실히 하기 위해
   - 복사 없이 array view 생성

2. **`ptu.from_numpy(...)`**
   - NumPy array → PyTorch tensor
   - 자동으로 GPU로 이동 (설정된 경우)
   ```python
   # ptu.from_numpy의 내부 구현 (간단화):
   def from_numpy(data):
       return torch.from_numpy(data).float().to(ptu.device)
   ```

3. **`[None]` 또는 `.unsqueeze(0)`**
   - Batch dimension 추가
   - Before: `(4,)` (CartPole) 또는 `(4, 84, 84)` (Atari)
   - After: `(1, 4)` 또는 `(1, 4, 84, 84)`
   - 왜? Neural network는 batch input을 기대

#### Epsilon-Greedy 구현

**알고리즘:**

$$
a = \begin{cases}
\text{random action} & \text{확률 } \epsilon \\
\arg\max_a Q(s, a) & \text{확률 } 1 - \epsilon
\end{cases}
$$

**코드 (Solution):**

```python
if np.random.random() < epsilon:
    # Exploration: 랜덤 행동
    action = np.random.randint(self.num_actions)
else:
    # Exploitation: greedy 행동
    with torch.no_grad():
        q_values = self.critic(observation)  # shape: (1, num_actions)
        action = torch.argmax(q_values, dim=1)  # shape: (1,)
```

**자세한 설명:**

1. **`np.random.random() < epsilon`**
   - `np.random.random()`: 0과 1 사이 uniform random float
   - 예: `epsilon=0.1`이면 10% 확률로 True

2. **Exploration 분기:**
   ```python
   action = np.random.randint(self.num_actions)
   ```
   - `0`부터 `num_actions-1` 사이의 정수 랜덤 선택
   - 모든 행동이 동등한 확률

3. **Exploitation 분기:**
   ```python
   with torch.no_grad():
       q_values = self.critic(observation)
       action = torch.argmax(q_values, dim=1)
   ```

   **`torch.no_grad()`의 의미:**
   - Gradient 계산을 비활성화
   - 왜? 여기서는 행동 선택만 하지, 학습하지 않음
   - 메모리 절약 + 속도 향상

   **`self.critic(observation)`:**
   - Input: `(1, 4)` 또는 `(1, 4, 84, 84)`
   - Output: `(1, num_actions)` - 각 행동의 Q-value
   - 예: `[[0.5, 0.8]]` (CartPole, 2 actions)

   **`torch.argmax(q_values, dim=1)`:**
   - `dim=1`: 각 row에서 최대값의 인덱스
   - Input: `[[0.5, 0.8]]`
   - Output: `[1]` (tensor)

#### 출력 후처리

```python
return ptu.to_numpy(action).squeeze(0).item()
```

1. **`ptu.to_numpy(action)`**
   - PyTorch tensor → NumPy array
   - GPU에 있으면 CPU로 이동
   - `[1]` (tensor) → `[1]` (ndarray)

2. **`.squeeze(0)`**
   - 첫 번째 차원 제거 (batch dimension)
   - `[1]` → `1` (scalar array)

3. **`.item()`**
   - NumPy array → Python int
   - `array(1)` → `1`
   - 환경이 Python int를 기대하기 때문

**전체 흐름 예시:**

```
Input: observation = [0.1, 0.2, 0.3, 0.4] (numpy)
       epsilon = 0.1

Step 1: to tensor + batch
  → tensor([[0.1, 0.2, 0.3, 0.4]])

Step 2: random check
  np.random.random() = 0.85 > 0.1 → greedy!

Step 3: Q-values
  critic(obs) = tensor([[0.5, 0.8]])

Step 4: argmax
  torch.argmax(..., dim=1) = tensor([1])

Step 5: to numpy + squeeze + item
  → array([1]) → array(1) → 1

Output: action = 1 (int)
```

---

### 5.3 update_critic() 구현 - DQN Loss

```python
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
```

#### 입력 텐서 구조

모든 입력은 **배치**로 주어집니다 (`batch_size=32`라고 가정):

- `obs`: `(32, 4)` 또는 `(32, 4, 84, 84)`
- `action`: `(32,)` - integer actions
- `reward`: `(32,)` - scalar rewards
- `next_obs`: `(32, 4)` 또는 `(32, 4, 84, 84)`
- `done`: `(32,)` - boolean (0 or 1)

**`(batch_size,) = reward.shape`의 의미:**

Tuple unpacking for assertion:
```python
reward.shape = (32,)
(batch_size,) = (32,)  # batch_size = 32
```

만약 shape이 `(32, 1)`이면 에러 발생 → shape 검증

#### Target Values 계산

```python
# Compute target values
with torch.no_grad():
    # TODO(student): compute target values
    next_qa_values = ...

    if self.use_double_q:
        # Choose action with argmax of critic network
        next_action = ...
    else:
        # Choose action with argmax of target critic network
        next_action = ...
    next_q_values = ... # see torch.gather
    target_values = ...
```

**Solution:**

```python
with torch.no_grad():
    # Step 1: 다음 상태의 모든 Q-values 계산 (target network 사용)
    next_qa_values = self.target_critic(next_obs)  # (batch_size, num_actions)

    if self.use_double_q:
        # Double DQN: online network로 행동 선택
        next_action = torch.argmax(self.critic(next_obs), dim=1)  # (batch_size,)
    else:
        # Vanilla DQN: target network로 행동 선택
        next_action = torch.argmax(next_qa_values, dim=1)  # (batch_size,)

    # Step 2: 선택된 행동의 Q-value 추출
    next_q_values = torch.gather(next_qa_values, 1, next_action.unsqueeze(1)).squeeze(1)
    # (batch_size,)

    # Step 3: TD target 계산
    target_values = reward + self.discount * (1 - done) * next_q_values
    # (batch_size,)
```

**자세한 설명:**

**Step 1: Next Q-values**

```python
next_qa_values = self.target_critic(next_obs)
```

- **Target network 사용** (중요!)
- Input: `(32, 4)` → Output: `(32, 2)` (CartPole 예시)
- 각 다음 상태에서 모든 행동의 Q-value

예시:
```python
next_qa_values = [
    [Q(s'₁, a₀), Q(s'₁, a₁)],  # 첫 번째 샘플
    [Q(s'₂, a₀), Q(s'₂, a₁)],  # 두 번째 샘플
    ...
]
```

**Step 2: Action Selection**

**Vanilla DQN:**
```python
next_action = torch.argmax(next_qa_values, dim=1)
```
- 각 row에서 최대 Q-value의 인덱스
- `[[0.5, 0.8], [0.9, 0.3]]` → `[1, 0]`

**Double DQN:**
```python
next_action = torch.argmax(self.critic(next_obs), dim=1)
```
- **Online network**로 행동 선택
- Overestimation bias 감소 (Section 5에서 설명)

**Step 3: torch.gather() - Q-value 추출**

`torch.gather()`는 고급 인덱싱 함수입니다.

**사용법:**
```python
torch.gather(input, dim, index)
```

**예시:**

```python
next_qa_values = tensor([
    [0.5, 0.8],  # row 0
    [0.9, 0.3],  # row 1
])
next_action = tensor([1, 0])  # row 0에서 index 1, row 1에서 index 0

# gather를 위해 shape 맞추기
next_action.unsqueeze(1)  # (2,) → (2, 1)
# tensor([[1], [0]])

# gather 실행
result = torch.gather(next_qa_values, 1, next_action.unsqueeze(1))
# dim=1: column 방향으로 인덱싱
# (2, 1) 출력
# [[0.8],  # row 0, column 1
#  [0.9]]  # row 1, column 0

result.squeeze(1)  # (2, 1) → (2,)
# tensor([0.8, 0.9])
```

**시각화:**

```
next_qa_values:          next_action:
┌───────┬───────┐        ┌───┐
│  0.5  │  0.8  │   ←───→│ 1 │  → select column 1 → 0.8
├───────┼───────┤        ├───┤
│  0.9  │  0.3  │   ←───→│ 0 │  → select column 0 → 0.9
└───────┴───────┘        └───┘
```

**왜 이렇게 복잡한가?**

간단한 인덱싱은 작동하지 않습니다:
```python
# 이건 안 됨:
next_q_values = next_qa_values[next_action]  # 틀린 결과!

# 왜? 이것은 row 인덱싱이 됨
next_qa_values[[1, 0]]  # row 1과 row 0 선택 (우리가 원하는 게 아님)
```

우리는 **각 row에서 다른 column**을 선택해야 합니다.

**Step 4: TD Target**

```python
target_values = reward + self.discount * (1 - done) * next_q_values
```

**TD (Temporal Difference) target:**

$$y = r + \gamma (1 - d) \max_{a'} Q_{\theta^-}(s', a')$$

- $r$: 즉각적인 보상
- $\gamma$: discount factor (0.99)
- $(1 - d)$: terminal mask
  - $d = 0$ (not done): 미래 보상 포함
  - $d = 1$ (done): 미래 보상 제외 (episode 끝)
- $\max_{a'} Q_{\theta^-}(s', a')$: 다음 상태의 최대 Q-value

**예시 계산:**

```python
reward = tensor([1.0, -1.0, 0.5])
discount = 0.99
done = tensor([0.0, 1.0, 0.0])  # 두 번째는 terminal
next_q_values = tensor([0.8, 0.5, 0.3])

target_values = reward + 0.99 * (1 - done) * next_q_values
              = [1.0, -1.0, 0.5] + 0.99 * [1.0, 0.0, 1.0] * [0.8, 0.5, 0.3]
              = [1.0, -1.0, 0.5] + [0.792, 0.0, 0.297]
              = [1.792, -1.0, 0.797]
```

두 번째 샘플에서 `done=1`이므로 미래 보상이 0이 됩니다.

#### Critic Training

```python
# TODO(student): train the critic with the target values
# Use self.critic_loss for calculating the loss
qa_values = ...
q_values = ... # Compute from the data actions; see torch.gather
loss = ...
```

**Solution:**

```python
# Step 1: 현재 Q-values (online network)
qa_values = self.critic(obs)  # (batch_size, num_actions)

# Step 2: 실제 선택한 행동의 Q-value 추출
q_values = torch.gather(qa_values, 1, action.unsqueeze(1)).squeeze(1)
# (batch_size,)

# Step 3: Loss 계산
loss = self.critic_loss(q_values, target_values)
```

**각 단계 설명:**

**Step 1: Current Q-values**

```python
qa_values = self.critic(obs)
```

- **Online network** 사용
- 현재 상태에서 모든 행동의 Q-value 예측
- 이것이 학습 대상

**Step 2: Action Values**

```python
q_values = torch.gather(qa_values, 1, action.unsqueeze(1)).squeeze(1)
```

- 실제로 **선택했던 행동**의 Q-value만 필요
- `action`은 replay buffer에서 온 실제 행동

예시:
```python
qa_values = [
    [0.5, 0.8],  # 첫 번째 샘플
    [0.9, 0.3],  # 두 번째 샘플
]
action = [0, 1]  # 실제 선택한 행동

q_values = [0.5, 0.3]  # 각 샘플에서 선택한 행동의 Q-value
```

**Step 3: MSE Loss**

```python
loss = self.critic_loss(q_values, target_values)
     = nn.MSELoss()(q_values, target_values)
```

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (Q_\theta(s_i, a_i) - y_i)^2$$

여기서:
- $Q_\theta(s_i, a_i)$: `q_values` - 현재 예측
- $y_i$: `target_values` - TD target

**예시 계산:**

```python
q_values = tensor([0.5, 0.3, 0.6])
target_values = tensor([1.792, -1.0, 0.797])

loss = MSELoss(q_values, target_values)
     = mean([(0.5-1.792)², (0.3-(-1.0))², (0.6-0.797)²])
     = mean([1.669, 1.690, 0.039])
     = 1.133
```

#### Gradient Descent

```python
self.critic_optimizer.zero_grad()
loss.backward()
grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
    self.critic.parameters(), self.clip_grad_norm or float("inf")
)
self.critic_optimizer.step()

self.lr_scheduler.step()
```

**각 줄 설명:**

**1. `self.critic_optimizer.zero_grad()`**

Gradient 버퍼를 0으로 초기화

PyTorch는 기본적으로 gradient를 **누적**합니다:
```python
# 만약 zero_grad() 안 하면:
loss1.backward()  # grad += ∂loss1/∂θ
loss2.backward()  # grad += ∂loss2/∂θ  (누적!)
```

우리는 각 batch에 대해 **새로운** gradient가 필요하므로 초기화 필수.

**2. `loss.backward()`**

Backpropagation 수행

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{2}{N} \sum_{i=1}^{N} (Q_\theta(s_i, a_i) - y_i) \frac{\partial Q_\theta(s_i, a_i)}{\partial \theta}$$

PyTorch의 autograd가 자동으로 계산:
- Computational graph를 역방향으로 탐색
- Chain rule 적용
- 각 파라미터의 gradient 저장

**3. Gradient Clipping**

```python
grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
    self.critic.parameters(),
    self.clip_grad_norm or float("inf")
)
```

**목적**: Gradient exploding 방지

**작동 원리:**

1. 모든 파라미터의 gradient norm 계산:
   $$\|g\| = \sqrt{\sum_i g_i^2}$$

2. Threshold보다 크면 scaling:
   $$g' = \frac{\text{threshold}}{\|g\|} \cdot g$$

**예시:**

```python
# Before clipping:
gradients = [3.0, 4.0]
norm = sqrt(3² + 4²) = 5.0

# threshold = 1.0
scaled_gradients = (1.0 / 5.0) * [3.0, 4.0] = [0.6, 0.8]
new_norm = sqrt(0.6² + 0.8²) = 1.0  # threshold로 제한됨
```

**왜 필요한가?**

Atari 게임에서 보상 scale이 크고 가변적:
- BankHeist: 보상이 10, 20, 50 등
- Q-values가 빠르게 커질 수 있음
- Gradient가 폭발 → 학습 불안정

CartPole은 보상이 항상 1이므로 clipping 불필요.

**4. `self.critic_optimizer.step()`**

파라미터 업데이트:

$$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$$

- $\alpha$: learning rate
- $\nabla_\theta \mathcal{L}$: gradient (backward()로 계산됨)

Adam optimizer의 경우 실제로는 더 복잡:
- Momentum 사용
- Adaptive learning rate
- Bias correction

**5. `self.lr_scheduler.step()`**

Learning rate 업데이트

예시 (Atari):
```python
schedule = PiecewiseSchedule([
    (0, 1.0),
    (20000, 1.0),
    (500000, 0.5),
])

# Step 0-20000: lr = lr₀ * 1.0
# Step 20000-500000: lr = lr₀ * (1.0 → 0.5) linearly
# Step 500000+: lr = lr₀ * 0.5
```

CartPole은 constant schedule (항상 1.0).

#### 반환값

```python
return {
    "critic_loss": loss.item(),
    "q_values": q_values.mean().item(),
    "target_values": target_values.mean().item(),
    "grad_norm": grad_norm.item(),
}
```

`.item()`: Tensor → Python scalar (로깅을 위해)

이 dictionary는 TensorBoard에 기록됩니다.

---

### 5.4 update() 구현 - Main Update Logic

```python
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
    critic_stats = ...
    return critic_stats
```

**Solution:**

```python
# Critic 업데이트
critic_stats = self.update_critic(obs, action, reward, next_obs, done)

# Target network 주기적 업데이트
if step % self.target_update_period == 0:
    self.update_target_critic()

return critic_stats
```

**설명:**

**1. Critic 업데이트**

```python
critic_stats = self.update_critic(obs, action, reward, next_obs, done)
```

매 step 실행:
- Batch에서 학습
- Online network의 파라미터 $\theta$ 업데이트

**2. Target Network 업데이트**

```python
if step % self.target_update_period == 0:
    self.update_target_critic()
```

**`step % self.target_update_period == 0`의 의미:**

- `target_update_period = 1000`이면
- Step 0, 1000, 2000, 3000, ... 에서만 True

**업데이트 과정:**

```python
def update_target_critic(self):
    self.target_critic.load_state_dict(self.critic.state_dict())
```

- Online network의 가중치를 target network로 **복사**
- $\theta^- \leftarrow \theta$ (hard update)

**Hard Update vs Soft Update:**

현재 구현은 **hard update**:
$$\theta^- \leftarrow \theta$$

Alternative: **soft update** (DDPG, SAC에서 사용):
$$\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$$

- $\tau = 0.001$ 같은 작은 값
- 매 step 조금씩 업데이트
- 더 부드러운 타겟 변화

DQN은 hard update 사용 (원 논문).

**왜 주기적 업데이트인가?**

1. **안정성**: 타겟이 일정 기간 고정
2. **계산 효율**: 복사는 비용이 큼 (매 step 불필요)
3. **경험적 성공**: 1000-2000 steps이 잘 작동

**전체 흐름:**

```
Step 0:
  update_critic() → θ 변경
  step % 1000 == 0 → target 업데이트: θ⁻ ← θ

Step 1-999:
  update_critic() → θ 계속 변경
  target 고정 (θ⁻ 변화 없음)

Step 1000:
  update_critic() → θ 변경
  step % 1000 == 0 → target 업데이트: θ⁻ ← θ

...
```

---
