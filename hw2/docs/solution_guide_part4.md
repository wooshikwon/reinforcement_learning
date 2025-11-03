# GCB6206 Homework 2: Policy Gradients 완전 해설 가이드 (Part 4)

## 6. Neural Network Baseline 구현

### 6.1 왜 Baseline이 필요한가?

#### 6.1.1 High Variance 문제 재방문

Policy Gradient의 가장 큰 문제는 **high variance**야. 예를 들어 CartPole에서:

```python
# Episode 1
states = [s₀, s₁, s₂, ..., s₁₉₉]
rewards = [1, 1, 1, ..., 1]  # 200 steps 생존
total_return = 200

# Episode 2 (똑같은 초기 상태, 똑같은 초기 행동)
states = [s₀, s₁, s₂, ..., s₉]
rewards = [1, 1, 1, ..., 1]  # 10 steps만 생존 (운이 나빠서)
total_return = 10
```

똑같은 행동 $a_0$인데도:
- Episode 1: $\nabla_\theta \log \pi(a_0|s_0) \cdot 200$ → 크게 증가!
- Episode 2: $\nabla_\theta \log \pi(a_0|s_0) \cdot 10$ → 약간만 증가

**이 차이가 variance**야. 같은 행동인데 평가가 20배 차이 나.

#### 6.1.2 Baseline의 아이디어

핵심 아이디어: **절대적인 return이 아니라 상대적인 좋고 나쁨을 평가하자!**

$$
\text{Advantage}(s_t, a_t) = Q(s_t, a_t) - \underbrace{V(s_t)}_{\text{baseline}}
$$

- $Q(s_t, a_t)$: 상태 $s_t$에서 행동 $a_t$를 했을 때의 가치
- $V(s_t)$: 상태 $s_t$의 평균적인 가치 (baseline)
- **Advantage**: "평균보다 얼마나 좋은가?"

**예시**:

상태 $s$에서:
- 행동 A의 Q-value = 50
- 행동 B의 Q-value = 30
- V(s) = 40 (평균)

Advantage:
- A: 50 - 40 = +10 (평균보다 좋음) ✓
- B: 30 - 40 = -10 (평균보다 나쁨) ✗

#### 6.1.3 수학적 정당화

**Baseline을 빼도 policy gradient는 unbiased**야! 증명:

$$
\begin{align}
\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)]
&= \sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) \cdot b(s) \\
&= \sum_a \nabla_\theta \pi_\theta(a|s) \cdot b(s) \\
&= b(s) \nabla_\theta \sum_a \pi_\theta(a|s) \\
&= b(s) \nabla_\theta 1 \\
&= 0
\end{align}
$$

중요한 건, $b(s)$가 **상태에만 의존**하고 **행동에 독립적**이어야 해.

---

### 6.2 Value Function의 정의

#### 6.2.1 이론적 정의

$$
V^\pi(s_t) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'} \mid s_t \right]
$$

**의미**: 상태 $s_t$에서 시작해서 정책 $\pi$를 따랐을 때 받을 기대 누적 보상

#### 6.2.2 Monte Carlo 추정

실제로는 기댓값을 모르니까, 샘플링한 trajectory로 추정:

$$
V^\pi(s_t) \approx \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}
$$

이게 바로 **reward-to-go**! 하지만 이것도 variance가 높아.

#### 6.2.3 Function Approximation

**더 나은 방법**: 신경망으로 $V^\pi$를 근사하자!

$$
V^\pi_\phi(s) \approx V^\pi(s)
$$

- $\phi$: Value network의 파라미터
- 입력: 상태 $s$
- 출력: 예상 누적 보상 (scalar)

**장점**:
- **Generalization**: 비슷한 상태에 대해 비슷한 value 예측
- **Lower variance**: 여러 trajectory의 정보를 평균 냄

---

### 6.3 ValueCritic 클래스 구현

#### 6.3.1 전체 코드 구조

```python
# critics.py

class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value."""

    def __init__(self, ob_dim, n_layers, layer_size, learning_rate):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,  # Value는 scalar
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        return None

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        # TODO: implement
        loss = None
        return {"Baseline Loss": ptu.to_numpy(loss)}
```

#### 6.3.2 `__init__` 메서드 상세

**`self.network = ptu.build_mlp(...)`**:
- **입력**: 상태 (observation)
- **출력**: Value 추정값 (scalar)

**왜 `output_size=1`?**
- Value function은 **하나의 숫자** (누적 보상 예측)
- 예: "이 상태에서 평균적으로 50의 보상을 받을 것"

**네트워크 구조 예시** (ob_dim=4, n_layers=2, layer_size=64):
```
Input (4개 상태 변수)
  → Linear(4, 64) → Tanh()
  → Linear(64, 64) → Tanh()
  → Linear(64, 1)
  → Output (1개 scalar: value)
```

**Actor vs Critic 비교**:

| | Actor (Policy) | Critic (Value) |
|---|---|---|
| 입력 | 상태 | 상태 |
| 출력 | 행동 분포 (ac_dim) | Value (1) |
| 목적 | 좋은 행동 선택 | 상태 가치 평가 |

#### 6.3.3 `forward()` 메서드 구현

```python
def forward(self, obs: torch.Tensor) -> torch.Tensor:
    value = self.network(obs)
    return value.squeeze(dim=-1)
```

**Line 1**: `value = self.network(obs)`
- **입력**: `obs` shape: `(batch_size, ob_dim)`
  - 예: `(128, 4)` (CartPole)
- **출력**: `value` shape: `(batch_size, 1)`
  - 예: `(128, 1)`

**Line 2**: `return value.squeeze(dim=-1)`

**`squeeze()` 함수**:
- 크기가 1인 차원을 제거
- `squeeze(dim=-1)`: 마지막 차원이 1이면 제거

예시:
```python
value = torch.tensor([[5.2], [3.8], [6.1]])  # shape: (3, 1)
value.squeeze(dim=-1)
# tensor([5.2, 3.8, 6.1])  # shape: (3,)
```

**왜 squeeze?**
- 나중에 Q-values와 빼기 연산할 때, shape이 같아야 해
- Q-values shape: `(batch_size,)`
- Values shape: `(batch_size,)` ← squeeze 후

**구체적 예시**:
```python
obs = torch.tensor([
    [0.1, 0.5, -0.2, 0.3],  # State 1
    [0.2, 0.6, -0.1, 0.2],  # State 2
])  # shape: (2, 4)

value = critic.forward(obs)
# value = tensor([12.5, 8.3])  # shape: (2,)
# "State 1에서 12.5의 보상 예상"
# "State 2에서 8.3의 보상 예상"
```

---

### 6.4 `update()` 메서드 구현

#### 6.4.1 Loss Function 설계

**목표**: Value network가 실제 return을 정확히 예측하도록 학습

**Loss**: Mean Squared Error (MSE)
$$
\mathcal{L}(\phi) = \frac{1}{N} \sum_{i=1}^N \left( V_\phi(s_i) - Q_i^{\text{target}} \right)^2
$$

여기서:
- $V_\phi(s_i)$: Network의 예측값
- $Q_i^{\text{target}}$: 실제 받은 return (reward-to-go)

**왜 MSE?**
- Regression 문제 (연속 값 예측)
- 예측과 실제 값의 차이를 최소화

#### 6.4.2 구현 코드

```python
def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
    obs = ptu.from_numpy(obs)
    q_values = ptu.from_numpy(q_values)

    assert obs.ndim == 2
    assert q_values.ndim == 1

    # TODO: update critic
    # Step 1: Predict values
    values = self.forward(obs)

    # Step 2: Calculate MSE loss
    loss = F.mse_loss(values, q_values)

    # Step 3: Backpropagation
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return {"Baseline Loss": ptu.to_numpy(loss)}
```

**한 줄씩 상세 설명:**

**Lines 1-2**: NumPy → PyTorch
```python
obs = ptu.from_numpy(obs)
q_values = ptu.from_numpy(q_values)
```

**Lines 4-5**: Shape assertions
```python
assert obs.ndim == 2        # (batch_size, ob_dim)
assert q_values.ndim == 1   # (batch_size,)
```

**Step 1**: Value 예측
```python
values = self.forward(obs)
# values shape: (batch_size,)
```

예시:
```python
obs.shape = (128, 4)
values = critic.forward(obs)
# values.shape = (128,)
# values = tensor([12.5, 8.3, 15.2, ...])
```

**Step 2**: MSE Loss
```python
loss = F.mse_loss(values, q_values)
```

**`F.mse_loss()` 함수**:
- `F`는 `torch.nn.functional`
- Mean Squared Error 계산:
  $$
  \text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
  $$

내부 구현 (단순화):
```python
def mse_loss(input, target):
    return ((input - target) ** 2).mean()
```

**구체적 예시**:
```python
values = tensor([12.5, 8.3, 15.2])    # 예측
q_values = tensor([10.0, 9.0, 14.0])  # 실제

loss = mse_loss(values, q_values)
# = ((12.5-10.0)² + (8.3-9.0)² + (15.2-14.0)²) / 3
# = (6.25 + 0.49 + 1.44) / 3
# = 8.18 / 3
# = 2.73
```

**Step 3**: Backpropagation
```python
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

Policy network 업데이트와 동일한 방식!

---

### 6.5 Multiple Gradient Steps

```python
# pg_agent.py의 update() 메서드에서

if self.critic is not None:
    # Update critic for `baseline_gradient_steps` times
    for _ in range(self.baseline_gradient_steps):
        critic_info = self.critic.update(obs, q_values)
```

**왜 여러 번 업데이트?**

1. **Value function은 느리게 학습**
   - Actor는 새로운 데이터로 한 번 업데이트
   - Critic은 같은 데이터로 여러 번 업데이트해서 따라잡기

2. **더 정확한 baseline 필요**
   - Baseline이 부정확하면 advantage 추정이 부정확
   - 여러 번 학습해서 fitting 개선

**Typical value**: `baseline_gradient_steps = 5`

**예시**:
```python
for step in range(5):
    loss = critic.update(obs, q_values)
    print(f"Step {step}: Loss = {loss:.3f}")

# Output:
# Step 0: Loss = 25.3
# Step 1: Loss = 18.7
# Step 2: Loss = 15.2
# Step 3: Loss = 13.8
# Step 4: Loss = 12.9
```

Loss가 점점 감소 → Fitting 개선

---

### 6.6 Advantage Estimation 구현

#### 6.6.1 `_estimate_advantage()` 메서드

```python
# pg_agent.py

def _estimate_advantage(
    self,
    obs: np.ndarray,
    rewards: np.ndarray,
    q_values: np.ndarray,
    terminals: np.ndarray,
) -> np.ndarray:
    """Computes advantages by (possibly) subtracting a baseline."""

    assert obs.ndim == 2

    if self.critic is None:
        # Case 1: No baseline
        advantages = None  # TODO
    else:
        # Case 2: With baseline
        values = None  # TODO

        if self.gae_lambda is None:
            # Simple baseline
            advantages = None  # TODO
        else:
            # GAE (나중에 다룸)
            pass

    return advantages
```

#### 6.6.2 Case 1: No Baseline

```python
if self.critic is None:
    advantages = q_values
```

**설명**:
- Baseline이 없으면 advantage = Q-value 그대로
- 이전 섹션에서 구현한 방식

#### 6.6.3 Case 2: With Baseline (Simple)

```python
else:
    # Get baseline values
    values = self.critic(ptu.from_numpy(obs))
    values = ptu.to_numpy(values)

    if self.gae_lambda is None:
        advantages = q_values - values
```

**한 줄씩 설명**:

**Line 1**: Critic network로 value 예측
```python
values = self.critic(ptu.from_numpy(obs))
```
- `obs` shape: `(batch_size, ob_dim)`
- `values` shape: `(batch_size,)` (squeeze됨)

**Line 2**: PyTorch → NumPy
```python
values = ptu.to_numpy(values)
```

**Line 3**: Advantage 계산
```python
advantages = q_values - values
```

**구체적 예시**:
```python
# CartPole episode
obs = [
    [0.1, 0.5, -0.2, 0.3],  # t=0
    [0.2, 0.6, -0.1, 0.2],  # t=1
    [0.3, 0.7,  0.0, 0.1],  # t=2
]

q_values = np.array([2.9701, 1.99, 1.0])  # reward-to-go (γ=0.99)

values = critic(obs)
# values = np.array([2.5, 2.0, 1.2])  # critic 예측

advantages = q_values - values
# advantages = [0.47, -0.01, -0.2]
```

**해석**:
- **t=0**: advantage = +0.47 → 평균보다 좋음! ✓
- **t=1**: advantage = -0.01 ≈ 0 → 평균 정도
- **t=2**: advantage = -0.2 → 평균보다 나쁨 ✗

---

### 6.7 전체 업데이트 흐름

#### 6.7.1 데이터 흐름 다이어그램

```
Trajectories 수집
    ↓
[rewards] → _calculate_q_vals() → [q_values]
    ↓
[obs, q_values] → _estimate_advantage()
    ↓
    ├─ critic(obs) → [values]
    ↓
    └─ advantages = q_values - values
    ↓
[obs, actions, advantages] → actor.update()
    └─ loss = -(log_probs * advantages).mean()
    └─ backprop

[obs, q_values] → critic.update() (여러 번)
    └─ loss = mse_loss(values, q_values)
    └─ backprop
```

#### 6.7.2 구체적 예시 (CartPole)

```python
# Step 1: Trajectory 수집
trajectory = {
    'observation': [[0.1, 0.5, -0.2, 0.3],
                    [0.2, 0.6, -0.1, 0.2],
                    [0.3, 0.7,  0.0, 0.1]],
    'action': [0, 1, 0],
    'reward': [1.0, 1.0, 1.0]
}

# Step 2: Q-values 계산 (γ=0.99)
q_values = _discounted_reward_to_go([1.0, 1.0, 1.0])
# q_values = [2.9701, 1.99, 1.0]

# Step 3: Advantages 추정
values = critic(obs)
# values = [2.5, 2.0, 1.2]

advantages = q_values - values
# advantages = [0.47, -0.01, -0.2]

# Step 4: Actor 업데이트
actor_loss = -(log_probs * advantages).mean()
# log_probs = [-0.5, -0.8, -0.6]
# loss = -(-0.5*0.47 + -0.8*(-0.01) + -0.6*(-0.2)) / 3
#      = -(-0.235 + 0.008 + 0.12) / 3
#      = 0.036

# Step 5: Critic 업데이트 (5번)
for _ in range(5):
    critic_loss = mse_loss(critic(obs), q_values)
    critic_loss.backward()
    critic_optimizer.step()
```

---

### 6.8 Baseline의 효과

#### 6.8.1 Variance Reduction

**Without Baseline**:
```python
# Episode 1: total_return = 200
advantages = [200, 200, 200, ...]

# Episode 2: total_return = 10
advantages = [10, 10, 10, ...]

# Variance = Var([200, 10]) = 9025 (매우 큼!)
```

**With Baseline**:
```python
# Episode 1
q_values = [200, 199, 198, ...]
values = [150, 149, 148, ...]  # baseline 예측
advantages = [50, 50, 50, ...]

# Episode 2
q_values = [10, 9, 8, ...]
values = [15, 14, 13, ...]
advantages = [-5, -5, -5, ...]

# Variance = Var([50, -5]) = 756 (훨씬 작음!)
```

Baseline이 "평균적인 값"을 빼주니까, 편차가 줄어들어.

#### 6.8.2 학습 안정성

**Without Baseline**: Gradient가 크게 흔들림
```
Iteration 1: gradient = 150
Iteration 2: gradient = -80
Iteration 3: gradient = 200
→ 학습 불안정
```

**With Baseline**: Gradient가 안정적
```
Iteration 1: gradient = 20
Iteration 2: gradient = -10
Iteration 3: gradient = 15
→ 학습 안정
```

---

### 6.9 Hyperparameter 설정

#### 6.9.1 Baseline Learning Rate

```python
# run_hw2.py
parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
```

**Typical values**: 1e-3 ~ 5e-3

**너무 크면**: Value function이 불안정하게 학습
**너무 작으면**: Value function이 느리게 학습 → 부정확한 baseline

#### 6.9.2 Baseline Gradient Steps

```python
parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
```

**Typical values**: 5 ~ 10

**Too few**: Underfitting → 부정확한 baseline
**Too many**: Overfitting + 시간 낭비

#### 6.9.3 실험적 권장사항

| Environment | -blr | -bgs |
|---|---|---|
| CartPole | 5e-3 | 5 |
| HalfCheetah | 1e-3 | 5 |
| Humanoid | 5e-4 | 10 |

복잡한 환경일수록:
- 낮은 learning rate
- 더 많은 gradient steps

---

### 6.10 디버깅 팁

#### 6.10.1 Baseline Loss 모니터링

TensorBoard에서 확인:
```
Baseline Loss should decrease over time
```

정상적인 학습:
```
Iter 0:  Loss = 150.3
Iter 10: Loss = 85.2
Iter 20: Loss = 45.7
Iter 50: Loss = 15.3
```

**문제 징후**:
- Loss가 감소하지 않음 → learning rate 너무 작음
- Loss가 발산 → learning rate 너무 큼
- Loss가 진동 → batch size 너무 작음

#### 6.10.2 Value 예측 정확도 체크

```python
# 디버깅 코드
values = critic(obs)
print(f"Predicted values: {values.mean():.2f} ± {values.std():.2f}")
print(f"Actual Q-values:  {q_values.mean():.2f} ± {q_values.std():.2f}")
```

**Good**: 평균과 분산이 비슷
```
Predicted values: 48.3 ± 15.2
Actual Q-values:  50.1 ± 16.8
```

**Bad**: 큰 차이
```
Predicted values: 30.0 ± 5.0
Actual Q-values:  50.0 ± 20.0
→ Critic이 제대로 학습 안 됨!
```

---

**Part 4 완료. 다음 파트에서 GAE (Generalized Advantage Estimator) 구현을 다룹니다.**
