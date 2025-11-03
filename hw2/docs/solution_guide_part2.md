# GCB6206 Homework 2: Policy Gradients 완전 해설 가이드 (Part 2)

## 4. Policy Gradient 구현: Reward 계산

### 4.1 Variance Reduction: Reward-to-go

#### 4.1.1 왜 Reward-to-go가 필요한가?

기본 Policy Gradient에서는 timestep t의 행동 $a_t$를 평가할 때, **전체 trajectory의 보상**을 사용해:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) \cdot \underbrace{\sum_{t'=0}^{T-1} r(s_{t'}^i, a_{t'}^i)}_{\text{전체 trajectory의 보상}}
$$

하지만 **인과성(causality)**을 생각해보면, **과거의 보상은 현재 행동과 무관**해. 예를 들어:

- Timestep t=5에서의 행동 $a_5$
- 과거 보상 (r₀, r₁, r₂, r₃, r₄)는 이미 받은 보상이야
- $a_5$가 아무리 좋아도 과거는 바꿀 수 없어

따라서 $a_5$를 평가할 때는 **미래 보상만** 고려해야 해:

$$
\text{Reward-to-go at } t=5: \quad r_5 + \gamma r_6 + \gamma^2 r_7 + \cdots
$$

#### 4.1.2 수학적 유도

Causality를 활용하면:

$$
\begin{align}
\nabla_\theta J(\theta) &= \mathbb{E}_{\tau} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \sum_{t'=0}^{T-1} r_{t'} \right] \\
&= \mathbb{E}_{\tau} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \sum_{t'=t}^{T-1} r_{t'} \right]
\end{align}
$$

왜 이렇게 바꿀 수 있냐면:

$$
\mathbb{E}_{\tau} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \sum_{t'=0}^{t-1} r_{t'} \right] = 0
$$

이유: $a_t$는 미래 행동이고, $r_{t'}$ (t' < t)는 과거 보상이라 독립적이야.

#### 4.1.3 Discounting과 결합

Discount factor $\gamma$를 적용하면:

**Case 1: Full trajectory (기본 PG)**
$$
Q(s_t, a_t) \approx \sum_{t'=0}^{T-1} \gamma^{t'} r_{t'}
$$

모든 timestep에서 같은 값!

**Case 2: Reward-to-go**
$$
Q(s_t, a_t) \approx \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}
$$

각 timestep마다 다른 값. 미래만 고려하고, discount는 현재 시점부터 계산.

---

### 4.2 코드 구현: `_calculate_q_vals()` 메서드

이제 실제 구현을 한 줄씩 뜯어보자.

#### 4.2.1 전체 코드 구조

```python
# pg_agent.py의 140-158번째 줄

def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    """Monte Carlo estimation of the Q function."""

    assert all(reward.ndim == 1 for reward in rewards)

    if not self.use_reward_to_go:
        # Case 1: trajectory-based PG
        q_values = None  # TODO: 구현 필요
    else:
        # Case 2: reward-to-go PG
        q_values = None  # TODO: 구현 필요

    return q_values
```

#### 4.2.2 입력 데이터 형식 이해

**`rewards` 파라미터**:
- 타입: `Sequence[np.ndarray]` = 리스트[넘파이 배열]
- 의미: 여러 trajectories의 reward 시퀀스들

예시:
```python
rewards = [
    np.array([1.0, 1.0, 1.0]),          # Trajectory 1: 3 timesteps
    np.array([1.0, 1.0, 1.0, 1.0, 1.0]) # Trajectory 2: 5 timesteps
]
```

CartPole 환경에서는 매 step마다 1.0의 보상을 받으니까, 위처럼 생겼을 거야.

**`assert all(reward.ndim == 1 for reward in rewards)`**:
- 각 trajectory의 reward가 1차원 배열인지 확인
- `ndim`: NumPy array의 차원 수
- 예: `np.array([1, 2, 3]).ndim == 1` (True)
- 예: `np.array([[1, 2, 3]]).ndim == 2` (False)

왜 이 체크가 필요하냐면, 실수로 2D array로 만들면 계산이 꼬이거든.

---

### 4.3 Helper Function: `_discounted_return()` 구현

#### 4.3.1 목표

```python
# Input:  rewards = [r₀, r₁, r₂]
# Output: ret = [R, R, R]  (모든 원소가 같은 값)
# 여기서 R = r₀ + γ·r₁ + γ²·r₂
```

#### 4.3.2 상세 구현 방법

```python
def _discounted_return(self, rewards: np.ndarray[float]) -> np.ndarray[float]:
    """
    Helper function which takes a list of rewards {r_0, r_1, ..., r_T-1}
    and returns a list where each index t contains sum_{t'=0}^{T-1} gamma^t' r_{t'}
    """
    assert rewards.ndim == 1

    # 방법 1: 명시적 계산
    T = len(rewards)
    discounted_sum = 0.0
    for t in range(T):
        discounted_sum += (self.gamma ** t) * rewards[t]

    ret = np.full(T, discounted_sum)  # 같은 값으로 채운 배열

    assert rewards.shape == ret.shape
    return ret
```

**한 줄씩 설명:**

**Line 1**: `assert rewards.ndim == 1`
- 입력이 1D 배열인지 확인
- 예: `[r₀, r₁, r₂]` (OK), `[[r₀], [r₁]]` (Error)

**Line 3**: `T = len(rewards)`
- Trajectory의 길이 (timesteps 개수)
- 예: `rewards = [1.0, 1.0, 1.0]` → T = 3

**Line 4-6**: For 루프
```python
for t in range(T):
    discounted_sum += (self.gamma ** t) * rewards[t]
```
- `range(T)`: 0부터 T-1까지 (0, 1, 2, ..., T-1)
- `self.gamma ** t`: γᵗ (gamma의 t제곱)
- `rewards[t]`: rₜ
- 누적: `discounted_sum = r₀ + γr₁ + γ²r₂ + ...`

**예시 계산 (γ=0.99)**:
```python
rewards = np.array([1.0, 2.0, 3.0])
gamma = 0.99

# t=0: 0.99^0 * 1.0 = 1.0 * 1.0 = 1.0
# t=1: 0.99^1 * 2.0 = 0.99 * 2.0 = 1.98
# t=2: 0.99^2 * 3.0 = 0.9801 * 3.0 = 2.9403

discounted_sum = 1.0 + 1.98 + 2.9403 = 5.9203
```

**Line 8**: `ret = np.full(T, discounted_sum)`
- `np.full(shape, fill_value)`: 주어진 값으로 채운 배열 생성
- `np.full(3, 5.9203)` → `[5.9203, 5.9203, 5.9203]`

왜 모든 원소가 같냐면, **전체 trajectory의 return은 어느 timestep에서나 같은 값**이거든.

**Line 10**: `assert rewards.shape == ret.shape`
- 출력 shape이 입력과 같은지 확인
- 입력: (T,), 출력: (T,) → OK

#### 4.3.3 더 효율적인 구현 (벡터화)

```python
def _discounted_return(self, rewards: np.ndarray[float]) -> np.ndarray[float]:
    T = len(rewards)

    # 방법 2: NumPy 벡터 연산
    gamma_powers = self.gamma ** np.arange(T)  # [γ⁰, γ¹, γ², ...]
    discounted_sum = np.sum(gamma_powers * rewards)

    ret = np.full(T, discounted_sum)
    return ret
```

**설명:**

**`gamma_powers = self.gamma ** np.arange(T)`**
- `np.arange(T)`: `[0, 1, 2, ..., T-1]`
- `self.gamma ** [0, 1, 2]`: element-wise 거듭제곱
- 결과: `[γ⁰, γ¹, γ²]` = `[1.0, 0.99, 0.9801]`

**`np.sum(gamma_powers * rewards)`**
- `gamma_powers * rewards`: element-wise 곱셈
  ```python
  [1.0, 0.99, 0.9801] * [1.0, 2.0, 3.0]
  = [1.0, 1.98, 2.9403]
  ```
- `np.sum([...])`: 전체 합
  ```python
  = 1.0 + 1.98 + 2.9403 = 5.9203
  ```

이 방법이 더 빠른 이유: NumPy의 C 구현 사용

---

### 4.4 Helper Function: `_discounted_reward_to_go()` 구현

#### 4.4.1 목표

```python
# Input:  rewards = [r₀, r₁, r₂]
# Output: ret = [R₀, R₁, R₂]  (각각 다른 값)
# 여기서:
#   R₀ = r₀ + γ·r₁ + γ²·r₂
#   R₁ = r₁ + γ·r₂
#   R₂ = r₂
```

#### 4.4.2 상세 구현 방법

```python
def _discounted_reward_to_go(self, rewards: np.ndarray[float]) -> np.ndarray[float]:
    """
    Helper function which takes rewards {r_0, r_1, ..., r_T-1}
    and returns a list where entry at index t is sum_{t'=t}^{T-1} gamma^{t'-t} * r_{t'}
    """
    assert rewards.ndim == 1

    # 방법 1: 역방향 동적 프로그래밍
    T = len(rewards)
    ret = np.zeros(T)

    ret[T-1] = rewards[T-1]  # 마지막 timestep: r_{T-1}

    for t in reversed(range(T-1)):
        ret[t] = rewards[t] + self.gamma * ret[t+1]

    assert rewards.shape == ret.shape
    return ret
```

**한 줄씩 설명:**

**Line 3**: `T = len(rewards)`
- Trajectory 길이
- 예: `rewards = [1.0, 2.0, 3.0]` → T = 3

**Line 4**: `ret = np.zeros(T)`
- 결과를 저장할 배열 초기화
- `np.zeros(3)` → `[0.0, 0.0, 0.0]`

**Line 6**: `ret[T-1] = rewards[T-1]`
- **Base case**: 마지막 timestep (t=T-1)
- 미래가 없으니까 현재 보상만 있어
- `ret[2] = rewards[2] = 3.0`

**Line 8-9**: For 루프 (역방향)
```python
for t in reversed(range(T-1)):
    ret[t] = rewards[t] + self.gamma * ret[t+1]
```

**`reversed(range(T-1))`**:
- `range(T-1)` = `range(2)` = `[0, 1]`
- `reversed([0, 1])` = `[1, 0]`
- 즉, t=1부터 t=0까지 역순으로

**왜 역방향이냐?**
- Reward-to-go는 **미래 보상**의 합
- 따라서 미래부터 계산해서 누적해야 해

**동적 프로그래밍 핵심 아이디어:**
$$
\text{ret}[t] = r_t + \gamma \cdot \text{ret}[t+1]
$$

이미 계산된 `ret[t+1]` (미래 reward-to-go)를 재사용!

**예시 계산 (γ=0.99)**:
```python
rewards = [1.0, 2.0, 3.0]

# Step 1: ret[2] = 3.0 (초기화)

# Step 2: t=1
ret[1] = rewards[1] + 0.99 * ret[2]
       = 2.0 + 0.99 * 3.0
       = 2.0 + 2.97
       = 4.97

# Step 3: t=0
ret[0] = rewards[0] + 0.99 * ret[1]
       = 1.0 + 0.99 * 4.97
       = 1.0 + 4.9203
       = 5.9203

# 최종: ret = [5.9203, 4.97, 3.0]
```

#### 4.4.3 검증

결과가 올바른지 직접 계산으로 확인:

$$
\begin{align}
\text{ret}[0] &= r_0 + \gamma r_1 + \gamma^2 r_2 \\
&= 1.0 + 0.99 \times 2.0 + 0.99^2 \times 3.0 \\
&= 1.0 + 1.98 + 2.9403 \\
&= 5.9203 \quad \checkmark
\end{align}
$$

$$
\begin{align}
\text{ret}[1] &= r_1 + \gamma r_2 \\
&= 2.0 + 0.99 \times 3.0 \\
&= 2.0 + 2.97 \\
&= 4.97 \quad \checkmark
\end{align}
$$

$$
\text{ret}[2] = r_2 = 3.0 \quad \checkmark
$$

#### 4.4.4 시간 복잡도 분석

- For 루프: O(T)
- 각 iteration: O(1)
- **총 시간 복잡도: O(T)**

Forward 방향으로 매번 전체 합을 계산하면 O(T²)인데, DP로 O(T)로 줄였어!

---

### 4.5 `_calculate_q_vals()` 완성

이제 헬퍼 함수를 사용해서 완성하자.

```python
def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    """Monte Carlo estimation of the Q function."""

    assert all(reward.ndim == 1 for reward in rewards)

    if not self.use_reward_to_go:
        # Case 1: Full trajectory return
        q_values = [self._discounted_return(r) for r in rewards]
    else:
        # Case 2: Reward-to-go
        q_values = [self._discounted_reward_to_go(r) for r in rewards]

    return q_values
```

**Line 7-8: Case 1**
```python
q_values = [self._discounted_return(r) for r in rewards]
```
- **List comprehension**: 각 trajectory에 대해 함수 적용
- `rewards`가 3개 trajectory면, `q_values`도 3개 배열

**예시:**
```python
rewards = [
    np.array([1.0, 1.0, 1.0]),     # Trajectory 1
    np.array([1.0, 1.0])            # Trajectory 2
]

# Case 1 출력:
q_values = [
    np.array([2.9701, 2.9701, 2.9701]),  # 모두 같은 값
    np.array([1.99, 1.99])               # 모두 같은 값
]
```

**Line 10-11: Case 2**
```python
q_values = [self._discounted_reward_to_go(r) for r in rewards]
```

**예시:**
```python
# Case 2 출력:
q_values = [
    np.array([2.9701, 1.99, 1.0]),  # 시간에 따라 감소
    np.array([1.99, 1.0])
]
```

---

### 4.6 Q-value의 의미 이해

#### 4.6.1 이론적 정의

**Q-function (Action-value function)**:
$$
Q^\pi(s_t, a_t) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'} \mid s_t, a_t \right]
$$

**의미**: 상태 $s_t$에서 행동 $a_t$를 했을 때, 그 이후 정책 $\pi$를 따랐을 때의 기대 누적 보상

#### 4.6.2 Monte Carlo 추정

우리는 기댓값을 모르니까, **실제로 샘플링한 trajectory**로 추정해:

$$
Q^\pi(s_t, a_t) \approx \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}
$$

이게 바로 `_discounted_reward_to_go()`가 계산하는 거야!

#### 4.6.3 Case 1 vs Case 2 비교

**Case 1 (Full trajectory)**:
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) \cdot \underbrace{\sum_{t'=0}^{T-1} \gamma^{t'} r_{t'}^i}_{\text{same for all } t}
$$

**장점**: 구현 간단
**단점**: High variance (과거 보상 포함)

**Case 2 (Reward-to-go)**:
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) \cdot \underbrace{\sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}^i}_{\text{different for each } t}
$$

**장점**: Lower variance (causality 활용)
**단점**: 약간 더 복잡한 구현

---

### 4.7 실제 사용 예시

CartPole 환경에서의 구체적 예:

```python
# Trajectory 수집 후
rewards = [
    np.array([1., 1., 1., 1., 1.]),  # 5 steps 생존
    np.array([1., 1., 1.])            # 3 steps 생존
]

# gamma = 0.95로 가정
agent = PGAgent(..., gamma=0.95, use_reward_to_go=True)

# Q-values 계산
q_values = agent._calculate_q_vals(rewards)

# 출력:
# q_values[0] = [
#     1 + 0.95*1 + 0.95²*1 + 0.95³*1 + 0.95⁴*1 = 4.5256,
#     1 + 0.95*1 + 0.95²*1 + 0.95³*1 = 3.7148,
#     1 + 0.95*1 + 0.95²*1 = 2.8525,
#     1 + 0.95*1 = 1.95,
#     1.0
# ]
```

**해석**:
- 초기 행동 (t=0): 높은 Q-value (4.5256) → 좋은 행동!
- 후기 행동 (t=4): 낮은 Q-value (1.0) → 영향력 적음

---

**Part 2 완료. 다음 파트에서 Policy Network 구현을 다룹니다.**
