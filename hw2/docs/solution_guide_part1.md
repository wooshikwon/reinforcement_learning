# GCB6206 Homework 2: Policy Gradients 완전 해설 가이드 (Part 1)

## 목차

1. [과제 개요 및 학습 목표](#1-과제-개요-및-학습-목표)
2. [전체 코드 구조 이해](#2-전체-코드-구조-이해)
3. [Policy Gradient 이론적 배경](#3-policy-gradient-이론적-배경)

---

## 1. 과제 개요 및 학습 목표

### 1.1 과제가 다루는 내용

이 과제는 **Policy Gradient Methods**(정책 경사 방법)를 깊이 이해하고 구현하는 것을 목표로 해. 단순히 코드를 작성하는 것을 넘어서, 왜 이런 방법들이 필요한지, 어떻게 작동하는지를 완전히 이해하는 게 핵심이야.

구체적으로 다음 내용들을 다뤄:

1. **기본 Policy Gradient (REINFORCE)**
   - 가장 기본적인 정책 경사 알고리즘
   - 전체 trajectory의 return을 사용

2. **Variance Reduction Techniques (분산 감소 기법)**
   - **Reward-to-go**: 과거 보상은 현재 행동에 영향을 주지 않는다는 causality 활용
   - **Baseline**: 상태 의존적인 기준선을 빼서 분산 감소
   - **GAE (Generalized Advantage Estimator)**: Bias-variance tradeoff를 조절할 수 있는 고급 기법

3. **PPO (Proximal Policy Optimization)**
   - 현대 강화학습에서 가장 널리 쓰이는 on-policy 알고리즘
   - Policy 업데이트가 너무 크게 일어나지 않도록 제약

### 1.2 왜 이런 기법들이 필요한가?

Policy Gradient의 가장 큰 문제는 **높은 분산(high variance)**이야. 같은 정책으로 여러 번 trajectory를 샘플링해도, 받는 reward가 매번 크게 달라질 수 있어. 이는 학습을 불안정하게 만들고, 많은 샘플이 필요하게 만들지.

이를 해결하기 위해:
- **Reward-to-go**: 불필요한 항을 제거해서 분산 감소
- **Baseline**: 절대적인 return이 아니라 상대적인 advantage를 사용
- **GAE**: n-step returns의 exponentially weighted average로 bias-variance 조절
- **PPO**: 너무 큰 policy 변화를 막아 안정적인 학습 보장

---

## 2. 전체 코드 구조 이해

### 2.1 디렉토리 구조

```
hw2/
├── gcb6206/
│   ├── agents/
│   │   └── pg_agent.py          # PGAgent 클래스: 전체 학습 로직
│   ├── networks/
│   │   ├── policies.py          # MLPPolicyPG: Actor 네트워크
│   │   └── critics.py           # ValueCritic: Baseline 네트워크
│   ├── infrastructure/
│   │   ├── utils.py             # Trajectory 수집 함수들
│   │   ├── logger.py            # TensorBoard 로깅
│   │   └── pytorch_util.py      # PyTorch 유틸리티
│   └── scripts/
│       └── run_hw2.py           # 메인 학습 스크립트
└── data/                        # 학습 로그 저장
```

### 2.2 핵심 클래스와 역할

#### 2.2.1 `PGAgent` (pg_agent.py)

**역할**: Policy Gradient 알고리즘의 전체 학습 로직을 담당하는 중심 클래스야.

**주요 컴포넌트**:
```python
class PGAgent(nn.Module):
    def __init__(...):
        self.actor = MLPPolicyPG(...)      # Policy 네트워크 (행동 선택)
        self.critic = ValueCritic(...)     # Value 네트워크 (baseline)
        self.gamma = gamma                 # Discount factor
        self.use_reward_to_go = ...        # Reward-to-go 사용 여부
        self.gae_lambda = ...              # GAE λ 파라미터
        self.use_ppo = ...                 # PPO 사용 여부
```

**주요 메서드**:
- `update()`: 한 번의 학습 iteration 수행
- `_calculate_q_vals()`: Reward로부터 Q-value 계산
- `_estimate_advantage()`: Advantage 추정
- `_discounted_return()`: 전체 trajectory의 discounted return
- `_discounted_reward_to_go()`: Reward-to-go 계산

#### 2.2.2 `MLPPolicyPG` (policies.py)

**역할**: 관찰(observation)을 입력받아 행동(action)을 출력하는 Actor 네트워크야.

**중요 특징**:
- **Discrete action space**: Categorical distribution 사용
- **Continuous action space**: Gaussian (Normal) distribution 사용

**주요 메서드**:
- `forward()`: 관찰 → 행동 분포(distribution) 반환
- `get_action()`: 관찰 → 실제 행동 샘플링
- `update()`: 일반 Policy Gradient 업데이트
- `ppo_update()`: PPO 방식 업데이트

#### 2.2.3 `ValueCritic` (critics.py)

**역할**: 상태의 가치를 추정하는 Critic 네트워크 (Baseline으로 사용)

**주요 메서드**:
- `forward()`: 상태 → 가치 추정값 반환
- `update()`: MSE loss로 value function 학습

### 2.3 학습 흐름도

전체 학습 과정은 다음과 같은 순서로 진행돼:

```
[run_hw2.py: run_training_loop()]
    ↓
1. Environment 초기화
    ↓
2. PGAgent 생성
    ↓
3. FOR each iteration:
    │
    ├─→ [utils.py] Trajectory 수집
    │       - rollout_trajectories()로 배치 크기만큼 데이터 수집
    │       - 각 trajectory는 (s, a, r, s', done) 시퀀스
    │
    ├─→ [pg_agent.py] Agent 업데이트
    │       │
    │       ├─→ _calculate_q_vals(): Reward → Q-value
    │       │       - Case 1: Full trajectory return
    │       │       - Case 2: Reward-to-go
    │       │
    │       ├─→ _estimate_advantage(): Q-value → Advantage
    │       │       - No baseline: advantage = Q-value
    │       │       - With baseline: advantage = Q-value - V(s)
    │       │       - With GAE: advantage = GAE formula
    │       │
    │       ├─→ [policies.py] Actor 업데이트
    │       │       - 일반 PG: MLPPolicyPG.update()
    │       │       - PPO: MLPPolicyPG.ppo_update()
    │       │
    │       └─→ [critics.py] Critic 업데이트 (if baseline)
    │               - ValueCritic.update()
    │
    └─→ 로깅 및 평가
```

### 2.4 데이터 흐름 이해

학습 과정에서 데이터가 어떻게 변환되는지 이해하는 게 매우 중요해.

#### Step 1: Trajectory 수집

```python
# utils.py: rollout_trajectories()
trajs = [
    {
        'observation': np.array([[s0], [s1], ..., [sT]]),  # shape: (T, obs_dim)
        'action': np.array([[a0], [a1], ..., [aT-1]]),     # shape: (T, act_dim)
        'reward': np.array([r0, r1, ..., rT-1]),           # shape: (T,)
        'terminal': np.array([0, 0, ..., 1])               # shape: (T,)
    },
    # ... 여러 trajectories
]
```

여기서 중요한 점:
- `observation`: 각 timestep의 상태. 2D array (시간, 상태 차원)
- `action`: 각 timestep의 행동
- `reward`: 각 timestep에서 받은 보상. 1D array
- `terminal`: 해당 timestep이 episode 끝인지 표시 (1이면 끝)

#### Step 2: Q-value 계산

```python
# pg_agent.py: _calculate_q_vals()
rewards = [
    np.array([r0, r1, r2]),      # trajectory 1
    np.array([r0, r1, r2, r3]),  # trajectory 2
]

# Case 1: Full trajectory return
q_values = [
    np.array([R, R, R]),         # 모든 timestep에 같은 값
    np.array([R', R', R', R'])
]
# 여기서 R = r0 + γr1 + γ²r2

# Case 2: Reward-to-go
q_values = [
    np.array([r0+γr1+γ²r2, r1+γr2, r2]),
    np.array([r0+γr1+γ²r2+γ³r3, r1+γr2+γ³r3, r2+γ³r3, r3])
]
```

#### Step 3: Flatten (평탄화)

여러 trajectories를 하나의 큰 배치로 합쳐:

```python
# Before: List of arrays
obs = [np.array([[s0], [s1]]), np.array([[s0], [s1], [s2]])]

# After: Single concatenated array
obs = np.array([[s0], [s1], [s0], [s1], [s2]])  # shape: (5, obs_dim)
```

이렇게 하면 벡터화된 연산이 가능해져서 효율적이야.

---

## 3. Policy Gradient 이론적 배경

### 3.1 강화학습의 목표

강화학습의 목표는 **기대 누적 보상(expected cumulative reward)**을 최대화하는 정책 $\pi_\theta$를 찾는 거야. 수식으로 쓰면:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta(\tau)}[r(\tau)]
$$

여기서:
- $\theta$: 정책 네트워크의 파라미터 (weights and biases)
- $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$: Trajectory (하나의 episode)
- $\pi_\theta(\tau)$: 정책 $\pi_\theta$를 따랐을 때 trajectory $\tau$가 나올 확률
- $r(\tau) = \sum_{t=0}^{T-1} r(s_t, a_t)$: Trajectory의 총 보상

### 3.2 Trajectory 확률 분해

Trajectory가 생성될 확률은:

$$
\pi_\theta(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t | s_t) p(s_{t+1} | s_t, a_t)
$$

이걸 풀어서 설명하면:
1. $p(s_0)$: 초기 상태 분포 (environment가 결정)
2. $\pi_\theta(a_t | s_t)$: 상태 $s_t$에서 행동 $a_t$를 선택할 확률 (우리가 학습)
3. $p(s_{t+1} | s_t, a_t)$: 환경의 dynamics (transition probability)

### 3.3 Policy Gradient 유도

목표 함수 $J(\theta)$를 $\theta$에 대해 미분하면:

$$
\begin{align}
\nabla_\theta J(\theta) &= \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta(\tau)}[r(\tau)] \\
&= \nabla_\theta \int \pi_\theta(\tau) r(\tau) d\tau \\
&= \int \nabla_\theta \pi_\theta(\tau) r(\tau) d\tau
\end{align}
$$

**핵심 트릭: Log-derivative trick**

$$
\nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau)
$$

이건 왜 성립하냐면:

$$
\nabla_\theta \log \pi_\theta(\tau) = \frac{\nabla_\theta \pi_\theta(\tau)}{\pi_\theta(\tau)}
$$

양변에 $\pi_\theta(\tau)$를 곱하면 위 식이 나와.

이걸 사용하면:

$$
\begin{align}
\nabla_\theta J(\theta) &= \int \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau) r(\tau) d\tau \\
&= \mathbb{E}_{\tau \sim \pi_\theta(\tau)}[\nabla_\theta \log \pi_\theta(\tau) r(\tau)]
\end{align}
$$

### 3.4 왜 이게 중요한가?

이 형태가 중요한 이유는:
1. **기댓값 형태**: Monte Carlo로 샘플링해서 추정 가능
2. **환경 dynamics 불필요**: $p(s_{t+1}|s_t, a_t)$를 모르더라도 gradient 계산 가능

### 3.5 Log probability 분해

$$
\log \pi_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T-1} [\log \pi_\theta(a_t | s_t) + \log p(s_{t+1} | s_t, a_t)]
$$

Gradient를 취하면:

$$
\nabla_\theta \log \pi_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t)
$$

왜냐하면:
- $\nabla_\theta \log p(s_0) = 0$ (초기 상태는 $\theta$와 무관)
- $\nabla_\theta \log p(s_{t+1} | s_t, a_t) = 0$ (환경 dynamics는 $\theta$와 무관)

### 3.6 최종 Policy Gradient 식

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta(\tau)} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot r(\tau) \right]
$$

실제로는 N개의 trajectory를 샘플링해서 근사해:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) \cdot r(\tau_i)
$$

여기서:
- $r(\tau_i) = \sum_{t'=0}^{T-1} r(s_{t'}^i, a_{t'}^i)$

### 3.7 직관적 이해

이 식의 의미를 직관적으로 이해해보자:

```
만약 trajectory τ의 총 보상 r(τ)가 크다면,
→ 해당 trajectory를 만든 행동들 (a₀, a₁, ..., aₜ)의 확률을 높여라
→ ∇_θ log π_θ(aₜ|sₜ) 방향으로 θ를 업데이트

만약 r(τ)가 작다면,
→ 해당 행동들의 확률을 낮춰라
```

### 3.8 문제점: High Variance

하지만 이 방법에는 큰 문제가 있어:

**같은 상태에서 같은 행동을 해도, 이후 어떤 일이 벌어지느냐에 따라 총 보상이 크게 달라질 수 있어.**

예를 들어, CartPole에서:
- Episode 1: 막대가 200 step 동안 유지됨 → r(τ) = 200
- Episode 2: 운이 나빠서 10 step 만에 넘어짐 → r(τ) = 10

똑같은 초기 행동인데도 결과가 20배 차이 나. 이게 **high variance** 문제야.

### 3.9 해결책 Preview

다음 섹션들에서 이 문제를 해결하는 기법들을 다룰 거야:

1. **Reward-to-go**: 미래만 고려 (과거는 무시)
2. **Baseline**: 상대적인 좋고 나쁨만 평가
3. **GAE**: Bias-variance tradeoff 조절

---

**Part 1 완료. 다음 파트에서 계속...**
