# GCB6206 Homework 2: Policy Gradients 완전 해설 가이드 (Part 5)

## 7. Generalized Advantage Estimator (GAE) 구현

### 7.1 GAE의 동기: Bias-Variance Tradeoff

#### 7.1.1 Advantage Estimation의 스펙트럼

Advantage를 추정하는 방법은 여러 가지야:

**Method 1: Monte Carlo (MC)**
$$
A^{\text{MC}}(s_t, a_t) = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'} - V(s_t)
$$

- **High variance**: 전체 trajectory의 randomness 포함
- **No bias** (unbiased): 기댓값이 정확

**Method 2: Temporal Difference (TD)**
$$
A^{\text{TD}}(s_t, a_t) = r_t + \gamma V(s_{t+1}) - V(s_t) = \delta_t
$$

- **Low variance**: 한 step의 randomness만
- **High bias**: $V(s_{t+1})$의 오차가 포함됨

**문제**: 둘 다 극단적이야. 중간은 없을까?

#### 7.1.2 N-step Returns

**아이디어**: n개의 실제 reward + 이후는 value function 사용

$$
A_n^{\pi}(s_t, a_t) = \sum_{t'=t}^{t+n-1} \gamma^{t'-t} r_{t'} + \gamma^n V(s_{t+n}) - V(s_t)
$$

**예시** (γ=1로 단순화):

**n=1** (TD):
```
A₁ = r_t + V(s_{t+1}) - V(s_t)
     └─ 1 real reward, rest from V
```

**n=2**:
```
A₂ = r_t + r_{t+1} + V(s_{t+2}) - V(s_t)
     └──┬──┘         └─ rest from V
      2 real
```

**n=3**:
```
A₃ = r_t + r_{t+1} + r_{t+2} + V(s_{t+3}) - V(s_t)
     └───────┬───────┘           └─ rest from V
          3 real
```

**n=∞** (MC):
```
A_∞ = r_t + r_{t+1} + ... + r_{T-1} - V(s_t)
      └────────────┬────────────┘
           all real rewards
```

#### 7.1.3 Bias-Variance Tradeoff

| n | Bias | Variance | 설명 |
|---|---|---|---|
| 1 (TD) | High | Low | V의 오차가 크게 영향 |
| 3 | Medium | Medium | 균형 잡힘 |
| 5 | Low | Medium-High | Real rewards 많이 사용 |
| ∞ (MC) | None | High | V 사용 안 함 |

**이상적**: n을 상황에 맞게 조절하고 싶다!

---

### 7.2 GAE의 핵심 아이디어

#### 7.2.1 Exponentially Weighted Average

**GAE**: 여러 n-step advantage들의 **exponentially weighted average**

$$
A^{\text{GAE}}_{\lambda}(s_t, a_t) = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} A_n(s_t, a_t)
$$

**$(1-\lambda)$ normalizing constant** 증명:
$$
(1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} = (1-\lambda) \cdot \frac{1}{1-\lambda} = 1
$$

#### 7.2.2 λ의 의미

**λ = 0**:
$$
A^{\text{GAE}}_0 = (1-0) \cdot A_1 = A_1 = \delta_t
$$
→ **Pure TD**: lowest variance, highest bias

**λ = 1**:
$$
A^{\text{GAE}}_1 = \sum_{n=1}^{\infty} A_n \approx A_{\infty}
$$
→ **Pure MC**: no bias, highest variance

**λ = 0.95** (common):
- 95% weight on later terms
- Balance bias-variance

#### 7.2.3 TD Error와의 관계

**TD error (temporal difference)**:
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**놀라운 사실**: GAE는 TD errors의 합으로 표현 가능!

$$
A^{\text{GAE}}_{\lambda}(s_t, a_t) = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

**증명 (간략)**:

n-step advantage:
$$
A_n = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}
$$

GAE:
$$
\begin{align}
A^{\text{GAE}}_\lambda &= (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} A_n \\
&= (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \sum_{l=0}^{n-1} \gamma^l \delta_{t+l} \\
&= \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
\end{align}
$$

---

### 7.3 Finite Horizon GAE

#### 7.3.1 실제 구현을 위한 수식

Infinite horizon은 비현실적이니까, **finite horizon**으로 근사:

$$
A^{\text{GAE}}_{\lambda}(s_t, a_t) = \sum_{l=0}^{T-1-t} (\gamma \lambda)^l \delta_{t+l}
$$

여기서:
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**Edge case** (episode 끝):
$$
\delta_{T-1} = r_{T-1} - V(s_{T-1})
$$
(다음 상태가 없으므로 $V(s_T) = 0$)

#### 7.3.2 Recursive Formulation

**핵심 관찰**:
$$
A^{\text{GAE}}_\lambda(s_t, a_t) = \delta_t + \gamma\lambda \cdot A^{\text{GAE}}_\lambda(s_{t+1}, a_{t+1})
$$

**증명**:
$$
\begin{align}
A_t^{\text{GAE}} &= \sum_{l=0}^{T-1-t} (\gamma\lambda)^l \delta_{t+l} \\
&= \delta_t + \sum_{l=1}^{T-1-t} (\gamma\lambda)^l \delta_{t+l} \\
&= \delta_t + \gamma\lambda \sum_{l=0}^{T-2-t} (\gamma\lambda)^l \delta_{t+1+l} \\
&= \delta_t + \gamma\lambda \cdot A_{t+1}^{\text{GAE}}
\end{align}
$$

**이게 중요한 이유**: **Dynamic Programming**으로 효율적 계산 가능!

---

### 7.4 코드 구현: GAE

#### 7.4.1 전체 코드 구조

```python
# pg_agent.py: _estimate_advantage() 메서드

def _estimate_advantage(
    self,
    obs: np.ndarray,
    rewards: np.ndarray,
    q_values: np.ndarray,
    terminals: np.ndarray,
) -> np.ndarray:
    if self.critic is None:
        advantages = q_values
    else:
        values = self.critic(ptu.from_numpy(obs))
        values = ptu.to_numpy(values)

        if self.gae_lambda is None:
            # Simple baseline
            advantages = q_values - values
        else:
            # GAE implementation
            batch_size = obs.shape[0]

            # Append dummy T+1 value
            values = np.append(values, [0])
            advantages = np.zeros(batch_size + 1)

            # Calculate deltas
            # TODO: implement

            # Recursive calculation (backward)
            for i in reversed(range(batch_size)):
                # TODO: implement
                pass

            # Remove dummy advantage
            advantages = advantages[:-1]

    return advantages
```

#### 7.4.2 Step 1: Dummy Value 추가

```python
values = np.append(values, [0])
advantages = np.zeros(batch_size + 1)
```

**왜 dummy value?**

Recursive formula:
$$
A_t = \delta_t + \gamma\lambda \cdot A_{t+1}
$$

마지막 timestep $t=T-1$에서:
$$
A_{T-1} = \delta_{T-1} + \gamma\lambda \cdot \underbrace{A_T}_{\text{필요!}}
$$

$A_T = 0$으로 설정 (episode 끝에는 미래 없음)

**구체적 예시**:
```python
batch_size = 3

values = np.array([12.5, 8.3, 5.1])
values = np.append(values, [0])
# values = array([12.5, 8.3, 5.1, 0.0])  ← dummy

advantages = np.zeros(batch_size + 1)
# advantages = array([0.0, 0.0, 0.0, 0.0])  ← 나중에 채움
```

#### 7.4.3 Step 2: TD Errors 계산

```python
# Calculate deltas (TD errors)
deltas = rewards + self.gamma * values[1:] * (1 - terminals) - values[:-1]
```

**한 부분씩 분해**:

**`values[1:]`**: 다음 상태의 values
```python
values = [V₀, V₁, V₂, V₃(=0)]
values[1:] = [V₁, V₂, V₃]
```

**`values[:-1]`**: 현재 상태의 values
```python
values[:-1] = [V₀, V₁, V₂]
```

**`rewards + self.gamma * values[1:]`**: $r_t + \gamma V(s_{t+1})$
```python
rewards = [r₀, r₁, r₂]
gamma * values[1:] = [γV₁, γV₂, γV₃]
rewards + ... = [r₀+γV₁, r₁+γV₂, r₂+γV₃]
```

**`(1 - terminals)`**: Terminal masking

**왜 필요?**

Episode가 끝나면 **다음 상태가 없어**. 따라서:
$$
\delta_t = \begin{cases}
r_t + \gamma V(s_{t+1}) - V(s_t) & \text{if not terminal} \\
r_t - V(s_t) & \text{if terminal}
\end{cases}
$$

이걸 벡터화:
$$
\delta_t = r_t + \gamma V(s_{t+1}) \cdot (1 - \text{terminal}_t) - V(s_t)
$$

**구체적 예시**:
```python
rewards = np.array([1.0, 1.0, 1.0])
values = np.array([12.5, 8.3, 5.1, 0.0])
terminals = np.array([0, 0, 1])  # 마지막이 terminal
gamma = 0.99

# Timestep 0:
delta_0 = 1.0 + 0.99 * 8.3 * (1-0) - 12.5
        = 1.0 + 8.217 - 12.5
        = -3.283

# Timestep 1:
delta_1 = 1.0 + 0.99 * 5.1 * (1-0) - 8.3
        = 1.0 + 5.049 - 8.3
        = -2.251

# Timestep 2 (terminal!):
delta_2 = 1.0 + 0.99 * 0.0 * (1-1) - 5.1
        = 1.0 + 0.0 - 5.1
        = -4.1

deltas = array([-3.283, -2.251, -4.1])
```

**Terminal masking 효과**:
- `(1-0) = 1`: 정상 계산
- `(1-1) = 0`: 다음 value 무시 ✓

#### 7.4.4 Step 3: Recursive GAE 계산

```python
for i in reversed(range(batch_size)):
    advantages[i] = deltas[i] + self.gamma * self.gae_lambda * (1 - terminals[i]) * advantages[i+1]
```

**한 줄씩 분석**:

**`reversed(range(batch_size))`**:
```python
batch_size = 3
range(3) = [0, 1, 2]
reversed([0, 1, 2]) = [2, 1, 0]
```

**왜 역순?**

Recursive formula:
$$
A_t = \delta_t + \gamma\lambda \cdot A_{t+1}
$$

$A_{t+1}$을 먼저 알아야 $A_t$ 계산 가능!

**Recursive update**:
```python
advantages[i] = deltas[i] + gamma * lambda * (1 - terminals[i]) * advantages[i+1]
```

**구조 분해**:
1. `deltas[i]`: 현재 timestep의 TD error
2. `gamma * lambda`: Discount factor
3. `(1 - terminals[i])`: Terminal masking
4. `advantages[i+1]`: 다음 timestep의 advantage (이미 계산됨)

**구체적 계산** (batch_size=3, γ=0.99, λ=0.95):

```python
deltas = np.array([-3.283, -2.251, -4.1])
terminals = np.array([0, 0, 1])

# Initialize
advantages = np.array([0.0, 0.0, 0.0, 0.0])

# Iteration 1: i=2 (마지막)
advantages[2] = deltas[2] + 0.99 * 0.95 * (1-1) * advantages[3]
              = -4.1 + 0.9405 * 0 * 0.0
              = -4.1

# Iteration 2: i=1
advantages[1] = deltas[1] + 0.99 * 0.95 * (1-0) * advantages[2]
              = -2.251 + 0.9405 * 1 * (-4.1)
              = -2.251 + (-3.856)
              = -6.107

# Iteration 3: i=0
advantages[0] = deltas[0] + 0.99 * 0.95 * (1-0) * advantages[1]
              = -3.283 + 0.9405 * 1 * (-6.107)
              = -3.283 + (-5.744)
              = -9.027

# Final
advantages = array([-9.027, -6.107, -4.1, 0.0])
```

#### 7.4.5 Step 4: Dummy 제거

```python
advantages = advantages[:-1]
```

마지막 dummy (0.0) 제거:
```python
advantages = array([-9.027, -6.107, -4.1])  # shape: (3,)
```

---

### 7.5 GAE 동작 원리 심화

#### 7.5.1 전개해서 이해하기

Recursive formula를 전개하면:

**i=2**:
$$
A_2 = \delta_2
$$

**i=1**:
$$
\begin{align}
A_1 &= \delta_1 + \gamma\lambda A_2 \\
&= \delta_1 + \gamma\lambda \delta_2
\end{align}
$$

**i=0**:
$$
\begin{align}
A_0 &= \delta_0 + \gamma\lambda A_1 \\
&= \delta_0 + \gamma\lambda (\delta_1 + \gamma\lambda \delta_2) \\
&= \delta_0 + \gamma\lambda \delta_1 + (\gamma\lambda)^2 \delta_2
\end{align}
$$

**일반화**:
$$
A_t = \sum_{l=0}^{T-1-t} (\gamma\lambda)^l \delta_{t+l}
$$

이게 바로 GAE 수식과 일치!

#### 7.5.2 λ의 영향 시각화

**λ = 0** (TD):
```python
A_0 = δ₀
# 오직 현재 TD error만 사용
```

**λ = 0.5**:
```python
A_0 = δ₀ + 0.495·δ₁ + 0.245·δ₂ + ...
# 지수적으로 감소하는 weight
```

**λ = 0.95**:
```python
A_0 = δ₀ + 0.941·δ₁ + 0.885·δ₂ + 0.832·δ₃ + ...
# 먼 미래도 상당한 weight
```

**λ = 1.0** (MC):
```python
A_0 = δ₀ + γ·δ₁ + γ²·δ₂ + ... + γᵀ⁻¹·δₜ₋₁
# 모든 미래 동등하게 (γ만큼 discount)
```

---

### 7.6 실제 사용 예시

#### 7.6.1 HumanoidStandup 실험

과제에서 요구하는 실험:

```bash
# λ = 0 (TD)
python run_hw2.py --env_name HumanoidStandup-v5 --gae_lambda 0 \
    --use_baseline --use_reward_to_go

# λ = 0.95 (Balanced)
python run_hw2.py --env_name HumanoidStandup-v5 --gae_lambda 0.95 \
    --use_baseline --use_reward_to_go

# λ = 1 (MC)
python run_hw2.py --env_name HumanoidStandup-v5 --gae_lambda 1 \
    --use_baseline --use_reward_to_go
```

**예상 결과**:

| λ | 학습 속도 | 최종 성능 | 안정성 |
|---|---|---|---|
| 0 | 빠름 | 낮음 | 높음 |
| 0.95 | 중간 | **최고** | 중간 |
| 1 | 느림 | 중간 | 낮음 |

**λ = 0.95**가 최선인 이유:
- Bias와 Variance의 좋은 균형
- 충분한 real rewards 사용
- 적절한 value function 활용

#### 7.6.2 Advantage 값 분석

```python
# λ = 0 (TD)
advantages_td = estimate_advantage(..., gae_lambda=0)
print(f"TD: mean={advantages_td.mean():.3f}, std={advantages_td.std():.3f}")
# TD: mean=0.123, std=2.5

# λ = 0.95 (GAE)
advantages_gae = estimate_advantage(..., gae_lambda=0.95)
print(f"GAE: mean={advantages_gae.mean():.3f}, std={advantages_gae.std():.3f}")
# GAE: mean=0.089, std=8.7

# λ = 1 (MC)
advantages_mc = estimate_advantage(..., gae_lambda=1)
print(f"MC: mean=0.051, std={advantages_mc.std():.3f}")
# MC: mean=0.051, std=15.3
```

**관찰**:
- TD: 가장 낮은 variance → 안정적이지만 bias 있음
- GAE: 중간 variance → 균형잡힘
- MC: 가장 높은 variance → 불안정하지만 unbiased

---

### 7.7 디버깅 및 검증

#### 7.7.1 Sanity Checks

**Test 1: λ=0은 δ와 같아야**
```python
def test_gae_lambda_0():
    agent = PGAgent(..., gae_lambda=0)
    advantages = agent._estimate_advantage(obs, rewards, q_values, terminals)

    # Manual delta calculation
    values = agent.critic(obs)
    deltas = rewards + gamma * values[1:] * (1 - terminals) - values[:-1]

    assert np.allclose(advantages, deltas)
```

**Test 2: λ=1은 MC와 같아야**
```python
def test_gae_lambda_1():
    agent = PGAgent(..., gae_lambda=1)
    advantages = agent._estimate_advantage(obs, rewards, q_values, terminals)

    # MC advantage
    mc_advantages = q_values - agent.critic(obs)

    assert np.allclose(advantages, mc_advantages)
```

#### 7.7.2 Common Bugs

**Bug 1: Terminal 처리 누락**
```python
# Wrong
advantages[i] = deltas[i] + gamma * lambda * advantages[i+1]

# Correct
advantages[i] = deltas[i] + gamma * lambda * (1 - terminals[i]) * advantages[i+1]
```

Episode 끝에서 미래 advantage를 0으로 만들어야 해!

**Bug 2: Dummy value 잊음**
```python
# Wrong
values = critic(obs)  # shape: (batch_size,)
advantages[i] = ... + advantages[i+1]  # IndexError when i=batch_size-1!

# Correct
values = np.append(critic(obs), [0])  # shape: (batch_size+1,)
```

**Bug 3: Forward iteration**
```python
# Wrong
for i in range(batch_size):  # Forward!
    advantages[i] = deltas[i] + ... * advantages[i+1]
    # advantages[i+1] is not calculated yet!

# Correct
for i in reversed(range(batch_size)):  # Backward!
```

---

### 7.8 GAE vs Simple Baseline 비교

#### 7.8.1 코드 차이

**Simple Baseline**:
```python
advantages = q_values - values
```
- 한 줄로 끝!
- Monte Carlo advantage

**GAE**:
```python
# 1. TD errors 계산
deltas = rewards + gamma * values[1:] * (1-terminals) - values[:-1]

# 2. Recursive accumulation
for i in reversed(range(batch_size)):
    advantages[i] = deltas[i] + gamma * lambda * (1-terminals[i]) * advantages[i+1]
```
- 더 복잡
- Bias-variance tradeoff 조절 가능

#### 7.8.2 성능 비교

**Simple Baseline**: $\lambda = 1$ (MC)와 동일
- Unbiased하지만 high variance
- 간단한 환경에서도 잘 작동

**GAE (λ=0.95)**:
- Lower variance, slight bias
- 복잡한 환경에서 더 안정적
- PPO와 함께 쓰면 최고!

---

### 7.9 Hyperparameter Tuning Guide

#### 7.9.1 λ 선택 가이드

| Environment Complexity | Recommended λ | 이유 |
|---|---|---|
| 간단 (CartPole) | 0.95 ~ 1.0 | Bias 문제 적음 |
| 중간 (HalfCheetah) | 0.90 ~ 0.97 | 균형 필요 |
| 복잡 (Humanoid) | 0.95 ~ 0.99 | Value function 활용 |

#### 7.9.2 실험적 접근

```python
# Grid search
for lambda_val in [0, 0.5, 0.9, 0.95, 0.97, 0.99, 1.0]:
    run_experiment(gae_lambda=lambda_val)
```

**TensorBoard로 비교**:
- Eval_AverageReturn: 최종 성능
- Train_StdReturn: 안정성
- 수렴 속도

---

**Part 5 완료. 다음 파트에서 PPO (Proximal Policy Optimization) 구현을 다룹니다.**
