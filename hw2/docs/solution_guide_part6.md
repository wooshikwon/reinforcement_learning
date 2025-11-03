# GCB6206 Homework 2: Policy Gradients 완전 해설 가이드 (Part 6)

## 8. Proximal Policy Optimization (PPO) 구현

### 8.1 왜 PPO가 필요한가?

#### 8.1.1 Policy Gradient의 근본적 문제

지금까지 구현한 방식의 문제점:

**Data Efficiency 문제**:
```python
# 1 iteration
trajectories = collect_data(policy)  # 환경과 상호작용 (비쌈!)
policy.update(trajectories)          # 한 번만 업데이트
# → trajectories 버림 (다시 쓸 수 없음)
```

**왜 버려야 하냐?**

Policy Gradient는 **on-policy** 알고리즘이야:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}[\cdots]
$$

**현재 정책 $\pi_\theta$로 수집한 데이터**만 사용 가능!

업데이트 후:
- $\theta \rightarrow \theta'$ (정책 변경)
- 이전 데이터는 $\pi_\theta$에서 수집됨
- 이제 정책은 $\pi_{\theta'}$ (다름!)
- **Distribution mismatch** → 데이터 재사용 불가 ✗

#### 8.1.2 PPO의 아이디어

**핵심**: "정책이 너무 많이 바뀌지 않으면, 데이터 재사용 가능하지 않을까?"

**Trust Region 개념**:
```
Old policy: π_old
New policy: π_new

IF π_new ≈ π_old (충분히 비슷)
THEN 이전 데이터를 여러 번 재사용 가능!
```

**PPO의 방법**: **Clipping**으로 policy 변화 제한

---

### 8.2 PPO-Clip Objective 유도

#### 8.2.1 Importance Sampling

**문제**: Old policy $\pi_{\theta_{\text{old}}}$로 수집한 데이터로, new policy $\pi_\theta$를 업데이트하고 싶어.

**Importance Sampling**:
$$
\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]
$$

Policy Gradient에 적용:
$$
\begin{align}
\nabla_\theta J(\theta) &= \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)] \\
&= \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)\right]
\end{align}
$$

#### 8.2.2 Probability Ratio

**정의**:
$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
$$

**의미**:
- $r_t = 1$: 두 정책이 동일
- $r_t > 1$: New policy가 해당 행동을 더 선호
- $r_t < 1$: New policy가 해당 행동을 덜 선호

**Surrogate objective**:
$$
L^{\text{CPI}}(\theta) = \mathbb{E}\left[r_t(\theta) \cdot A_t\right]
$$

CPI = Conservative Policy Iteration

**문제**: $r_t$가 너무 커지면?
- Old policy와 너무 달라짐
- Importance sampling 부정확
- 학습 불안정

#### 8.2.3 Clipped Surrogate Objective

**PPO-Clip**:
$$
L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) A_t,\, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$

**Clip 함수**:
$$
\text{clip}(r, 1-\epsilon, 1+\epsilon) = \begin{cases}
1-\epsilon & \text{if } r < 1-\epsilon \\
r & \text{if } 1-\epsilon \le r \le 1+\epsilon \\
1+\epsilon & \text{if } r > 1+\epsilon
\end{cases}
$$

**예시** (ε=0.2):
```python
r < 0.8:  clip(r) = 0.8
0.8 ≤ r ≤ 1.2:  clip(r) = r  (변화 없음)
r > 1.2:  clip(r) = 1.2
```

#### 8.2.4 Min 연산의 의미

**경우 1: Advantage > 0** (좋은 행동)

목표: $r_t$를 증가시켜서 해당 행동의 확률 높이기

```python
A = +2.0  (좋은 행동!)

# r_t가 작을 때 (new policy가 덜 선호)
r = 0.5
r * A = 0.5 * 2.0 = 1.0
clip(r) * A = 0.8 * 2.0 = 1.6
min(1.0, 1.6) = 1.0  ← unclipped 선택

# r_t가 적당할 때
r = 1.1
r * A = 1.1 * 2.0 = 2.2
clip(r) * A = 1.1 * 2.0 = 2.2  (clipping 안 됨)
min(2.2, 2.2) = 2.2  ← 둘 다 같음

# r_t가 클 때 (new policy가 훨씬 더 선호)
r = 2.0
r * A = 2.0 * 2.0 = 4.0
clip(r) * A = 1.2 * 2.0 = 2.4
min(4.0, 2.4) = 2.4  ← clipped! (제한됨)
```

**효과**: $r_t$가 너무 커지는 걸 방지 (1.2 이상 못 감)

**경우 2: Advantage < 0** (나쁜 행동)

목표: $r_t$를 감소시켜서 해당 행동의 확률 낮추기

```python
A = -2.0  (나쁜 행동!)

# r_t가 작을 때 (new policy가 이미 덜 선호)
r = 0.5
r * A = 0.5 * (-2.0) = -1.0
clip(r) * A = 0.8 * (-2.0) = -1.6
min(-1.0, -1.6) = -1.6  ← clipped! (더 감소 제한)

# r_t가 적당할 때
r = 1.1
r * A = 1.1 * (-2.0) = -2.2
clip(r) * A = 1.1 * (-2.0) = -2.2
min(-2.2, -2.2) = -2.2

# r_t가 클 때 (new policy가 여전히 선호)
r = 2.0
r * A = 2.0 * (-2.0) = -4.0
clip(r) * A = 1.2 * (-2.0) = -2.4
min(-4.0, -2.4) = -4.0  ← unclipped (더 감소하도록)
```

**효과**: $r_t$가 너무 작아지는 걸 방지 (0.8 이하 못 감)

---

### 8.3 코드 구현: `ppo_update()`

#### 8.3.1 전체 코드 구조

```python
# policies.py: MLPPolicyPG.ppo_update()

def ppo_update(
    self,
    obs: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    old_logp: np.ndarray,
    ppo_cliprange: float = 0.2,
) -> dict:
    """Implements PPO update."""
    assert obs.ndim == 2
    assert advantages.ndim == 1
    assert old_logp.ndim == 1
    assert advantages.shape == old_logp.shape

    obs = ptu.from_numpy(obs)
    actions = ptu.from_numpy(actions)
    advantages = ptu.from_numpy(advantages)
    old_logp = ptu.from_numpy(old_logp)

    # TODO: Implement PPO update
    loss = None

    return {"PPO Loss": ptu.to_numpy(loss)}
```

#### 8.3.2 Step-by-Step 구현

```python
def ppo_update(self, obs, actions, advantages, old_logp, ppo_cliprange=0.2):
    # Convert to tensors (이미 위에서 함)

    # Step 1: Calculate new log probabilities
    dist = self.forward(obs)
    new_logp = dist.log_prob(actions)

    # For continuous actions, sum over action dims
    if not self.discrete:
        new_logp = new_logp.sum(dim=-1)

    # Step 2: Calculate probability ratio
    ratio = torch.exp(new_logp - old_logp)

    # Step 3: Calculate clipped ratio
    clipped_ratio = torch.clamp(ratio, 1 - ppo_cliprange, 1 + ppo_cliprange)

    # Step 4: Calculate PPO loss
    loss_unclipped = ratio * advantages
    loss_clipped = clipped_ratio * advantages
    loss = -torch.min(loss_unclipped, loss_clipped).mean()

    # Step 5: Backpropagation
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return {"PPO Loss": ptu.to_numpy(loss)}
```

**한 줄씩 상세 설명:**

#### 8.3.3 Step 1: New Log Probabilities

```python
dist = self.forward(obs)
new_logp = dist.log_prob(actions)

if not self.discrete:
    new_logp = new_logp.sum(dim=-1)
```

**설명**:
- `obs` shape: `(batch_size, ob_dim)` → 예: `(128, 4)`
- `dist`: 현재 정책의 행동 분포
- `new_logp`: **새 정책**이 해당 행동을 선택할 log probability

**구체적 예시**:
```python
# Discrete (CartPole)
obs = tensor([[0.1, 0.5, -0.2, 0.3], ...])  # (128, 4)
actions = tensor([0, 1, 0, ...])             # (128,)

dist = Categorical(logits=logits_net(obs))
new_logp = dist.log_prob(actions)
# new_logp = tensor([-0.52, -1.13, -0.48, ...])  # (128,)

# Continuous (HalfCheetah)
obs = tensor([[...], ...])                   # (128, 17)
actions = tensor([[...], ...])               # (128, 6)

dist = Normal(mean_net(obs), exp(logstd))
new_logp = dist.log_prob(actions)            # (128, 6)
new_logp = new_logp.sum(dim=-1)              # (128,) ← 합산!
```

#### 8.3.4 Step 2: Probability Ratio

```python
ratio = torch.exp(new_logp - old_logp)
```

**수식**:
$$
r = \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} = \exp(\log \pi_\theta(a|s) - \log \pi_{\theta_{\text{old}}}(a|s))
$$

**왜 exp?**

Log probability 차이:
$$
\log \pi_\theta - \log \pi_{\theta_{\text{old}}} = \log \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}
$$

Exp를 취하면:
$$
\exp(\log \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}) = \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}
$$

**구체적 예시**:
```python
new_logp = tensor([-0.5, -1.2, -0.3])
old_logp = tensor([-0.6, -1.0, -0.4])

ratio = exp(new_logp - old_logp)
      = exp(tensor([0.1, -0.2, 0.1]))
      = tensor([1.105, 0.819, 1.105])
```

**해석**:
- `ratio[0] = 1.105`: New policy가 행동 0을 10.5% 더 선호
- `ratio[1] = 0.819`: New policy가 행동 1을 18.1% 덜 선호
- `ratio[2] = 1.105`: New policy가 행동 2를 10.5% 더 선호

#### 8.3.5 Step 3: Clipped Ratio

```python
clipped_ratio = torch.clamp(ratio, 1 - ppo_cliprange, 1 + ppo_cliprange)
```

**`torch.clamp()` 함수**:
```python
torch.clamp(input, min, max)
# input의 각 원소를 [min, max] 범위로 제한
```

내부 동작:
```python
def clamp(x, min_val, max_val):
    return max(min(x, max_val), min_val)
```

**예시** (ppo_cliprange=0.2):
```python
ratio = tensor([0.5, 0.9, 1.1, 1.5, 2.0])

clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
# clipped_ratio = tensor([0.8, 0.9, 1.1, 1.2, 1.2])
#                          ↑                 ↑    ↑
#                       clipped          clipped clipped
```

**구체적 계산**:
```python
ppo_cliprange = 0.2
ratio = tensor([1.105, 0.819, 1.105])

clipped_ratio = clamp(ratio, 1-0.2, 1+0.2)
              = clamp(ratio, 0.8, 1.2)
              = tensor([1.105, 0.819, 1.105])
# 모두 [0.8, 1.2] 범위 안 → clipping 없음
```

#### 8.3.6 Step 4: PPO Loss

```python
loss_unclipped = ratio * advantages
loss_clipped = clipped_ratio * advantages
loss = -torch.min(loss_unclipped, loss_clipped).mean()
```

**Line 1-2**: 두 loss 계산
```python
ratio = tensor([1.105, 0.819, 1.105])
clipped_ratio = tensor([1.105, 0.819, 1.105])
advantages = tensor([2.0, -1.5, 3.0])

loss_unclipped = ratio * advantages
               = tensor([2.21, -1.229, 3.315])

loss_clipped = clipped_ratio * advantages
             = tensor([2.21, -1.229, 3.315])
```

**Line 3**: Element-wise minimum
```python
torch.min(loss_unclipped, loss_clipped)
# = tensor([2.21, -1.229, 3.315])  (이 경우 같음)
```

**왜 min?**

**Pessimistic objective**: 두 값 중 **작은 값** 선택
- Advantage > 0: 더 보수적으로 증가
- Advantage < 0: 더 적극적으로 감소

**예시 (clipping 발생)**:
```python
ratio = tensor([2.0])           # 너무 큼!
clipped_ratio = tensor([1.2])   # 제한됨
advantages = tensor([1.5])      # 좋은 행동

loss_unclipped = 2.0 * 1.5 = 3.0
loss_clipped = 1.2 * 1.5 = 1.8

min(3.0, 1.8) = 1.8  ← clipped 선택 (더 보수적)
```

**Negative 부호**:
```python
loss = -torch.min(...).mean()
```

Policy Gradient와 동일한 이유: Loss 최소화 = Objective 최대화

#### 8.3.7 Step 5: Backpropagation

```python
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

일반 Policy Gradient와 동일!

---

### 8.4 Old Log Probabilities 계산

#### 8.4.1 `_calculate_log_probs()` 메서드

```python
# pg_agent.py

def _calculate_log_probs(self, obs: np.ndarray, actions: np.ndarray):
    """Calculate log probabilities of actions."""
    assert obs.ndim == 2

    obs = ptu.from_numpy(obs)
    actions = ptu.from_numpy(actions)

    dist = self.actor.forward(obs)
    logp = dist.log_prob(actions)

    if not self.actor.discrete:
        logp = logp.sum(dim=-1)

    logp = ptu.to_numpy(logp)

    assert logp.ndim == 1 and logp.shape[0] == obs.shape[0]
    return logp
```

**언제 호출?**

PPO update **전에** 한 번만:
```python
# pg_agent.py: update() 메서드

if self.use_ppo:
    # Calculate old log probs ONCE
    logp = self._calculate_log_probs(obs, actions)

    for epoch in range(n_ppo_epochs):
        for minibatch in minibatches:
            # PPO update with OLD logp
            actor.ppo_update(obs_mb, actions_mb, adv_mb, logp_mb, ...)
```

**왜 한 번만?**

Old policy = **업데이트 전 정책**
- 여러 epoch 동안 고정
- 매번 재계산하면 "new policy"가 됨!

---

### 8.5 Multiple Epochs and Minibatches

#### 8.5.1 PPO 학습 루프

```python
# pg_agent.py: update() 메서드

if self.use_ppo:
    logp = self._calculate_log_probs(obs, actions)

    n_batch = len(obs)
    inds = np.arange(n_batch)

    for epoch in range(self.n_ppo_epochs):
        np.random.shuffle(inds)  # 매 epoch마다 섞기

        minibatch_size = (n_batch + self.n_ppo_minibatches - 1) // self.n_ppo_minibatches

        for start in range(0, n_batch, minibatch_size):
            end = start + minibatch_size

            # Minibatch 추출
            obs_mb = obs[inds[start:end]]
            actions_mb = actions[inds[start:end]]
            advantages_mb = advantages[inds[start:end]]
            logp_mb = logp[inds[start:end]]

            # Advantage normalization
            if self.normalize_advantages:
                mean = advantages_mb.mean()
                std = advantages_mb.std() + 1e-8
                advantages_mb = (advantages_mb - mean) / std

            # PPO update
            info = self.actor.ppo_update(
                obs_mb, actions_mb, advantages_mb, logp_mb, self.ppo_cliprange
            )
```

**한 부분씩 설명:**

#### 8.5.2 인덱스 섞기

```python
n_batch = len(obs)  # 예: 5000
inds = np.arange(n_batch)  # [0, 1, 2, ..., 4999]

for epoch in range(n_ppo_epochs):
    np.random.shuffle(inds)
```

**왜 섞냐?**

같은 데이터를 여러 번 사용하는데, **순서를 바꿔서 overfitting 방지**

**예시**:
```python
# Epoch 0
inds = [2, 4, 0, 1, 3, ...]

# Epoch 1
inds = [3, 0, 2, 4, 1, ...]  # 다른 순서!
```

#### 8.5.3 Minibatch Size 계산

```python
minibatch_size = (n_batch + self.n_ppo_minibatches - 1) // self.n_ppo_minibatches
```

**왜 이렇게?**

Ceiling division: $\lceil \frac{n}{k} \rceil$

**예시**:
```python
n_batch = 5000
n_ppo_minibatches = 4

# 일반 나눗셈
5000 / 4 = 1250

# Ceiling division
(5000 + 4 - 1) // 4 = 5003 // 4 = 1250

# 만약 n_batch = 5001이면?
(5001 + 4 - 1) // 4 = 5004 // 4 = 1251  (올림!)
```

**결과**: 5000개를 4개 minibatch로 나누면, 각 1250개

#### 8.5.4 Minibatch 추출

```python
for start in range(0, n_batch, minibatch_size):
    end = start + minibatch_size

    obs_mb = obs[inds[start:end]]
```

**예시**:
```python
n_batch = 5000
minibatch_size = 1250

# Iteration 1: start=0
start=0, end=1250
obs_mb = obs[inds[0:1250]]  # 1250 samples

# Iteration 2: start=1250
start=1250, end=2500
obs_mb = obs[inds[1250:2500]]  # 1250 samples

# Iteration 3: start=2500
start=2500, end=3750
obs_mb = obs[inds[2500:3750]]  # 1250 samples

# Iteration 4: start=3750
start=3750, end=5000
obs_mb = obs[inds[3750:5000]]  # 1250 samples
```

---

### 8.6 PPO vs Vanilla PG 비교

#### 8.6.1 데이터 효율성

**Vanilla PG**:
```python
# 1 iteration
data = collect(5000 steps)  # 환경과 상호작용
policy.update(data, 1 step) # 1번 업데이트
# → 5000 env steps per 1 policy update
```

**PPO**:
```python
# 1 iteration
data = collect(5000 steps)  # 환경과 상호작용
for epoch in range(4):      # 4 epochs
    for mb in minibatches:  # 4 minibatches
        policy.update(mb)   # 4×4 = 16번 업데이트!
# → 5000 env steps per 16 policy updates
```

**16배 더 효율적!**

#### 8.6.2 성능 비교 (Reacher-v4)

과제 실험 결과 예상:

| Method | Final Return | Training Time | Stability |
|---|---|---|---|
| PG | -15 ~ -20 | 100 iters | 불안정 |
| PPO | **-8 ~ -10** | **100 iters** | **안정** |

PPO가 더 빠르고 안정적으로 수렴!

---

### 8.7 Hyperparameter 설정

#### 8.7.1 Clip Range (ε)

```python
parser.add_argument("--ppo_cliprange", type=float, default=0.2)
```

**Typical values**: 0.1 ~ 0.3

**ε = 0.1**: 매우 보수적 (정책 변화 ±10%)
**ε = 0.2**: **권장** (정책 변화 ±20%)
**ε = 0.3**: 덜 보수적

#### 8.7.2 Number of Epochs

```python
parser.add_argument("--n_ppo_epochs", type=int, default=4)
```

**Typical values**: 3 ~ 10

**Too few** (1-2): 데이터 활용 부족
**Good** (3-5): 균형 잡힘
**Too many** (>10): Overfitting 위험

#### 8.7.3 Number of Minibatches

```python
parser.add_argument("--n_ppo_minibatches", type=int, default=4)
```

**Typical values**: 4 ~ 16

**계산**:
```
minibatch_size = batch_size / n_ppo_minibatches
```

**예**:
- batch_size=5000, n_ppo_minibatches=4 → 1250 samples/mb
- batch_size=5000, n_ppo_minibatches=10 → 500 samples/mb

**Trade-off**:
- **더 많은 minibatches**: 더 많은 업데이트, 느린 학습
- **더 적은 minibatches**: 적은 업데이트, 빠른 학습

---

### 8.8 디버깅 및 검증

#### 8.8.1 Ratio 모니터링

```python
# ppo_update() 내부에 추가
ratio_mean = ratio.mean().item()
ratio_std = ratio.std().item()
print(f"Ratio: {ratio_mean:.3f} ± {ratio_std:.3f}")

# 정상:
# Ratio: 1.005 ± 0.15  (1에 가까움)

# 문제:
# Ratio: 2.5 ± 1.8  (너무 큼! Clipping 많이 발생)
```

#### 8.8.2 Clipping Fraction

```python
# 얼마나 많은 samples이 clipped 되었는지 측정
clipped = (ratio < 1 - ppo_cliprange) | (ratio > 1 + ppo_cliprange)
clip_frac = clipped.float().mean().item()
print(f"Clipping fraction: {clip_frac:.3f}")

# 정상:
# Clipping fraction: 0.05 ~ 0.2

# 문제:
# Clipping fraction: 0.8  (너무 많이 clip됨 → ε 너무 작음)
# Clipping fraction: 0.0  (전혀 clip 안 됨 → ε 의미 없음)
```

#### 8.8.3 KL Divergence (참고)

PPO 논문에서는 KL divergence도 모니터링 권장:

$$
\text{KL}(\pi_{\text{old}} \| \pi_{\text{new}}) \approx \frac{1}{N} \sum_{i=1}^N \left(\log \pi_{\text{old}}(a_i|s_i) - \log \pi_{\text{new}}(a_i|s_i)\right)
$$

```python
approx_kl = (old_logp - new_logp).mean().item()
print(f"Approx KL: {approx_kl:.4f}")

# 좋음: 0.01 ~ 0.05
# 너무 큼: > 0.1 (정책이 너무 많이 변함)
```

---

### 8.9 왜 PPO가 잘 작동하는가?

#### 8.9.1 Trust Region의 직관

**문제**: Policy gradient는 local information만 사용
- Gradient는 "현재 위치에서 어느 방향"만 알려줌
- 얼마나 멀리 가야 할지는 모름

**해결**: Trust region으로 **안전한 영역** 제한
- Old policy 근처에서만 탐색
- 큰 실수 방지

#### 8.9.2 Data Reuse의 이점

**Sample efficiency**:
```
Vanilla PG: 100k env steps, 20 policy updates
PPO:        100k env steps, 320 policy updates (16배!)
```

**Robotics/시뮬레이션**에서 특히 중요:
- 실제 로봇: 시간, 마모
- 시뮬레이션: 계산 비용

#### 8.9.3 현업에서의 PPO

**가장 널리 쓰이는 on-policy RL 알고리즘**:
- OpenAI Five (Dota 2)
- DeepMind AlphaStar (StarCraft II)
- Robotics: 보행, 조작
- 자율주행 시뮬레이션

**이유**:
- 구현 간단
- 안정적
- 다양한 환경에서 잘 작동
- 하이퍼파라미터에 robust

---

**Part 6 완료. 다음 파트에서 전체 Training Loop를 다룹니다.**
