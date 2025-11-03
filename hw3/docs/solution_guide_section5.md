# Homework 3 Solution Guide - Section 5

## 7. Double Q-Learning Implementation

### 7.1 Overestimation Bias 문제

**왜 Double DQN이 필요한가?**

Vanilla DQN은 **overestimation bias** (과대평가 편향) 문제가 있습니다.

#### Vanilla DQN의 Target 계산

$$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$

여기서 `max` 연산이 문제입니다.

**Max Operator와 Noise:**

Q-function은 noise를 포함합니다 (estimation error):

$$Q_{\theta}(s, a) = Q^*(s, a) + \epsilon$$

여기서 $\epsilon$은 error (+ 또는 -).

**예시:**

진짜 Q-values:
```python
Q*(s', a0) = 5.0
Q*(s', a1) = 5.0  # 둘 다 같음
```

Noisy estimates:
```python
Q(s', a0) = 5.0 + (-0.3) = 4.7
Q(s', a1) = 5.0 + (+0.5) = 5.5  # noise가 양수!
```

Max 연산:
```python
max(4.7, 5.5) = 5.5 > 5.0  # 과대평가!
```

**왜 항상 양쪽이 안 될까?**

- Max는 **positive noise를 선호**
- Negative noise는 무시됨 (max에서 선택 안 됨)
- 평균적으로 overestimation

**수학적 증명:**

$$\mathbb{E}[\max(Q_1, Q_2)] \geq \max(\mathbb{E}[Q_1], \mathbb{E}[Q_2])$$

이것은 **Jensen's inequality** (max는 convex function).

#### Overestimation의 문제점

1. **잘못된 행동 선택**:
   - 나쁜 행동이 운 좋게 높게 평가될 수 있음
   - 그 행동을 계속 선택 → 나쁜 정책

2. **불안정한 학습**:
   - Q-values가 계속 부풀려짐
   - Divergence 위험

3. **Suboptimal policy**:
   - 진짜 좋은 행동을 못 찾음

---

### 7.2 Double DQN의 해결책

**핵심 아이디어**: "선택"과 "평가"를 분리

#### Vanilla DQN (단일 네트워크):

$$y = r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_{\theta^-}(s', a'))$$

- **Selection**: $\arg\max_{a'} Q_{\theta^-}(s', a')$ - target network
- **Evaluation**: $Q_{\theta^-}(s', a')$ - target network
- **같은 네트워크**로 선택하고 평가 → bias!

#### Double DQN (두 네트워크):

$$a^* = \arg\max_{a'} Q_{\theta}(s', a')$$
$$y = r + \gamma Q_{\theta^-}(s', a^*)$$

- **Selection**: $Q_{\theta}$ (online network)
- **Evaluation**: $Q_{\theta^-}$ (target network)
- **다른 네트워크** 사용 → bias 감소!

**왜 작동하는가?**

두 네트워크의 noise가 **independent**라고 가정:

```python
# Online network errors:
ε_online(s', a0) = +0.3
ε_online(s', a1) = +0.5  # maximum! → select a1

# Target network errors (independent):
ε_target(s', a0) = +0.2
ε_target(s', a1) = -0.1  # 이번엔 negative!

# Double DQN:
a* = argmax_a Q_θ(s', a) = a1  # online으로 선택
y = Q_θ⁻(s', a1) = 5.0 + (-0.1) = 4.9  # target으로 평가

# Result: 4.9 < 5.5 (vanilla DQN)
```

평균적으로 overestimation이 감소합니다.

---

### 7.3 구현 (dqn_agent.py)

**이미 구현된 부분:**

```python
def update_critic(self, ...):
    with torch.no_grad():
        next_qa_values = self.target_critic(next_obs)

        if self.use_double_q:
            # Choose action with argmax of critic network
            next_action = ...
        else:
            # Choose action with argmax of target critic network
            next_action = ...
```

Section 3에서 이미 solution을 제공했습니다:

```python
if self.use_double_q:
    # Double DQN: online network로 행동 선택
    next_action = torch.argmax(self.critic(next_obs), dim=1)
else:
    # Vanilla DQN: target network로 행동 선택
    next_action = torch.argmax(next_qa_values, dim=1)
```

**코드 분석:**

**Vanilla DQN (`use_double_q=False`):**

```python
next_qa_values = self.target_critic(next_obs)  # θ⁻
next_action = torch.argmax(next_qa_values, dim=1)  # argmax로 θ⁻ 사용
next_q_values = torch.gather(next_qa_values, 1, next_action.unsqueeze(1)).squeeze(1)
# 결과: max_{a'} Q_θ⁻(s', a')
```

**Double DQN (`use_double_q=True`):**

```python
next_qa_values = self.target_critic(next_obs)  # θ⁻ (evaluation용)
next_action = torch.argmax(self.critic(next_obs), dim=1)  # θ (selection용)
next_q_values = torch.gather(next_qa_values, 1, next_action.unsqueeze(1)).squeeze(1)
# 결과: Q_θ⁻(s', argmax_{a'} Q_θ(s', a'))
```

**차이점:**

- Vanilla: `self.target_critic`로 `argmax`
- Double: `self.critic`로 `argmax`, `self.target_critic`로 평가

**단 한 줄만 다릅니다!**

---

### 7.4 실험 설정

#### DQN 실험 (Section 4.2)

**CartPole-v1:**

```bash
python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/cartpole.yaml --seed 1
```

**설정:**

```yaml
# experiments/dqn/cartpole.yaml
env_name: CartPole-v1
exp_name: dqn
total_steps: 300000
learning_rate: 0.001
discount: 0.99
target_update_period: 1000
use_double_q: false  # Vanilla DQN
```

**기대 결과:**

- Training time: ~15 minutes (GPU) / ~30 minutes (CPU)
- Final eval_return: ~500 (maximum)
- Learning curve: 빠른 상승 후 plateau

**Plot 요구사항:**

- X-axis: environment steps (0-300K)
- Y-axis: eval_return
- Caption: "DQN on CartPole-v1: Learning curve shows convergence to optimal policy (return ~500) around 300K steps"

**Learning Rate 비교 실험:**

```yaml
# experiments/dqn/cartpole_lr_5e-2.yaml
learning_rate: 0.05  # 50x larger!
```

```bash
python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/cartpole_lr_5e-2.yaml --seed 1
```

**예상 현상:**

**(a) Predicted Q-values:**

- **Default LR (0.001)**: Stable, gradual increase
- **High LR (0.05)**: Unstable, oscillating, often diverging

**이유**:
- Learning rate가 크면 각 gradient step이 큼
- Q-values가 급격하게 변함
- Oscillation 또는 divergence

**(b) Critic Error:**

- **Default LR**: Decreasing over time
- **High LR**: High variance, not decreasing

**이유**:
- Overshooting the minimum
- Gradient descent가 불안정

**(c) Eval Returns:**

- **Default LR**: Smooth improvement to ~500
- **High LR**: Erratic, may not converge

**이유**:
- Unstable Q-function → poor policy
- May learn sub-optimal policy or fail to learn

**관련 강의 주제:**

- **Optimization**: Learning rate scheduling
- **Stability**: Step size와 convergence
- **Deep RL**: Function approximation의 instability

Plot 요구사항: 같은 axes에 두 curves를 다른 색으로.

---

#### BankHeist 실험 (Section 5.2)

**Vanilla DQN:**

```bash
python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 1
python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 2
python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 3
```

**설정:**

```yaml
# experiments/dqn/bankheist.yaml
env_name: BankHeist-v5
total_steps: 1000000
learning_rate: 0.0001
target_update_period: 2000
use_double_q: false  # Vanilla
```

**기대 결과:**

- Training time: ~2 hours/seed (GPU) / ~4 hours (CPU)
- Final eval_return: ~150
- Total time: ~6 hours (GPU) / ~12 hours (CPU)

**Double DQN:**

```bash
python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 1
python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 2
python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 3
```

**설정:**

```yaml
# experiments/dqn/bankheist_ddqn.yaml
env_name: BankHeist-v5
use_double_q: true  # Double DQN!
# 다른 설정은 동일
```

**기대 결과:**

- Final eval_return: ~300 (2x better!)
- More stable learning
- (Disclaimer: seed에 따라 variation 있음)

**Plot 요구사항:**

- 같은 axes에 6 curves:
  - DQN: 3 seeds (blue)
  - Double DQN: 3 seeds (red)
- X-axis: environment steps (0-1M)
- Y-axis: eval_return
- Caption에 비교 설명 포함

**예상 관찰:**

1. **Double DQN이 더 높은 return** (평균적으로)
2. **더 안정적인 학습** (덜 oscillating)
3. **더 빠른 수렴** (같은 performance에 빨리 도달)

**설명 예시:**

"Double DQN shows significantly better performance (~300 vs ~150 return) due to reduced overestimation bias. By decoupling action selection (online network) and value estimation (target network), Double DQN avoids selecting actions that are overestimated due to noise. This is particularly important in complex environments like Atari where Q-value estimates are noisy. The three seeds show consistent improvement, demonstrating the robustness of the approach."

---

### 7.5 DQN vs Double DQN 비교표

| Aspect | Vanilla DQN | Double DQN |
|--------|-------------|------------|
| **Action Selection** | $\arg\max_{a'} Q_{\theta^-}(s', a')$ | $\arg\max_{a'} Q_{\theta}(s', a')$ |
| **Value Estimation** | $Q_{\theta^-}(s', a^*)$ | $Q_{\theta^-}(s', a^*)$ |
| **Overestimation** | High | Reduced |
| **CartPole Performance** | ~500 | ~500 (similar) |
| **BankHeist Performance** | ~150 | ~300 (better) |
| **Stability** | Moderate | Better |
| **Code Change** | 1 line | 1 line |
| **Computational Cost** | Same | Same |

**왜 CartPole에서는 차이가 적은가?**

- 간단한 환경 → Q-values가 덜 noisy
- Overestimation bias가 작음
- 둘 다 optimal에 수렴

**왜 BankHeist에서는 차이가 큰가?**

- 복잡한 환경 → Q-values가 매우 noisy
- Pixel input → high-dimensional, 학습 어려움
- Overestimation이 심각 → Double DQN이 중요

---

## 8. Hyperparameter Experimentation (Section 6)

### 8.1 목적

Q-learning의 **hyperparameter sensitivity**를 분석합니다.

**선택 가능한 hyperparameters:**

1. **Learning rate** (`learning_rate`)
2. **Network architecture** (`hidden_size`, `num_layers`)
3. **Exploration schedule** (`exploration_schedule`)
4. **Discount factor** (`discount`)
5. **Target update period** (`target_update_period`)
6. **Batch size** (`batch_size`)

**요구사항:**

- 하나의 hyperparameter 선택
- 4가지 값 실험 (default 포함)
- CartPole-v1 환경
- Config 파일 생성: `experiments/dqn/hyperparameters/`

---

### 8.2 추천: Exploration Schedule 실험

**왜 exploration schedule이 흥미로운가?**

- RL의 핵심 tradeoff (exploration vs exploitation)
- 결과가 극적으로 다를 수 있음
- 개념적으로 이해하기 쉬움

**실험 설정:**

**Config 1: Default**

```yaml
# experiments/dqn/hyperparameters/exploration_default.yaml
base: experiments/dqn/cartpole.yaml
exploration_schedule:
  type: piecewise
  endpoints:
    - [0, 1.0]
    - [30000, 0.02]
  outside_value: 0.02
```

**Config 2: Fast Decay**

```yaml
# experiments/dqn/hyperparameters/exploration_fast.yaml
base: experiments/dqn/cartpole.yaml
exploration_schedule:
  endpoints:
    - [0, 1.0]
    - [10000, 0.02]  # 3x faster decay!
```

**Config 3: Slow Decay**

```yaml
# experiments/dqn/hyperparameters/exploration_slow.yaml
base: experiments/dqn/cartpole.yaml
exploration_schedule:
  endpoints:
    - [0, 1.0]
    - [100000, 0.02]  # 3x slower decay
```

**Config 4: High Final Epsilon**

```yaml
# experiments/dqn/hyperparameters/exploration_high.yaml
base: experiments/dqn/cartpole.yaml
exploration_schedule:
  endpoints:
    - [0, 1.0]
    - [30000, 0.1]  # Keep exploring!
  outside_value: 0.1
```

**예상 결과:**

**(1) Fast Decay:**
- 빠르게 exploitation으로 전환
- 초기 학습은 빠를 수 있음
- 하지만 충분한 exploration 부족 가능
- Final performance: 좋을 수도, 나쁠 수도

**(2) Default:**
- Balanced
- Best overall performance

**(3) Slow Decay:**
- 오래 explore
- 학습이 느림 (too much randomness)
- 하지만 data diversity는 좋음
- Final: default와 비슷하거나 약간 나쁨

**(4) High Final Epsilon:**
- 계속 10% random action
- Evaluation에서도 suboptimal action
- Final: 명확하게 나쁨 (~450 instead of 500)

**Plot:**

- 4 curves (다른 색)
- Legend로 구분
- X-axis: steps, Y-axis: eval_return

**Caption 예시:**

"Sensitivity of DQN to exploration schedule on CartPole-v1. Default (orange, ε: 1.0→0.02 over 30K steps) achieves best performance. Fast decay (blue, 10K steps) converges quickly but may not explore enough. Slow decay (green, 100K steps) takes longer to exploit learned policy. High final epsilon (red, ε=0.1) performs suboptimally due to persistent random actions during evaluation."

---

### 8.3 다른 Hyperparameter 옵션

**Option 1: Learning Rate**

```yaml
# learning_rate: [1e-4, 1e-3, 1e-2, 5e-2]
```

- 너무 작으면: 학습 느림
- 너무 크면: 불안정, divergence
- Section 4.2에서 이미 일부 탐색

**Option 2: Network Architecture**

```yaml
# hidden_size: [32, 64, 128, 256]
# 또는
# num_layers: [1, 2, 3, 4]
```

- 작은 network: underfit
- 큰 network: slower, may overfit (DQN에서는 덜 중요)
- CartPole은 간단해서 차이 적을 수 있음

**Option 3: Target Update Period**

```yaml
# target_update_period: [500, 1000, 2000, 5000]
```

- 작은 값: 자주 업데이트, 덜 stable
- 큰 값: 느리게 학습
- Trade-off between stability and speed

**Option 4: Batch Size**

```yaml
# batch_size: [32, 64, 128, 256]
```

- 작은 batch: noisy gradients, 빠른 iteration
- 큰 batch: stable gradients, 느린 iteration
- Memory constraint

**선택 기준:**

1. **흥미로운 trade-off**가 있는가?
2. **결과 차이**가 명확한가?
3. **설명**하기 쉬운가?

Exploration schedule을 추천하는 이유는 위 3가지를 모두 만족하기 때문입니다.

---

### 8.4 실험 실행

```bash
# 각 config로 실험
python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/hyperparameters/exploration_default.yaml --seed 1
python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/hyperparameters/exploration_fast.yaml --seed 1
python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/hyperparameters/exploration_slow.yaml --seed 1
python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/hyperparameters/exploration_high.yaml --seed 1
```

**시간:**

- 각 실험: ~15분
- 총: ~1시간

**TensorBoard:**

```bash
tensorboard --logdir data/
```

각 실험의 `eval_return` curve를 비교.

**Plotting:**

```python
# gcb6206/scripts/parse_tensorboard.py 사용
python gcb6206/scripts/parse_tensorboard.py data/

# 또는 직접 plot:
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for name, color in [("default", "orange"), ("fast", "blue"), ...]:
    # Load data
    steps, returns = load_tensorboard(f"data/hw3_dqn_{name}")
    plt.plot(steps, returns, label=name, color=color)

plt.xlabel("Environment Steps")
plt.ylabel("Evaluation Return")
plt.legend()
plt.title("DQN Hyperparameter Sensitivity: Exploration Schedule")
plt.grid(True)
plt.savefig("hyperparameter_analysis.png")
```

---

### 8.5 분석 및 설명

**보고서에 포함할 내용:**

1. **Hyperparameter 선택 이유**:
   - "I chose exploration schedule because it directly relates to the fundamental exploration-exploitation trade-off in RL."

2. **실험 설정**:
   - 4가지 설정 나열
   - 각각의 의미 설명

3. **결과 관찰**:
   - Plot 제시
   - 각 curve의 특징 설명

4. **이론적 설명**:
   - 왜 이런 결과가 나왔는가?
   - RL 이론과 연결

5. **결론**:
   - 최적의 설정은?
   - Trade-off는?
   - 다른 환경에서는 어떨까?

**예시 설명:**

"The default schedule (ε: 1.0→0.02 over 30K steps) balances exploration and exploitation effectively. Fast decay leads to premature exploitation—the agent commits to a potentially suboptimal policy before exploring sufficient state-action pairs. Slow decay maintains high randomness for too long, delaying the exploitation of learned knowledge. High final epsilon permanently handicaps the policy by forcing 10% random actions even after learning, resulting in lower returns. This demonstrates that careful tuning of exploration is crucial for RL algorithms."

---

## 9. Summary (요약)

### 9.1 핵심 구현 요약

**DQN Agent (dqn_agent.py):**

1. **get_action()**: Epsilon-greedy 행동 선택
2. **update_critic()**: TD loss로 Q-network 업데이트
3. **update()**: Critic + target network 업데이트

**Training Loop (run_hw3.py):**

1. **Action**: `agent.get_action(obs, epsilon)`
2. **Step**: `env.step(action)`
3. **Store**: `replay_buffer.insert(...)`
4. **Sample**: `batch = replay_buffer.sample(batch_size)`
5. **Update**: `agent.update(batch, step)`
6. **Evaluate**: 주기적으로 성능 측정

**Utils (utils.py):**

1. **sample_trajectory()**: 하나의 episode 실행
2. **Terminated vs Truncated** 구분

**Double DQN:**

- Single line change: action selection으로 online network 사용

### 9.2 주요 개념

1. **Off-policy Learning**: Replay buffer로 과거 데이터 재사용
2. **Target Network**: Moving target 문제 해결
3. **Epsilon-Greedy**: Exploration-exploitation balance
4. **Experience Replay**: i.i.d. data로 SGD
5. **Double Q-Learning**: Overestimation bias 감소
6. **Frame Stacking**: Temporal information (Atari)
7. **Memory Efficiency**: Frame을 한 번만 저장

### 9.3 실험 결과 체크리스트

- [ ] Section 4.2: CartPole DQN (eval_return ~500)
- [ ] Section 4.2: CartPole learning rate 비교 (3 plots)
- [ ] Section 5.2: BankHeist DQN (3 seeds, ~150)
- [ ] Section 5.2: BankHeist Double DQN (3 seeds, ~300)
- [ ] Section 5.2: DQN vs DDQN 비교 plot
- [ ] Section 6: Hyperparameter 실험 (4 values)
- [ ] 모든 plot에 caption 추가
- [ ] 결과 설명 작성

### 9.4 디버깅 팁

**Q-values가 diverge하는 경우:**

- Learning rate 줄이기
- Gradient clipping 확인
- Target update period 줄이기

**학습이 너무 느린 경우:**

- Learning rate 높이기
- Batch size 확인
- Replay buffer에 충분한 데이터 있는지 확인

**Performance가 낮은 경우:**

- Exploration schedule 확인
- Network architecture 확인
- Hyperparameter tuning

**Memory error:**

- Batch size 줄이기
- Replay buffer capacity 줄이기
- GPU memory 확인

---
