# Homework 3 Solution Guide - Section 1

## 1. Introduction (소개)

이 과제의 목표는 Deep Q-Network (DQN)과 Double DQN을 이해하고 구현하는 것입니다. Q-learning은 value-based 강화학습 방법으로, 정책(policy)을 직접 학습하는 대신 state-action value function인 Q-function을 학습합니다.

### 왜 DQN인가?

전통적인 Q-learning은 테이블 형태로 모든 state-action 쌍의 Q-value를 저장했습니다. 하지만 이 방식은 다음과 같은 문제가 있습니다:

1. **상태 공간이 크거나 연속적인 경우** (예: 이미지 입력) 테이블로 표현 불가능
2. **메모리 요구량이 지수적으로 증가**
3. **일반화(generalization) 능력 부족** - 비슷한 상태에 대해서도 각각 학습해야 함

DQN은 이런 문제를 **neural network**를 사용해 Q-function을 근사(approximate)함으로써 해결합니다. 신경망은 다음과 같은 장점을 제공합니다:

- **함수 근사(Function Approximation)**: 무한히 많은 상태를 유한한 파라미터로 표현
- **일반화**: 비슷한 상태에 대해 비슷한 Q-value 예측
- **표현력**: 복잡한 패턴과 특징 학습 가능 (특히 이미지에서)

---

## 2. Deep Q-Network Quiz (DQN 퀴즈 정답 및 해설)

### I. Q-Learning cannot leverage off-policy samples, resulting in poor sample efficiency.

**정답: False (거짓)**

**상세 해설:**

이 문제는 Q-learning의 가장 중요한 특징 중 하나를 다룹니다. Q-learning은 **off-policy** 알고리즘입니다.

**Off-policy vs On-policy란?**

- **On-policy**: 현재 학습 중인 정책(policy)으로 데이터를 수집하고, 그 정책을 개선합니다.
  - 예: SARSA, Policy Gradient
  - 행동 정책 = 타겟 정책

- **Off-policy**: 다른 정책(behavior policy)으로 데이터를 수집하고, 목표 정책(target policy)을 학습합니다.
  - 예: Q-learning, DQN
  - 행동 정책 ≠ 타겟 정책

**Q-learning이 off-policy인 이유:**

Q-learning의 업데이트 식을 보면:

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

여기서 핵심은 $\max_{a'} Q(s', a')$ 부분입니다. 이것은 **greedy policy**(탐욕적 정책)에 따른 최대 Q-value를 사용합니다. 즉:

1. **실제 행동**: $\epsilon$-greedy 정책으로 선택 (exploration 포함)
2. **타겟 계산**: greedy 정책 사용 (exploitation만)

이 덕분에 Q-learning은 **replay buffer**를 사용할 수 있습니다. 과거에 어떤 정책으로 수집했든, 그 데이터를 재사용해서 현재 Q-function을 학습할 수 있습니다.

**Sample Efficiency (샘플 효율성):**

Off-policy의 장점은 **높은 sample efficiency**입니다:

- **데이터 재사용**: Replay buffer에서 같은 transition을 여러 번 학습
- **과거 데이터 활용**: 오래된 경험도 계속 사용 가능
- **탐험 데이터 활용**: 랜덤하게 수집한 데이터도 최적 정책 학습에 사용

반면 on-policy 방법(예: Policy Gradient)은 각 데이터를 한 번만 사용하고 버리므로 sample efficiency가 낮습니다.

---

### II. Without an actor, evaluating Q-values for all possible actions is infeasible with continuous action space.

**정답: True (참)**

**상세 해설:**

이 문제는 DQN의 근본적인 한계를 다룹니다.

**DQN의 행동 선택 메커니즘:**

DQN은 다음과 같이 행동을 선택합니다:

$$a^* = \arg\max_a Q_\theta(s, a)$$

이를 위해서는 **모든 가능한 행동 $a$에 대해 Q-value를 계산**하고 최대값을 찾아야 합니다.

**이산 행동 공간 (Discrete Action Space):**

CartPole: 행동 = {왼쪽, 오른쪽} → 2개만 평가하면 됨
```python
q_values = critic(state)  # shape: [num_actions]
# q_values = [Q(s, left), Q(s, right)]
action = torch.argmax(q_values)  # 단순 비교
```

**연속 행동 공간 (Continuous Action Space):**

로봇 팔 제어: 행동 = 각 관절 각도 ∈ ℝ^n → **무한히 많은 행동**

이 경우 모든 행동을 평가하는 것은:
1. **계산적으로 불가능** (무한대를 다 확인할 수 없음)
2. **최적화 문제로 변환 필요** (gradient ascent 등)

```python
# 이런 식으로는 할 수 없음
q_values = []
for a in infinite_continuous_actions:  # 무한 루프!
    q_values.append(critic(state, a))
best_action = actions[argmax(q_values)]
```

**해결책: Actor-Critic 방법**

연속 행동 공간에서는 **actor network**가 필요합니다:

1. **Actor**: 상태를 받아 행동을 직접 출력
   - $\mu_\phi(s) \rightarrow a$

2. **Critic**: 그 행동의 가치를 평가
   - $Q_\theta(s, \mu_\phi(s))$

이런 방법의 예:
- **DDPG** (Deep Deterministic Policy Gradient)
- **TD3** (Twin Delayed DDPG)
- **SAC** (Soft Actor-Critic)

**왜 DQN은 Atari 게임에 잘 작동하는가?**

Atari 게임들은 **이산 행동 공간**을 가지고 있기 때문입니다:
- Breakout: {불발사, 왼쪽, 오른쪽} → 3개
- BankHeist: 18개의 조이스틱 조합
- 모두 유한하고 작은 수의 행동

---

### III. One of the main challenges in DQN is the moving target, which happens when the agent estimates Q-values and target value using the same neural network. To avoid this, we can use the fixed target network within an inner loop.

**정답: True (참)**

**상세 해설:**

이것은 DQN의 가장 중요한 기술적 혁신 중 하나입니다.

**Moving Target 문제:**

일반적인 supervised learning에서는:
- **입력 (X)**: 고정된 데이터
- **타겟 (Y)**: 고정된 레이블

하지만 Q-learning에서는:

$$\text{Loss} = \mathbb{E}[(r + \gamma \max_{a'} Q_\theta(s', a') - Q_\theta(s, a))^2]$$

여기서 문제는:
1. **예측값**: $Q_\theta(s, a)$ - 파라미터 $\theta$로 계산
2. **타겟값**: $r + \gamma \max_{a'} Q_\theta(s', a')$ - **같은 파라미터 $\theta$**로 계산

매 업데이트마다 $\theta$가 변하면:
- 예측값이 변함
- **타겟값도 같이 변함** ← 이것이 문제!

**비유로 이해하기:**

활을 쏘는데 과녁이 계속 움직인다고 상상해보세요:
- 과녁을 향해 화살을 쏘면
- 과녁이 움직여서
- 다시 조준하면
- 또 과녁이 움직임
- → **수렴이 어렵거나 불안정**

**Target Network 해결책:**

DQN은 **두 개의 네트워크**를 사용합니다:

1. **Online Network (Q-network)**: $Q_\theta$
   - 매 스텝 업데이트
   - 행동 선택에 사용
   - 예측값 계산

2. **Target Network**: $Q_{\theta^-}$
   - **일정 주기(예: 1000 steps)마다만 업데이트**
   - 타겟값 계산에만 사용
   - $\theta^- \leftarrow \theta$ (복사)

**업데이트 과정:**

```python
# 타겟 계산 (target network 사용)
with torch.no_grad():  # gradient 계산 안 함
    next_q_values = target_critic(next_obs)  # θ⁻ 사용
    next_q = torch.max(next_q_values, dim=1)
    target = reward + gamma * (1 - done) * next_q

# 예측 (online network 사용)
q_values = critic(obs)  # θ 사용
q_value = q_values.gather(1, action)

# Loss 계산 및 online network만 업데이트
loss = MSELoss(q_value, target)
loss.backward()  # θ만 업데이트, θ⁻는 고정
optimizer.step()

# 일정 주기마다 target network 업데이트
if step % target_update_period == 0:
    target_critic.load_state_dict(critic.state_dict())  # θ⁻ ← θ
```

**왜 이것이 작동하는가?**

1. **타겟 안정성**: 타겟값이 일정 기간 동안 고정되어 있음
2. **학습 안정성**: 네트워크가 "움직이는 과녁"을 쫓지 않아도 됨
3. **수렴 보장**: 타겟이 천천히 변하므로 점진적 개선 가능

**Inner Loop에서의 의미:**

"within an inner loop"는 다음을 의미합니다:
- **Outer loop**: target network 업데이트 (예: 1000 steps마다)
- **Inner loop**: online network 업데이트 (매 step)

Inner loop 동안 target network는 **고정**되어 있어 안정적인 타겟을 제공합니다.

---

### IV. We often use epsilon scheduling to encourage more exploration over time.

**정답: False (거짓)**

**상세 해설:**

이 문제는 exploration-exploitation trade-off와 epsilon scheduling의 정확한 이해를 묻습니다.

**Epsilon-Greedy 정책:**

$$
a = \begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q(s, a) & \text{with probability } 1 - \epsilon
\end{cases}
$$

- **$\epsilon$ 높음** (예: 0.9): 90% 확률로 랜덤 → **많은 exploration**
- **$\epsilon$ 낮음** (예: 0.02): 2% 확률로 랜덤 → **적은 exploration**

**Epsilon Scheduling의 실제 방향:**

문제의 진술과 반대로, 우리는 시간이 지남에 따라 **exploration을 줄입니다**:

$$\epsilon: 1.0 \rightarrow 0.02$$

**왜 exploration을 줄이는가?**

강화학습은 다음 두 단계로 진행됩니다:

1. **초기 (Early Training)**:
   - Q-function이 부정확함
   - 환경에 대해 모름
   - **많은 exploration 필요** → $\epsilon = 1.0$ (거의 랜덤)
   - 목적: 다양한 상태-행동 경험 수집

2. **후기 (Late Training)**:
   - Q-function이 정확해짐
   - 좋은 행동을 학습함
   - **적은 exploration 필요** → $\epsilon = 0.02$ (대부분 greedy)
   - 목적: 학습한 정책 활용 (exploitation)

**코드에서의 구현:**

```python
# gcb6206/env_configs/dqn_basic_config.py
exploration_schedule = PiecewiseSchedule(
    [
        (0, 1.0),                    # 시작: ε = 1.0 (100% 랜덤)
        (total_steps * 0.1, 0.02),   # 10% 지점: ε = 0.02 (2% 랜덤)
    ],
    outside_value=0.02,              # 이후: ε = 0.02 유지
)
```

이것은 다음을 의미합니다:
- **Step 0**: $\epsilon = 1.0$ (전체 exploration)
- **Step 0 ~ 30,000**: $\epsilon$ 선형 감소 $1.0 \rightarrow 0.02$
- **Step 30,000+**: $\epsilon = 0.02$ (거의 exploitation)

**시각화:**

```
ε
1.0 |●
    |  ╲
    |    ╲
    |      ╲
0.02|        ●━━━━━━━━━━━━━
    └─────────────────────→ time
    0   30k            300k steps

    많은        →      적은
  exploration     exploration
```

**다른 Exploration 전략:**

Epsilon-greedy 외에도 다양한 방법들이 있습니다:

1. **Boltzmann Exploration**:
   $$P(a|s) \propto \exp(Q(s,a) / \tau)$$
   - $\tau$ (temperature)를 감소시킴

2. **Upper Confidence Bound (UCB)**:
   - 불확실성이 높은 행동 선호

3. **Noise-based** (DDPG, TD3):
   - 행동에 노이즈 추가
   - 노이즈 크기를 감소

모두 공통점은: **시간이 지남에 따라 exploration 감소**

**정답 요약:**

문제 진술은 "encourage more exploration over time"(시간이 지남에 따라 더 많은 exploration)이라고 했지만, 실제로는 **"encourage less exploration over time"**이 맞습니다. 따라서 답은 **False**입니다.

---
