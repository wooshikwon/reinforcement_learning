# Lecture 07: Off-Policy Actor-Critic Methods

## 목차

- [강의 개요](#강의-개요)
- [복습: Fitted Q-iteration (Q-learning)](#복습-fitted-q-iteration-q-learning)
- [DDPG (Deep Deterministic Policy Gradient)](#ddpg-deep-deterministic-policy-gradient)
  - [연속 행동 공간에서의 DQN 문제](#연속-행동-공간에서의-dqn-문제)
  - [연속 행동 공간에서의 탐욕 정책 찾기](#연속-행동-공간에서의-탐욕-정책-찾기)
  - [DDPG의 수학적 원리](#ddpg의-수학적-원리)
  - [소프트 타겟 업데이트](#소프트-타겟-업데이트)
  - [DDPG 알고리즘 요약](#ddpg-알고리즘-요약)
  - [DDPG 의사 코드](#ddpg-의사-코드)
  - [DDPG 요약](#ddpg-요약)
- [TD3 (Twin Delayed Deep Deterministic Policy Gradient)](#td3-twin-delayed-deep-deterministic-policy-gradient)
  - [DDPG의 Q-value 과대평가 문제](#ddpg의-q-value-과대평가-문제)
  - [Q-value 과대평가 실험 결과](#q-value-과대평가-실험-결과)
  - [TD3 의사 코드](#td3-의사-코드)
  - [TD3 성능 비교](#td3-성능-비교)
  - [TD3 요약](#td3-요약)
- [SAC (Soft Actor-Critic)](#sac-soft-actor-critic)
  - [확률론적 정책의 필요성](#확률론적-정책의-필요성)
  - [최대 엔트로피 강화학습](#최대-엔트로피-강화학습)
  - [왜 더 무작위적인 정책이 선호될까?](#왜-더-무작위적인-정책이-선호될까)
  - [소프트 Q-가치](#소프트-q-가치)
  - [소프트 정책 반복](#소프트-정책-반복)
  - [α 자동 튜닝](#α-자동-튜닝)
  - [SAC 의사 코드](#sac-의사-코드)
  - [SAC 요약](#sac-요약)
  - [TD3 vs SAC](#td3-vs-sac)
- [강화학습 벤치마크](#강화학습-벤치마크)
  - [RL 알고리즘 비교 방법](#rl-알고리즘-비교-방법)
  - [유명한 RL 벤치마크](#유명한-rl-벤치마크)
  - [OpenAI Gym & DeepMind Control Suite](#openai-gym--deepmind-control-suite)
  - [Atari](#atari)
  - [고급 로봇 벤치마크](#고급-로봇-벤치마크)
- [요약](#요약)

---

## 강의 개요

오늘은 지난 시간에 배운 DQN과 같은 가치 기반 방법론(Value-based methods)과 정책 경사 하강법(Policy Gradient)을 결합한, 소위 **액터-크리틱(Actor-Critic)** 방법론의 심화 과정을 다룹니다.

특히, 오늘 다룰 알고리즘들은 **연속적인 행동 공간(continuous action spaces)**에서 매우 강력한 성능을 보이는 것들입니다.

**주요 학습 내용:**

1. **DDPG (Deep Deterministic Policy Gradient)**: DQN을 연속 행동 공간으로 확장한, 아주 고전적이면서도 중요한 오프-폴리시(off-policy) 액터-크리틱 알고리즘입니다.
2. **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**: DDPG가 가진 고질적인 문제, 특히 Q-value를 과대평가(overestimation)하는 문제를 해결하여 성능을 획기적으로 개선한 알고리즘입니다.
3. **SAC (Soft Actor-Critic)**: 앞선 두 알고리즘이 결정론적(deterministic) 정책을 사용한 것과 달리, 확률적(stochastic) 정책을 사용하며 '최대 엔트로피(maximum entropy)'라는 독특한 개념을 도입하여 탐험(exploration)과 안정성을 모두 잡으려 한, 현재까지도 매우 널리 쓰이는 알고리즘입니다.
4. **RL 벤치마크**: 이런 알고리즘들의 성능을 비교하고 검증하는 데 사용되는 환경들에 대해 간단히 소개합니다.

---

## 복습: Fitted Q-iteration (Q-learning)

### 🧠 Fitted Q-iteration 알고리즘

DDPG를 배우기 전에 그 근간이 되는 Q-learning, 특히 딥러닝과 결합된 'Fitted Q-iteration' 알고리즘을 다시 한번 짚고 넘어가겠습니다. DQN을 생각하시면 됩니다.

Q-learning의 핵심 아이디어는 현재 상태 $s$에서 행동 $a$를 했을 때의 가치, 즉 $Q(s, a)$를 학습하는 것입니다.

**알고리즘 단계:**

1. **데이터 수집 (Roll-out policy)**: 먼저, (어떤 정책이든) 현재 정책을 따라서 환경과 상호작용하며 $(s, a, s', r)$ 튜플들을 수집합니다.

2. **타겟 설정 (Estimate return)**: 수집한 데이터로 Q-function을 업데이트할 '정답지(target)'를 만듭니다.

$$y_i = r_i + \max_{a'} Q_{\phi}(s_i', a_i')$$

이 수식의 의미는 "현재 받은 보상 $r_i$에다가, 다음 상태 $s_i'$에서 할 수 있는 *모든* 행동 $a'$ 중에서 Q-value를 *최대*로 만드는 값을 더한 것"입니다. 이것이 바로 우리가 기대하는 '최적의' 총 보상입니다.

3. **모델 학습 (Fit model)**: 이제 우리의 Q-network $Q_{\phi}(s_i, a_i)$가 이 타겟 $y_i$와 같아지도록, 즉 $Q_{\phi}(s_i, a_i) \approx y_i$가 되도록 모델 파라미터 $\phi$를 학습시킵니다. 보통은 MSE(Mean Squared Error) 손실 함수를 사용하죠.

### 암시적 정책 개선 (Implicit Policy Improvement)

**중요한 지점**은 바로 '정책 개선(Improve policy)' 부분입니다. Q-learning에서 정책 $\pi(s)$는 **암시적(implicitly)**으로 정의됩니다.

$$\pi(s) = \arg \max_a Q_{\phi}(s, a)$$

즉, "학습된 $Q_{\phi}$를 보고, 현재 상태 $s$에서 Q-value를 최대로 만드는 행동 $a$를 선택하는 것"이 곧 정책입니다.

---

## DDPG (Deep Deterministic Policy Gradient)

### 🚀 개요

이제 오늘의 첫 번째 주제인 **Deep Deterministic Policy Gradient (DDPG)** 입니다. 이름이 길지만, 'Deep' (딥러닝 사용), 'Deterministic Policy' (결정론적 정책 사용), 'Policy Gradient' (정책 경사 하강법 사용) 이렇게 끊어서 이해하면 됩니다.

---

### 🤔 연속 행동 공간에서의 DQN 문제

앞선 Fitted Q-iteration에서 Q-learning의 문제점을 눈치채셨나요?

바로 $\max_{a'} Q_{\phi}(s', a')$ 이 부분입니다.

**이산 행동 공간 (Discrete actions)**:
- 예: DQN이 쓰인 아타리 게임
- 행동이 '왼쪽', '오른쪽', '점프' 같이 몇 개로 정해져 있습니다
- $\max$ 연산? 그냥 모든 행동에 대한 Q-value를 계산해서 제일 큰 값을 고르면 됩니다. 간단하죠.

**연속 행동 공간 (Continuous actions)**:
- 예: 로봇 팔의 관절 각도, 자동차의 핸들 각도
- 행동 $a'$가 실숫값입니다. $-1.0$과 $+1.0$ 사이의 모든 실수처럼요.

이 $\max_{a'} Q_{\phi}(s', a')$를 어떻게 계산할까요?

**(1)** 가능한 많은 행동을 샘플링해서 비교하거나
**(2)** $Q_{\phi}$를 $a'$에 대해 미분해서 0이 되는 지점을 찾는 최적화(optimization) 과정을 거쳐야 합니다.

**문제점**: 이 두 가지 방법 모두 **매우 느리고 부정확합니다.** 매 스텝마다 이 최적화 문제를 푸는 것은 거의 불가능에 가깝죠.

---

### 💡 연속 행동 공간에서의 탐욕 정책 찾기

DDPG는 이 $\arg \max_a Q_{\phi}(s, a)$ 문제를 아주 현명하게 해결합니다.

**핵심 아이디어:**

> "매번 $\max$ 연산을 힘들게 최적화해서 *찾지 말고* (finding),
> 이 $\arg \max$ 값을 *예측하는* 별도의 함수를 하나 *학습시키자* (learning)!"

즉, 우리는 $Q_{\phi}(s, a)$를 최대화하는 $a$를 근사하는 **결정론적 정책(deterministic policy)** 함수 $\pi_{\theta}(s)$를 만듭니다.

- 이 $\pi_{\theta}$가 바로 **액터(Actor)**입니다. $s$를 입력받아 $a$를 바로 출력하죠.
- 기존의 $Q_{\phi}$는 $s$와 $a$를 입력받아 가치를 평가하므로 **크리틱(Critic)**이라고 부릅니다.

그래서 $\max_a Q_{\phi}(s, a)$라는 어려운 계산 대신, $\pi_{\theta}(s)$가 예측한 행동 $a$를 Q-함수에 넣은 값, 즉 $Q_{\phi}(s, \pi_{\theta}(s))$를 사용합니다.

**이 방식의 장점:**

1. $Q_{\phi}$가 조금 변하면, $\pi_{\theta}$도 조금만 변하게 되어 학습이 안정적입니다.
2. $a$가 연속적이므로 $Q_{\phi}(s, a)$가 $a$에 대해 미분 가능합니다.

**직관적 이해:**

$a$축에 대해 $Q(s, a)$의 값이 볼록한 곡선을 그리고 있다고 생각해보세요. 정책 $\pi_{\theta}(s)$가 $Q$값이 최대가 아닌 지점을 가리키고 있을 때, 우리는 이 정책 $\pi$를 $Q$값이 더 높은 지점으로 이동시키고 싶습니다.

어떻게 이동시킬까요? 바로 $\theta$에 대한 경사 상승법(gradient ascent)을 사용합니다.

우리의 목표는 $\mathbb{E}_{s \sim \mathcal{B}}[Q_{\phi}(s, \pi_{\theta}(s))]$를 최대화하는 $\theta$를 찾는 것입니다. 즉, 액터 $\pi_{\theta}$가 크리틱 $Q_{\phi}$로부터 "잘했다"는 높은 점수를 받도록 학습시키는 것이죠.

---

### 📈 DDPG의 수학적 원리

그럼 액터 $\pi_{\theta}$를 업데이트하기 위한 경사(gradient)를 어떻게 계산할까요?

우리의 목표 함수(Objective function) $J(\theta)$는 액터가 만든 행동 $\pi_{\theta}(s)$를 크리틱 $Q_{\phi}$가 평가한 값의 기댓값입니다.

$$J(\theta) = \mathbb{E}_{s \sim \mathcal{B}}[Q_{\phi}(s, \pi_{\theta}(s))]$$

우리는 $J(\theta)$를 $\theta$에 대해 미분하고 싶습니다.

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{s \sim \mathcal{B}}[\nabla_{\theta} Q_{\phi}(s, a) \vert_{a=\pi_{\theta}(s)}]$$

여기서 **연쇄 법칙(Chain Rule)**이 사용됩니다. $Q_{\phi}$는 $\theta$에 직접적으로 의존하는 것이 아니라, $\theta$로 만들어진 $a = \pi_{\theta}(s)$를 통해 간접적으로 의존합니다.

따라서 $\theta$에 대한 $Q$의 미분은 ($a$에 대한 $Q$의 미분) $\times$ ($\theta$에 대한 $a$의 미분) 이 됩니다.

$$\nabla_{\theta} Q_{\phi}(s, \pi_{\theta}(s)) = \nabla_a Q_{\phi}(s, a) \vert_{a=\pi_{\theta}(s)} \cdot \nabla_{\theta} \pi_{\theta}(s)$$

이것이 바로 **결정론적 정책 경사(Deterministic Policy Gradient, DPG)** 정리입니다.

**구성 요소:**

- $\nabla_{\theta} \pi_{\theta}(s)$: 액터 네트워크를 $\theta$로 미분한 값. (액터가 $\theta$의 변화에 따라 행동을 얼마나 바꾸는지)
- $\nabla_a Q_{\phi}(s, a)$: 크리틱 네트워크를 $a$로 미분한 값. (크리틱이 $a$의 변화에 따라 가치를 얼마나 바꾸는지)

**직관적 해석:**

> "크리틱이 '이 방향($\nabla_a Q$)으로 $a$를 바꾸면 Q-value가 높아져!'라고 알려주면, 액터는 '알았어, 내 파라미터 $\theta$를 $\nabla_{\theta} \pi_{\theta}$ 만큼 조절해서 $a$를 그 방향으로 바꿀게!' "

**Policy Gradient와의 차이:**

"Policy Gradient"라는 이름 때문에 REINFORCE 같은 고분산(high variance) 알고리즘을 떠올릴 수 있지만, DDPG는 다릅니다.

1. **결정론적 행동(deterministic actions)**을 사용하고,
2. **크리틱으로부터 직접(directly from critic)** 경사를 받기 때문에 분산이 훨씬 적고 안정적입니다.

---

### 🎯 소프트 타겟 업데이트

DQN에서 학습을 안정화시킨 핵심 기술 중 하나가 '타겟 네트워크(target network)'였습니다.

**DQN의 하드 타겟 업데이트 (Hard Target Update):**

DQN은 일정 스텝($L$ steps)마다 타겟 네트워크 $\phi^-$의 가중치를 현재 네트워크 $\phi$의 가중치로 통째로 복사($\phi^- \leftarrow \phi$)했죠.

**DDPG의 소프트 타겟 업데이트 (Soft Target Update):**

DDPG는 이보다 더 부드러운 **'소프트 타겟 업데이트(soft target update)'**를 사용합니다. 이는 **Polyak 평균 (Polyak average)** 또는 **지수 이동 평균 (exponential moving average)**이라고도 불립니다.

$$\phi^- \leftarrow \rho \phi^- + (1 - \rho) \phi$$

여기서 $\rho$는 매우 1에 가까운 값 (예: 0.99) 입니다.

매 스텝마다 타겟 네트워크 $\phi^-$는:
1. 99%의 기존 가중치를 유지하고,
2. 1%의 새로운 현재 네트워크 $\phi$의 가중치를 받아들입니다.

이렇게 하면 타겟 $y$값이 급격하게 변하는 것을 막아주어 학습 과정 전체가 매우 안정적으로 수렴하게 됩니다.

**중요**: DDPG는 이 소프트 타겟 업데이트를 **크리틱($\phi$)뿐만 아니라 액터($\theta$)에도 모두 적용**합니다. (즉, $\theta^- \leftarrow \rho \theta^- + (1 - \rho) \theta$)

---

### 🧩 DDPG 알고리즘 요약

이제 DDPG의 전체 그림을 요약해봅시다. DDPG는 액터와 크리틱, 두 개의 네트워크가 상호작용하며 학습합니다.

**1. 크리틱($Q_{\phi}$) 업데이트 (Fit model):**

DQN과 매우 유사합니다.

**타겟:**

$$y_i = r_i + Q_{\phi^-}(s_i', \pi_{\theta^-}(s_i'))$$

여기서 중요한 점! 타겟을 계산할 때 현재 네트워크($\phi, \theta$)가 아니라 **타겟 네트워크($\phi^-, \theta^-$)**를 사용합니다. 즉, "미래의 가치($Q_{\phi^-}$)"는 "미래의 정책($\pi_{\theta^-}$)"을 기반으로 계산해야 합니다.

$Q_{\phi}(s_i, a_i)$가 이 타겟 $y_i$에 가까워지도록 $\phi$를 업데이트합니다 (MSE 손실 최소화).

**2. 액터($\pi_{\theta}$) 업데이트 (Improve policy):**

앞서 유도한 DPG를 사용합니다.

목표는 $\mathbb{E}_{s \sim \mathcal{B}}[Q_{\phi}(s, \pi_{\theta}(s))]$를 최대화하는 것입니다.

즉, 현재 크리틱 $Q_{\phi}$가 액터의 행동 $\pi_{\theta}(s)$에 대해 "좋다"고 말하는 방향으로 액터의 파라미터 $\theta$를 경사 *상승*(ascent)시킵니다.

$$\nabla_{\theta}J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N}[\nabla_a Q_{\phi}(s_i, a) \vert_{a=\pi_{\theta}(s_i)} \nabla_{\theta} \pi_{\theta}(s_i)]$$

**3. 타겟 네트워크 업데이트:**

마지막으로, 액터와 크리틱의 타겟 네트워크들을 소프트 업데이트합니다.

---

### 💻 DDPG 의사 코드

알고리즘 의사 코드를 한 줄씩 뜯어보며 DDPG의 작동 방식을 완벽하게 이해해봅시다.

**의사 코드:**

```pseudo
1:  Initialize actor μ_θ and critic Q_φ
2:  Initialize target networks μ_θ_targ ← θ, Q_φ_targ ← φ
    Initialize replay buffer D ← ∅

3:  repeat until convergence:
4:      Observe state s
5:      a ← clip(μ_θ(s) + ε, a_Low, a_High)
6:      Execute a in environment
7:      Observe s', r, d
8:      D ← D ∪ {(s, a, r, s', d)}
9:      if s' is terminal then reset environment

10:     if it's time to update then
11:         Sample minibatch B from D
12:         y(r, s', d) ← r + γ(1 - d) Q_φ_targ(s', μ_θ_targ(s'))
13:         Update critic: ∇_φ (1/|B|) Σ (Q_φ(s, a) - y)²
14:         Update actor: ∇_θ (1/|B|) Σ Q_φ(s, μ_θ(s))
15:         Update target networks:
            φ_targ ← ρ φ_targ + (1 - ρ) φ
            θ_targ ← ρ θ_targ + (1 - ρ) θ
```

**알고리즘 설명:**

**1-2행 (초기화):**
- 액터 정책 $\mu_{\theta}$와 크리틱 Q-함수 $Q_{\phi}$를 초기화합니다.
- 타겟 네트워크 $\mu_{\theta_{targ}}$와 $Q_{\phi_{targ}}$를 만들고, 현재 네트워크와 동일하게 가중치를 복사합니다 ($\theta_{targ} \leftarrow \theta$, $\phi_{targ} \leftarrow \phi$).
- 리플레이 버퍼 $\mathcal{D}$를 비웁니다.

**3-9행 (데이터 수집 루프):**
- `repeat ... until convergence`: 학습이 끝날 때까지 반복합니다.
- **5행**: 행동을 선택합니다. $a = clip(\mu_{\theta}(s) + \epsilon, a_{Low}, a_{High})$
  - $\mu_{\theta}(s)$: 현재 액터가 결정론적으로 제안하는 행동입니다.
  - $+\epsilon$: **탐험(Exploration)**을 위해 노이즈 $\epsilon$ (예: $\mathcal{N}(0, \sigma)$ 정규분포 노이즈)를 추가합니다. DDPG는 결정론적 정책이라 스스로 탐험을 못하기 때문에, 행동에 직접 노이즈를 줍니다.
  - $clip(...)$: 행동이 환경의 허용 범위(예: -1.0 ~ 1.0)를 벗어나지 않게 잘라줍니다.
- **6-7행**: 행동 $a$를 환경에서 실행하고, 다음 상태 $s'$, 보상 $r$, 종료 여부 $d$를 관찰합니다.
- **8행**: 이 $(s, a, r, s', d)$ 튜플을 리플레이 버퍼 $\mathcal{D}$에 저장합니다. (Off-policy의 핵심이죠)
- **9행**: $s'$가 종료 상태이면 환경을 리셋합니다.

**10-15행 (업데이트 루프):**
- `if it's time to update then`: (예: 매 스텝마다 또는 N 스텝마다)
- **11행**: 리플레이 버퍼 $\mathcal{D}$에서 미니배치 $B$를 랜덤하게 샘플링합니다.
- **12행 (크리틱 타겟 계산)**: $y(r, s', d) = r + \gamma(1 - d) Q_{\phi_{targ}}(s', \mu_{\theta_{targ}}(s'))$
  - 이것이 Q-learning의 타겟입니다.
  - 중요한 것은 $Q$와 $\mu$ 모두 **타겟 네트워크($\phi_{targ}, \theta_{targ}$)**를 사용한다는 것입니다!
- **13행 (크리틱 업데이트)**: 크리틱 $Q_{\phi}$를 업데이트합니다.
  - $\nabla_{\phi} \frac{1}{|B|} \sum (Q_{\phi}(s, a) - y)^2$
  - 배치에 포함된 $(s, a)$에 대한 현재 Q-value $Q_{\phi}(s, a)$와 우리가 계산한 타겟 $y$ 사이의 MSE 손실을 최소화하는 방향으로 $\phi$를 경사 하강(gradient descent)합니다.
- **14행 (액터 업데이트)**: 액터 $\mu_{\theta}$를 업데이트합니다.
  - $\nabla_{\theta} \frac{1}{|B|} \sum Q_{\phi}(s, \mu_{\theta}(s))$
  - 배치에 포함된 $s$에 대해, 액터가 제안한 행동 $\mu_{\theta}(s)$를 크리틱 $Q_{\phi}$가 평가한 값 $Q_{\phi}(s, \mu_{\theta}(s))$를 **최대화**하는 방향으로 $\theta$를 경사 *상승*(gradient ascent)합니다.
- **15행 (타겟 네트워크 업데이트)**: 소프트 타겟 업데이트를 수행합니다.
  - $\phi_{targ} \leftarrow \rho \phi_{targ} + (1 - \rho) \phi$
  - $\theta_{targ} \leftarrow \rho \theta_{targ} + (1 - \rho) \theta$

---

### 📝 DDPG 요약

**DDPG의 특징:**

- **오프-폴리시(Off-policy), 액터-크리틱(Actor-Critic)** RL 알고리즘입니다. (리플레이 버퍼를 사용하므로 오프-폴리시, 정책 네트워크와 가치 네트워크를 모두 사용하므로 액터-크리틱)
- DQN을 **연속 행동 공간(continuous action spaces)** 버전으로 만든 것입니다. ($\max_a Q$ 문제를 $\pi_{\theta}$로 근사하여 해결)
- 하지만 DDPG는 **하이퍼파라미터에 매우 민감합니다.** (탐험 노이즈 $\epsilon$, 타겟 업데이트 비율 $\rho$, 학습률(learning rate) 등) 튜닝하기가 꽤 까다롭다는 단점이 있습니다.

---

## TD3 (Twin Delayed Deep Deterministic Policy Gradient)

### 🛠️ 개요

DDPG는 강력하지만 불안정하고 하이퍼파라미터에 민감하다고 했죠. 특히 DDPG는 Q-learning에 기반하기 때문에, DQN이 가졌던 **Q-value 과대평가(overestimation) 문제**를 그대로 물려받았습니다.

**TD3**는 이 DDPG의 문제를 해결하기 위해 세 가지 핵심 기술을 도입한 알고리즘입니다.

---

### 💥 DDPG의 Q-value 과대평가 문제

DDPG (및 Q-learning)는 왜 Q-value를 과대평가할까요?

#### 문제 #1: 타겟 값의 과대평가

타겟 $y_i = r_i + Q_{\phi^{-}}(s_i', \pi_{\theta^{-}}(s_i'))$를 계산할 때, $Q_{\phi^-}$ 네트워크에 근사 오차(approximation error)가 있을 수 있습니다.

만약 $Q_{\phi^-}$가 특정 $(s', a')$ 쌍에 대해 실수로 높은 값을 출력하면, $\pi_{\theta^-}$는 그 잘못된 높은 값을 향해 빠르게 학습합니다. 이 잘못된 타겟값이 벨만 방정식을 통해 계속 전파되면서 Q-value가 실제보다 훨씬 높게 부풀려집니다.

**해결책 #1: Clipped Double Q-learning**

Double DQN의 아이디어를 차용합니다.

- 크리틱(Q-function)을 **두 개($Q_{\phi_1}, Q_{\phi_2}$)** 만듭니다.
- 타겟을 계산할 때, 두 타겟 크리틱 $Q_{\phi_1^-}$와 $Q_{\phi_2^-}$가 예측한 값 중 **더 작은(minimum) 값**을 사용합니다.

$$y_i = r + \gamma \min_{i=1, 2} Q_{\phi_i^-}(s', \pi_{\theta^-}(s'))$$

이렇게 하면, 둘 중 하나가 실수로 과대평가하더라도 다른 하나가 제대로 평가한다면 그 '실수'가 전파되는 것을 막아줍니다. (그래서 'Clipped' Double Q-learning)

#### 문제 #2: 결정론적 정책의 과적합

결정론적 정책 $\pi_{\theta}$는 Q-function의 매우 좁고 뾰족한 피크(peak)에 쉽게 과적합(overfit)될 수 있습니다. 만약 이 피크가 $Q$의 근사 오차로 인해 생긴 가짜 피크라면, 정책은 완전히 잘못된 방향으로 학습하게 됩니다.

**해결책 #2a: 지연된 정책 업데이트 (Delayed policy updates)**

- 액터($\pi_{\theta}$)를 크리틱($Q_{\phi}$)보다 **더 느리게(less frequently)** 업데이트합니다.
- 예를 들어, 크리틱을 2번 업데이트할 때 액터는 1번만 업데이트합니다.
- 이는 Q-value가 (가짜 피크가 아닌) 좀 더 안정적이고 정확한 값으로 수렴할 시간을 준 뒤에, 액터가 그 안정된 Q-value를 기반으로 업데이트하게 만들기 위함입니다.

**해결책 #2b: 타겟 정책 스무딩 (Smooth target policy)**

- 타겟을 계산할 때, 타겟 액터 $\pi_{\theta^-}$가 만든 행동에 **노이즈**를 추가합니다.

$$a' = \pi_{\theta^-}(s') + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma)$$

- 이는 $Q$ 함수의 값들을 주변으로 스무딩(smoothing)하는 효과를 줍니다.
- $Q$ 랜드스케이프가 부드러워지면, 정책 $\pi$가 좁고 잘못된 피크에 과적합되기 어려워집니다.

---

### 📊 Q-value 과대평가 실험 결과

이 그래프가 TD3의 핵심 아이디어(Clipped Double Q-learning)가 왜 중요한지 명확하게 보여줍니다.

**그래프 분석:**

- 가로축은 학습 스텝(Time steps), 세로축은 가치(Average Value)입니다.
- **주황색 (DDPG)**:
  - 실선(DDPG)은 알고리즘이 *추정한* Q-value입니다.
  - 점선(True DDPG)은 *실제* 환경에서 받은 리턴(보상)입니다.
  - 학습 초반부터 추정치(실선)가 실제(점선)보다 훨씬 높게 치솟는 것이 보이죠? 이것이 **과대평가(overestimation bias)**입니다.
- **파란색 (CDQ - Clipped Double Q-learning)**:
  - 실선(CDQ)과 점선(True CDQ)이 거의 붙어서 올라갑니다.
  - 이는 알고리즘이 추정한 Q-value가 실제 리턴과 매우 유사하다는 뜻입니다.

**결론**: Clipped Double Q-learning이 과대평가 편향을 성공적으로 해결했다는 것을 시각적으로 확인할 수 있습니다.

---

### 💻 TD3 의사 코드

TD3의 의사 코드를 DDPG와 비교하며 살펴보겠습니다.

**의사 코드:**

```pseudo
1:  Initialize actor μ_θ and Twin critics Q_φ1, Q_φ2
    Initialize target networks: μ_θ_targ, Q_φ_targ,1, Q_φ_targ,2
    Initialize replay buffer D ← ∅

... (lines 3-11 same as DDPG)

12: a'(s') ← clip(μ_θ_targ(s') + clip(ε, -c, c), a_Low, a_High)
13: y(...) ← r + γ(1 - d) min_{i=1,2} Q_φ_targ,i(s', a'(s'))
14:
15: Update both critics: ∇_φi (1/|B|) Σ (Q_φi(s, a) - y)²  for i=1,2
16:
17: if j mod policy_delay == 0 then
18:     Update actor: ∇_θ (1/|B|) Σ Q_φ1(s, μ_θ(s))
19:     Update target networks (soft update)
```

**알고리즘 설명:**

**1행 (초기화):**
- DDPG와 달리, Q-function($\phi_1, \phi_2$)과 타겟 Q-function($\phi_{targ, 1}, \phi_{targ, 2}$)이 **두 쌍(Twin)**으로 시작합니다.

**12행 (타겟 행동 계산):**
- $a'(s') = clip(\mu_{\theta_{targ}}(s') + clip(\epsilon, -c, c), a_{Low}, a_{High})$
- 타겟 액터 $\mu_{\theta_{targ}}$가 만든 행동에 노이즈 $\epsilon$을 더합니다.
- 이것이 바로 **(Solution 2b) 타겟 정책 스무딩**입니다.

**13행 (크리틱 타겟 계산):**
- $y(...) = r + \gamma(1 - d) \min_{i=1, 2} Q_{\phi_{targ, i}}(s', a'(s'))$
- 두 타겟 Q-네트워크 중 **최솟값(min)**을 사용합니다.
- 이것이 바로 **(Solution 1) Clipped Double Q-learning**입니다.

**15행 (크리틱 업데이트):**
- 두 개의 Q-function $Q_{\phi_1}, Q_{\phi_2}$ **모두** 동일한 타겟 $y$를 향해 업데이트합니다.

**17-19행 (액터 및 타겟 업데이트):**
- `if j mod policy_delay == 0 then`
- 크리틱은 매 `j` 스텝마다 업데이트되지만, 액터와 타겟 네트워크는 `policy_delay` 스텝(예: 2 스텝)마다 한 번씩만 업데이트됩니다.
- 이것이 바로 **(Solution 2a) 지연된 정책 업데이트**입니다.
- 액터를 업데이트할 때는 두 크리틱 중 하나(여기서는 $Q_{\phi_1}$)만 사용합니다.

---

### 🏆 TD3 성능 비교

이 그래프들은 TD3의 성능을 다른 알고리즘들과 비교한 것입니다.

**그래프 분석:**

- 파란색 선이 **TD3**입니다.
- 주황색 선이 **DDPG**입니다.
- (a) HalfCheetah, (b) Hopper, (c) Walker2d 등 다양한 MuJoCo 벤치마크 환경에서,
- TD3(파란색)가 DDPG(주황색)보다 훨씬 더 높고 안정적인 리턴(보상)을 달성하는 것을 볼 수 있습니다.
- PPO, TRPO 같은 (당시 SOTA였던) 온-폴리시 알고리즘들과 비교해도 대등하거나 더 나은 성능을 보여줍니다.

---

### 📝 TD3 요약

**TD3의 특징:**

- TD3는 **DDPG를 개선한(Improved DDPG)** 알고리즘입니다.
- 학습 안정성을 높이기 위해 3가지 핵심 기술을 사용합니다:
  1. **Clipped Double Q-learning** (Q-value 과대평가 방지)
  2. **지연된 정책 업데이트 (Delayed policy updates)** (안정된 Q-value 기반 학습)
  3. **타겟 정책 스무딩 (Target policy smoothing)** (Q-function 과적합 방지)
- 이러한 개선 덕분에 **실제 환경에서 매우 잘 작동하며(Works well in practice!)** 당시 연속 행동 공간을 위한 오프-폴리시 RL 알고리즘의 **SOTA(state-of-the-art)**가 되었습니다.

---

## SAC (Soft Actor-Critic)

### 💡 개요

이제 DDPG와 TD3와는 약간 다른 철학을 가진 알고리즘, **Soft Actor-Critic (SAC)**에 대해 알아보겠습니다. 이름 그대로 '소프트(Soft)'한 액터-크리틱입니다.

---

### 🎲 확률론적 정책의 필요성

DDPG와 TD3는 모두 **결정론적 정책(deterministic policy)** $\pi_{\theta}(s)$를 사용했습니다. 즉, 상태 $s$가 주어지면 *항상* 똑같은 행동 $a$를 출력합니다. (탐험을 위해 인위적으로 노이즈를 더했죠.)

이번에는 **확률론적 정책(stochastic policy)** $\pi_{\theta}(a|s)$를 사용해봅시다.

예를 들어, 연속 행동 공간에서는 $\pi_{\theta}$가 평균 $\mu(s)$와 표준편차 $\sigma(s)$를 갖는 정규분포 $\mathcal{N}(\mu(s), \sigma(s))$를 출력하게 할 수 있습니다. 실제 행동 $a$는 이 분포에서 샘플링하고요.

확률론적 정책은 탐험(exploration)에 자연스러운 이점이 있습니다.

**문제점:**

일반적인 RL의 목표는 '보상(reward)의 기댓값을 최대화'하는 것입니다.

만약 우리가 $\mathbb{E}[\sum r(s, a)]$ 만을 최대화하도록 정책을 학습시킨다면, 알고리즘은 가장 보상이 높은 *하나의* 행동을 찾은 뒤, 그 행동만 확신을 가지고 선택하기 위해 표준편차 $\sigma$를 0으로 줄여버릴 것입니다.

결국, 확률론적 정책이 다시 결정론적 정책으로 수렴(collapse)해버리는 문제가 발생합니다.

SAC는 이 문제를 어떻게 해결할까요?

---

### 🎁 최대 엔트로피 강화학습

SAC의 핵심 아이디어는 바로 **"최대 엔트로피 강화학습 (Maximum Entropy RL)"**입니다.

DQN은 $\epsilon$-greedy로, DDPG는 액션 노이즈로 탐험을 했습니다. SAC는 아예 **목표 함수(Objective function)** 자체에 탐험을 장려하는 텀을 추가합니다.

**기존 RL 목표**: 보상의 합을 최대화

$$\mathbb{E}_{\tau \sim \pi_{\theta}(\tau)} [\sum_t r(s_t, a_t)]$$

**최대 엔트로피 RL 목표**: 보상의 합 + ($\alpha$ $\times$ **정책의 엔트로피**)의 합을 최대화

$$\mathbb{E}_{\tau \sim \pi_{\theta}(\tau)} [\sum_t r(s_t, a_t) + \alpha H(\pi(\cdot | s_t))]$$

여기서 **엔트로피(Entropy)** $H$는 정책의 **무작위성(randomness)**을 측정하는 지표입니다.

$$H(\pi) = \mathbb{E}[-\log \pi(a_t | s_t)]$$

정책이 특정 행동만 확신(낮은 엔트로피)하면 이 값은 낮아지고, 여러 행동을 골고루 고려(높은 엔트로피)하면 이 값은 높아집니다.

$\alpha$는 '온도 파라미터(temperature parameter)'로, 보상 최대화와 엔트로피 최대화(탐험) 사이의 균형을 조절합니다.

**핵심 개념:**

> SAC의 정책은 **"보상을 최대로 받으면서, *동시에* 가능한 한 무작위적인 행동(높은 엔트로피)을 하도록"** 학습됩니다.

이것이 SAC를 'Soft' Actor-Critic이라고 부르는 이유입니다. $\max$ 대신 'soft' $\max$를 수행하는 것과 수학적으로 유사하기 때문입니다.

---

### 🤔 왜 더 무작위적인 정책이 선호될까?

"그냥 보상만 최대로 받으면 되지, 왜 굳이 '무작위성(엔트로피)'까지 챙겨야 하나요?" 아주 좋은 질문입니다.

**직관적 예시:**

에이전트가 장애물을 피해 목표까지 가는 경로를 찾는 상황을 생각해봅시다.

**보상만 최대화하는 경우:**
- 에이전트가 장애물을 피해 목표까지 가는 *하나의* 경로를 찾았습니다.
- 이 경로가 최적이라고 판단하고 이 경로만 100% 확신을 가지고 수행할 것입니다. (즉, 엔트로피가 0에 수렴하겠죠)
- **문제**: 만약 학습이 끝난 뒤 실제 환경에서 갑자기 장애물이 나타나 이 경로를 막아버리면? 에이전트는 이 변화에 대응하지 못하고 실패할 것입니다.

**최대 엔트로피를 추구하는 경우:**
- 에이전트는 보상을 받는 경로가 하나(위쪽 경로) 있다는 것을 알면서도, $\alpha H$ 항 때문에 계속해서 다른 경로를 탐험(exploration)하려 합니다.
- 그 결과, 아래쪽으로 돌아가는 *또 다른* 경로도 발견하게 되죠.

**장점:**

1. **다양한 솔루션(different modes of solutions)**을 찾도록 유도합니다.
2. 이를 통해 예기치 않은 환경 변화에 **강건한(robust)** 정책을 학습하게 만듭니다.
3. 그리고 이 과정 자체가 매우 **효율적인 탐험(exploration)**을 장려합니다.

이는 우리가 강화학습의 근본적인 문제, 즉 '탐험(Exploration)과 활용(Exploitation)의 균형' 문제를 다루는 매우 세련된 방식입니다.

---

### 🧮 소프트 Q-가치

그러면 이 '최대 엔트로피' 목표가 우리의 가치 함수(Value function)를 어떻게 바꾸는지 수학적으로 살펴봅시다.

**기존의 벨만 방정식:**

$$Q(s_t, a_t) \leftarrow r(s_t, a_t) + \gamma \mathbb{E}[V(s_{t+1})]$$

"현재 Q값은 (즉각적인 보상 $r$) + (할인된 미래의 가치 $V(s_{t+1})$) 이다."

이때, 기존의 상태 가치 함수 $V(s)$는 무엇이었죠?

$$V(s_t) = \mathbb{E}_{a_t \sim \pi(\cdot | s_t)} [Q(s_t, a_t)]$$

"상태 $s_t$의 가치는, 현재 정책 $\pi$를 따를 때 기대되는 Q-value이다."

**소프트(Soft) Q-iteration에서의 새로운 V(s):**

이제 **'소프트(Soft)' Q-iteration**에서는 이 $V(s)$가 바뀝니다.

$$V(s_t) = \mathbb{E}_{a_t \sim \pi(\cdot | s_t)} [Q(s_t, a_t) - \alpha \log \pi(a_t | s_t)]$$

이 수식을 잘 뜯어보세요.

"상태 $s_t$의 가치는 (현재 정책 $\pi$를 따를 때 기대되는 Q-value) + (현재 정책 $\pi$의 엔트로피 보너스) 이다."

기억하세요. $H(\pi) = \mathbb{E}[-\log \pi(a|s)]$였으니, $-\log \pi(a|s)$ 항은 $s_t$에서 $a_t$를 샘플링했을 때 얻는 '엔트로피 보너스' 그 자체입니다.

$\alpha$는 이 보너스를 얼마나 중요하게 여길지 결정하는 '온도(temperature)' 파라미터입니다.

**결론**: $V(s)$가 높아지려면 $Q(s, a)$가 높거나, 아니면 $\pi(a|s)$의 엔트로피가 높아야(즉, 정책이 무작위적이어야) 합니다.

---

### 🔄 소프트 정책 반복

이제 이 새로운 '소프트 가치 함수'를 어떻게 학습시킬지, 즉 '소프트 정책 반복' 과정을 봅시다. 정책 반복(Policy Iteration)은 '정책 평가(Policy Evaluation)'와 '정책 개선(Policy Improvement)' 두 단계로 나뉘죠.

#### 1. 소프트 Q-업데이트 (정책 평가 / 크리틱 업데이트)

이건 TD3와 비슷합니다. 크리틱 $Q_{\phi}$를 타겟 $y_t$와 가까워지도록 MSE 손실을 최소화합니다.

$$J_Q(\phi) = \mathbb{E}_{(s_t, a_t) \sim \mathcal{D}} [\frac{1}{2}(Q_{\phi}(s_t, a_t) - y_t)^2]$$

중요한 것은 타겟 $y_t$의 정의입니다.

$$y_t = r(s_t, a_t) + \gamma (Q_{\phi^-}(s_{t+1}, a_{t+1}) - \alpha \log \pi_{\theta}(a_{t+1} | s_{t+1}))$$

(이때 $a_{t+1}$은 다음 상태 $s_{t+1}$에서 *현재* 정책 $\pi_{\theta}$로부터 샘플링된 행동입니다.)

#### 2. 정책 업데이트 (정책 개선 / 액터 업데이트)

액터 $\pi_{\theta}$는 어떤 목표를 향해 개선되어야 할까요?

바로 앞서 본 $V(s)$를 최대화하는 방향입니다.

즉, $\mathbb{E}_{a_t \sim \pi_{\theta}} [Q_{\phi}(s_t, a_t) - \alpha \log \pi_{\theta}(a_t | s_t)]$를 최대화해야 합니다.

이는 $J_{\pi}(\theta) = \mathbb{E}_{s_t \sim \mathcal{D}} [\mathbb{E}_{a_t \sim \pi_{\theta}} [\alpha \log \pi_{\theta}(a_t | s_t) - Q_{\phi}(s_t, a_t)]]$를 **최소화**하는 것과 같습니다.

- $Q_{\phi}$가 높은 행동 $a_t$는 $\log \pi_{\theta}$ (확률)가 높아져야 전체 값이 작아집니다.
- $\log \pi_{\theta}$가 작은 행동 (엔트로피가 높은, 즉 무작위적인)은 그 자체로 보너스($\alpha \log \pi$)를 받아 전체 값을 작게 만듭니다.

#### 리파라미터화 트릭 (Reparameterization Trick)

**기술적인 문제:**

$a_t \sim \pi_{\theta}(\cdot | s_t)$ 이 부분. 즉, $a_t$가 $\pi_{\theta}$라는 확률 분포에서 '샘플링'됩니다.

샘플링 연산은 미분이 불가능합니다. DDPG는 $\pi_{\theta}(s)$가 결정론적이라 연쇄 법칙(chain rule)으로 간단히 해결했는데, 확률론적 정책은 어떻게 $\theta$로 미분할까요?

**해결책: 리파라미터화 트릭 (Reparameterization Trick)**

"미분 불가능한 샘플링 과정을 네트워크 외부로 분리하자!"

$a_t$를 $\pi_{\theta}$에서 직접 샘플링하는 대신:

1. $\theta$와 무관한 표준 정규분포 $\mathcal{N}(0, 1)$에서 노이즈 $\epsilon_t$를 샘플링합니다.
2. 액터 $\pi_{\theta}$는 결정론적인 평균 $\mu_{\theta}(s_t)$과 표준편차 $\sigma_{\theta}(s_t)$를 출력합니다.
3. $a_t = \mu_{\theta}(s_t) + \epsilon_t \cdot \sigma_{\theta}(s_t)$ (이것을 $f_{\theta}(s_t, \epsilon_t)$라고 합시다)

이제 $a_t$는 $\theta$에 대해 미분 가능한 함수 $f_{\theta}$가 되었습니다. 샘플링 $\epsilon_t$는 $\theta$와 무관하므로, $J_{\pi}(\theta)$의 기댓값 안으로 $\nabla_{\theta}$를 옮길 수 있습니다.

---

### 🌡️ α 자동 튜닝

우리가 지금까지 $\alpha$를 '온도'라고 불렀는데, 이 $\alpha$ 값은 하이퍼파라미터입니다.

**문제점:**

- 이 값이 너무 크면($\alpha \uparrow$), 정책은 보상(reward)은 무시하고 엔트로피(무작위성)만 극대화하려 할 겁니다. (그냥 마구잡이로 행동)
- 반대로 너무 작으면($\alpha \downarrow$), 엔트로피 보너스가 사라져 탐험을 안 하고, 결국 결정론적 정책으로 수렴(collapse)해버릴 수 있습니다. (SAC의 장점이 사라짐)

매번 환경마다 이 $\alpha$를 손으로 튜닝하는 것은 매우 고된 일입니다.

**해결책: α를 학습 가능한 파라미터로 만들기**

그래서 SAC 논문은 $\alpha$ 역시 **학습 가능한 파라미터**로 만들자고 제안합니다.

**아이디어:**

"$\alpha$를 튜닝하지 말고, 우리가 원하는 **'최소 엔트로피 타겟($\mathcal{H}$)'**을 정하자."

목표를 '제약이 있는 최적화 문제(constrained optimization)'로 다시 정의합니다.

- **목표**: $\max_{\pi} \sum \mathbb{E}[r(s_t, a_t)]$ (보상을 최대화하라)
- **제약**: $\mathbb{E}[-\log \pi(a_t | s_t)] \ge \mathcal{H}$ (단, 정책의 평균 엔트로피는 타겟 $\mathcal{H}$보다 크거나 같아야 한다)

이 문제는 '라그랑주 승수법(Lagrangian dual problem)'을 통해 풀 수 있습니다.

$$\min_{\alpha \ge 0} \max_{\pi} \sum \mathbb{E}[r(s_t, a_t) - \alpha(\log \pi(a_t | s_t) + \mathcal{H})]$$

여기서 라그랑주 승수 $\alpha$가 바로 우리가 찾던 온도 파라미터입니다!

이제 우리는 $\pi$ (액터)에 대해서는 이 값을 *최대화*하고, $\alpha$에 대해서는 *최소화*하면 됩니다.

**α 업데이트:**

$\alpha$를 최소화하는 목적 함수 $J(\alpha)$는 다음과 같습니다:

$$J(\alpha) = \mathbb{E}_{a \sim \pi} [-\alpha (\log \pi(a|s) + \mathcal{H})]$$

이를 $\alpha$에 대해 미분하면 (경사 하강법을 위해):

$$\nabla_{\alpha} J(\alpha) = \mathbb{E}_{a \sim \pi} [-\log \pi(a|s) - \mathcal{H}]$$

(기대 엔트로피 - 타겟 엔트로피)

- 만약 현재 엔트로피($\mathbb{E}[-\log \pi]$)가 타겟 $\mathcal{H}$보다 **낮으면** (탐험 부족), $\nabla_{\alpha} J(\alpha)$는 음수가 됩니다. $\alpha$는 경사 *하강* $\alpha \leftarrow \alpha - \eta \nabla_{\alpha} J(\alpha)$에 의해 **증가**합니다. ($\alpha$가 커지면 엔트로피 항이 중요해져 탐험을 더 하겠죠)
- 만약 현재 엔트로피가 타겟 $\mathcal{H}$보다 **높으면** (탐험 과다), $\nabla_{\alpha} J(\alpha)$는 양수가 됩니다. $\alpha$는 **감소**합니다. ($\alpha$가 작아지면 보상 항이 중요해져 활용을 더 하겠죠)

이렇게 $\alpha$가 동적으로 조절됩니다.

---

### 💻 SAC 의사 코드

이제 TD3의 장점과 SAC의 철학이 모두 합쳐진 최종 SAC 알고리즘의 의사 코드를 살펴봅시다.

**의사 코드:**

```pseudo
1:  Initialize actor π_θ
2:  Initialize Twin critics Q_φ1, Q_φ2
    Initialize Twin target critics Q_φ_targ,1, Q_φ_targ,2
    Initialize replay buffer D ← ∅

4:  repeat until convergence:
5:      Observe state s
6:      Sample a ~ π_θ(·|s)  (not deterministic + noise!)
7:      Execute a, observe s', r, d
8:      D ← D ∪ {(s, a, r, s', d)}
9:      if s' is terminal then reset environment
10:
11:     if it's time to update then
12:         Sample minibatch B from D
13:         ã' ~ π_θ(·|s')  (sample action for next state)
14:         y(...) ← r + γ(1-d) (min_{i=1,2} Q_φ_targ,i(s', ã') - α log π_θ(ã'|s'))
15:
16:         Update both critics: ∇_φi (1/|B|) Σ (Q_φi(s, a) - y)²
17:
18:         Update actor: ∇_θ (1/|B|) Σ (α log π_θ(ã_θ(s)|s) - min_{i=1,2} Q_φi(s, ã_θ(s)))
19:         (where ã_θ(s) is sampled via reparameterization trick)
20:
21:         (Optional) Update α
22:
23:         Update target networks (soft update)
```

**알고리즘 설명:**

**1-2행 (초기화):**
- 액터 $\pi_{\theta}$
- **"Twin" 크리틱** $Q_{\phi_1}, Q_{\phi_2}$ (TD3처럼 두 개를 씁니다)
- **"Twin" 타겟 크리틱** $Q_{\phi_{targ, 1}}, Q_{\phi_{targ, 2}}$

**4-8행 (데이터 수집):**
- DDPG/TD3와 달리, 노이즈를 더하는 것이 아니라 **확률론적 정책 $\pi_{\theta}(\cdot | s)$에서 직접 샘플링**하여 행동 $a$를 선택합니다.
- $(s, a, r, s', d)$를 리플레이 버퍼 $\mathcal{D}$에 저장합니다.

**11-14행 (샘플링 및 타겟 계산):**
- 버퍼 $\mathcal{D}$에서 미니배치 $B$를 샘플링합니다.
- **타겟 $y$ 계산**:

$$y(r, s', d) = r + \gamma(1-d) (\min_{i=1, 2} Q_{\phi_{targ, i}}(s', \tilde{a}') - \alpha \log \pi_{\theta}(\tilde{a}' | s'))$$

이 한 줄에 모든 핵심이 다 들어있습니다.
- **(TD3) Clipped Double Q-learning**: $\min_{i=1, 2} Q_{\phi_{targ, i}}$
- **(SAC) Soft Q-value**: $... -\alpha \log \pi_{\theta}(\tilde{a}' | s')$
- (이때 $\tilde{a}'$는 $s'$에서 현재 정책 $\pi_{\theta}$를 통해 샘플링된 행동)

**16행 (크리틱 업데이트):**
- 두 개의 크리틱 $Q_{\phi_1}, Q_{\phi_2}$ 모두가 이 타겟 $y$에 가까워지도록 MSE 손실을 줄입니다. (TD3와 동일)

**18행 (액터 업데이트):**

$$\nabla_{\theta} \frac{1}{|B|} \sum (\alpha \log \pi_{\theta}(\tilde{a}_{\theta}(s) | s) - \min_{i=1, 2} Q_{\phi_i}(s, \tilde{a}_{\theta}(s)))$$

- 액터는 '소프트 Q-value'를 최대화하도록 업데이트됩니다.
- 이때 Q-value는 TD3처럼 **두 크리틱 중 더 작은 값($\min$)**을 사용합니다. (더 안정적)
- $\tilde{a}_{\theta}(s)$는 리파라미터화 트릭을 통해 샘플링됩니다.

**21행 (α 업데이트):**
- (만약 자동 튜닝을 쓴다면) 앞서 유도한 $\nabla_{\alpha} J(\alpha)$를 사용해 $\alpha$를 업데이트합니다.

**23행 (타겟 네트워크 업데이트):**
- 크리틱 타겟 네트워크 $Q_{\phi_{targ, i}}$만 소프트 업데이트합니다. (TD3와 동일)

---

### 📝 SAC 요약

**SAC의 특징:**

- 오프-폴리시(Off-policy), 액터-크리틱(Actor-Critic) RL 알고리즘입니다.
- **최대 엔트로피 RL(Maximum Entropy RL)**을 통해 탐험(exploration) 성능을 극대화합니다.
- **Clipped Double Q-learning** (TD3와 동일)을 사용하여 학습 안정성을 향상시킵니다.
- (단점) 자동 튜닝을 쓰지 않으면 $\alpha$ (또는 $\mathcal{H}$)를 튜닝해야 하는 부담이 있습니다.
- TD3와 더불어, 현재(SOTA) 연속 행동 공간에서 **가장 성능이 좋고 널리 쓰이는(Works well in practice!)** 알고리즘 중 하나입니다.

---

### 🆚 TD3 vs SAC

이 비교는 TD3와 SAC의 성능을 보여주는 그래프들을 분석한 것입니다.

**그래프 분석:**

- **위쪽 그래프들 (TD3 논문에서 발췌)**: TD3(파란색)가 SAC(초록색)를 포함한 다른 알고리즘들보다 성능이 더 좋게 나옵니다.
- **아래쪽 그래프들 (SAC 논문에서 발췌)**: SAC(빨간색, 보라색)가 TD3(파란색)보다 성능이 더 좋게 나옵니다.

**결론:**

논문이라는 것이 보통 자신의 알고리즘이 더 잘 나오도록 하이퍼파라미터 튜닝 등에 공을 들이기 마련입니다.

이 두 알고리즘, TD3와 SAC는 사실상 현존하는 오프-폴리시 연속 제어 알고리즘의 양대 산맥입니다. 성능은 우열을 가리기 힘들 정도로 둘 다 매우 뛰어납니다.

다만, SAC가 (특히 $\alpha$ 자동 튜닝을 사용할 경우) 하이퍼파라미터에 조금 덜 민감하고 더 안정적인 탐험을 제공한다는 평을 받으며 최근에는 더 널리 사용되는 추세입니다.

---

## 강화학습 벤치마크

### 🧑‍🏫 지금까지 배운 내용

오늘까지 해서 우리 강화학습 수업의 핵심적인 알고리즘들을 거의 다 배웠습니다. 한번 정리해볼까요?

1. **강화학습 문제 정의**: MDP, 벤치마크
2. **모방 학습 (Imitation Learning)**: BC, DAgger 등 (HW1)
3. **정책 경사 (Policy Gradient)** (On-policy): REINFORCE, PPO, GAE 등 (HW2)
4. **가치 기반 (Value-based)** (Off-policy): DQN, Double DQN 등 (HW3)
5. **오프-폴리시 액터-크리틱 (Off-policy Actor-Critic)** (Off-policy): **DDPG, TD3, SAC** (오늘 배운 내용, HW4)

이제 오늘 강의 계획에서 마지막 주제 하나만 남았습니다.

---

### 📊 RL 알고리즘 비교 방법

우리가 "알고리즘 A가 B보다 좋다"라고 말할 때, 그 기준은 무엇일까요?

**두 가지 주요 기준:**

1. **최종 성능 (Asymptotic performance)**: 충분히 오래 (예: 100만 스텝) 학습시켰을 때, **누가 더 높은 최종 점수**에 도달하는가?

2. **샘플 효율성 (Sample efficiency)**: **얼마나 빨리** 학습하는가? 즉, *더 적은* 학습 스텝(데이터)을 사용하고도 높은 점수에 도달하는가?

**그래프 해석:**
- 가로축: 학습 스텝(Time steps)
- 세로축: 평가 점수(Average Value)
- 초록색 선이 주황색 선보다 최종 점수가 높음 (더 나은 최종 성능)
- 초록색 선이 주황색 선보다 훨씬 가파르게 상승 (더 나은 샘플 효율성)

**이상적인 알고리즘**: 샘플 효율성도 좋고 (빨리 배우고), 최종 성능도 좋은 (높게 배우는) 알고리즘일 것입니다.

---

### 🎮 유명한 RL 벤치마크

우리가 논문에서 보는 수많은 성능 그래프는 대부분 표준화된 '시험 환경', 즉 벤치마크에서 측정된 것입니다.

**연속 제어 (Continuous control):**
- **DeepMind Control Suite (DMC)**
- **OpenAI Gym** (요즘은 Gymnasium으로 불림)
- (DDPG, TD3, SAC가 주로 여기서 테스트됩니다)

**이산 제어 (Discrete control):**
- **Atari**: (DQN이 유명해진 그 게임 환경)
- **MiniGrid**: (간단한 격자 세계)

**최근의 어려운 벤치마크들:**

**Robotics:**
- robosuite
- RLBench
- Meta-World

**Long-horizon (장기 기억/복잡한 작업):**
- FrankaKitchen
- **FurnitureBench**
- **HumanoidBench**

**Open-ended:**
- Minecraft

---

### 🏃‍♂️ OpenAI Gym & DeepMind Control Suite

우리가 오늘 배운 DDPG, TD3, SAC 논문에서 봤던 그래프들(HalfCheetah, Hopper, Walker2d...)이 바로 이 환경들입니다.

**특징:**
- MuJoCo라는 물리 엔진을 사용하여 로봇의 움직임을 시뮬레이션합니다.
- 연속적인 행동 공간(관절 토크 등)을 가지며, 알고리즘의 성능을 비교하는 표준 벤치마크입니다.
- 관절 각도 같은 '상태(state)'를 직접 입력받는 설정(proprioception)과, '픽셀(pixel)' 이미지를 입력받는 설정(vision)이 있습니다.

---

### 🕹️ Atari

이산 행동 공간(discrete action spaces)의 표준 벤치마크입니다. (예: 상, 하, 좌, 우, 발사 등 18개 행동)

DQN, Rainbow 등이 여기서 테스트됩니다.

**두 가지 표준 설정:**

앞서 배운 두 가지 기준을 테스트하기 위해, 아타리 벤치마크는 두 가지 표준 설정이 있습니다:

1. **Atari200M**: 2억 프레임(매우 긴 학습) 동안 학습시켜 **'최종 성능(asymptotic performance)'**을 봅니다.
2. **Atari100K**: 10만 스텝(매우 짧은 학습)만 학습시켜 **'샘플 효율성(sample efficiency)'**을 봅니다.

---

### 🤖 고급 로봇 벤치마크

이 벤치마크들은 최근 강화학습 연구가 얼마나 더 복잡하고 어려운 문제로 나아가고 있는지를 보여줍니다.

**기본 로봇 벤치마크:**
- **OpenAI Gym Robotics**
- **RLBench**
- **Meta-World**
- 주로 로봇 팔(manipulator)을 이용한 다양한 집기/밀기/문열기 과제들

**FurnitureBench:**
- 더 나아가, 단순히 물건을 집는 것이 아니라 IKEA 가구를 '조립'하는 것과 같은 **장기적이고 복잡한(long-horizon complex manipulation)** 과제를 다룹니다.

**HumanoidBench:**
- 가장 복잡한 로봇 중 하나인 '휴머노이드(인간형 로봇)'를 제어하는 벤치마크입니다.
- 단순히 걷는 것(walk, hurdle)뿐만 아니라, 물건을 들고(truck), 선반을 밀고(cabinet), 창문을 닦는(window) 등 **전신 이동(locomotion)과 조작(manipulation)을 동시에** 수행해야 하는 매우 어려운 과제들을 포함합니다.

---

## 요약

### 학습 내용 정리

**1. 핵심 개념:**

- **DDPG (Deep Deterministic Policy Gradient)**:
  - 연속 행동 공간을 위한 결정론적 액터-크리틱 알고리즘
  - $\max_a Q$ 문제를 액터 네트워크 $\pi_{\theta}$로 근사하여 해결
  - 소프트 타겟 업데이트 사용 (Polyak averaging)
  - 하이퍼파라미터에 민감

- **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**:
  - DDPG의 Q-value 과대평가 문제 해결
  - 세 가지 핵심 기술:
    1. Clipped Double Q-learning (Twin critics)
    2. Delayed policy updates
    3. Target policy smoothing
  - DDPG보다 안정적이고 성능이 우수

- **SAC (Soft Actor-Critic)**:
  - 최대 엔트로피 강화학습 프레임워크
  - 확률론적 정책 사용
  - 목표: 보상 최대화 + 엔트로피 최대화
  - $\alpha$ 자동 튜닝으로 탐험-활용 균형 자동 조절
  - 강건성과 탐험 성능이 우수

**2. 주요 알고리즘 비교:**

| 알고리즘 | 정책 유형 | 핵심 특징 | 장점 | 단점 |
|---------|----------|----------|------|------|
| DDPG | 결정론적 | DQN의 연속 행동 확장 | 간단한 구조 | 불안정, 과대평가 |
| TD3 | 결정론적 | Twin critics + Delayed update | 안정적 학습 | 하이퍼파라미터 튜닝 필요 |
| SAC | 확률론적 | Maximum entropy | 강건성, 자동 탐험 | $\alpha$ 튜닝 필요 (선택적) |

**3. 수학적 핵심:**

**DDPG의 DPG 정리:**

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{s \sim \mathcal{B}}[\nabla_a Q_{\phi}(s, a) \vert_{a=\pi_{\theta}(s)} \cdot \nabla_{\theta} \pi_{\theta}(s)]$$

**TD3의 Clipped Double Q-learning:**

$$y = r + \gamma \min_{i=1, 2} Q_{\phi_i^-}(s', \pi_{\theta^-}(s'))$$

**SAC의 소프트 Q-value:**

$$V(s) = \mathbb{E}_{a \sim \pi}[Q(s, a) - \alpha \log \pi(a|s)]$$

**4. RL 벤치마크:**

**평가 기준:**
- **최종 성능 (Asymptotic performance)**: 충분히 학습한 후 최고 점수
- **샘플 효율성 (Sample efficiency)**: 적은 데이터로 빠르게 학습

**주요 벤치마크:**
- **연속 제어**: MuJoCo (Gym, DMC)
- **이산 제어**: Atari (Atari200M, Atari100K)
- **로봇 조작**: RLBench, Meta-World, FurnitureBench
- **복잡한 제어**: HumanoidBench

**5. 중요 포인트:**

**연속 행동 공간의 도전:**
- 이산 공간: $\max_a Q(s,a)$를 모든 $a$에 대해 계산 (간단)
- 연속 공간: $\max_a Q(s,a)$를 최적화 문제로 풀어야 함 (어려움)
- 해결책: 액터 네트워크로 $\arg \max_a Q(s,a)$를 학습

**Q-value 과대평가 문제:**
- 원인: 근사 오차가 벨만 방정식을 통해 전파
- 해결: Double Q-learning으로 두 크리틱 중 최솟값 사용

**탐험 전략:**
- DDPG/TD3: 행동에 노이즈 추가 (외부 탐험)
- SAC: 엔트로피를 목표 함수에 포함 (내부 탐험)

**안정성 기법:**
- 리플레이 버퍼 (Off-policy learning)
- 타겟 네트워크 (소프트 업데이트)
- Clipped Double Q-learning
- 지연된 정책 업데이트

**실무 적용 시 고려사항:**
- TD3: 하이퍼파라미터 튜닝에 시간 투자 필요, 결정론적 정책
- SAC: 자동 튜닝 활용 가능, 확률론적 정책으로 더 강건
- 둘 다 현재 SOTA 수준의 성능, 환경에 따라 선택

---

### 다음 강의 예고

다음 시간에는 **오프라인 강화학습 (Offline RL)**에 대해 배울 것입니다.

오늘 배운 DDPG, TD3, SAC는 모두 '오프-폴리시(Off-policy)'였습니다. 리플레이 버퍼 $\mathcal{D}$를 사용했죠. 하지만 이들은 여전히 '온라인(Online)' 학습입니다. 왜냐하면, 학습하는 중에도 계속 환경과 상호작용을 하며 *새로운* 데이터를 버퍼에 추가하기 때문입니다.

**오프라인 RL (Offline RL)**, 혹은 **배치 RL (Batch RL)**은 여기서 한 걸음 더 나아갑니다. 만약, 우리가 환경과 *전혀* 상호작용할 수 없고, 과거에 다른 정책이나 사람이 수집해 놓은 **고정된(static) 데이터셋 $\mathcal{D}$**만 가지고 있다면 어떨까요?

(예: 병원의 과거 환자 진료 기록, 자율주행 차량의 과거 주행 로그)

이 데이터셋만 보고 최고의 정책을 학습해야 합니다.

이 경우, 오늘 배운 SAC나 TD3를 그대로 쓰면 심각한 문제가 발생합니다. 왜 그런 문제가 발생하며, 이를 어떻게 해결하는지가 다음 시간의 주제가 되겠습니다.
