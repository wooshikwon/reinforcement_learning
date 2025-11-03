## Lecture 03-1: Log-derivative Trick (로그-미분 트릭)

안녕하세요! 질문해주신 수식은 **로그-미분 트릭(Log-derivative Trick)** 또는 **스코어 함수(Score Function)**의 핵심 아이디어라고 불리며, 통계학과 머신러닝, 특히 강화학습(Policy Gradient)이나 변분 추론(Variational Inference) 같은 분야에서 아주 중요하게 사용됩니다.

이 식이 어떻게 유도되는지, 가장 기초적인 미분 개념부터 차근차근, 단계별로 엄밀하게 설명해 드릴게요.

------

### 1단계: 모든 것의 시작, 로그 함수의 미분 📈

가장 먼저 알아야 할 기초는 자연로그 함수 $y = \log(x)$ (또는 $\ln(x)$)의 미분입니다. 고등학교 수학에서 배우는 기본 공식이죠.

$x$에 대해 $\log(x)$를 미분하면 그 결과는 $\frac{1}{x}$이 됩니다.

$$\frac{d}{dx} \log(x) = \frac{1}{x} $$이것이 모든 설명의 출발점입니다. 간단히 말해, 로그 함수의 특정 지점에서의 변화율은 그 지점의 x값에 반비례한다는 뜻입니다. 

---

### 2단계: 핵심 연결고리, 연쇄 법칙 (Chain Rule) 🔗 

이제 변수가 그냥 $x$가 아니라, $x$에 대한 어떤 함수, 예를 들어 $p(x)$일 때를 생각해 보겠습니다. 즉, $y = \log(p(x))$를 미분하고 싶습니다. 

이처럼 함수 안에 또 다른 함수가 들어있는 형태를 **합성 함수**라고 부릅니다.

 합성 함수를 미분할 때는 \*\*연쇄 법칙(Chain Rule)\*\*을 사용해야 합니다. 연쇄 법칙은 다음과 같습니다. 

> 함수 $g(u)$와 $u=f(x)$의 합성 함수 $g(f(x))$를 $x$에 대해 미분하면, **겉 함수를 미분하고(속 함수는 그대로) 속 함수를 미분한 것을 곱한다.** 
> $$
> \frac{d}{dx} g(f(x)) = g'(f(x)) \cdot f'(x)
> $$

이것을 우리의 경우인 $\log(p(x))$에 적용해 봅시다.   

**겉 함수** $g(u) = \log(u)$  * **속 함수** $u = p(x)$ 연쇄 법칙에 따라 단계적으로 미분해 보겠습니다. 

1. **겉 함수를 미분합니다:** $g'(u) = \frac{d}{du}\log(u) = \frac{1}{u}$ 입니다. 여기에 속 함수 $p(x)$를 그대로 대입하면 $\frac{1}{p(x)}$가 됩니다. 
2. **속 함수를 미분합니다:** $p(x)$를 $x$에 대해 미분하면 $p'(x)$ 또는 $\frac{d}{dx}p(x)$ 입니다. 
3. **두 결과를 곱합니다:** 위의 두 결과를 곱하면 최종 미분 결과가 나옵니다. 

$$
\frac{d}{dx} \log(p(x)) = \frac{1}{p(x)} \cdot p'(x) = \frac{p'(x)}{p(x)}
$$

이것이 로그-미분 트릭의 가장 핵심적인 수학적 관계입니다. 어떤 함수의 로그를 미분하면, 원래 함수 분의 원래 함수의 도함수(미분)가 된다는 것입니다.

------

### 3단계: 등식의 첫 부분, 항등 변형 💡

이제 원래 질문의 등식을 다시 살펴봅시다.
$$
\nabla p(x) = p(x) \frac{\nabla p(x)}{p(x)} = p(x) \nabla \log p(x)
$$
여기서 첫 번째 등호(=) 부분은 사실 미분이 아니라 단순한 대수적 트릭입니다.

$$\nabla p(x) = p(x) \frac{\nabla p(x)}{p(x)}$$우변을 보면, $\nabla p(x)$에 $p(x)$를 곱하고 다시 $p(x)$로 나누어 주었습니다. 

$p(x) \neq 0$ 이라면, 어떤 값에 특정 수를 곱했다가 다시 나누면 원래 값이 그대로 남습니다. 

즉, 이 식은 $\nabla p(x) = \nabla p(x)$ 라는 당연한 말을 다르게 표현한 것뿐입니다. 

이런 변형을 하는 이유는 바로 **2단계에서 유도한 형태인 $\frac{\nabla p(x)}{p(x)}$를 만들어내기 위함**입니다. 

-----

### 4단계: 모든 조각 맞추기, 최종 증명 🧩 

이제 모든 조각이 준비되었습니다. 차근차근 증명을 완성해 봅시다. 

1. 우리는 2단계에서 연쇄 법칙을 통해 다음을 증명했습니다. 
   $$
   \nabla \log p(x) = \frac{\nabla p(x)}{p(x)}
   $$

​	(여기서 미분 연산자 $\frac{d}{dx}$를 벡터 미분 연산자인 $\nabla$로 일반화해도 똑같이 성립합니다. 이 부분은 5단계에서 자세히 설명하겠습니다.)

2. 이제 3단계의 식 $\nabla p(x) = p(x) \frac{\nabla p(x)}{p(x)}$ 에서 $\frac{\nabla p(x)}{p(x)}$ 부분을 위 1번 결과로 **치환**합니다.

$$
\\ \nabla p(x) = p(x) \underbrace{\frac{\nabla p(x)}{p(x)}}\_{\text{이 부분을 } \nabla \log p(x) \text{ 로 치환}}
$$



3. 치환하면 최종적으로 우리가 증명하려던 식이 완성됩니다.

$$
\\ \nabla p(x) = p(x) \nabla \log p(x)
$$

이로써 모든 증명이 끝났습니다. 첫 번째 등호는 대수적 트릭이고, 두 번째 등호는 로그 미분의 연쇄 법칙을 이용한 것입니다.

------

### 5단계: 다변수로의 확장 (d/dx에서 ∇로) 🌐

지금까지는 변수가 하나인 경우($x$)를 가정하고 $\frac{d}{dx}$로 설명했지만, 원래 질문에는 다변수 벡터를 의미하는 **그래디언트(gradient) 연산자** $\nabla$가 사용되었습니다.

결론부터 말하면, **다변수에서도 위 관계는 똑같이 성립합니다.**

$x$가 벡터 $x = [x_1, x_2, ..., x_n]^T$ 라고 할 때, 그래디언트 $\nabla$는 각 성분에 대한 편미분(partial derivative)을 원소로 가지는 벡터입니다.

$$\nabla = \left[ \frac{\partial}{\partial x_1}, \frac{\partial}{\partial x_2}, \dots, \frac{\partial}{\partial x_n} \right]^T $$

$\nabla \log p(x)$의 각 성분($i$번째)을 계산해 보면, 연쇄 법칙에 의해 다음과 같이 됩니다.

$$\frac{\partial}{\partial x_i} \log p(x) = \frac{1}{p(x)} \frac{\partial p(x)}{\partial x_i}$$

이것을 모든 성분 $i=1, \dots, n$에 대해 벡터로 묶으면 다음과 같습니다.

$$\nabla \log p(x) = \begin{bmatrix} \frac{1}{p(x)} \frac{\partial p(x)}{\partial x_1} \\ \frac{1}{p(x)} \frac{\partial p(x)}{\partial x_2} \\ \vdots \\ \frac{1}{p(x)} \frac{\partial p(x)}{\partial x_n} \end{bmatrix} = \frac{1}{p(x)} \begin{bmatrix} \frac{\partial p(x)}{\partial x_1} \\ \frac{\partial p(x)}{\partial x_2} \\ \vdots \\ \frac{\partial p(x)}{\partial x_n} \end{bmatrix} = \frac{1}{p(x)} \nabla p(x)$$

따라서 $\nabla \log p(x) = \frac{\nabla p(x)}{p(x)}$ 라는 관계가 다변수 환경에서도 동일하게 성립하며, 양변에 $p(x)$를 곱하면 우리가 원하는 최종 식을 얻게 됩니다.

$$
p(x) \nabla \log p(x) = \nabla p(x)
$$

------

### 6단계: 강화학습의 목표 설정 - 보상 최대화하기 🎯

이제부터는 이 수학적 도구가 어떻게 실제 문제, 특히 **강화학습(Reinforcement Learning)**에 적용되는지 알아보겠습니다. 강화학습에서 우리의 주인공인 **에이전트(agent)**는 **환경(environment)** 속에서 최적의 행동 방식을 학습합니다.

- **정책(Policy) $\pi_{\theta}$**: 에이전트의 행동 지침입니다. 특정 **상태(state)** $s$에서 어떤 **행동(action)** $a$를 할지 결정하는 함수이며, 파라미터 $\theta$로 정의됩니다. (예: $\theta$는 신경망의 가중치)
- **궤적(Trajectory) $\tau$**: 에이전트가 환경과 상호작용하며 만들어내는 한 편의 에피소드입니다. 상태, 행동, 보상의 연속적인 시퀀스 $\tau = (s_1, a_1, r_1, s_2, a_2, r_2, \dots)$로 구성됩니다.
- **보상(Reward) $r$**: 각 행동의 결과로 환경으로부터 받는 즉각적인 신호입니다.

강화학습의 목표는 명확합니다. **"누적 보상의 기댓값을 최대화하는 정책 $\pi_{\theta}$를 찾는 것"** 입니다. 이를 위한 **목표 함수(Objective Function) $J(\theta)$**는 다음과 같이 정의됩니다.

$$\theta^* = \arg\max_{\theta} J(\theta) = \arg\max_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_t r(s_t, a_t) \right]$$

- $\theta^*$는 우리가 찾고 싶은 최적의 정책 파라미터입니다.
- $\sum_t r(s_t, a_t)$는 하나의 궤적 $\tau$에서 받은 보상의 총합, 즉 **반환값(Return)**입니다. 간단히 $r(\tau)$라고도 씁니다.
- $\mathbb{E}*{\tau \sim \pi*{\theta}}[\cdot]$는 현재 정책 $\pi_{\theta}$를 따를 때 나타날 수 있는 모든 가능한 궤적에 대한 기댓값을 의미합니다.

우리의 임무는 이 $J(\theta)$를 가장 크게 만드는 $\theta$를 찾는 것입니다. 이를 위해 경사 상승법(Gradient Ascent)을 사용할 것이며, 그러려면 먼저 $J(\theta)$를 $\theta$에 대해 미분한 **경사도(gradient) $\nabla_{\theta}J(\theta)$**를 구해야 합니다.

------

### 7단계: 핵심 난관 돌파 - 정책 경사도 유도하기 🌊

자, 이제 목표 함수 $J(\theta) = \mathbb{E}*{\tau \sim \pi*{\theta}}[r(\tau)]$를 미분해 봅시다.
 $\nabla_{\theta} J(\theta) = \nabla_{\theta},\mathbb{E}*{\tau \sim \pi*{\theta}}[r(\tau)]$

여기서 큰 문제가 발생합니다. 미분 연산자 $\nabla_{\theta}$를 기댓값 안으로 그냥 넣을 수가 없습니다. 왜냐하면 기댓값을 계산하는 대상인 확률 분포, 즉 궤적의 분포($\tau \sim \pi_{\theta}$) 자체가 미분하려는 변수 $\theta$에 따라 변하기 때문입니다. 정책이 바뀌면 당연히 에이전트가 만들어내는 궤적의 분포도 달라지겠죠.

바로 이 난관을 해결하기 위해 1~5단계에서 공들여 유도한 **로그-미분 트릭**이 화려하게 등판합니다.

먼저, 기댓값을 적분(또는 이산 확률에서는 시그마) 형태로 풀어 써봅시다.
$$
J(\theta) = \int \pi_{\theta}(\tau) r(\tau) d\tau
$$
이제 양변을 $\theta$로 미분합니다. 적분과 미분의 순서는 바꿀 수 있으므로(Leibniz integral rule), 미분 연산자를 안으로 넣을 수 있습니다.
$$
\nabla_{\theta} J(\theta) = \nabla_{\theta} \int \pi_{\theta}(\tau), r(\tau), d\tau = \int \nabla_{\theta}\pi_{\theta}(\tau) r(\tau) d\tau
$$
바로 이 지점, $\nabla_{\theta} \pi_{\theta}(\tau)$에 로그-미분 트릭($\nabla p(x) = p(x), \nabla \log p(x)$)을 적용합니다.
$$
\nabla_{\theta}\pi_{\theta}(\tau) = \pi_{\theta}(\tau) \nabla_{\theta}\log \pi_{\theta}(\tau)
$$
따라서
$$
\nabla_{\theta} J(\theta) = \int \pi_{\theta}(\tau) \nabla_{\theta}\log \pi_{\theta}(\tau) r(\tau) d\tau
$$
자, 이제 이 복잡해 보이는 적분식을 다시 기댓값(Expectation) 형태로 되돌릴 차례입니다. 

이것이 어떻게 가능할까요? 그 근거는 기댓값의 수학적 정의 자체에 있습니다. 

어떤 확률 변수 $x$가 확률 분포 $p(x)$를 따를 때, 함수 $f(x)$의 기댓값 $\mathbb{E}*{x \sim p(x)}[f(x)]$는 다음과 같이 정의됩니다.
$$
\mathbb{E}*{x \sim p(x)}[f(x)] = \int p(x) f(x) dx
$$
이 정의를 우리의 정책 경사도 적분식과 나란히 놓고 보면, 그 구조가 완벽하게 일치하는 것을 알 수 있습니다.
$$
\nabla_{\theta} J(\theta) = \int \pi_{\theta}(\tau)\big[\nabla_{\theta}\log \pi_{\theta}(\tau) r(\tau)\big] d\tau
$$
여기서 각 요소를 하나씩 짝지어 보면,
 $p(x)\longleftrightarrow\pi_{\theta}(\tau) \quad f(x)\longleftrightarrow\nabla_{\theta}\log \pi_{\theta}(\tau) r(\tau) \quad x\longleftrightarrow\tau.$

따라서 기댓값의 정의에 의해, 위 적분식은 정확히 “확률 분포 $\pi_{\theta}(\tau)$에 대한 함수 $\nabla_{\theta}\log \pi_{\theta}(\tau), r(\tau)$의 기댓값”으로 표현할 수 있습니다.
$$
\nabla_{\theta} J(\theta) ;=; \mathbb{E}*{\tau \sim \pi*{\theta}}\big[,\nabla_{\theta}\log \pi_{\theta}(\tau), r(\tau),\big]
$$
이것이 바로 **정책 경사도 정리(Policy Gradient Theorem)**입니다. 

정말 놀라운 결과입니다! 이 변환이 중요한 이유는 이론을 현실로 가져오기 때문입니다. 

모든 궤적에 대한 적분($\int \dots d\tau$)은 사실상 계산이 불가능하지만, 

기댓값($\mathbb{E}[\dots]$) 형태는 **샘플링을 통한 근사**를 허용합니다. 

즉, 우리가 현재 정책 $\pi_{\theta}$를 이용해 여러 궤적 $\tau$를 샘플링하고, 그 샘플들의 평균으로 경사도를 ‘추정’할 수 있는 실용적인 길이 열린 것입니다.
$$
\nabla_{\theta} J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}\Big(\nabla_{\theta}\log \pi_{\theta}\big(\tau^{(i)}\big) r\big(\tau^{(i)}\big)\Big)
$$


------

### 8단계: 경사도 단순화 - 환경을 몰라도 괜찮아! 🤖

방금 유도한 식의 핵심인 $\nabla_{\theta} \log \pi_{\theta}(\tau)$는 구체적으로 어떻게 계산할까요?

먼저 궤적 $\tau$가 나타날 확률 $\pi_{\theta}(\tau)$는 다음과 같이 여러 확률의 곱으로 이루어집니다.

$$\pi_{\theta}(\tau) = p(s_1) \prod_{t=1}^{T} \pi_{\theta}(a_t|s_t) p(s_{t+1}|s_t, a_t)$$

- $p(s_1)$: 초기 상태 분포 (에피소드가 어떤 상태에서 시작될 확률)
- $\pi_{\theta}(a_t|s_t)$: **정책**. 상태 $s_t$에서 행동 $a_t$를 할 확률. **이 부분만 파라미터 $\theta$에 의존합니다.**
- $p(s_{t+1}|s_t, a_t)$: **환경의 동역학(Dynamics)**. 상태 $s_t$, 행동 $a_t$ 이후 다음 상태 $s_{t+1}$이 나올 확률.

이제 양변에 로그를 취하면, 복잡한 곱셈이 간단한 덧셈으로 바뀝니다.

$$\log \pi_{\theta}(\tau) = \log p(s_1) + \sum_{t=1}^{T} \log \pi_{\theta}(a_t|s_t) + \sum_{t=1}^{T} \log p(s_{t+1}|s_t, a_t)$$

우리가 구하고 싶은 것은 이 식을 $\theta$에 대해 미분한 $\nabla_{\theta} \log \pi_{\theta}(\tau)$입니다. 위 식의 각 항을 살펴보면, $\log p(s_1)$와 $\log p(s_{t+1}|s_t, a_t)$ 항들은 정책 파라미터 $\theta$와 아무 관련이 없습니다. 따라서 미분하면 0이 되어 사라집니다. 그 결과, 아주 깔끔하고 중요한 형태만 남게 됩니다.

$$\nabla_{\theta} \log \pi_{\theta}(\tau) = \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t)$$

이것은 정책 경사도 방법의 또 다른 강력한 장점을 보여줍니다. 경사도를 계산하는 데 **환경의 동역학($p(s_{t+1}|s_t, a_t)$)에 대한 정보가 전혀 필요 없다**는 것입니다! 에이전트는 환경이 어떻게 작동하는지 전혀 몰라도, 오직 자신의 정책과 그 결과로 받은 보상만으로 학습을 진행할 수 있습니다. 이런 방식을 **모델-프리(Model-Free)** 방식이라고 부릅니다.

------

### 9단계: 최종 레시피 - REINFORCE 알고리즘 👨‍🍳

이제 7단계와 8단계의 결과를 합쳐봅시다. 최종적인 정책 경사도(Policy Gradient) 공식은 다음과 같습니다.

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \left( \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \right) \left( \sum_{t=1}^{T} r(s_t, a_t) \right) \right]$$

이 수식의 의미는 아주 직관적입니다.

- $(\sum_{t=1}^{T} r(s_t, a_t))$: 한 궤적의 **총 보상(반환값)**입니다. 이 값이 크면 좋은 궤적, 작으면 나쁜 궤적이라는 의미입니다.
- $(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t))$: 그 궤적에 포함된 모든 행동들에 대해, 해당 행동이 나올 **로그 확률을 높이는 방향**의 벡터입니다.

따라서 이 둘을 곱한다는 것은 **"좋은 궤적(총 보상이 높은)에서 했던 행동들의 확률은 높이고, 나쁜 궤적(총 보상이 낮은)에서 했던 행동들의 확률은 낮추도록 정책 $\theta$를 업데이트하라"**는 의미입니다.

실제로는 이 기댓값을 정확히 계산할 수 없으므로, **몬테카를로 샘플링**으로 근사합니다. 즉, 현재 정책으로 N개의 궤적을 샘플링하고, 그 결과를 평균 내어 경사도를 추정합니다. 이 원리를 그대로 구현한 것이 바로 **REINFORCE 알고리즘**입니다.

1. **샘플링**: 현재 정책 $\pi_{\theta}(a|s)$를 사용해 환경과 상호작용하며 N개의 궤적(에피소드) $\tau^1, \tau^2, \dots, \tau^N$을 수집합니다.

2. **경사도 계산**: 수집된 궤적들을 이용해 경사도를 근사 계산합니다.

   $$\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{i,t}|s_{i,t}) \right) \left( \sum_{t=1}^{T} r(s_{i,t}, a_{i,t}) \right)$$

3. **파라미터 업데이트**: 경사 상승법을 이용해 정책 파라미터를 업데이트합니다. (보상을 최대화하는 것이므로 경사도를 더해줍니다)

   $$\theta \leftarrow \theta + \alpha \nabla_{\theta} J(\theta)$$

   여기서 $\alpha$는 학습률(learning rate)입니다. 이 세 단계를 계속 반복하면 정책은 점진적으로 더 높은 보상을 받는 방향으로 개선됩니다.

------

### 10단계: 개선의 순환 고리 - REINFORCE 흐름도 🔄

REINFORCE 알고리즘의 전체적인 흐름을 그림으로 보면 더 명확하게 이해할 수 있습니다. 이는 '정책 실행 → 평가 및 학습 → 정책 개선'이라는 끊임없는 순환 고리입니다.

1. **정책 실행 (Roll-out policy)**
   - 현재의 정책 $\pi_{\theta}$를 가지고 실제 환경에서 에피소드를 여러 번(N번) 끝까지 진행시켜 봅니다.
   - 이 과정에서 $(s_1, a_1, r_1), (s_2, a_2, r_2), \dots$ 와 같은 데이터 궤적들이 수집됩니다.
2. **반환값 추정 및 모델 학습 (Estimate return / Fit model)**
   - 수집된 궤적들 각각에 대해 보상의 총합, 즉 반환값 $r(\tau^i)$을 계산합니다.
   - 이 반환값과 로그 확률의 경사도를 이용해 9단계에서 본 공식으로 최종 정책 경사도 $\nabla_{\theta}J(\theta)$를 계산합니다.
3. **정책 개선 (Improve policy)**
   - 계산된 경사도의 방향으로 정책의 파라미터 $\theta$를 조금 업데이트합니다 ($\theta \leftarrow \theta + \alpha \nabla_{\theta} J(\theta)$).
   - 이제 에이전트는 이전보다 조금 더 나은 정책을 갖게 됩니다.

그리고 개선된 정책을 가지고 다시 1번으로 돌아가 이 과정을 무한히 반복하는 것입니다. 이 순환을 거듭할수록 에이전트의 정책은 점차 최적의 정책으로 수렴해 갑니다.