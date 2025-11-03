# GCB6206 Homework 2: Policy Gradients 완전 해설 가이드 (Part 3)

## 5. Policy Network 구현

### 5.1 MLPPolicy 클래스 구조

#### 5.1.1 클래스 개요

`MLPPolicy`는 관찰(observation)을 입력받아 행동(action)을 출력하는 신경망이야. MLP는 Multi-Layer Perceptron의 약자로, 여러 층의 fully-connected layers로 구성된 네트워크를 의미해.

```python
class MLPPolicy(nn.Module):
    def __init__(self, ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate):
        # 네트워크 구조 정의

    def forward(self, obs):
        # 관찰 → 행동 분포 반환

    def get_action(self, obs):
        # 관찰 → 실제 행동 샘플링
```

#### 5.1.2 `__init__` 메서드 상세 분석

```python
def __init__(
    self,
    ac_dim: int,      # Action space의 차원
    ob_dim: int,      # Observation space의 차원
    discrete: bool,   # Discrete action space인지 여부
    n_layers: int,    # 은닉층 개수
    layer_size: int,  # 각 은닉층의 뉴런 개수
    learning_rate: float,  # 학습률
):
    super().__init__()
```

**파라미터 설명:**

**`ac_dim` (Action Dimension)**:
- **Discrete 환경** (예: CartPole): 가능한 행동의 개수
  - CartPole: `ac_dim = 2` (왼쪽/오른쪽)
- **Continuous 환경** (예: HalfCheetah): 행동 벡터의 차원
  - HalfCheetah: `ac_dim = 6` (6개 관절의 토크)

**`ob_dim` (Observation Dimension)**:
- 환경 상태를 나타내는 벡터의 차원
- CartPole: `ob_dim = 4` (위치, 속도, 각도, 각속도)
- HalfCheetah: `ob_dim = 17` (관절 각도, 속도 등)

**`discrete`**:
- `True`: Categorical distribution 사용 (분류 문제)
- `False`: Gaussian distribution 사용 (회귀 문제)

**`n_layers`**:
- 은닉층 개수 (입력/출력층 제외)
- 예: `n_layers=2` → [입력] → [은닉1] → [은닉2] → [출력]

**`layer_size`**:
- 각 은닉층의 뉴런 개수
- 예: `layer_size=64` → 각 은닉층이 64개 뉴런

---

### 5.2 Discrete Action Space 구현

```python
if discrete:
    self.logits_net = ptu.build_mlp(
        input_size=ob_dim,
        output_size=ac_dim,
        n_layers=n_layers,
        size=layer_size,
    ).to(ptu.device)
    parameters = self.logits_net.parameters()
```

#### 5.2.1 `ptu.build_mlp()` 함수

이 함수는 MLP 네트워크를 자동으로 생성해줘. 내부 구현은 다음과 같아:

```python
# pytorch_util.py에 정의됨
def build_mlp(input_size, output_size, n_layers, size, activation='tanh'):
    layers = []

    # 첫 번째 층
    layers.append(nn.Linear(input_size, size))
    layers.append(nn.Tanh())

    # 은닉층들
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(size, size))
        layers.append(nn.Tanh())

    # 출력층
    layers.append(nn.Linear(size, output_size))

    return nn.Sequential(*layers)
```

**예시: CartPole (ob_dim=4, ac_dim=2, n_layers=2, layer_size=64)**

생성되는 네트워크:
```
Input (4)
  → Linear(4, 64) → Tanh()
  → Linear(64, 64) → Tanh()
  → Linear(64, 2)
  → Output (2)  [왼쪽/오른쪽의 logits]
```

#### 5.2.2 Logits의 의미

**Logits**: Softmax를 적용하기 전의 raw scores

예시:
```python
obs = [0.1, 0.5, -0.2, 0.3]  # CartPole 상태

logits = logits_net(obs)
# logits = [2.1, -0.8]  (raw scores)

# 확률로 변환:
probs = softmax(logits)
# probs = [0.91, 0.09]
# → 왼쪽 이동 확률 91%, 오른쪽 이동 확률 9%
```

#### 5.2.3 `.to(ptu.device)` 의미

```python
.to(ptu.device)
```

- **GPU 사용 시**: 네트워크를 GPU 메모리로 이동
- **CPU 사용 시**: CPU에 유지
- `ptu.device`는 `pytorch_util.py`에서 설정:
  ```python
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  ```

왜 필요하냐면, PyTorch에서는 **텐서와 모델이 같은 device에 있어야** 연산 가능하거든.

---

### 5.3 Continuous Action Space 구현

```python
else:
    self.mean_net = ptu.build_mlp(
        input_size=ob_dim,
        output_size=ac_dim,
        n_layers=n_layers,
        size=layer_size,
    ).to(ptu.device)

    self.logstd = nn.Parameter(
        torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
    )

    parameters = itertools.chain([self.logstd], self.mean_net.parameters())
```

#### 5.3.1 Gaussian Policy 구조

Continuous action은 **Gaussian (정규) 분포**로 모델링해:

$$
a_t \sim \mathcal{N}(\mu_\theta(s_t), \sigma^2)
$$

- $\mu_\theta(s_t)$: 신경망이 예측한 평균 (mean)
- $\sigma^2$: 분산 (학습 가능한 파라미터)

#### 5.3.2 `self.mean_net` 설명

```python
self.mean_net = ptu.build_mlp(...)
```

- **입력**: 상태 (observation)
- **출력**: 행동의 **평균값**

**예시: HalfCheetah (ob_dim=17, ac_dim=6)**
```
Input (17개 상태 변수)
  → MLP
  → Output (6개 관절의 평균 토크)
```

출력 예시:
```python
obs = [...]  # 17-dim state
mean = mean_net(obs)
# mean = [0.5, -0.3, 0.8, 0.1, -0.2, 0.6]  (각 관절의 평균 토크)
```

#### 5.3.3 `self.logstd` 설명

```python
self.logstd = nn.Parameter(
    torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
)
```

**한 줄씩 분석:**

**`nn.Parameter(...)`**:
- PyTorch의 학습 가능한 파라미터로 등록
- `nn.Module`에 포함되면 자동으로 `parameters()`에 추가됨
- Gradient descent로 학습됨

**`torch.zeros(ac_dim, ...)`**:
- 초기값을 0으로 설정
- `ac_dim=6`이면: `[0., 0., 0., 0., 0., 0.]`

**왜 log(std)를 학습하냐?**

Standard deviation $\sigma$는 항상 양수여야 해. 그런데 신경망 파라미터는 음수도 될 수 있잖아? 그래서:

$$
\log \sigma = \text{학습 가능한 파라미터 (제약 없음)}
$$
$$
\sigma = \exp(\log \sigma) \quad \text{(항상 양수!)}
$$

**초기값 0의 의미**:
$$
\log \sigma = 0 \quad \Rightarrow \quad \sigma = e^0 = 1
$$

초기에 분산이 1인 분포로 시작.

#### 5.3.4 `itertools.chain()` 설명

```python
parameters = itertools.chain([self.logstd], self.mean_net.parameters())
```

**`itertools.chain()`**: 여러 iterable을 하나로 연결

예시:
```python
import itertools

list1 = [1, 2]
list2 = [3, 4, 5]
combined = list(itertools.chain(list1, list2))
# combined = [1, 2, 3, 4, 5]
```

우리 경우:
```python
# 결합 대상:
# 1) [self.logstd]  → log standard deviation 파라미터
# 2) self.mean_net.parameters()  → MLP의 모든 weights/biases

# 결과: 모든 학습 가능한 파라미터를 하나의 iterator로
```

이걸 optimizer에 전달하면, **mean network와 std 파라미터를 모두 학습**해.

---

### 5.4 Optimizer 초기화

```python
self.optimizer = optim.Adam(parameters, learning_rate)
```

#### 5.4.1 Adam Optimizer

**Adam (Adaptive Moment Estimation)**:
- 가장 널리 쓰이는 optimizer
- Momentum + RMSprop의 장점 결합
- 각 파라미터마다 적응적 학습률 사용

**내부 동작 (간단히)**:
```python
# 1st moment (평균)
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t

# 2nd moment (분산)
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²

# 파라미터 업데이트
θ_t = θ_{t-1} - α * m_t / (√v_t + ε)
```

기본 하이퍼파라미터:
- β₁ = 0.9
- β₂ = 0.999
- ε = 1e-8

---

### 5.5 `forward()` 메서드 구현

```python
def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
    if self.discrete:
        # TODO: Categorical distribution
        dist = None
    else:
        # TODO: Normal distribution
        dist = None

    return dist
```

#### 5.5.1 Discrete 구현

```python
if self.discrete:
    logits = self.logits_net(obs)
    dist = distributions.Categorical(logits=logits)
```

**`logits = self.logits_net(obs)`**:
- **입력**: `obs` shape: `(batch_size, ob_dim)`
  - 예: `(128, 4)` (128개 상태, 각 4차원)
- **출력**: `logits` shape: `(batch_size, ac_dim)`
  - 예: `(128, 2)` (128개, 각 2개 행동의 score)

**`distributions.Categorical(logits=logits)`**:
- PyTorch의 Categorical 분포 객체 생성
- **Logits → 자동으로 softmax 적용 → 확률 분포**

예시:
```python
obs = torch.tensor([[0.1, 0.5, -0.2, 0.3]])  # 1개 상태

logits = logits_net(obs)
# logits = tensor([[2.1, -0.8]])

dist = distributions.Categorical(logits=logits)

# 내부적으로 softmax 적용:
# probs = softmax([2.1, -0.8]) = [0.91, 0.09]

# 샘플링:
action = dist.sample()
# action = tensor([0])  (확률 91%로 행동 0 선택)

# Log probability:
log_prob = dist.log_prob(action)
# log_prob = tensor([-0.094])  # log(0.91) ≈ -0.094
```

#### 5.5.2 `torch.distributions.Categorical` 클래스 심화

**주요 메서드**:

1. **`sample()`**: 분포에서 행동 샘플링
   ```python
   action = dist.sample()  # shape: (batch_size,)
   ```

2. **`log_prob(value)`**: 주어진 행동의 log probability
   ```python
   log_p = dist.log_prob(action)
   ```
   Policy Gradient에서 $\log \pi_\theta(a|s)$를 계산할 때 사용!

3. **`entropy()`**: 분포의 엔트로피 (탐험 정도)
   ```python
   H = dist.entropy()
   ```
   높은 엔트로피 = 더 uniform한 분포 = 더 많은 탐험

#### 5.5.3 Continuous 구현

```python
else:
    mean = self.mean_net(obs)
    std = torch.exp(self.logstd)
    dist = distributions.Normal(mean, std)
```

**Line 1**: `mean = self.mean_net(obs)`
- **입력**: `obs` shape: `(batch_size, ob_dim)`
  - 예: `(128, 17)` (HalfCheetah)
- **출력**: `mean` shape: `(batch_size, ac_dim)`
  - 예: `(128, 6)` (각 상태마다 6개 관절의 평균 토크)

**Line 2**: `std = torch.exp(self.logstd)`
- `self.logstd` shape: `(ac_dim,)`
  - 예: `[0.1, -0.2, 0.3, 0.0, 0.15, -0.1]`
- `std = exp(logstd)`:
  - 예: `[1.105, 0.819, 1.350, 1.000, 1.162, 0.905]`
- **모든 상태에 같은 std 사용** (state-independent)

**Broadcasting 이해**:
```python
mean.shape = (128, 6)
std.shape = (6,)

# distributions.Normal(mean, std)에서 자동 broadcasting:
# std → (1, 6) → (128, 6)
```

**Line 3**: `dist = distributions.Normal(mean, std)`
- Gaussian distribution 생성
- 각 batch, 각 action dimension마다 독립적인 Gaussian

예시:
```python
obs = torch.tensor([[...]])  # (1, 17)

mean = mean_net(obs)
# mean = tensor([[0.5, -0.3, 0.8, 0.1, -0.2, 0.6]])

std = exp(logstd)
# std = tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

dist = Normal(mean, std)
# 6개의 독립 Gaussian:
# action[0] ~ N(0.5, 1.0²)
# action[1] ~ N(-0.3, 1.0²)
# ...
# action[5] ~ N(0.6, 1.0²)

action = dist.sample()
# action = tensor([0.7, -0.1, 1.2, -0.3, 0.1, 0.9])
```

#### 5.5.4 `torch.distributions.Normal` 클래스 심화

**주요 메서드**:

1. **`sample()`**: Gaussian에서 샘플링
   ```python
   action = dist.sample()
   ```
   내부: `action = mean + std * epsilon`, where `epsilon ~ N(0, 1)`

2. **`log_prob(value)`**: Log probability density
   ```python
   log_p = dist.log_prob(action)
   ```

   수식:
   $$
   \log p(a) = -\frac{1}{2}\left[\log(2\pi\sigma^2) + \frac{(a - \mu)^2}{\sigma^2}\right]
   $$

   **중요**: `log_prob` returns shape `(batch_size, ac_dim)`

   각 action dimension의 log prob를 합쳐야 해:
   ```python
   log_p_total = dist.log_prob(action).sum(dim=-1)
   ```

3. **`rsample()`**: Reparameterization trick으로 샘플링
   - `sample()`은 gradient가 흐르지 않음
   - `rsample()`은 gradient가 흐름 (backprop 가능)

   하지만 우리는 Policy Gradient에서 `log_prob`을 사용하므로 `sample()`로 충분해.

---

### 5.6 `get_action()` 메서드 구현

```python
@torch.no_grad()
def get_action(self, obs: np.ndarray) -> np.ndarray:
    """Takes a single observation and returns a single action."""
    # TODO: implement
    action = None
    return action
```

#### 5.6.1 구현 방법

```python
@torch.no_grad()
def get_action(self, obs: np.ndarray) -> np.ndarray:
    obs = ptu.from_numpy(obs)          # NumPy → PyTorch tensor
    dist = self.forward(obs)           # 관찰 → 분포
    action = dist.sample()             # 분포에서 샘플링
    action = ptu.to_numpy(action)      # Tensor → NumPy
    return action
```

**한 줄씩 설명:**

**`@torch.no_grad()`**:
- Decorator: gradient 계산 비활성화
- 추론(inference) 시에는 gradient 불필요
- 메모리 절약 + 속도 향상

내부 동작:
```python
with torch.no_grad():
    # 이 블록 안에서는 autograd 꺼짐
    # requires_grad=True인 텐서도 gradient 추적 안 함
```

**Line 1**: `obs = ptu.from_numpy(obs)`
- `ptu.from_numpy()` 함수:
  ```python
  def from_numpy(data):
      return torch.from_numpy(data).float().to(device)
  ```
- NumPy array → PyTorch tensor
- `.float()`: float32 타입으로 변환
- `.to(device)`: GPU/CPU로 이동

**입력 예시**:
```python
# NumPy array
obs_np = np.array([0.1, 0.5, -0.2, 0.3])  # CartPole 상태

# PyTorch tensor
obs_torch = torch.tensor([0.1, 0.5, -0.2, 0.3],
                         dtype=torch.float32,
                         device=device)
```

**Line 2**: `dist = self.forward(obs)`
- 앞서 구현한 `forward()` 메서드 호출
- Discrete: `Categorical` distribution 반환
- Continuous: `Normal` distribution 반환

**Line 3**: `action = dist.sample()`
- 분포에서 하나의 행동 샘플링
- **Stochastic policy**: 같은 상태에서도 매번 다른 행동 가능

**Discrete 예시**:
```python
dist = Categorical(probs=[0.7, 0.3])
action = dist.sample()
# action = tensor(0)  또는  tensor(1)
```

**Continuous 예시**:
```python
dist = Normal(mean=tensor([0.5, -0.3]), std=tensor([1.0, 1.0]))
action = dist.sample()
# action = tensor([0.7, -0.1])  (랜덤)
```

**Line 4**: `action = ptu.to_numpy(action)`
- `ptu.to_numpy()` 함수:
  ```python
  def to_numpy(tensor):
      return tensor.cpu().detach().numpy()
  ```
- `.cpu()`: GPU tensor → CPU로 이동
- `.detach()`: Computation graph에서 분리 (gradient 끊음)
- `.numpy()`: PyTorch tensor → NumPy array

**왜 NumPy로 변환?**
- Gym 환경은 NumPy array를 받아
- `env.step(action)` ← action은 NumPy여야 함

---

### 5.7 `update()` 메서드 구현 (Policy Gradient)

```python
def update(
    self,
    obs: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
) -> dict:
    """Implements the policy gradient actor update."""
    assert obs.ndim == 2
    assert advantages.ndim == 1
    assert obs.shape[0] == actions.shape[0] == advantages.shape[0]

    obs = ptu.from_numpy(obs)
    actions = ptu.from_numpy(actions)
    advantages = ptu.from_numpy(advantages)

    # TODO: implement policy gradient update
    loss = None

    return {"Actor Loss": ptu.to_numpy(loss)}
```

#### 5.7.1 Policy Gradient Loss 유도

**목표**: $\nabla_\theta J(\theta)$ 방향으로 파라미터 업데이트

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A_t \right]
$$

여기서 $A_t$는 advantage (나중에 자세히).

**PyTorch는 gradient descent를 기본으로 하므로**, loss를 최소화하는 방향으로 학습해. 우리는 **reward를 최대화**하고 싶으니까:

$$
\text{Loss} = -\mathbb{E} \left[ \log \pi_\theta(a_t | s_t) \cdot A_t \right]
$$

Negative를 붙여서 **loss 최소화 = reward 최대화**

배치 평균:
$$
\text{Loss} = -\frac{1}{N} \sum_{i=1}^N \log \pi_\theta(a_i | s_i) \cdot A_i
$$

#### 5.7.2 구현 코드

```python
def update(self, obs, actions, advantages):
    # ... assertions and conversions ...

    # Step 1: Get distribution
    dist = self.forward(obs)

    # Step 2: Calculate log probabilities
    log_probs = dist.log_prob(actions)

    # Step 3: Handle continuous actions (sum over action dims)
    if not self.discrete:
        log_probs = log_probs.sum(dim=-1)

    # Step 4: Calculate loss
    loss = -(log_probs * advantages).mean()

    # Step 5: Backpropagation
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return {"Actor Loss": ptu.to_numpy(loss)}
```

**한 줄씩 상세 설명:**

**Step 1**: `dist = self.forward(obs)`
- `obs` shape: `(batch_size, ob_dim)` → 예: `(128, 4)`
- `dist`: Categorical 또는 Normal distribution

**Step 2**: `log_probs = dist.log_prob(actions)`
- `actions` shape:
  - Discrete: `(batch_size,)` → 예: `tensor([0, 1, 0, ...])`
  - Continuous: `(batch_size, ac_dim)` → 예: `(128, 6)`

- `log_probs` shape:
  - Discrete: `(batch_size,)`
  - Continuous: `(batch_size, ac_dim)` ← **주의!**

**Discrete 예시**:
```python
# obs.shape = (3, 4)
dist = Categorical(logits=[[2.0, -1.0], [1.0, 1.5], [-0.5, 0.8]])
actions = tensor([0, 1, 1])

log_probs = dist.log_prob(actions)
# log_probs = tensor([-0.12, -0.37, -0.25])
# 각 sample의 선택된 action의 log probability
```

**Continuous 예시**:
```python
# obs.shape = (2, 17)
dist = Normal(mean=[[0.5, -0.3]], std=[[1.0, 1.0]])
actions = tensor([[0.7, -0.1]])

log_probs = dist.log_prob(actions)
# log_probs = tensor([[-0.94, -0.96]])  # shape: (2, 2)
# 각 action dimension의 log probability
```

**Step 3**: Continuous action의 경우 합산
```python
if not self.discrete:
    log_probs = log_probs.sum(dim=-1)
```

**왜 합산?**

Gaussian policy는 각 action dimension이 **독립**이라고 가정:

$$
\pi_\theta(a | s) = \prod_{i=1}^{\text{ac\_dim}} \mathcal{N}(a_i | \mu_i(s), \sigma_i^2)
$$

Log probability:
$$
\log \pi_\theta(a | s) = \sum_{i=1}^{\text{ac\_dim}} \log \mathcal{N}(a_i | \mu_i, \sigma_i^2)
$$

따라서 **각 dimension의 log prob를 합쳐야** 전체 log probability가 돼.

**`.sum(dim=-1)` 설명**:
```python
log_probs.shape = (batch_size, ac_dim)

log_probs.sum(dim=-1)  # 마지막 차원에 대해 합
# 결과 shape: (batch_size,)
```

예시:
```python
log_probs = tensor([
    [-0.9, -1.0, -0.8],  # Sample 1
    [-1.1, -0.7, -0.9]   # Sample 2
])

log_probs.sum(dim=-1)
# tensor([-2.7, -2.7])
```

**Step 4**: Loss 계산
```python
loss = -(log_probs * advantages).mean()
```

**Element-wise 곱셈**:
```python
log_probs.shape = (batch_size,)
advantages.shape = (batch_size,)

log_probs * advantages  # element-wise
# shape: (batch_size,)
```

예시:
```python
log_probs = tensor([-0.5, -0.3, -0.8])
advantages = tensor([2.0, -1.5, 3.0])

log_probs * advantages
# tensor([-1.0, 0.45, -2.4])
```

**`.mean()`**: 배치 평균
```python
(log_probs * advantages).mean()
# = (-1.0 + 0.45 + (-2.4)) / 3
# = -0.983
```

**Negative 부호**:
```python
loss = -(log_probs * advantages).mean()
# = -(-0.983) = 0.983
```

**직관적 이해**:
- `advantage > 0` (좋은 행동) + `negative` → **loss 감소** → log_prob 증가 → 해당 행동 확률 증가 ✓
- `advantage < 0` (나쁜 행동) + `negative` → **loss 증가** → log_prob 감소 → 해당 행동 확률 감소 ✓

**Step 5**: Backpropagation
```python
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

**Line 1**: `self.optimizer.zero_grad()`
- 이전 iteration의 gradient 초기화
- PyTorch는 기본적으로 gradient를 **누적**하기 때문에 매번 초기화 필요

**왜 누적?**
- RNN 같은 경우 유용
- 하지만 우리는 매 iteration마다 새로 계산해야 하니까 초기화

**Line 2**: `loss.backward()`
- **Autograd**: 자동 미분
- Computation graph를 역순으로 따라가며 gradient 계산
- 모든 `requires_grad=True`인 파라미터에 대해 `∂loss/∂θ` 계산

내부 동작:
```python
# 모든 파라미터 θ에 대해:
θ.grad = ∂loss/∂θ
```

**Line 3**: `self.optimizer.step()`
- 계산된 gradient로 파라미터 업데이트
- Adam optimizer 사용:
  ```python
  for θ in parameters:
      θ = θ - learning_rate * adam_update(θ.grad)
  ```

---

### 5.8 전체 흐름 예시

CartPole 환경에서의 구체적 예:

```python
# 1. Trajectory 수집 후
obs = np.array([
    [0.1, 0.5, -0.2, 0.3],  # Timestep 1
    [0.2, 0.6, -0.1, 0.2],  # Timestep 2
    ...
])  # shape: (128, 4)

actions = np.array([0, 1, 0, ...])  # shape: (128,)
advantages = np.array([1.5, -0.8, 2.3, ...])  # shape: (128,)

# 2. Update 호출
info = policy.update(obs, actions, advantages)

# 3. 내부 동작:
# dist = Categorical(logits=logits_net(obs))
# log_probs = dist.log_prob(actions)
#   → tensor([-0.5, -1.2, -0.3, ...])
# loss = -(log_probs * advantages).mean()
#   → tensor(0.85)
# loss.backward()
# optimizer.step()

# 4. 결과:
# 네트워크 파라미터 업데이트됨
# advantage가 높은 행동의 확률 증가
# advantage가 낮은 행동의 확률 감소
```

---

**Part 3 완료. 다음 파트에서 Baseline (Critic Network) 구현을 다룹니다.**
