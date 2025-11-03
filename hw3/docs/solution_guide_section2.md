# Homework 3 Solution Guide - Section 2

## 3. Code Structure Overview (코드 구조 개요)

### 3.1 전체 아키텍처

이 과제의 코드는 모듈화된 구조로 설계되어 있습니다:

```
gcb6206/
├── agents/
│   └── dqn_agent.py          # DQN 에이전트 클래스
├── scripts/
│   └── run_hw3.py             # 메인 훈련 루프
├── infrastructure/
│   ├── replay_buffer.py       # 경험 재생 버퍼
│   ├── utils.py               # 유틸리티 함수 (trajectory sampling)
│   └── atari_wrappers.py      # Atari 환경 래퍼
└── env_configs/
    ├── dqn_basic_config.py    # CartPole, LunarLander 설정
    ├── dqn_atari_config.py    # Atari 게임 설정
    └── schedule.py            # Epsilon scheduling 클래스
```

### 3.2 실행 흐름 (Execution Flow)

```
1. run_hw3.py 시작
   ↓
2. Config 파일 로드 (YAML → basic_dqn_config 또는 atari_dqn_config)
   ↓
3. 환경 생성 (env, eval_env, render_env)
   ↓
4. DQN Agent 생성 (critic, target_critic, optimizer)
   ↓
5. Replay Buffer 초기화
   ↓
6. Training Loop (run_training_loop)
   ├── Action 선택 (ε-greedy)
   ├── Environment step
   ├── Replay buffer에 저장
   ├── Batch sampling
   ├── Agent update (critic + target)
   └── Evaluation
```

---

## 4. Replay Buffer Implementation (리플레이 버퍼 구현)

Replay buffer는 DQN의 핵심 구성 요소입니다. 두 가지 버전이 있습니다:

### 4.1 Regular Replay Buffer (일반 리플레이 버퍼)

**파일 위치**: `gcb6206/infrastructure/replay_buffer.py`

**목적**: Off-policy 학습을 위한 경험 저장 및 샘플링

#### 클래스 구조와 초기화

```python
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.max_size = capacity
        self.size = 0
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dones = None
```

**각 속성의 의미:**

- `max_size`: 버퍼의 최대 용량 (기본 100만 transitions)
  - 왜 100만? Atari 논문에서 경험적으로 좋은 성능을 보인 값
  - 메모리가 부족하면 줄일 수 있음

- `size`: 현재 저장된 transition 수
  - 처음에는 0부터 시작
  - `insert()`할 때마다 1씩 증가
  - `max_size`에 도달하면 오래된 데이터 덮어쓰기 (circular buffer)

- `observations`, `actions` 등: 실제 데이터 저장소
  - 처음에는 `None` (lazy initialization)
  - 첫 `insert()` 시 numpy array로 초기화
  - 이유: 처음에는 observation의 shape를 모르기 때문

#### insert() 메서드 - 데이터 저장

```python
def insert(
    self,
    /,  # position-only parameters (Python 3.8+)
    observation: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    next_observation: np.ndarray,
    done: np.ndarray,
):
```

**`/` 의 의미**: Position-only parameter marker
- 이 기호 앞의 파라미터들은 키워드 인자로 호출할 수 없음
- 하지만 이 코드에서는 실제로 키워드 인자로 사용됨 (주석 참고)
- 실제 사용: `replay_buffer.insert(observation=..., action=...)`

**타입 변환 로직:**

```python
if isinstance(reward, (float, int)):
    reward = np.array(reward)
if isinstance(done, bool):
    done = np.array(done)
if isinstance(action, int):
    action = np.array(action, dtype=np.int64)
```

이 부분은 편의를 위한 것입니다:
- `reward`를 `5.0` (scalar)으로 전달해도 `np.array(5.0)` (ndarray)로 변환
- `done`을 `True`로 전달해도 `np.array(True)`로 변환
- `action`을 `2`로 전달해도 `np.array(2, dtype=np.int64)`로 변환

**Lazy Initialization:**

```python
if self.observations is None:
    self.observations = np.empty(
        (self.max_size, *observation.shape), dtype=observation.dtype
    )
    self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
    # ... 다른 버퍼들도 동일
```

**`*observation.shape`의 의미:**
- `observation.shape = (4,)`이면
- `(self.max_size, *observation.shape)` = `(1000000, 4)`
- Unpacking operator로 shape를 확장

**왜 Lazy Initialization?**
- 처음에는 observation의 차원을 모름
- CartPole: `(4,)` - 저차원 벡터
- Atari: `(4, 84, 84)` - 이미지
- 첫 데이터를 보고 나서야 shape를 알 수 있음

**Circular Buffer 저장:**

```python
self.observations[self.size % self.max_size] = observation
self.actions[self.size % self.max_size] = action
# ...
self.size += 1
```

**`self.size % self.max_size`의 작동 원리:**

```
max_size = 5일 때:

size=0: 0 % 5 = 0 → index 0에 저장
size=1: 1 % 5 = 1 → index 1에 저장
size=2: 2 % 5 = 2 → index 2에 저장
size=3: 3 % 5 = 3 → index 3에 저장
size=4: 4 % 5 = 4 → index 4에 저장
size=5: 5 % 5 = 0 → index 0에 덮어쓰기 (가장 오래된 데이터)
size=6: 6 % 5 = 1 → index 1에 덮어쓰기
...
```

이것을 **circular buffer** (원형 버퍼)라고 합니다.

#### sample() 메서드 - 배치 샘플링

```python
def sample(self, batch_size):
    rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
    return {
        "observations": self.observations[rand_indices],
        "actions": self.actions[rand_indices],
        "rewards": self.rewards[rand_indices],
        "next_observations": self.next_observations[rand_indices],
        "dones": self.dones[rand_indices],
    }
```

**샘플링 로직 분석:**

```python
rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
```

이 한 줄은 두 부분으로 나뉩니다:

1. `np.random.randint(0, self.size, size=(batch_size,))`
   - 0부터 `self.size-1` 사이의 랜덤 정수 `batch_size`개 생성
   - 예: `batch_size=3, size=100` → `[23, 67, 89]`

2. `% self.max_size`
   - 왜 필요한가? `self.size > self.max_size`일 수 있기 때문
   - 예: `size=1500, max_size=1000`일 때
   - `rand_indices = [234, 1234, 567]`
   - `% 1000` → `[234, 234, 567]` (유효한 인덱스)

**uniform random sampling의 의미:**

모든 transition이 동등한 확률로 샘플링됩니다:
- 최근 경험이든 과거 경험이든 상관없음
- 이것이 i.i.d. (independent and identically distributed) 가정을 만족
- SGD (확률적 경사 하강법)의 요구사항

**왜 Dictionary로 반환하는가?**

```python
return {
    "observations": ...,
    "actions": ...,
    ...
}
```

- 명시적이고 가독성이 좋음
- 순서를 기억할 필요 없음
- PyTorch tensor로 변환하기 쉬움:
  ```python
  batch = replay_buffer.sample(32)
  batch = ptu.from_numpy(batch)
  # batch["observations"]는 torch.Tensor가 됨
  ```

---

### 4.2 Memory-Efficient Replay Buffer (메모리 효율적 리플레이 버퍼)

**파일 위치**: `gcb6206/infrastructure/replay_buffer.py`

**목적**: Frame stacking을 사용하는 Atari 환경에서 메모리 절약

#### 왜 필요한가?

Atari 환경에서는 **frame stacking**을 사용합니다:
- 단일 프레임: 84×84 grayscale 이미지
- Stacked observation: 최근 4 프레임을 쌓음 → `(4, 84, 84)`

**문제점 (Regular Replay Buffer 사용 시):**

```python
# 한 transition 저장
observation:      (4, 84, 84)  # 4 프레임
next_observation: (4, 84, 84)  # 4 프레임 (3개는 중복!)
```

실제로는:
```
observation:      [frame1, frame2, frame3, frame4]
next_observation: [frame2, frame3, frame4, frame5]
                   ^^^^^^^^^^^^^^^^^^^^^^ 중복!
```

**메모리 낭비 계산:**

- 한 transition: `2 * 4 * 84 * 84 = 56,448 bytes ≈ 55 KB`
- 1M transitions: `55 KB * 1M = 55 GB!`

하지만 실제 고유 프레임 수는 훨씬 적습니다.

#### MemoryEfficientReplayBuffer의 아이디어

**핵심 아이디어**: 프레임을 개별적으로 저장하고, 인덱스로 참조

```
Regular Buffer:
  transition 0: [f0, f1, f2, f3] → [f1, f2, f3, f4]
  transition 1: [f1, f2, f3, f4] → [f2, f3, f4, f5]
  transition 2: [f2, f3, f4, f5] → [f3, f4, f5, f6]
  → 프레임 중복 저장

Memory-Efficient Buffer:
  framebuffer: [f0, f1, f2, f3, f4, f5, f6, ...]
  transition 0: indices [0,1,2,3] → [1,2,3,4]
  transition 1: indices [1,2,3,4] → [2,3,4,5]
  transition 2: indices [2,3,4,5] → [3,4,5,6]
  → 각 프레임 한 번만 저장!
```

#### 클래스 구조

```python
class MemoryEfficientReplayBuffer:
    def __init__(self, frame_history_len: int, capacity=1000000):
        self.max_size = capacity
        self.max_framebuffer_size = 2 * capacity
        self.frame_history_len = frame_history_len  # 보통 4

        # Transition 데이터
        self.actions = None
        self.rewards = None
        self.dones = None

        # 프레임 관련
        self.framebuffer = None  # 실제 프레임 저장소
        self.observation_framebuffer_idcs = None  # 인덱스 참조
        self.next_observation_framebuffer_idcs = None

        # 현재 trajectory 추적
        self.current_trajectory_begin = None
        self.current_trajectory_framebuffer_begin = None
        self.framebuffer_idx = None
        self.recent_observation_framebuffer_idcs = None
```

**`max_framebuffer_size = 2 * capacity`의 이유:**

```python
# Technically we need max_size*2 to support both obs and next_obs.
# Otherwise we'll end up overwriting old observations' frames, but the
# corresponding next_observation_framebuffer_idcs will still point to the old frames.
```

예시:
- `max_size = 1M` transitions
- 평균적으로 한 transition은 1-2개의 새 프레임 추가
- 최악의 경우 (모든 episode가 길이 1): 2M 프레임 필요
- 안전하게 `2 * capacity` 할당

실제로는 OS가 **page out**하므로 실제 메모리 사용량은 적습니다.

#### on_reset() 메서드 - Episode 시작

```python
def on_reset(
    self,
    /,
    observation: np.ndarray,  # 단일 프레임! (H, W)
):
    assert observation.ndim == 2, "Single-frame observation should have dimensions (H, W)"
    assert observation.dtype == np.uint8, "Observation should be uint8 (0-255)"
```

**중요**: `observation`은 **단일 프레임** `(84, 84)`입니다!
- Regular buffer: 스택된 observation `(4, 84, 84)`
- Memory-efficient buffer: 마지막 프레임만 `(84, 84)`

**초기화 로직:**

```python
if self.observation_framebuffer_idcs is None:
    self.observation_framebuffer_idcs = np.empty(
        (self.max_size, self.frame_history_len), dtype=np.int64
    )
    self.next_observation_framebuffer_idcs = np.empty(
        (self.max_size, self.frame_history_len), dtype=np.int64
    )
    self.framebuffer = np.empty(
        (self.max_framebuffer_size, *observation.shape), dtype=observation.dtype
    )
    self.framebuffer_idx = 0
    self.current_trajectory_begin = 0
    self.current_trajectory_framebuffer_begin = 0
```

**각 버퍼의 shape:**

- `observation_framebuffer_idcs`: `(1M, 4)` - 각 transition의 observation을 구성하는 4개 프레임의 인덱스
- `next_observation_framebuffer_idcs`: `(1M, 4)` - next_observation의 인덱스
- `framebuffer`: `(2M, 84, 84)` - 실제 프레임 데이터

**메모리 절약 계산:**

Regular buffer:
```
1M transitions * 2 observations * 4 frames * 84 * 84 = 56 GB
```

Memory-efficient buffer:
```
프레임: 2M * 84 * 84 = 14 GB
인덱스: 1M * 2 * 4 * 8 bytes = 64 MB
합계: ~14 GB
```

**약 4배 메모리 절약!**

**첫 프레임 삽입:**

```python
self.current_trajectory_begin = self.size  # transition 인덱스 기록
self.current_trajectory_framebuffer_begin = self._insert_frame(observation)  # 프레임 삽입
self.recent_observation_framebuffer_idcs = self._compute_frame_history_idcs(
    self.current_trajectory_framebuffer_begin,
    self.current_trajectory_framebuffer_begin,
)
```

#### _insert_frame() 메서드

```python
def _insert_frame(self, frame: np.ndarray) -> int:
    assert frame.ndim == 2, "Single-frame observation should have dimensions (H, W)"
    assert frame.dtype == np.uint8, "Observation should be uint8 (0-255)"

    self.framebuffer[self.framebuffer_idx] = frame
    frame_idx = self.framebuffer_idx
    self.framebuffer_idx = self.framebuffer_idx + 1

    return frame_idx  # 저장된 위치 반환
```

- 프레임을 framebuffer에 저장
- 현재 인덱스 반환 (나중에 참조하기 위해)
- `framebuffer_idx` 증가 (다음 프레임을 위해)

#### _compute_frame_history_idcs() 메서드

```python
def _compute_frame_history_idcs(
    self, latest_framebuffer_idx: int, trajectory_begin_framebuffer_idx: int
) -> np.ndarray:
    return np.maximum(
        np.arange(-self.frame_history_len + 1, 1) + latest_framebuffer_idx,
        trajectory_begin_framebuffer_idx,
    )
```

**이 함수의 목적**: 4개 프레임 인덱스 계산

**예시로 이해하기:**

```python
frame_history_len = 4
latest_framebuffer_idx = 10
trajectory_begin_framebuffer_idx = 8

# Step 1: np.arange(-3, 1) = [-3, -2, -1, 0]
# Step 2: + 10 = [7, 8, 9, 10]
# Step 3: np.maximum([7,8,9,10], 8) = [8,8,9,10]
```

**왜 maximum이 필요한가?**

Episode 초반에는 4개 프레임이 없을 수 있습니다:

```
Episode 시작:
  프레임 0: [f8, f8, f8, f8]  # 첫 프레임을 4번 반복
  프레임 1: [f8, f8, f8, f9]  # 2개 프레임만 있음
  프레임 2: [f8, f8, f9, f10] # 3개 프레임
  프레임 3: [f8, f9, f10, f11] # 4개 프레임 (정상)
```

`maximum()`은 trajectory 시작 이전의 인덱스를 방지합니다.

#### insert() 메서드

```python
def insert(
    self,
    /,
    action: np.ndarray,
    reward: np.ndarray,
    next_observation: np.ndarray,  # 단일 프레임!
    done: np.ndarray,
):
```

**주의**: `next_observation`도 단일 프레임 `(84, 84)`입니다!

**저장 과정:**

```python
# 1. 현재 observation의 인덱스 저장
self.observation_framebuffer_idcs[self.size % self.max_size] = \
    self.recent_observation_framebuffer_idcs

# 2. action, reward, done 저장
self.actions[self.size % self.max_size] = action
self.rewards[self.size % self.max_size] = reward
self.dones[self.size % self.max_size] = done

# 3. next_observation 프레임 추가
next_frame_idx = self._insert_frame(next_observation)

# 4. next_observation의 인덱스 계산
next_framebuffer_idcs = self._compute_frame_history_idcs(
    next_frame_idx, self.current_trajectory_framebuffer_begin
)
self.next_observation_framebuffer_idcs[self.size % self.max_size] = \
    next_framebuffer_idcs

# 5. 다음 step을 위한 준비
self.size += 1
self.recent_observation_framebuffer_idcs = next_framebuffer_idcs
```

**상태 전이 예시:**

```
Step 0 (reset):
  framebuffer: [f0, ?, ?, ...]
  recent_observation_framebuffer_idcs: [0, 0, 0, 0]

Step 1 (insert):
  action=2, reward=1.0, next_observation=f1, done=False

  framebuffer: [f0, f1, ?, ...]
  transition 0:
    observation_framebuffer_idcs: [0, 0, 0, 0]
    action: 2
    reward: 1.0
    next_observation_framebuffer_idcs: [0, 0, 0, 1]
    done: False

  recent_observation_framebuffer_idcs: [0, 0, 0, 1]

Step 2 (insert):
  action=1, reward=2.0, next_observation=f2, done=False

  framebuffer: [f0, f1, f2, ...]
  transition 1:
    observation_framebuffer_idcs: [0, 0, 0, 1]
    action: 1
    reward: 2.0
    next_observation_framebuffer_idcs: [0, 0, 1, 2]
    done: False
```

#### sample() 메서드

```python
def sample(self, batch_size):
    rand_indices = (
        np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
    )

    observation_framebuffer_idcs = (
        self.observation_framebuffer_idcs[rand_indices] % self.max_framebuffer_size
    )
    next_observation_framebuffer_idcs = (
        self.next_observation_framebuffer_idcs[rand_indices] % self.max_framebuffer_size
    )

    return {
        "observations": self.framebuffer[observation_framebuffer_idcs],
        "actions": self.actions[rand_indices],
        "rewards": self.rewards[rand_indices],
        "next_observations": self.framebuffer[next_observation_framebuffer_idcs],
        "dones": self.dones[rand_indices],
    }
```

**고급 인덱싱 (Advanced Indexing):**

```python
observation_framebuffer_idcs.shape = (batch_size, 4)
# 예: [[8,8,9,10], [15,16,17,18], [20,21,22,23]]

self.framebuffer[observation_framebuffer_idcs]
# Shape: (batch_size, 4, 84, 84)
```

NumPy는 이런 multi-dimensional indexing을 지원합니다:
- 각 행은 독립적으로 인덱싱됨
- 결과는 자동으로 stacking됨

**반환 값:**

```python
{
    "observations": (batch_size, 4, 84, 84),  # Stacked frames
    "actions": (batch_size,),
    "rewards": (batch_size,),
    "next_observations": (batch_size, 4, 84, 84),
    "dones": (batch_size,),
}
```

---
