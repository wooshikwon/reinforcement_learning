# 강화학습 HW1 완전 가이드: Behavioral Cloning과 DAgger

이 문서는 `run_hw1.py`의 전체 파이프라인을 처음부터 끝까지 상세하게 설명하는 강화학습 학습자를 위한 완전한 가이드입니다. 코드의 각 줄, 각 메서드의 동작 원리와 의도, 그리고 강화학습의 핵심 개념들을 함께 다룹니다.

> **📌 중요 안내:**
> 이 가이드는 TODO로 표시된 미구현 부분들에 대해 **구현 예시**를 제공합니다.
> **[구현 예시 - 학습 참고용]**으로 표시된 코드는 실제 코드베이스에는 구현되지 않은 부분이며, 학습과 이해를 돕기 위한 참고 자료입니다.

## 목차
1. [전체 구조 개요](#1-전체-구조-개요)
2. [파이프라인 시작: main() 함수](#2-파이프라인-시작-main-함수)
3. [실험 설정: run_bc() 함수](#3-실험-설정-run_bc-함수)
4. [학습 관리자: BCTrainer 초기화](#4-학습-관리자-bctrainer-초기화)
5. [에이전트: BCAgent 초기화](#5-에이전트-bcagent-초기화)
6. [정책 네트워크: MLPPolicySL 초기화](#6-정책-네트워크-mlppolicysl-초기화)
7. [신경망 구축: build_mlp() 함수](#7-신경망-구축-build_mlp-함수)
8. [메인 학습 루프: run_training_loop()](#8-메인-학습-루프-run_training_loop)
9. [데이터 수집: collect_training_trajectories()](#9-데이터-수집-collect_training_trajectories)
10. [환경 상호작용: rollout_trajectory()](#10-환경-상호작용-rollout_trajectory)
11. [전문가 라벨링: do_relabel_with_expert()](#11-전문가-라벨링-do_relabel_with_expert)
12. [경험 저장: ReplayBuffer.add_rollouts()](#12-경험-저장-replaybufferadd_rollouts)
13. [에이전트 학습: train_agent()](#13-에이전트-학습-train_agent)
14. [데이터 샘플링: ReplayBuffer.sample_random_data()](#14-데이터-샘플링-replaybuffersample_random_data)
15. [정책 업데이트: MLPPolicySL 핵심 메서드들](#15-정책-업데이트-mlppolicysl-핵심-메서드들)

---

## 1. 전체 구조 개요

### 1.1 이 코드가 하는 일

이 코드베이스는 **Behavioral Cloning (BC)**과 **DAgger (Dataset Aggregation)** 두 가지 모방 학습(Imitation Learning) 알고리즘을 구현합니다.

**Behavioral Cloning이란?**
- 전문가(expert)가 플레이한 데이터를 보고, 그대로 따라하도록 학습하는 방법이야
- 마치 운전을 배울 때 옆에서 숙련된 운전자가 운전하는 걸 보고 따라하는 것과 같아
- 수학적으로는: 전문가의 (상태, 행동) 쌍들을 학습 데이터로 사용해서 supervised learning을 하는 거지

**DAgger란?**
- BC의 문제점을 해결하기 위한 방법이야
- BC는 전문가가 방문한 상태에서만 학습하는데, 학습 중인 에이전트가 실수하면 전문가가 가보지 않은 상태에 도달하게 돼
- 그럼 뭘 해야 할지 모르게 되는 거지 (이걸 **distributional shift** 문제라고 해)
- DAgger는 이렇게 해결해: "내가 실수해서 도달한 이 상태에서, 전문가라면 뭘 했을까?" 하고 전문가에게 물어봐서 그것도 학습해

### 1.2 코드 구조

```
run_hw1.py (메인 스크립트)
    ↓
BCTrainer (학습 루프 관리)
    ↓
BCAgent (에이전트)
    ├── MLPPolicySL (정책 네트워크)
    └── ReplayBuffer (경험 저장소)
```

**핵심 클래스들:**
- `BCTrainer`: 전체 학습 과정을 orchestrate하는 지휘자 역할
- `BCAgent`: 행동을 결정하고 학습하는 에이전트
- `MLPPolicySL`: 관찰을 받아서 행동을 출력하는 신경망 (정책)
- `ReplayBuffer`: 과거 경험들을 저장하는 메모리

---

## 2. 파이프라인 시작: main() 함수

`run_hw1.py`의 66-159번 줄

### 2.1 커맨드라인 인자 파싱

```python
parser = argparse.ArgumentParser()
parser.add_argument("--expert_policy_file", "-epf", type=str, required=True)
parser.add_argument("--expert_data", "-ed", type=str, required=True)
parser.add_argument("--env_name", "-env", type=str, required=True)
```

**argparse란?**
- Python의 표준 라이브러리로, 커맨드라인에서 실행할 때 인자들을 받는 도구야
- 예를 들어: `python run_hw1.py --expert_data data.pkl --env_name Ant-v4`
- 이렇게 실행하면 `args.expert_data`는 "data.pkl"이 되는 거지

**주요 인자들:**
- `expert_policy_file`: 이미 학습된 전문가 정책이 저장된 파일 경로
- `expert_data`: 전문가가 플레이한 데이터 (pickle 파일)
- `env_name`: 어떤 환경에서 학습할지 (예: "Ant-v4", "Walker2d-v4")
- `do_dagger`: DAgger를 사용할지 여부 (이건 flag라서 있으면 True)
- `n_iter`: 몇 번 반복할지

### 2.2 BC vs DAgger 구분

```python
if args.do_dagger:
    logdir_prefix = "q2_"
    assert args.n_iter > 1, "DAgger needs more than 1 iteration..."
else:
    logdir_prefix = "q1_"
    assert args.n_iter == 1, "Vanilla behavioral cloning collects expert data just once"
```

**왜 이렇게 구분할까?**

**BC (Behavioral Cloning):**
- `n_iter == 1`: 딱 한 번만 돌아
- 전문가 데이터 로드 → 그걸로 학습 → 끝
- 새로운 데이터 수집 안 함

**DAgger:**
- `n_iter > 1`: 여러 번 반복해야 해
- Iteration 0: 전문가 데이터로 시작
- Iteration 1~N:
  1. 현재 정책으로 데이터 수집
  2. 전문가에게 "이 상태에서 뭘 해야 돼?" 물어봄
  3. 그 답변으로 다시 학습
  4. 반복

**assert문이란?**
- 조건이 False면 에러를 발생시키는 Python 문법
- 여기서는 잘못된 설정을 미리 잡기 위한 안전장치야
- DAgger인데 n_iter=1이면? "어? 한 번만 돌면 DAgger가 아니잖아!" 하고 에러 발생

### 2.3 로그 디렉토리 생성

```python
data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")
if not os.path.exists(data_path):
    os.makedirs(data_path)

logdir = logdir_prefix + args.exp_name + "_" + args.env_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
logdir = os.path.join(data_path, logdir)
```

**os.path 함수들 설명:**
- `os.path.realpath(__file__)`: 현재 실행 중인 파일의 절대 경로를 가져와
- `os.path.dirname()`: 경로에서 디렉토리 부분만 추출
- `os.path.join()`: 경로들을 OS에 맞게 합쳐줘 (Windows는 `\`, Linux/Mac은 `/`)

**time.strftime() 설명:**
- 현재 시간을 문자열로 포맷팅해
- `"%d-%m-%Y_%H-%M-%S"`: "일-월-년_시-분-초" 형식
- 예: "04-10-2025_14-30-45"
- 왜? 같은 실험을 여러 번 돌려도 로그가 섞이지 않게!

**결과 예시:**
```
logdir = "data/q2_my_experiment_Ant-v4_04-10-2025_14-30-45"
```

---

## 3. 실험 설정: run_bc() 함수

`run_hw1.py`의 16-63번 줄

### 3.1 Agent 파라미터 설정

```python
agent_params = {
    "n_layers": params["n_layers"],
    "size": params["size"],
    "learning_rate": params["learning_rate"],
    "max_replay_buffer_size": params["max_replay_buffer_size"],
}
params["agent_class"] = BCAgent
params["agent_params"] = agent_params
```

**여기서 뭘 하는 거야?**
- 신경망 구조와 학습 설정을 딕셔너리로 정리해
- 나중에 BCAgent를 만들 때 이 파라미터들을 전달할 거야

**각 파라미터 의미:**
- `n_layers`: 신경망의 은닉층(hidden layer) 개수
  - 예: n_layers=2면 input → hidden1 → hidden2 → output
- `size`: 각 은닉층의 뉴런(neuron) 개수
  - 예: size=64면 각 은닉층에 64개 뉴런
- `learning_rate`: 학습률 (얼마나 크게 파라미터를 업데이트할지)
  - 예: 0.001이면 천천히, 0.1이면 빠르게 학습
- `max_replay_buffer_size`: 버퍼에 최대 몇 개의 transition을 저장할지
  - 예: 1000000이면 백만 개까지 저장

### 3.2 환경 파라미터 설정

```python
params["env_kwargs"] = MJ_ENV_KWARGS[params["env_name"]]
```

**MJ_ENV_KWARGS는 뭐야?**
```python
# utils.py에 정의됨
MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]
MJ_ENV_KWARGS = {name: {"render_mode": "rgb_array"} for name in MJ_ENV_NAMES}
MJ_ENV_KWARGS["Ant-v4"]["use_contact_forces"] = True
```

- **딕셔너리 컴프리헨션**: `{key: value for item in list}` 문법
- 각 환경 이름을 key로, 환경 설정을 value로 하는 딕셔너리 생성
- `render_mode="rgb_array"`: 환경을 이미지(RGB 배열)로 렌더링
- Ant-v4는 추가로 `use_contact_forces=True`: 접촉력 정보 사용

### 3.3 Expert Policy 로드

```python
print("Loading expert policy from...", params["expert_policy_file"])
loaded_expert_policy = LoadedGaussianPolicy(params["expert_policy_file"])
print("Done restoring expert policy...")
```

**LoadedGaussianPolicy란?**
- 이미 학습된 전문가 정책을 파일에서 불러오는 클래스야
- pickle 파일에서 신경망 가중치를 읽어서 복원해
- **Gaussian Policy**: 정규분포를 사용하는 확률적 정책
  - 관찰 s를 받아서 → 행동 분포 N(μ, σ²)를 출력
  - 여기서 행동을 샘플링해

**왜 전문가 정책이 필요해?**
- BC: 전문가 데이터를 제공하기 위해
- DAgger: 새로 수집한 상태에서 정답 행동을 라벨링하기 위해

### 3.4 학습 시작

```python
trainer = BCTrainer(params)
trainer.run_training_loop(
    n_iter=params["n_iter"],
    initial_expertdata=params["expert_data"],
    collect_policy=trainer.agent.actor,
    eval_policy=trainer.agent.actor,
    relabel_with_expert=params["do_dagger"],
    expert_policy=loaded_expert_policy,
)
```

**인자 설명:**
- `n_iter`: 총 몇 iteration 돌릴지
- `initial_expertdata`: 첫 iteration에 사용할 전문가 데이터 파일 경로
- `collect_policy`: 데이터 수집에 사용할 정책 (우리가 학습 중인 정책)
- `eval_policy`: 평가에 사용할 정책 (똑같이 우리 정책)
- `relabel_with_expert`: DAgger 모드인지 (True면 전문가가 다시 라벨링)
- `expert_policy`: 라벨링에 사용할 전문가 정책

**trainer.agent.actor는 뭐야?**
- `trainer`: BCTrainer 인스턴스
- `trainer.agent`: BCAgent 인스턴스 (나중에 만들어짐)
- `trainer.agent.actor`: MLPPolicySL 인스턴스 (실제 정책 신경망)

---

## 4. 학습 관리자: BCTrainer 초기화

`bc_trainer.py`의 52-106번 줄

### 4.1 기본 설정

```python
def __init__(self, params):
    self.params = params
    self.logger = Logger(self.params["logdir"])
```

**__init__은 뭐야?**
- Python의 특별 메서드(magic method)야
- 클래스의 인스턴스를 만들 때 자동으로 호출돼
- `trainer = BCTrainer(params)` 하면 이 함수가 실행되는 거지

**Logger는 뭐하는 거야?**
- TensorBoard에 학습 과정을 기록하는 클래스
- 손실(loss), 보상(reward), 비디오 등을 저장
- 나중에 TensorBoard로 시각화해서 볼 수 있어

### 4.2 Random Seed 설정

```python
seed = self.params["seed"]
np.random.seed(seed)
torch.manual_seed(seed)
ptu.init_gpu(use_gpu=not self.params["no_gpu"], gpu_id=self.params["which_gpu"])
```

**왜 seed를 설정해?**
- **재현성(Reproducibility)**: 같은 seed면 같은 랜덤 결과가 나와
- 딥러닝은 랜덤 요소가 많아:
  - 가중치 초기화
  - 데이터 셔플링
  - dropout 등
- Seed를 고정하면 실험을 정확히 재현할 수 있어

**각 seed 함수:**
- `np.random.seed(seed)`: NumPy의 랜덤 생성기 초기화
- `torch.manual_seed(seed)`: PyTorch의 랜덤 생성기 초기화
- `ptu.init_gpu()`: GPU 설정 및 CUDA seed 초기화

**ptu.init_gpu() 자세히:**
```python
def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")
```

- `torch.cuda.is_available()`: CUDA(NVIDIA GPU) 사용 가능한지 체크
- `torch.device()`: 연산을 어디서 할지 (GPU or CPU)
- global device: 전역 변수로 설정해서 어디서든 사용 가능

### 4.3 환경(Environment) 설정

```python
self.env = gym.make(self.params["env_name"], **self.params["env_kwargs"])
self.env.reset(seed=seed)
```

**gym.make()란?**
- OpenAI Gym/Gymnasium의 함수
- 환경 이름을 주면 그 환경을 만들어줘
- `**env_kwargs`: 딕셔너리를 keyword arguments로 언팩
  - `gym.make("Ant-v4", render_mode="rgb_array", use_contact_forces=True)` 와 동일

**env.reset()이란?**
- 환경을 초기 상태로 리셋
- 첫 observation과 info를 반환
- seed를 주면 환경의 랜덤성도 고정돼

### 4.4 환경 정보 추출

```python
self.params["ep_len"] = self.params["ep_len"] or self.env.spec.max_episode_steps
```

**`or` 연산자 트릭:**
- `A or B`: A가 False(또는 None, 0)면 B를 반환
- `params["ep_len"]`이 None이면 → `env.spec.max_episode_steps` 사용
- 사용자가 지정 안 했으면 환경 기본값 사용

```python
discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
```

**isinstance()란?**
- 객체가 특정 클래스의 인스턴스인지 확인
- `gym.spaces.Discrete`: 이산 행동 공간 (0, 1, 2, ... 같은 정수)
- `gym.spaces.Box`: 연속 행동 공간 (실수 벡터)

**이산 vs 연속:**
- 이산: 체스 (64칸 중 하나 선택)
- 연속: 로봇 제어 (관절 각도를 연속적으로 조절)

```python
ob_dim = self.env.observation_space.shape[0]
ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
```

**shape[0]은 뭐야?**
- NumPy 배열의 첫 번째 차원 크기
- 예: observation이 17차원 벡터면 `shape = (17,)` → `shape[0] = 17`

**ac_dim 계산:**
- 이산: `action_space.n` (행동 개수, 예: 4개 방향)
- 연속: `action_space.shape[0]` (행동 벡터 차원, 예: 8개 관절)

### 4.5 FPS 설정

```python
if "model" in dir(self.env):
    self.fps = 1 / self.env.model.opt.timestep
else:
    self.fps = self.env.env.metadata["render_fps"]
```

**dir() 함수:**
- 객체가 가진 속성(attribute)과 메서드 리스트 반환
- `"model" in dir(self.env)`: env에 model 속성이 있는지 확인

**왜 이렇게 해?**
- MuJoCo 환경: `env.model.opt.timestep`에서 시뮬레이션 timestep 가져와 → fps 계산
- 다른 환경: metadata에서 직접 fps 가져와
- 비디오 저장할 때 올바른 fps 필요!

### 4.6 Agent 생성

```python
agent_class = self.params["agent_class"]  # BCAgent
self.agent = agent_class(self.env, self.params["agent_params"])
```

**동적 클래스 호출:**
- `agent_class`는 변수에 저장된 클래스 (BCAgent)
- `agent_class(...)`: 그 클래스의 인스턴스 생성
- 왜 이렇게? 나중에 다른 agent도 쉽게 사용하려고 (확장성)

---

## 5. 에이전트: BCAgent 초기화

`bc_agent.py`의 25-43번 줄

### 5.1 클래스 구조

```python
class BCAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(BCAgent, self).__init__()
```

**상속(Inheritance):**
- `class BCAgent(BaseAgent)`: BCAgent는 BaseAgent를 상속
- BaseAgent는 추상 클래스(abstract class)로 인터페이스 정의
- `super().__init__()`: 부모 클래스의 초기화 호출

**BaseAgent 살펴보기:**
```python
class BaseAgent(object):
    def train(self) -> dict:
        raise NotImplementedError
    def add_to_replay_buffer(self, trajs):
        raise NotImplementedError
    def sample(self, batch_size):
        raise NotImplementedError
```

- `NotImplementedError`: 자식 클래스에서 구현하라는 뜻
- 이게 **추상 메서드 패턴**이야

### 5.2 정책(Actor) 생성

```python
self.actor = MLPPolicySL(
    self.agent_params["ac_dim"],
    self.agent_params["ob_dim"],
    self.agent_params["n_layers"],
    self.agent_params["size"],
    discrete=self.agent_params["discrete"],
    learning_rate=self.agent_params["learning_rate"],
)
```

**MLPPolicySL 인자:**
- `ac_dim`: 행동 차원 (예: 8개 관절)
- `ob_dim`: 관찰 차원 (예: 17차원 벡터)
- `n_layers`: 은닉층 개수 (예: 2)
- `size`: 각 층 크기 (예: 64)
- `discrete`: 이산/연속 (여기선 False, 연속)
- `learning_rate`: 학습률 (예: 0.001)

**왜 actor라고 불러?**
- Actor-Critic 구조에서 유래
- Actor: 행동을 결정하는 정책
- Critic: 가치를 평가하는 함수
- BC는 actor만 있고 critic은 없어

### 5.3 Replay Buffer 생성

```python
self.replay_buffer = ReplayBuffer(self.agent_params["max_replay_buffer_size"])
```

**Replay Buffer란?**
- 과거 경험들을 저장하는 메모리
- (observation, action, reward, next_observation, done) 튜플들을 저장
- 나중에 무작위로 샘플링해서 학습에 사용

**왜 필요해?**
- **Off-policy 학습**: 과거 데이터 재사용
- **Sample efficiency**: 데이터를 여러 번 사용해 학습 효율↑
- **Breaking correlation**: 시간적 상관관계 제거

### 5.4 BCAgent 메서드들

```python
def train(self, ob_no, ac_na):
    log = self.actor.update(ob_no, ac_na)
    return log
```

**변수명 규칙:**
- `ob_no`: observation, batch_size × ob_dim
- `ac_na`: action, batch_size × ac_dim
- `_no`, `_na`는 차원 표시 관례 (n=batch, o=obs, a=action)

```python
def add_to_replay_buffer(self, trajs):
    self.replay_buffer.add_rollouts(trajs)
```

**단순 위임(delegation):**
- BCAgent가 직접 하지 않고 replay_buffer에게 넘김
- **관심사 분리(Separation of Concerns)** 원칙

```python
def sample(self, batch_size):
    return self.replay_buffer.sample_random_data(batch_size)
```

**샘플링 메서드:**
- 버퍼에서 batch_size만큼 무작위 샘플 추출
- 5개 배열 반환: obs, actions, rewards, next_obs, terminals

---

## 6. 정책 네트워크: MLPPolicySL 초기화

`MLP_policy.py`의 48-99번 줄

### 6.1 클래스 정의

```python
class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
```

**다중 상속(Multiple Inheritance):**
- `BasePolicy`: 정책 인터페이스
- `nn.Module`: PyTorch 신경망 기본 클래스
- `metaclass=abc.ABCMeta`: 추상 클래스 메타클래스

**nn.Module이란?**
- PyTorch의 모든 신경망은 nn.Module 상속
- 제공 기능:
  - `.parameters()`: 학습 가능한 파라미터 관리
  - `.to(device)`: GPU/CPU 이동
  - `.train()`, `.eval()`: 학습/평가 모드 전환
  - forward() 자동 호출

### 6.2 초기화 파라미터

```python
def __init__(self, ac_dim, ob_dim, n_layers, size, discrete=False,
             learning_rate=1e-4, training=True, nn_baseline=False, **kwargs):
    super().__init__(**kwargs)

    self.ac_dim = ac_dim
    self.ob_dim = ob_dim
    self.n_layers = n_layers
    self.discrete = discrete
    self.size = size
    self.learning_rate = learning_rate
```

**super().__init__(**kwargs):**
- 부모 클래스들의 초기화 호출
- `**kwargs`: 추가 keyword arguments를 받아서 전달

### 6.3 연속 행동 공간 설정

```python
if self.discrete:
    # ... 이산 행동 공간 (우리는 안 씀)
else:
    self.logits_na = None
    self.mean_net = ptu.build_mlp(
        input_size=self.ob_dim,
        output_size=self.ac_dim,
        n_layers=self.n_layers,
        size=self.size,
    )
    self.mean_net.to(ptu.device)
```

**mean_net이란?**
- 관찰 → 행동 평균을 출력하는 신경망
- 입력: ob_dim 차원 벡터 (예: 17차원 상태)
- 출력: ac_dim 차원 벡터 (예: 8차원 행동 평균)

**ptu.build_mlp():**
- MLP(Multi-Layer Perceptron) 생성 함수
- 나중에 자세히 볼게

**.to(ptu.device):**
- 신경망을 GPU 또는 CPU로 이동
- ptu.device는 초기화 때 설정한 전역 변수

### 6.4 Log Standard Deviation

```python
self.logstd = nn.Parameter(
    torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
)
self.logstd.to(ptu.device)
```

**nn.Parameter란?**
- PyTorch에서 학습 가능한 파라미터를 나타내는 특별한 Tensor
- 신경망에 등록되어 `.parameters()`에 포함됨
- optimizer가 자동으로 업데이트함

**왜 log std를 사용해?**
- std는 항상 양수여야 해 (표준편차니까)
- logstd는 unbounded: -∞ ~ +∞ 범위
- `std = exp(logstd)` 하면:
  - logstd가 음수여도 std는 양수
  - logstd = 0 → std = 1
  - logstd = -2 → std ≈ 0.135
  - logstd = 2 → std ≈ 7.39
- **수치적 안정성**: log space에서 학습하면 gradient flow가 좋아

**초기값이 0인 이유:**
- logstd = 0 → std = exp(0) = 1
- 초기에 적당한 exploration 제공

### 6.5 Optimizer 설정

```python
self.optimizer = optim.Adam(
    itertools.chain([self.logstd], self.mean_net.parameters()),
    self.learning_rate,
)
```

**itertools.chain():**
- 여러 iterable을 하나로 연결
- `[self.logstd]`: 리스트에 logstd 파라미터
- `self.mean_net.parameters()`: mean_net의 모든 파라미터
- 결과: 모든 학습 파라미터를 하나의 iterator로

**Adam optimizer:**
- Adaptive Moment Estimation
- 각 파라미터마다 적응적 학습률 사용
- 왜 Adam?
  - SGD보다 빠른 수렴
  - Learning rate에 덜 민감
  - 모멘텀과 RMSProp의 장점 결합

**작동 원리 (간단히):**
```
m = β₁ · m + (1-β₁) · gradient     # 1차 모멘트 (평균)
v = β₂ · v + (1-β₂) · gradient²    # 2차 모멘트 (분산)
θ = θ - lr · m / (√v + ε)          # 파라미터 업데이트
```

---

## 7. 신경망 구축: build_mlp() 함수

`pytorch_util.py`의 25-59번 줄

### 7.1 함수 시그니처

```python
def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation: Activation = "tanh",
    output_activation: Activation = "identity",
) -> nn.Module:
```

**Type hints:**
- `input_size: int`: input_size는 int 타입이어야 함
- `-> nn.Module`: 반환값은 nn.Module 타입
- Python 3.5+의 기능, 실행에는 영향 없지만 코드 가독성↑

### 7.2 활성화 함수 처리

```python
if isinstance(activation, str):
    activation = _str_to_activation[activation]
if isinstance(output_activation, str):
    output_activation = _str_to_activation[output_activation]
```

**_str_to_activation 딕셔너리:**
```python
_str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}
```

**활성화 함수들:**
- **ReLU**: max(0, x) - 가장 많이 쓰임
- **Tanh**: 쌍곡탄젠트, 범위 [-1, 1]
- **LeakyReLU**: max(0.01x, x) - ReLU의 dying 문제 해결
- **Sigmoid**: 1/(1+e^-x), 범위 [0, 1]
- **SELU**: Self-normalizing 효과
- **Softplus**: log(1+e^x) - ReLU의 부드러운 버전
- **Identity**: f(x) = x - 그대로 통과

### 7.3 MLP 구조 생성

**[구현 예시 - 학습 참고용]**

```python
layers = []

# 입력층 → 첫 은닉층
layers.append(nn.Linear(input_size, size))
layers.append(activation)

# 은닉층들
for _ in range(n_layers - 1):
    layers.append(nn.Linear(size, size))
    layers.append(activation)

# 출력층
layers.append(nn.Linear(size, output_size))
layers.append(output_activation)

return nn.Sequential(*layers)
```

**nn.Linear란?**
```python
class Linear(nn.Module):
    def forward(self, x):
        return x @ self.weight.T + self.bias
```
- 선형 변환: y = Wx + b
- W: (output_size, input_size) 행렬
- b: (output_size,) 벡터

**구조 예시:**
n_layers=2, input_size=17, size=64, output_size=8, activation=tanh

```
Layer 1: Linear(17, 64)  → [batch, 17] → [batch, 64]
         Tanh()          → [batch, 64] → [batch, 64]
Layer 2: Linear(64, 64)  → [batch, 64] → [batch, 64]
         Tanh()          → [batch, 64] → [batch, 64]
Output:  Linear(64, 8)   → [batch, 64] → [batch, 8]
         Identity()      → [batch, 8]  → [batch, 8]
```

**nn.Sequential():**
- 레이어들을 순서대로 실행
- `*layers`: 리스트를 positional arguments로 언팩
- 사용 예: `output = model(input)` → 순서대로 통과

**왜 Tanh를 기본값으로?**
- 범위: [-1, 1] → 행동이 보통 normalize 돼있어
- Zero-centered: 평균이 0 근처 → gradient flow 좋음
- ReLU보다 부드러움

**왜 출력층은 Identity?**
- 연속 행동: unbounded 출력 필요
- 나중에 평균(mean)으로 사용할 거라 제한 불필요

---

## 8. 메인 학습 루프: run_training_loop()

`bc_trainer.py`의 108-181번 줄

### 8.1 초기화

```python
def run_training_loop(
    self,
    n_iter,
    collect_policy,
    eval_policy,
    initial_expertdata=None,
    relabel_with_expert=False,
    start_relabel_with_expert=1,
    expert_policy=None,
):
    self.total_envsteps = 0
    self.start_time = time.time()
```

**time.time():**
- 현재 시각을 초 단위로 반환 (Unix timestamp)
- 나중에 `time.time() - self.start_time`으로 경과 시간 계산

### 8.2 Iteration 루프

```python
for itr in range(n_iter):
    print("\n\n********** Iteration %i ************" % itr)
```

**string formatting:**
- `%i`: 정수 placeholder
- 예: itr=0이면 "Iteration 0" 출력
- 다른 방법: f"Iteration {itr}" (Python 3.6+)

### 8.3 로깅 빈도 제어

```python
if (itr % self.params["video_log_freq"] == 0 and
    self.params["video_log_freq"] != -1):
    self.log_video = True
else:
    self.log_video = False
```

**% 연산자 (modulo):**
- `itr % video_log_freq`: itr을 video_log_freq로 나눈 나머지
- 예: video_log_freq=5일 때
  - itr=0: 0%5=0 → True (비디오 저장)
  - itr=1: 1%5=1 → False
  - itr=5: 5%5=0 → True (비디오 저장)
  - itr=10: 10%5=0 → True

**왜 이렇게 해?**
- 매 iteration마다 비디오 저장하면 용량↑
- 주기적으로만 저장해서 효율↑
- `-1`: 비디오 안 저장

```python
if itr % self.params["scalar_log_freq"] == 0:
    self.log_metrics = True
```

**scalar vs video:**
- scalar: 숫자 메트릭 (loss, reward 등)
- video: 실제 플레이 영상
- 보통 scalar는 매번, video는 가끔

### 8.4 학습 데이터 수집

```python
training_returns = self.collect_training_trajectories(
    itr, collect_policy, initial_expertdata
)
trajs, envsteps_this_batch, train_video_trajs = training_returns
self.total_envsteps += envsteps_this_batch
```

**반환값 언패킹:**
- `training_returns`는 튜플: (trajs, envsteps, videos)
- 한 줄에 3개 변수에 할당

**self.total_envsteps:**
- 누적 환경 step 수
- 학습 progress 추적용

### 8.5 전문가 라벨링 (DAgger)

```python
if relabel_with_expert and itr >= start_relabel_with_expert:
    trajs = self.do_relabel_with_expert(expert_policy, trajs)
```

**조건:**
- `relabel_with_expert=True`: DAgger 모드
- `itr >= start_relabel_with_expert`: 특정 iteration부터 시작
  - 기본값 1: iteration 0은 전문가 데이터 그대로, 1부터 relabel

**왜 나중부터 relabel?**
- Iteration 0: 전문가 데이터로 warm-start
- Iteration 1+: 학습한 정책으로 데이터 수집 → relabel

### 8.6 버퍼에 추가 및 학습

```python
self.agent.add_to_replay_buffer(trajs)
training_logs = self.train_agent()
```

**순서가 중요해:**
1. 버퍼에 새 데이터 추가
2. 버퍼에서 샘플링해서 학습
3. 이전 데이터도 함께 학습 (replay)

### 8.7 로깅 및 저장

```python
if self.log_video or self.log_metrics:
    print("\nBeginning logging procedure...")
    self.perform_logging(itr, trajs, eval_policy, train_video_trajs, training_logs)

    if self.params["save_params"]:
        print("\nSaving agent params")
        self.agent.save("{}/policy_itr_{}.pt".format(self.params["logdir"], itr))
```

**파일명 포맷:**
- `policy_itr_0.pt`: iteration 0의 정책
- `.pt`: PyTorch 모델 파일 확장자

**왜 매 iteration마다 저장?**
- 중간 체크포인트: 나중에 특정 iteration 모델 로드 가능
- 학습 과정 추적
- 최고 성능 모델 보존

---

## 9. 데이터 수집: collect_training_trajectories()

`bc_trainer.py`의 186-222번 줄

### 9.1 함수 로직

```python
def collect_training_trajectories(
    self, itr, collect_policy, load_initial_expertdata=None
):
```

**두 가지 경로:**

**경로 1: 전문가 데이터 로드 (itr == 0)**
```python
if itr == 0 and load_initial_expertdata is not None:
    with open(load_initial_expertdata, 'rb') as f:
        loaded_trajs = pickle.load(f)
    return loaded_trajs, 0, None
```

**pickle이란?**
- Python 객체를 파일로 직렬화(serialize)
- `pickle.dump(obj, file)`: 저장
- `pickle.load(file)`: 로드
- 주의: 보안 위험 (신뢰할 수 있는 파일만!)

**with 문:**
```python
with open(file, mode) as f:
    # f 사용
# 자동으로 f.close() 호출
```
- **Context manager**: 자원 관리 자동화
- 파일 닫기 보장 (에러나도)

**경로 2: 환경에서 데이터 수집**

**[구현 예시 - 학습 참고용]**

```python
if itr == 0:
    batch_size = self.params['batch_size_initial']
else:
    batch_size = self.params['batch_size']

trajs, envsteps = utils.rollout_trajectories(
    self.env,
    collect_policy,
    batch_size,
    self.params['ep_len']
)
```

**batch_size 차이:**
- `batch_size_initial`: 첫 수집 (보통 더 많이)
- `batch_size`: 이후 수집 (적당히)
- 왜? 초기엔 데이터 많이 필요, 이후엔 점진적

### 9.2 비디오 수집

```python
train_video_trajs = None
if self.log_video:
    print("\nCollecting train rollouts to be used for saving videos...")
    train_video_trajs = utils.rollout_n_trajectories(
        self.env,
        collect_policy,
        MAX_NVIDEO,
        MAX_VIDEO_LEN,
        render=True
    )
```

**MAX_NVIDEO, MAX_VIDEO_LEN:**
```python
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # 나중에 ep_len으로 덮어씀
```

- 2개의 trajectory만 비디오로 저장
- 각 최대 40 steps
- `render=True`: 이미지 렌더링 활성화

**rollout_n_trajectories vs rollout_trajectories:**
- `rollout_n_trajectories`: 정확히 N개 trajectory
- `rollout_trajectories`: 최소 N개 timesteps

---

## 10. 환경 상호작용: rollout_trajectory()

`utils.py`의 21-68번 줄

### 10.1 환경 초기화

**[구현 예시 - 학습 참고용]**

```python
def rollout_trajectory(env, policy, max_traj_length, render=False):
    ob, _ = env.reset()
```

**env.reset() 반환값:**
- Gymnasium (새 버전): `(observation, info)` 튜플
- `_`: info는 안 쓰니까 언더스코어로 무시

**observation이란?**
- 에이전트가 관찰하는 환경 상태
- 예시 (Ant-v4):
  - 관절 각도: 8차원
  - 관절 속도: 8차원
  - 기타: 1차원
  - 총 17차원 벡터

### 10.2 데이터 저장 리스트

```python
obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
steps = 0
```

**각 리스트 역할:**
- `obs`: 각 step의 observation
- `acs`: 각 step의 action
- `rewards`: 각 step의 reward
- `next_obs`: 각 step의 다음 observation
- `terminals`: 각 step의 종료 여부
- `image_obs`: 각 step의 렌더링 이미지 (선택)

### 10.3 메인 루프

```python
while True:
    if render:
        if hasattr(env, "sim"):
            image_obs.append(
                env.sim.render(camera_name="track", height=500, width=500)[::-1]
            )
        else:
            image_obs.append(env.render())
```

**hasattr() 함수:**
- 객체에 특정 속성이 있는지 확인
- `hasattr(env, "sim")`: env에 sim 속성 있나?

**MuJoCo 렌더링:**
- `env.sim.render()`: MuJoCo 시뮬레이터 직접 렌더링
- `camera_name="track"`: 추적 카메라 사용
- `[::-1]`: 이미지 상하 반전 (OpenGL 좌표계 때문)

**다른 환경:**
- `env.render()`: 표준 렌더링 메서드

### 10.4 행동 선택

```python
obs.append(ob)
ac = policy.get_action(ob)
acs.append(ac)
```

**policy.get_action(ob):**
1. ob를 torch tensor로 변환
2. 신경망 forward pass
3. 행동 분포에서 샘플링
4. numpy array로 반환

**예시 값:**
- ob: `[0.1, -0.3, 0.5, ..., 0.2]` (17차원)
- ac: `[0.05, -0.15, 0.02, ..., -0.1]` (8차원)

### 10.5 환경 step

```python
ob, rew, terminated, truncated, _ = env.step(ac)
```

**env.step() 반환값:**
- `ob`: 다음 observation (s')
- `rew`: 즉시 보상 (r)
- `terminated`: 환경이 종료 상태 도달 (예: 로봇 넘어짐)
- `truncated`: 시간 제한 도달
- `_`: info (안 씀)

**terminated vs truncated (Gymnasium):**
- terminated: 진짜 끝 (성공 or 실패)
- truncated: 시간 다 됨
- 왜 구분? 가치 함수 계산 때 다르게 처리

### 10.6 결과 기록

```python
steps += 1
next_obs.append(ob)
rewards.append(rew)

rollout_done = (terminated or truncated) or (steps >= max_traj_length)
terminals.append(rollout_done)

if rollout_done:
    break
```

**종료 조건 3가지:**
1. `terminated`: 환경이 끝남
2. `truncated`: 시간 제한
3. `steps >= max_traj_length`: 강제 종료

**terminals 값:**
- 중간: 0 (또는 False)
- 마지막: 1 (또는 True)

### 10.7 Trajectory 반환

```python
return Traj(obs, image_obs, acs, rewards, next_obs, terminals)
```

**Traj 함수:**
```python
def Traj(obs, image_obs, acs, rewards, next_obs, terminals):
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }
```

**np.stack():**
- 리스트의 배열들을 새 차원으로 쌓음
- `axis=0`: 첫 번째 차원에 쌓기
- 예: [(H,W,3), (H,W,3), ...] → (T,H,W,3)

**dtype 설명:**
- `float32`: 메모리 절약 (float64보다 절반)
- `uint8`: 이미지 (0-255 범위)

**반환 딕셔너리 shape:**
```python
{
    "observation": (T, ob_dim),      # 예: (100, 17)
    "action": (T, ac_dim),           # 예: (100, 8)
    "reward": (T,),                  # 예: (100,)
    "next_observation": (T, ob_dim),
    "terminal": (T,),
    "image_obs": (T, H, W, 3) or []
}
```

### 10.8 rollout_trajectories 구현

**[구현 예시 - 학습 참고용]**

```python
def rollout_trajectories(env, policy, min_timesteps_per_batch, max_traj_length, render=False):
    timesteps_this_batch = 0
    trajs = []

    while timesteps_this_batch < min_timesteps_per_batch:
        traj = rollout_trajectory(env, policy, max_traj_length, render)
        trajs.append(traj)
        timesteps_this_batch += len(traj["reward"])

    return trajs, timesteps_this_batch
```

**로직:**
1. timesteps 카운터 초기화
2. 목표 timesteps 도달까지 반복
3. 각 trajectory 수집
4. timesteps 누적
5. trajs 리스트와 총 timesteps 반환

**왜 trajectory 개수가 아니라 timesteps로?**
- Trajectory마다 길이가 다름
- 학습 데이터 양을 일정하게 유지
- 예: batch_size=1000
  - 짧은 trajs: 20개 필요 (각 50 steps)
  - 긴 trajs: 10개 필요 (각 100 steps)

### 10.9 rollout_n_trajectories 구현

**[구현 예시 - 학습 참고용]**

```python
def rollout_n_trajectories(env, policy, ntraj, max_traj_length, render=False):
    trajs = []
    for _ in range(ntraj):
        traj = rollout_trajectory(env, policy, max_traj_length, render)
        trajs.append(traj)
    return trajs
```

**간단한 반복:**
- 정확히 ntraj개 수집
- 길이 상관없이 개수만 중요
- 비디오/평가용

---

## 11. 전문가 라벨링: do_relabel_with_expert()

`bc_trainer.py`의 249-264번 줄

### 11.1 DAgger의 핵심 아이디어

**문제: Distributional Shift**
- 학습 중인 정책이 전문가와 다른 행동 → 전문가가 안 가본 상태 도달
- 그 상태에서 뭘 해야 할지 모름
- 점점 더 이상한 상태로...

**해결: DAgger**
- 학습 정책이 방문한 상태에서 전문가에게 물어봄
- "내가 여기 왔는데, 당신이라면 뭘 할 건가요?"
- 그 답변으로 학습 데이터 보강

### 11.2 구현

**[구현 예시 - 학습 참고용]**

```python
def do_relabel_with_expert(self, expert_policy, trajs):
    print("\nRelabelling collected observations with labels from an expert policy...")

    for i in range(len(trajs)):
        observations = trajs[i]["observation"]
        expert_actions = expert_policy.get_action(observations)
        trajs[i]["action"] = expert_actions

    return trajs
```

**과정:**
1. 각 trajectory 순회
2. observation 추출 (학습 정책이 방문한 상태)
3. expert_policy.get_action() 호출
4. 기존 action을 expert action으로 교체

**예시:**
```python
# 원본 trajectory (학습 정책이 수집)
traj = {
    "observation": [[0.1, 0.2, ...], [0.3, 0.1, ...], ...],
    "action": [[0.5, -0.2, ...], [0.1, 0.3, ...], ...],  # 학습 정책의 행동
}

# 전문가 라벨링 후
expert_actions = expert_policy.get_action(traj["observation"])
# expert_actions = [[0.3, -0.1, ...], [0.05, 0.2, ...], ...]

traj["action"] = expert_actions  # 전문가 행동으로 교체
```

### 11.3 왜 이게 효과적인가?

**수학적 설명:**
- BC: D_expert = {(s, a) | s ~ π_expert}에서 학습
  - 전문가가 방문한 상태만
- DAgger: D_agg = {(s, a) | s ~ π_learner, a = π_expert(s)}로 확장
  - 학습 정책이 방문한 상태 + 전문가의 정답

**반복 학습:**
```
Iteration 0: 전문가 데이터로 학습 → π₀
Iteration 1: π₀로 데이터 수집 → 전문가가 라벨링 → 학습 → π₁
Iteration 2: π₁로 데이터 수집 → 전문가가 라벨링 → 학습 → π₂
...
```

**수렴:**
- π가 전문가에 가까워질수록 방문 상태 분포도 비슷해짐
- 결국 distributional shift 감소

---

## 12. 경험 저장: ReplayBuffer.add_rollouts()

`replay_buffer.py`의 59-98번 줄

### 12.1 Trajectory 추가

```python
for traj in trajs:
    self.trajs.append(traj)
```

**self.trajs:**
- 전체 trajectory를 딕셔너리 형태로 저장
- 나중에 분석이나 재사용 가능

### 12.2 성분별 변환

```python
observations, actions, rewards, next_observations, terminals = (
    convert_listofrollouts(trajs, concat_rew)
)
```

**convert_listofrollouts() 함수:**
```python
def convert_listofrollouts(trajs, concat_rew=True):
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    if concat_rew:
        rewards = np.concatenate([traj["reward"] for traj in trajs])
    else:
        rewards = [traj["reward"] for traj in trajs]
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    return observations, actions, rewards, next_observations, terminals
```

**np.concatenate() 자세히:**
- 여러 배열을 첫 번째 차원(axis=0)을 따라 연결
- 예시:
```python
traj1["observation"]: (50, 17)
traj2["observation"]: (30, 17)
traj3["observation"]: (40, 17)

concatenated: (120, 17)  # 50+30+40 = 120
```

**List comprehension:**
- `[traj["observation"] for traj in trajs]`
- trajs의 각 traj에서 "observation" 추출하여 리스트 생성

**concat_rew 파라미터:**
- True: reward를 하나의 배열로 합침
- False: reward를 리스트로 유지 (trajectory별로)
- 여기선 True (BC는 reward 안 씀)

### 12.3 버퍼 초기화 또는 추가

**첫 추가 (버퍼 비어있음):**
```python
if self.obs is None:
    self.obs = observations[-self.max_size :]
    self.acs = actions[-self.max_size :]
    self.rews = rewards[-self.max_size :]
    self.next_obs = next_observations[-self.max_size :]
    self.terminals = terminals[-self.max_size :]
```

**[-self.max_size:]란?**
- 배열의 마지막 max_size개 요소만 가져오기
- 예: max_size=1000000
  - observations가 500000개면? → 500000개 전부
  - observations가 1500000개면? → 마지막 1000000개만
- 왜? 버퍼 크기 제한 미리 적용

**기존 데이터 있음:**
```python
else:
    self.obs = np.concatenate([self.obs, observations])[-self.max_size :]
    self.acs = np.concatenate([self.acs, actions])[-self.max_size :]
    # ... 나머지도 동일
```

**FIFO (First In First Out):**
1. 기존 데이터 + 새 데이터 합침
2. 마지막 max_size개만 유지
3. 오래된 데이터 자동 제거

**예시:**
```python
# 기존 버퍼
self.obs: (800000, 17)

# 새 데이터
observations: (300000, 17)

# 합치기
concatenated: (1100000, 17)

# max_size = 1000000
self.obs = concatenated[-1000000:]  # (1000000, 17)
# 앞의 100000개 제거됨
```

### 12.4 __len__ 메서드

```python
def __len__(self):
    if self.obs is not None:
        return self.obs.shape[0]
    else:
        return 0
```

**__len__은 특별 메서드:**
- `len(buffer)` 호출 시 실행
- 버퍼의 transition 개수 반환
- `.shape[0]`: 첫 번째 차원 크기

---

## 13. 에이전트 학습: train_agent()

`bc_trainer.py`의 224-247번 줄

### 13.1 학습 루프

**[구현 예시 - 학습 참고용]**

```python
def train_agent(self):
    print("\nTraining agent using sampled data from replay buffer...")
    all_logs = []

    for train_step in range(self.params["num_agent_train_steps_per_iter"]):
        ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = (
            self.agent.sample(self.params["train_batch_size"])
        )

        train_log = self.agent.train(ob_batch, ac_batch)
        all_logs.append(train_log)

    return all_logs
```

**num_agent_train_steps_per_iter:**
- 한 iteration에 몇 번 gradient step 할지
- 예: 1000이면 1000번 파라미터 업데이트
- 많을수록: 학습 충분, 시간↑
- 적을수록: 빠름, 학습 부족

**train_batch_size:**
- 각 gradient step에서 사용할 샘플 개수
- 예: 100이면 버퍼에서 100개 transition 샘플링
- 크면: 안정적, 메모리↑
- 작으면: 빠름, 노이즈↑

### 13.2 BCAgent.sample()

```python
def sample(self, batch_size):
    return self.replay_buffer.sample_random_data(batch_size)
```

**반환값:**
- ob_batch: (batch_size, ob_dim)
- ac_batch: (batch_size, ac_dim)
- re_batch: (batch_size,) - BC는 안 씀
- next_ob_batch: (batch_size, ob_dim) - BC는 안 씀
- terminal_batch: (batch_size,) - BC는 안 씀

### 13.3 BCAgent.train()

```python
def train(self, ob_no, ac_na):
    log = self.actor.update(ob_no, ac_na)
    return log
```

**변수명 규칙 복습:**
- ob_no: observation, (n)umber × (o)bservation dimension
- ac_na: action, (n)umber × (a)ction dimension

**단순 위임:**
- BCAgent는 actor(정책)에게 업데이트 맡김
- 실제 학습은 MLPPolicySL.update()에서

---

## 14. 데이터 샘플링: ReplayBuffer.sample_random_data()

`replay_buffer.py`의 103-132번 줄

### 14.1 구현

**[구현 예시 - 학습 참고용]**

```python
def sample_random_data(self, batch_size):
    assert (
        self.obs.shape[0]
        == self.acs.shape[0]
        == self.rews.shape[0]
        == self.next_obs.shape[0]
        == self.terminals.shape[0]
    )

    indices = np.random.choice(
        len(self.obs),
        size=batch_size,
        replace=False
    )

    return (
        self.obs[indices],
        self.acs[indices],
        self.rews[indices],
        self.next_obs[indices],
        self.terminals[indices]
    )
```

### 14.2 Assert 검증

```python
assert (
    self.obs.shape[0] == self.acs.shape[0] == ...
)
```

**연쇄 비교:**
- Python의 편리한 문법
- 모든 배열의 첫 차원 크기가 같은지 확인
- 데이터 정합성 검증

**왜 필요?**
- 버그 조기 발견
- 예: obs 1000개, acs 999개 → 에러 발생
- 디버깅 시간 절약

### 14.3 무작위 인덱스 샘플링

```python
indices = np.random.choice(
    len(self.obs),
    size=batch_size,
    replace=False
)
```

**np.random.choice() 상세:**
- 첫 인자: 0부터 len(self.obs)-1까지 범위
- size: 몇 개 샘플링할지
- replace: 복원 추출 여부
  - False: 중복 없이 (without replacement)
  - True: 중복 허용 (with replacement)

**예시:**
```python
len(self.obs) = 10000
batch_size = 100

# indices: [3472, 189, 7834, 215, ..., 9001]  (100개)
# 0~9999 범위에서 무작위로 100개, 중복 없음
```

### 14.4 왜 replace=False인가?

**Without replacement (replace=False):**
- 같은 샘플 여러 번 안 뽑힘
- 한 배치 내 다양성 보장
- 일반적으로 권장

**With replacement (replace=True):**
- 같은 샘플 여러 번 뽑힐 수 있음
- 버퍼 크기 < batch_size일 때 필요
- 보통 안 씀

### 14.5 인덱싱

```python
return (
    self.obs[indices],
    ...
)
```

**NumPy 고급 인덱싱:**
- 배열로 인덱싱하면 해당 인덱스들의 요소 추출
- 예:
```python
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 2, 4])
arr[indices]  # array([10, 30, 50])
```

**2D 배열:**
```python
self.obs: (10000, 17)
indices: (100,)

self.obs[indices]: (100, 17)
# indices에 해당하는 100개 행 추출
```

### 14.6 왜 무작위 샘플링인가?

**i.i.d. (Independent and Identically Distributed) 가정:**
- 머신러닝은 데이터가 독립적이고 동일 분포라고 가정
- Trajectory는 시간적 상관관계 있음:
  - s_t와 s_{t+1}은 연관됨
  - 순차적으로 샘플링하면 상관관계 유지
- 무작위 샘플링으로 상관관계 제거

**학습 안정성:**
- 연속된 샘플: gradient 편향
- 무작위 샘플: 다양한 상황 학습
- 일반화 성능↑

---

## 15. 정책 업데이트: MLPPolicySL 핵심 메서드들

### 15.1 forward() - 행동 분포 생성

`MLP_policy.py`의 130-147번 줄

**[구현 예시 - 학습 참고용]**

```python
def forward(self, observation: torch.FloatTensor) -> Any:
    mean = self.mean_net(observation)
    std = torch.exp(self.logstd)
    return torch.distributions.Normal(mean, std)
```

**입력:**
- observation: (batch_size, ob_dim) 또는 (ob_dim,) 텐서
- 예: (100, 17) - 100개 샘플, 17차원 관찰

**mean_net(observation):**
- MLP를 통과
- 입력: (100, 17)
- 출력: (100, ac_dim) 예: (100, 8)
- 각 샘플마다 ac_dim차원 평균 벡터

**torch.exp(self.logstd):**
- logstd: (ac_dim,) 예: (8,)
- exp() 적용: 항상 양수
- std: (ac_dim,) 예: (8,)

**왜 exp를 사용?**
```python
logstd = -2.0 → std = exp(-2.0) ≈ 0.135  (작은 탐색)
logstd = 0.0  → std = exp(0.0) = 1.0     (보통 탐색)
logstd = 2.0  → std = exp(2.0) ≈ 7.39    (큰 탐색)
```
- logstd는 unbounded (-∞ ~ +∞)
- std는 항상 양수
- 수치적 안정성

**torch.distributions.Normal:**
```python
Normal(loc=mean, scale=std)
```
- loc: 평균 (mean), shape (100, 8)
- scale: 표준편차 (std), shape (8,)
- **Broadcasting**: std가 모든 샘플에 적용됨

**Broadcasting 예시:**
```python
mean: (100, 8)
std:  (8,)      # 자동으로 (100, 8)로 확장

# 각 샘플마다 독립적인 정규분포
distribution[0]: N(mean[0], std)  # 8차원 분포
distribution[1]: N(mean[1], std)  # 8차원 분포
...
```

**반환 Distribution 객체:**
```python
distribution = Normal(mean, std)

# 사용 가능 메서드:
distribution.sample()       # 샘플링
distribution.log_prob(x)    # 로그 확률
distribution.entropy()      # 엔트로피
distribution.mean          # 평균
distribution.stddev        # 표준편차
```

### 15.2 get_action() - 행동 샘플링

`MLP_policy.py`의 111-128번 줄

**[구현 예시 - 학습 참고용]**

```python
def get_action(self, obs: np.ndarray) -> np.ndarray:
    if len(obs.shape) > 1:
        observation = obs
    else:
        observation = obs[None]

    observation = ptu.from_numpy(observation)
    action_distribution = self.forward(observation)
    action = action_distribution.sample()
    return ptu.to_numpy(action)
```

**배치 차원 처리:**
```python
# 단일 샘플
obs.shape = (17,)
obs[None] = (1, 17)  # 배치 차원 추가

# 여러 샘플
obs.shape = (100, 17)
# 그대로 사용
```

**obs[None]이란?**
- None은 np.newaxis의 별칭
- 새 차원 추가
- 예:
```python
arr = np.array([1, 2, 3])     # shape: (3,)
arr[None]                      # shape: (1, 3)
arr[:, None]                   # shape: (3, 1)
```

**ptu.from_numpy():**
```python
def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)
```
1. numpy → torch tensor
2. float64 → float32 (`.float()`)
3. CPU → GPU (`.to(device)`)

**action_distribution.sample():**
```python
# distribution: Normal(mean, std)
# mean: (1, 8), std: (8,)

action = distribution.sample()  # (1, 8)

# 각 차원 독립적으로 샘플링
# action[0][i] ~ N(mean[0][i], std[i])
```

**ptu.to_numpy():**
```python
def to_numpy(tensor):
    return tensor.to("cpu").detach().numpy()
```
1. GPU → CPU (`.to("cpu")`)
2. gradient 끊기 (`.detach()`)
3. torch → numpy (`.numpy()`)

**왜 detach()?**
- gradient 계산 불필요 (inference만)
- 메모리 절약
- computational graph에서 분리

**전체 흐름 예시:**
```python
# 입력
obs: (17,) numpy array, CPU

# 1. 배치 차원 추가
observation: (1, 17)

# 2. torch 변환
observation: (1, 17) tensor, GPU

# 3. forward
mean: (1, 8) tensor, GPU
std: (8,) tensor, GPU
distribution: Normal(mean, std)

# 4. 샘플링
action: (1, 8) tensor, GPU

# 5. numpy 변환
action: (1, 8) numpy array, CPU

# 환경은 (8,)을 기대하므로 squeeze 필요
# (실제론 [0]으로 첫 샘플만 사용)
```

### 15.3 update() - 정책 학습

`MLP_policy.py`의 149-166번 줄

**[구현 예시 - 학습 참고용]**

```python
def update(self, observations, actions):
    self.optimizer.zero_grad()

    observations = ptu.from_numpy(observations)
    actions = ptu.from_numpy(actions)

    action_distribution = self.forward(observations)
    log_prob = action_distribution.log_prob(actions).sum(dim=-1)
    loss = -log_prob.mean()

    loss.backward()
    self.optimizer.step()

    return {
        "Training Loss": ptu.to_numpy(loss),
    }
```

**optimizer.zero_grad():**
- 이전 gradient 초기화
- PyTorch는 gradient를 누적하기 때문
- 매 step마다 필요

**왜 누적되는가?**
```python
# step 1
loss1.backward()  # grad += ∂loss1/∂θ

# step 2 (zero_grad 안 하면)
loss2.backward()  # grad += ∂loss2/∂θ
# grad = ∂loss1/∂θ + ∂loss2/∂θ (잘못됨!)

# 올바른 방법
optimizer.zero_grad()  # grad = 0
loss2.backward()       # grad = ∂loss2/∂θ
```

**Numpy → Torch 변환:**
```python
observations: (100, 17) numpy → (100, 17) tensor, GPU
actions: (100, 8) numpy → (100, 8) tensor, GPU
```

**Forward pass:**
```python
action_distribution = self.forward(observations)
# Normal(mean=(100,8), std=(8,))
```

**Log probability 계산:**
```python
log_prob = action_distribution.log_prob(actions)
# shape: (100, 8)
```

**log_prob(actions)란?**
- 주어진 actions의 로그 확률 계산
- 각 차원 독립적으로:
```python
log_prob[i][j] = log P(actions[i][j] | mean[i][j], std[j])
                = log (1/√(2πσ²)) - (x-μ)²/(2σ²)
```

**sum(dim=-1):**
```python
log_prob: (100, 8)
log_prob.sum(dim=-1): (100,)
```
- 마지막 차원(행동 차원)을 따라 합산
- 왜? 각 행동 차원이 독립이면:
```
P(a₁, a₂, ..., a₈) = P(a₁) × P(a₂) × ... × P(a₈)
log P(a₁, ..., a₈) = log P(a₁) + ... + log P(a₈)
```

**dim 인자 설명:**
```python
tensor: (100, 8)

sum(dim=0): (8,)     # 샘플 축으로 합
sum(dim=1): (100,)   # 행동 축으로 합
sum(dim=-1): (100,)  # 마지막 축 (dim=1과 동일)
```

**Loss 계산:**
```python
loss = -log_prob.mean()
```

**왜 negative?**
- 목표: log P(a|s) 최대화
- optimizer는 최소화하므로
- -log P(a|s) 최소화 = log P(a|s) 최대화

**Maximum Likelihood Estimation (MLE):**
```
θ* = argmax Σ log P(aᵢ|sᵢ; θ)
   = argmin Σ -log P(aᵢ|sᵢ; θ)
   = argmin E[-log P(a|s; θ)]
```

**mean():**
- 배치 평균
- log_prob: (100,) → scalar
- 전체 손실의 기댓값 추정

**Backward pass:**
```python
loss.backward()
```

**무슨 일이?**
1. Computational graph 역방향 탐색
2. Chain rule로 gradient 계산:
```
∂loss/∂mean_net.weight
∂loss/∂logstd
```
3. 각 파라미터의 .grad에 저장

**Computational graph 예시:**
```
observations → mean_net → mean ┐
logstd → exp → std             ├→ Normal → log_prob → sum → mean → neg → loss
actions ────────────────────────┘
```

**Optimizer step:**
```python
self.optimizer.step()
```

**Adam update (간략):**
```python
for param in parameters:
    m = β₁*m + (1-β₁)*param.grad
    v = β₂*v + (1-β₂)*param.grad²
    param -= lr * m / (√v + ε)
```

**반환 로그:**
```python
return {
    "Training Loss": ptu.to_numpy(loss),
}
```
- 손실 값을 numpy로 변환해서 반환
- 로깅/모니터링용

### 15.4 학습 과정 전체 흐름

**1회 Update 전체:**
```python
# 1. 버퍼에서 샘플링
obs, actions = replay_buffer.sample(100)
# obs: (100, 17), actions: (100, 8)

# 2. Gradient 초기화
optimizer.zero_grad()

# 3. Forward
obs_tensor = from_numpy(obs)              # (100, 17) GPU
actions_tensor = from_numpy(actions)      # (100, 8) GPU

mean = mean_net(obs_tensor)               # (100, 8)
std = exp(logstd)                         # (8,)
distribution = Normal(mean, std)

# 4. Loss
log_prob = distribution.log_prob(actions) # (100, 8)
log_prob_sum = log_prob.sum(-1)          # (100,)
loss = -log_prob_sum.mean()              # scalar

# 5. Backward
loss.backward()
# mean_net.weight.grad, logstd.grad 계산됨

# 6. Update
optimizer.step()
# mean_net.weight -= lr * ...
# logstd -= lr * ...
```

**1000 steps 후:**
- 정책이 전문가 행동을 잘 모방
- mean_net이 전문가처럼 행동 출력
- std는 적절한 exploration 수준 학습

---

## 16. 전체 파이프라인 요약

### 16.1 초기화 단계

```
1. main() - 인자 파싱, 로그 디렉토리 생성
   ↓
2. run_bc() - 파라미터 설정, expert policy 로드
   ↓
3. BCTrainer.__init__() - 환경 생성, seed 설정
   ↓
4. BCAgent.__init__() - actor, replay buffer 생성
   ↓
5. MLPPolicySL.__init__() - 신경망, optimizer 생성
   ↓
6. build_mlp() - MLP 구조 구축
```

### 16.2 학습 루프 (n_iter 반복)

```
for itr in range(n_iter):

    1. collect_training_trajectories()
       - itr==0: 전문가 데이터 로드
       - itr>0: 현재 정책으로 데이터 수집
         ↓
       rollout_trajectories()
         ↓
       rollout_trajectory() (여러 번)
         - env.reset()
         - while not done:
             policy.get_action() → env.step()

    2. do_relabel_with_expert() [DAgger만]
       - 수집한 obs에 대해 전문가 행동 쿼리
       - action을 전문가 action으로 교체

    3. agent.add_to_replay_buffer()
       - trajectory → 성분별 배열 변환
       - 버퍼에 추가 (FIFO)

    4. train_agent()
       - for _ in range(num_train_steps):
           - replay_buffer.sample_random_data()
             ↓
           - agent.train()
             ↓
           - actor.update()
             - forward() → distribution
             - log_prob → loss
             - backward() → optimizer.step()

    5. perform_logging()
       - 평가 데이터 수집
       - 메트릭 계산 및 로깅
       - 모델 저장
```

### 16.3 핵심 데이터 흐름

**Environment → Trajectories:**
```
env.reset() → observation (17,)
  ↓
policy.get_action() → action (8,)
  ↓
env.step() → next_obs, reward, done
  ↓
반복 → trajectory
{
    "observation": (T, 17),
    "action": (T, 8),
    "reward": (T,),
    "next_observation": (T, 17),
    "terminal": (T,)
}
```

**Trajectories → Replay Buffer:**
```
List[trajectory] → convert_listofrollouts()
  ↓
obs: (N, 17)
acs: (N, 8)
...
  ↓
replay_buffer.add_rollouts()
```

**Replay Buffer → Training:**
```
sample_random_data(100)
  ↓
obs_batch: (100, 17)
ac_batch: (100, 8)
  ↓
actor.update()
  ↓
loss.backward()
optimizer.step()
```

### 16.4 주요 객체 Shape 정리

**환경 관련:**
- observation: (ob_dim,) = (17,)
- action: (ac_dim,) = (8,)
- reward: scalar
- done: boolean

**Trajectory:**
- observations: (T, ob_dim)
- actions: (T, ac_dim)
- rewards: (T,)
- next_observations: (T, ob_dim)
- terminals: (T,)

**Replay Buffer:**
- obs: (N, ob_dim) - N은 총 transition 수
- acs: (N, ac_dim)
- ...

**Training Batch:**
- ob_batch: (batch_size, ob_dim) = (100, 17)
- ac_batch: (batch_size, ac_dim) = (100, 8)

**신경망:**
- mean_net 입력: (batch, ob_dim)
- mean_net 출력: (batch, ac_dim)
- logstd: (ac_dim,)
- std: (ac_dim,)
- distribution: Normal(mean=(batch, ac_dim), std=(ac_dim,))

---

## 17. 강화학습 핵심 개념 정리

### 17.1 Imitation Learning

**정의:**
- 전문가의 행동을 관찰하고 모방하는 학습

**종류:**
1. **Behavioral Cloning (BC)**
   - Supervised learning으로 직접 모방
   - 장점: 간단, 빠름
   - 단점: Distributional shift

2. **DAgger**
   - 반복적으로 전문가에게 라벨 요청
   - 장점: Distributional shift 해결
   - 단점: 전문가 필요

### 17.2 Distributional Shift

**문제:**
```
학습: s ~ π_expert → 전문가가 방문한 상태
실행: s ~ π_학습 → 다른 상태 방문 가능
```

**예시:**
- 자율주행 학습: 도로 중앙 데이터만
- 실행: 약간 벗어남 → 복구 방법 모름 → 더 벗어남 → 충돌

**해결: DAgger**
- 학습 정책이 방문한 상태에서도 전문가 행동 학습
- 점진적으로 분포 일치

### 17.3 Policy Representation

**Stochastic Policy:**
```
π(a|s) = P(action=a | state=s)
```

**Gaussian Policy:**
```
π(a|s) = N(μ(s), Σ)
μ(s) = mean_net(s)
Σ = diag(σ²) where σ = exp(logstd)
```

**왜 확률적?**
- Exploration: 다양한 행동 시도
- Robustness: 비슷한 상태에서 약간 다른 행동
- Expressiveness: 복잡한 정책 표현

### 17.4 Supervised Learning for BC

**목표:**
```
θ* = argmin E_{(s,a)~D} [loss(π_θ(s), a)]
```

**Negative Log Likelihood Loss:**
```
loss = -log π_θ(a|s)
     = -log P(a | s; θ)
```

**Gradient:**
```
∇_θ loss = -∇_θ log P(a|s; θ)
```

### 17.5 Replay Buffer

**목적:**
1. **Experience reuse**: 데이터 재사용
2. **Breaking correlation**: 시간 상관관계 제거
3. **Sample efficiency**: 효율적 학습

**동작:**
```
1. Collect: 환경에서 데이터 수집
2. Store: 버퍼에 저장
3. Sample: 무작위 추출
4. Train: 샘플로 학습
```

**FIFO:**
```
[오래된 데이터] ... [새 데이터]
         ↓ 버퍼 가득 차면
[        새 데이터만        ]
```

### 17.6 Neural Network 구조

**MLP (Multi-Layer Perceptron):**
```
x → Linear → Activation → ... → Linear → y
```

**역할:**
- Linear: 선형 변환
- Activation: 비선형성 추가

**왜 비선형?**
- Linear만: 전체가 Linear
- Activation 추가: 복잡한 함수 근사 가능

### 17.7 Optimization

**Gradient Descent:**
```
θ_{t+1} = θ_t - α ∇_θ loss
```

**Adam:**
- Adaptive learning rate
- Momentum 사용
- 빠른 수렴

**Backpropagation:**
- Chain rule로 gradient 계산
- Computational graph 사용
- 자동 미분

---

## 18. 실전 팁 및 디버깅

### 18.1 자주 하는 실수

1. **Gradient 초기화 안 함**
```python
# 잘못
loss.backward()
optimizer.step()

# 올바름
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

2. **Shape 안 맞음**
```python
# 단일 샘플 처리
obs.shape = (17,)
obs[None]  # (1, 17)로 만들기
```

3. **Device 불일치**
```python
# 잘못
model.to('cuda')
input.to('cpu')  # 에러!

# 올바름
model.to('cuda')
input.to('cuda')
```

### 18.2 디버깅 체크리스트

- [ ] 모든 데이터 shape 확인
- [ ] Gradient 초기화 확인
- [ ] Device 일치 확인
- [ ] Loss가 감소하는지
- [ ] Replay buffer에 데이터 있는지
- [ ] 샘플링이 제대로 되는지

### 18.3 성능 향상

**하이퍼파라미터 튜닝:**
- learning_rate: 0.001, 0.0001, ...
- batch_size: 32, 64, 128, ...
- n_layers: 2, 3, 4
- size: 64, 128, 256

**데이터:**
- 더 많은 expert 데이터
- DAgger iterations 증가
- 더 긴 training steps

---

## 마무리

이 가이드는 run_hw1.py의 전체 파이프라인을 처음부터 끝까지 상세히 다뤘습니다.

**핵심 요약:**
1. BC는 전문가 모방, DAgger는 반복적 개선
2. 신경망으로 정책 표현 (Gaussian policy)
3. Replay buffer로 효율적 학습
4. MLE로 supervised learning
5. PyTorch로 구현

**다음 단계:**
- 코드 직접 구현해보기
- 다른 환경에서 실험
- 성능 비교 및 분석
- 논문 읽기 (DAgger, GAIL 등)

이제 여러분은 Imitation Learning의 핵심을 완전히 이해했습니다! 🎉
