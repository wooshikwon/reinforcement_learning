# Homework 3 Solution Guide - Section 4

## 6. Training Loop Implementation (훈련 루프 구현)

### 6.1 sample_trajectory() 구현

**파일 위치**: `gcb6206/infrastructure/utils.py`

이 함수는 환경에서 **하나의 episode**를 실행하고 데이터를 수집합니다.

```python
def sample_trajectory(
    env: gym.Env, agent: DQNAgent, max_length: int, render: bool = False
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob, _ = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
```

#### 초기화

```python
ob, _ = env.reset()
```

**Gymnasium API (새 버전):**

```python
observation, info = env.reset()
```

- `observation`: 초기 상태
- `info`: 추가 정보 dictionary (여기서는 사용 안 함)

**이전 Gym API와 차이:**

```python
# Old (Gym):
ob = env.reset()

# New (Gymnasium):
ob, info = env.reset()
```

우리는 `info`를 사용하지 않으므로 `_`로 버립니다.

**빈 리스트 초기화:**

- `obs`: observations
- `acs`: actions
- `rewards`: rewards
- `next_obs`: next observations
- `terminals`: done flags
- `image_obs`: 렌더링된 이미지 (비디오 생성용)

#### Main Loop

```python
while True:
    # render an image
    if render:
        if hasattr(env, "sim"):
            img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
        else:
            img = env.render(mode="rgb_array")

        if isinstance(img, list):
            img = img[0]

        image_obs.append(
            cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
        )
```

**렌더링 로직 (Optional):**

- `hasattr(env, "sim")`: MuJoCo 환경인지 확인
- `env.render(mode="rgb_array")`: RGB 이미지 반환
- `[::-1]`: 이미지 상하 반전 (MuJoCo 좌표계 문제)
- `cv2.resize(..., (250, 250))`: 크기 조정 (비디오 용량 절약)

평가 시 비디오를 저장하기 위해 사용됩니다.

#### TODO 1: Action Selection

```python
# TODO(student): use the most recent ob to decide what to do
# HINT: agent.get_action()
ac = ...
```

**Solution:**

```python
ac = agent.get_action(ob)
```

**설명:**

- `ob`: 현재 observation
- `agent.get_action()`: epsilon-greedy로 행동 선택
- Evaluation 시: `epsilon=0.02` (거의 greedy)
- Training 시: `epsilon`은 schedule에 따라 변함

**내부 동작:**

```python
# agent.get_action() 내부:
if np.random.random() < epsilon:
    ac = np.random.randint(num_actions)  # Explore
else:
    with torch.no_grad():
        q_values = self.critic(ob)
        ac = torch.argmax(q_values).item()  # Exploit
```

#### TODO 2: Environment Step

```python
# TODO(student): take that action and get reward and next obs from the environment
# HINT: use env.step()
next_ob, rew, terminated, truncated, info = ...
```

**Solution:**

```python
next_ob, rew, terminated, truncated, info = env.step(ac)
```

**Gymnasium Step API:**

```python
observation, reward, terminated, truncated, info = env.step(action)
```

**5개 반환값 설명:**

1. **`observation` (next_ob)**:
   - 행동 후의 다음 상태
   - Type: numpy array

2. **`reward` (rew)**:
   - 이번 step의 보상
   - Type: float
   - CartPole: 항상 1.0 (살아있는 동안)
   - Atari: 게임마다 다름 (점수 변화량)

3. **`terminated`**:
   - **진짜 종료** (catastrophic failure)
   - CartPole: 막대가 넘어지거나 벗어남
   - Atari: 목숨을 모두 잃음
   - Type: bool

4. **`truncated`**:
   - **인위적 종료** (time limit 등)
   - CartPole: 500 steps 도달
   - Atari: 일반적으로 False (시간 제한 없음)
   - Type: bool

5. **`info`**:
   - 추가 정보 dictionary
   - `info["episode"]`: episode 통계 (return, length)

**Terminated vs Truncated:**

이 구분은 **bootstrapping**에 중요합니다:

```python
# Terminated (진짜 끝):
done = True → V(s') = 0
target = reward + 0 * V(s') = reward

# Truncated (시간 제한):
done = False → V(s')를 추정해야 함
target = reward + γ * V(s')
```

CartPole 예시:
- Step 499: reward=1, next_state=valid
- Step 500: **truncated=True**
  - 아직 넘어지지 않았지만 시간 제한
  - done=False로 처리해야 함 (계속 살 수 있었음)

#### TODO 3: Rollout Done Flag

```python
# TODO(student): rollout can end due to termination, or truncation because it reached the maximum number of steps.
rollout_done = ... # HINT: this is either 0 or 1
```

**Solution:**

```python
rollout_done = terminated or truncated
```

**설명:**

Episode가 끝나는 두 가지 경우:

1. **Terminated**: 환경이 종료 상태 도달
2. **Truncated**: 최대 길이 도달

둘 중 하나라도 True면 rollout 종료.

**왜 boolean 연산?**

Python은 `or` 연산을 지원:
```python
True or False = True
False or True = True
False or False = False
```

결과는 `True` 또는 `False` (Python bool).

**주의**: 주석에는 "0 or 1"이라고 되어 있지만, Python bool도 정수처럼 작동:
```python
True == 1   # True
False == 0  # True
```

#### 데이터 수집

```python
steps += 1

# record result of taking that action
obs.append(ob)
acs.append(ac)
rewards.append(rew)
next_obs.append(next_ob)
terminals.append(terminated)  # 중요: terminated만 저장!

ob = next_ob  # jump to next timestep

# end the rollout if the rollout ended
if rollout_done:
    break
```

**핵심: `terminals.append(terminated)`**

**Truncated는 저장하지 않습니다!**

이유:
- Replay buffer의 `done` flag는 TD target 계산에 사용
- Truncated는 "실제 종료"가 아님
- Target: $r + \gamma (1 - \text{done}) \max Q(s', a')$
  - Truncated: $\text{done} = 0$ → 미래 가치 포함
  - Terminated: $\text{done} = 1$ → 미래 가치 제외

예시:
```python
# Step 499 (마지막 step):
reward = 1.0
terminated = False
truncated = True
next_state = [0.1, 0.2, 0.3, 0.4]

# Replay buffer에 저장:
terminals.append(False)  # terminated만!

# TD target:
target = 1.0 + 0.99 * (1 - 0) * max(Q(next_state, a))
       = 1.0 + 0.99 * Q_max(next_state)
# 계속 살 수 있었으므로 미래 가치 포함!
```

#### 반환값

```python
episode_statistics = {"l": steps, "r": np.sum(rewards)}
if "episode" in info:
    episode_statistics.update(info["episode"])

env.close()

return {
    "observation": np.array(obs, dtype=np.float32),
    "image_obs": np.array(image_obs, dtype=np.uint8),
    "reward": np.array(rewards, dtype=np.float32),
    "action": np.array(acs, dtype=np.float32),
    "next_observation": np.array(next_obs, dtype=np.float32),
    "terminal": np.array(terminals, dtype=np.float32),
    "episode_statistics": episode_statistics,
}
```

**`episode_statistics`:**

- `"l"`: length (episode 길이)
- `"r"`: return (총 보상)
- `info["episode"]`: 환경이 제공하는 추가 통계

**왜 numpy array로 변환?**

- List는 메모리 비효율적
- NumPy array는 연속 메모리, 빠른 연산
- Batch 연산에 적합

**dtype 지정:**

- `observation`, `reward`, `action`, etc.: `float32`
  - GPU는 float32가 빠름
  - float64는 불필요한 정밀도
- `image_obs`: `uint8`
  - 이미지는 0-255 정수
  - 메모리 절약 (1 byte per pixel)
- `terminal`: `float32`
  - 계산에 사용되므로 float

---

### 6.2 run_hw3.py - Training Loop

**파일 위치**: `gcb6206/scripts/run_hw3.py`

#### 초기화 부분

```python
def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)
    exploration_schedule = config["exploration_schedule"]
```

**세 개의 환경:**

1. **`env`**: Training 환경
   - 데이터 수집용
   - 매 step 사용

2. **`eval_env`**: Evaluation 환경
   - 성능 평가용
   - 주기적으로 사용 (10K steps마다)

3. **`render_env`**: Rendering 환경
   - 비디오 생성용
   - `render=True` 플래그

**왜 분리하는가?**

- 환경마다 독립적인 random seed
- Training과 evaluation 분리 (scientific practice)
- Rendering 오버헤드 없이 training

**Exploration Schedule:**

```python
exploration_schedule = PiecewiseSchedule([
    (0, 1.0),                    # 초기: ε=1
    (total_steps * 0.1, 0.02),   # 10% 후: ε=0.02
], outside_value=0.02)
```

#### Agent 생성

```python
agent = DQNAgent(
    env.observation_space.shape,
    env.action_space.n,
    **config["agent_kwargs"],
)
```

**`env.observation_space.shape`:**

- CartPole: `(4,)` - [position, velocity, angle, angular_velocity]
- Atari: `(4, 84, 84)` - stacked frames

**`env.action_space.n`:**

- CartPole: 2 - [left, right]
- BankHeist: 18 - joystick combinations

**`**config["agent_kwargs"]`:**

Dictionary unpacking:
```python
config["agent_kwargs"] = {
    "make_critic": ...,
    "make_optimizer": ...,
    "discount": 0.99,
    ...
}

# Equivalent to:
DQNAgent(
    ...,
    make_critic=...,
    make_optimizer=...,
    discount=0.99,
    ...
)
```

#### Replay Buffer 초기화

```python
# Replay buffer
if len(env.observation_space.shape) == 3:
    stacked_frames = True
    frame_history_len = env.observation_space.shape[0]
    assert frame_history_len == 4, "only support 4 stacked frames"
    replay_buffer = MemoryEfficientReplayBuffer(
        frame_history_len=frame_history_len
    )
elif len(env.observation_space.shape) == 1:
    stacked_frames = False
    replay_buffer = ReplayBuffer()
else:
    raise ValueError(
        f"Unsupported observation space shape: {env.observation_space.shape}"
    )
```

**Shape 기반 선택:**

- **3D** `(C, H, W)`: 이미지 → `MemoryEfficientReplayBuffer`
  - Atari: `(4, 84, 84)`
  - Frame stacking 활용

- **1D** `(D,)`: 벡터 → `ReplayBuffer`
  - CartPole: `(4,)`
  - 간단한 버퍼로 충분

- **기타**: 에러 (지원 안 함)

#### Reset 함수

```python
def reset_env_training():
    nonlocal observation

    observation, _ = env.reset()
    observation = np.asarray(observation)

    if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
        replay_buffer.on_reset(observation=observation[-1, ...])

reset_env_training()
```

**`nonlocal observation`:**

- Nested function에서 outer scope 변수 수정
- `observation`은 `run_training_loop`의 변수

**`observation[-1, ...]`:**

- Memory-efficient buffer는 **단일 프레임**만 필요
- `observation.shape = (4, 84, 84)`
- `observation[-1, ...]` = `(84, 84)` - 마지막 프레임

`...` (Ellipsis):
- 나머지 모든 차원
- `observation[-1, ...]` = `observation[-1, :, :]`

#### Main Training Loop - TODO 구현

```python
for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
    epsilon = exploration_schedule.value(step)

    # TODO(student): Compute action
    # HINT: use agent.get_action() with epsilon
    action = ...

    # TODO(student): Step the environment
    # HINT: use env.step()
    next_observation, reward, terminated, truncated, info = ...
```

**TODO 1: Action Selection**

**Solution:**

```python
action = agent.get_action(observation, epsilon=epsilon)
```

**설명:**

- `observation`: 현재 상태
- `epsilon`: 현재 step의 exploration rate
  - Step 0: ε=1.0
  - Step 30000: ε=0.02
  - 이후: ε=0.02

**`epsilon`을 명시적으로 전달:**

```python
def get_action(self, observation, epsilon=0.02):
    if np.random.random() < epsilon:  # ← 여기서 사용
        ...
```

Default `epsilon=0.02`는 evaluation용이고, training은 schedule 사용.

**TODO 2: Environment Step**

**Solution:**

```python
next_observation, reward, terminated, truncated, info = env.step(action)
```

이전 `sample_trajectory()`와 동일.

#### Replay Buffer 삽입

```python
next_observation = np.asarray(next_observation)

# Add the data to the replay buffer if not truncated
if not truncated:
    if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
        # We're using the memory-efficient replay buffer,
        # So we do not insert observation, but only the last frame of the next_observation
        replay_buffer.insert(action, reward, next_observation[-1], terminated)
    else:
        # TODO(student):
        # We're using the regular replay buffer
        # Simply insert all obs (not observation[-1])
        # replay_buffer.insert(...)
```

**Memory-Efficient Buffer:**

```python
replay_buffer.insert(action, reward, next_observation[-1], terminated)
```

- `next_observation[-1]`: 마지막 프레임만 `(84, 84)`
- Buffer가 내부적으로 frame history 관리

**Regular Buffer (TODO):**

**Solution:**

```python
replay_buffer.insert(
    observation=observation,
    action=action,
    reward=reward,
    next_observation=next_observation,
    done=terminated,
)
```

**주의사항:**

1. **`observation[-1]` 아님**: 전체 observation
2. **`done=terminated`**: truncated 아님!
3. **Keyword arguments**: 명시적으로 전달

**왜 `if not truncated`?**

Truncated step은 버퍼에 저장하지 않습니다:

- Time limit로 인한 종료는 "실제 transition"이 아님
- 다음 episode의 첫 state와 연결되면 안 됨
- 데이터 품질 유지

예시:
```
Episode 1:
  Step 0-499: 저장 ✓
  Step 500: truncated=True → 저장 ✗

Episode 2:
  Step 0: 새 episode 시작
```

#### Episode 종료 처리

```python
# Handle episode termination
if terminated or truncated:
    reset_env_training()

    logger.log_scalar(info["episode"]["r"], "train_return", step)
    logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
else:
    observation = next_observation
```

**Episode 끝:**

- 환경 reset
- Episode 통계 로깅 (`info["episode"]`에서)
  - `"r"`: total return
  - `"l"`: episode length

**Episode 계속:**

- `observation` 업데이트 (다음 step을 위해)

**`info["episode"]`는 언제 생기나?**

`RecordEpisodeStatistics` wrapper가 제공:
```python
env = RecordEpisodeStatistics(gym.make("CartPole-v1"))
```

Episode 종료 시에만 `info["episode"]`가 생성됨.

#### DQN Training (TODO)

```python
# Main DQN training loop
if step >= config["learning_starts"]:
    # TODO(student): Sample config["batch_size"] samples from the replay buffer
    # HINT: Use replay_buffer.sample()
    batch = ...

    # Convert to PyTorch tensors
    batch = ptu.from_numpy(batch)

    # TODO(student): Train the agent. `batch` is a dictionary of numpy arrays.
    # HINT: agent.update
    update_info = ...
```

**`if step >= config["learning_starts"]`:**

- `learning_starts`: 보통 20,000 steps
- 충분한 데이터를 모으기 전에는 학습하지 않음
- 이유:
  1. Replay buffer가 거의 비어있음 → 다양성 부족
  2. 초기 랜덤 데이터로 학습 → 나쁜 Q-function

**TODO 1: Batch Sampling**

**Solution:**

```python
batch = replay_buffer.sample(config["batch_size"])
```

- `batch_size`: 보통 32 (Atari) 또는 128 (CartPole)
- 반환: dictionary of numpy arrays

**TODO 2: Agent Update**

**Solution:**

```python
update_info = agent.update(
    obs=batch["observations"],
    action=batch["actions"],
    reward=batch["rewards"],
    next_obs=batch["next_observations"],
    done=batch["dones"],
    step=step,
)
```

**각 배열의 shape:**

CartPole (batch_size=128):
```python
batch["observations"]:      (128, 4)
batch["actions"]:           (128,)
batch["rewards"]:           (128,)
batch["next_observations"]: (128, 4)
batch["dones"]:             (128,)
```

Atari (batch_size=32):
```python
batch["observations"]:      (32, 4, 84, 84)
batch["actions"]:           (32,)
batch["rewards"]:           (32,)
batch["next_observations"]: (32, 4, 84, 84)
batch["dones"]:             (32,)
```

**`ptu.from_numpy(batch)`의 역할:**

Dictionary의 모든 값을 numpy → torch tensor:
```python
# Before:
batch = {
    "observations": np.array(...),
    "actions": np.array(...),
    ...
}

# After:
batch = {
    "observations": torch.tensor(...).to(device),
    "actions": torch.tensor(...).to(device),
    ...
}
```

자동으로 GPU로 이동 (설정된 경우).

#### Logging

```python
# Logging code
update_info["epsilon"] = epsilon
update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]

if step % args.log_interval == 0:
    for k, v in update_info.items():
        logger.log_scalar(v, k, step)
    logger.flush()
```

**`update_info` dictionary:**

```python
{
    "critic_loss": 0.123,
    "q_values": 5.67,
    "target_values": 5.89,
    "grad_norm": 2.34,
    "epsilon": 0.5,       # 추가
    "lr": 0.0001,         # 추가
}
```

**TensorBoard 기록:**

- `log_interval`: 1000 steps
- 매번 로깅하면 오버헤드
- TensorBoard에서 curves 확인 가능

#### Evaluation

```python
if step % args.eval_interval == 0:
    # Evaluate
    trajectories = utils.sample_n_trajectories(
        eval_env,
        agent,
        args.num_eval_trajectories,
        ep_len,
    )
    returns = [t["episode_statistics"]["r"] for t in trajectories]
    ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

    logger.log_scalar(np.mean(returns), "eval_return", step)
    logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)
```

**Evaluation 과정:**

1. **`eval_interval`**: 10,000 steps마다
2. **`num_eval_trajectories`**: 10 episodes
3. **Deterministic policy**: epsilon=0.02 (거의 greedy)
4. **통계 계산**:
   - Mean return
   - Mean episode length
   - Std, max, min (if > 1 episode)

**왜 여러 episode?**

- 단일 episode는 noisy
- 10 episodes의 평균 = 안정적인 성능 지표
- Stochasticity 고려 (초기 상태, epsilon)

**Eval vs Train 환경:**

- **Eval**: 성능 측정용 (다른 random seed)
- **Train**: 데이터 수집용
- 분리를 통해 공정한 평가

---

### 6.3 전체 훈련 과정 요약

**Phase 1: Exploration (Step 0-20K)**

```
for step in 0..20000:
    ε = 1.0 → 0.2 (linear decay)
    action = random with high probability
    env.step(action)
    replay_buffer.insert(...)
    # No learning yet!
```

- 목적: Diverse data 수집
- Replay buffer 채우기
- Q-function은 업데이트 안 함

**Phase 2: Learning (Step 20K-300K)**

```
for step in 20000..300000:
    ε = 0.2 → 0.02 (linear decay)
    action = ε-greedy(Q)
    env.step(action)
    replay_buffer.insert(...)

    if step >= 20000:
        batch = replay_buffer.sample(128)
        agent.update(batch)  # Q-function 학습!

        if step % 1000 == 0:
            target_network.copy(q_network)

    if step % 10000 == 0:
        evaluate()  # 성능 측정
```

- 목적: Q-function 학습
- Exploration 점진적 감소
- Target network 주기적 업데이트
- 주기적 evaluation

**Phase 3: Exploitation (Step 30K+)**

- ε=0.02로 고정
- 대부분 greedy 행동
- 계속 학습 (off-policy 장점)

**수렴 판단:**

CartPole: eval_return ≈ 500 (maximum)
BankHeist: eval_return > 150 (good performance)

---
