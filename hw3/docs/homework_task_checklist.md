# Homework 3: Q-Learning 완료 작업 리스트

## 과제 개요
이 문서는 GCB6206 Homework 3 (Q-Learning) 과제를 완료하기 위해 해야 할 모든 작업을 순서대로 정리한 체크리스트입니다.

---

## Section 1: 이론 퀴즈 (5분)

### 1. DQN 퀴즈 4문제 True/False 답변 작성
- [ ] **Quiz I**: False (Q-learning은 off-policy로 높은 sample efficiency)
- [ ] **Quiz II**: True (연속 행동 공간에서는 actor 필요)
- [ ] **Quiz III**: True (Moving target 문제 해결위해 target network 사용)
- [ ] **Quiz IV**: False (시간에 따라 exploration을 줄임, 늘리는 게 아님)

---

## Section 2: 코드 구조 이해 (30분)

### 2. 주요 파일들 읽고 이해하기
- [ ] `gcb6206/env_configs/dqn_basic_config.py` 읽기
- [ ] `gcb6206/env_configs/dqn_atari_config.py` 읽기
- [ ] `gcb6206/infrastructure/replay_buffer.py` 읽기
- [ ] `gcb6206/infrastructure/atari_wrappers.py` 읽기

---

## Section 3: DQN 구현 (2-3시간)

### 3.1 DQN Agent 구현 (`gcb6206/agents/dqn_agent.py`)

#### 3. `get_action()` 메서드 구현 - Epsilon-greedy 행동 선택
- [ ] Epsilon-greedy 로직 구현
  ```python
  if np.random.random() < epsilon:
      action = np.random.randint(self.num_actions)  # Exploration
  else:
      with torch.no_grad():
          q_values = self.critic(observation)
          action = torch.argmax(q_values, dim=1)  # Exploitation
  ```

#### 4. `update_critic()` 메서드 구현 - DQN critic 업데이트
- [ ] Target values 계산
  ```python
  with torch.no_grad():
      next_qa_values = self.target_critic(next_obs)
      if self.use_double_q:
          next_action = torch.argmax(self.critic(next_obs), dim=1)
      else:
          next_action = torch.argmax(next_qa_values, dim=1)
      next_q_values = torch.gather(next_qa_values, 1, next_action.unsqueeze(1)).squeeze(1)
      target_values = reward + self.discount * (1 - done) * next_q_values
  ```
- [ ] Q-values 계산
  ```python
  qa_values = self.critic(obs)
  q_values = torch.gather(qa_values, 1, action.unsqueeze(1)).squeeze(1)
  ```
- [ ] Loss 계산 및 backpropagation
  ```python
  loss = self.critic_loss(q_values, target_values)
  ```

#### 5. `update()` 메서드 구현 - Main update logic
- [ ] Critic 업데이트 호출
- [ ] Target network 주기적 업데이트
  ```python
  critic_stats = self.update_critic(obs, action, reward, next_obs, done)
  if step % self.target_update_period == 0:
      self.update_target_critic()
  return critic_stats
  ```

### 3.2 Training Loop 구현 (`gcb6206/scripts/run_hw3.py`)

#### 6. Training loop TODOs 구현
- [ ] Action 계산
  ```python
  action = agent.get_action(observation, epsilon=epsilon)
  ```
- [ ] Environment step
  ```python
  next_observation, reward, terminated, truncated, info = env.step(action)
  ```
- [ ] Replay buffer insertion (regular buffer)
  ```python
  replay_buffer.insert(
      observation=observation,
      action=action,
      reward=reward,
      next_observation=next_observation,
      done=terminated,
  )
  ```
- [ ] Batch sampling
  ```python
  batch = replay_buffer.sample(config["batch_size"])
  ```
- [ ] Agent update
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

### 3.3 Trajectory Sampling 구현 (`gcb6206/infrastructure/utils.py`)

#### 7. `sample_trajectory()` TODOs 구현
- [ ] Action selection
  ```python
  ac = agent.get_action(ob)
  ```
- [ ] Environment step
  ```python
  next_ob, rew, terminated, truncated, info = env.step(ac)
  ```
- [ ] Rollout done flag (terminated or truncated)
  ```python
  rollout_done = terminated or truncated
  ```

---

## Section 4: DQN 실험 (1-2시간)

### 4.1 CartPole 기본 실험

#### 8. CartPole-v1 실험 실행 (~15분)
- [ ] 실험 실행
  ```bash
  python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/cartpole.yaml --seed 1
  ```
- [ ] 목표: eval_return ~500

#### 9. Learning curve plot 생성
- [ ] X축: environment steps
- [ ] Y축: eval_return
- [ ] Caption 작성
- [ ] Plot 저장

### 4.2 Learning Rate 비교 실험

#### 10. Config 파일 생성
- [ ] `experiments/dqn/cartpole_lr_5e-2.yaml` 생성
- [ ] learning_rate: 0.05로 변경

#### 11. 높은 LR 실험 실행 (~15분)
- [ ] 실험 실행
  ```bash
  python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/cartpole_lr_5e-2.yaml --seed 1
  ```

#### 12. 3개 비교 plot 생성
- [ ] (a) Predicted Q-values 비교
- [ ] (b) Critic error 비교
- [ ] (c) Eval returns 비교
- [ ] 각 plot에 caption 추가

#### 13. 결과 분석 및 설명 작성
- [ ] 높은 LR의 영향 설명
- [ ] 강의 내용과 연결
- [ ] Trade-off 분석

---

## Section 5: Double DQN 구현 및 실험 (12-24시간)

### 5.1 Double DQN 구현

#### 14. `update_critic()` 내 Double Q-Learning 로직 구현
- [ ] Online network로 action selection
- [ ] Target network로 value estimation
- [ ] 구현 확인 (이미 Section 3.1에서 완료됨)

### 5.2 BankHeist 실험

#### 15. Vanilla DQN 실험 (3 seeds) (~6시간 GPU / 12시간 CPU)
- [ ] Seed 1 실험
  ```bash
  python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 1
  ```
- [ ] Seed 2 실험
  ```bash
  python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 2
  ```
- [ ] Seed 3 실험
  ```bash
  python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 3
  ```
- [ ] 목표: eval_return ~150

#### 16. Double DQN 실험 (3 seeds) (~6시간 GPU / 12시간 CPU)
- [ ] Seed 1 실험
  ```bash
  python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 1
  ```
- [ ] Seed 2 실험
  ```bash
  python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 2
  ```
- [ ] Seed 3 실험
  ```bash
  python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 3
  ```
- [ ] 목표: eval_return ~300

#### 17. 비교 plot 생성
- [ ] DQN 3 curves (blue)
- [ ] Double DQN 3 curves (red)
- [ ] 같은 axes에 표시
- [ ] Legend 추가
- [ ] Caption 작성

#### 18. 결과 분석 및 설명 작성
- [ ] Double DQN의 성능 향상 설명
- [ ] Overestimation bias 감소 메커니즘 설명
- [ ] 3 seeds 결과의 일관성 논의

---

## Section 6: Hyperparameter 실험 (1-2시간)

#### 19. Hyperparameter 선택
- [ ] 선택한 hyperparameter 결정 (예: exploration schedule, learning rate, network architecture 등)
- [ ] 선택 이유 작성

#### 20. 4개 config 파일 생성
- [ ] `experiments/dqn/hyperparameters/` 디렉토리 생성
- [ ] Config 1 (default) 생성
- [ ] Config 2 생성
- [ ] Config 3 생성
- [ ] Config 4 생성

#### 21. 4가지 실험 실행 (~1시간)
- [ ] Config 1 실험
- [ ] Config 2 실험
- [ ] Config 3 실험
- [ ] Config 4 실험

#### 22. 비교 plot 생성
- [ ] 4 curves 표시
- [ ] 다른 색상 사용
- [ ] Legend 추가
- [ ] Caption 작성

#### 23. 분석 및 설명 작성
- [ ] Hyperparameter 선택 이유
- [ ] 각 설정의 의미 설명
- [ ] 결과 관찰 및 분석
- [ ] 이론적 설명
- [ ] 결론 및 최적 설정

---

## Section 7: 최종 제출 준비 (1시간)

#### 24. 모든 plot 검토 및 caption 작성
- [ ] Section 4.2: CartPole DQN plot
- [ ] Section 4.2: Learning rate 비교 plots (3개)
- [ ] Section 5.2: BankHeist DQN vs Double DQN plot
- [ ] Section 6: Hyperparameter 비교 plot
- [ ] 모든 caption 완성도 확인

#### 25. 설명/분석 텍스트 작성
- [ ] Quiz 답변 완료
- [ ] CartPole LR 실험 설명
- [ ] BankHeist 비교 설명
- [ ] Hyperparameter 분석

#### 26. PDF 보고서 작성
- [ ] 이름/학번 입력
- [ ] Section 2: Quiz 답변
- [ ] Section 4.2: CartPole plots + 설명
- [ ] Section 4.2: LR 비교 plots + 설명
- [ ] Section 5.2: BankHeist plot + 설명
- [ ] Section 6: Hyperparameter plot + 설명
- [ ] 전체 formatting 확인

#### 27. 제출 파일 구조화
- [ ] 파일 구조 확인
  ```
  hw3_[YourStudentID].zip
  ├── hw3_[YourStudentID].pdf
  ├── gcb6206/
  │   ├── agents/
  │   │   └── dqn_agent.py
  │   ├── scripts/
  │   │   └── run_hw3.py
  │   ├── infrastructure/
  │   │   └── utils.py
  │   └── ...codes
  └── data/
      ├── hw3_dqn_cartpole/
      ├── hw3_dqn_bankheist/
      └── ...
          └── events.out.tfevents....
  ```

#### 28. 파일 크기 확인
- [ ] 총 크기 < 50MB
- [ ] 비디오 파일 제외 확인
- [ ] 불필요한 파일 삭제

#### 29. 최종 제출
- [ ] ZIP 파일 생성
- [ ] 파일명 확인: `hw3_[YourStudentID].zip`
- [ ] 제출

---

## 추정 총 소요 시간

| 작업 단계 | 예상 시간 |
|---------|---------|
| **구현** | 3-4시간 |
| **CartPole 실험** | 1시간 |
| **BankHeist 실험** | 12-24시간 (GPU/CPU) |
| **Hyperparameter 실험** | 1-2시간 |
| **보고서 작성** | 1-2시간 |
| **총계** | **~18-33시간** |

---

## 중요 참고 사항

### ⚠️ 우선순위 높은 작업
1. **Section 5.2 BankHeist 실험을 최대한 빨리 시작**
   - 실험 시간이 가장 오래 걸림 (12-24시간)
   - GPU 사용 권장 (VESSL AI 또는 Colab)
   - 3 seeds × 2 methods = 총 6개 실험

### 💡 팁
- **병렬 실행**: 여러 seed를 동시에 다른 GPU/머신에서 실행
- **체크포인트**: 실험 중간중간 결과 확인
- **디버깅**: CartPole에서 먼저 구현 검증 후 Atari로 진행
- **TensorBoard**: 실시간으로 학습 곡선 모니터링
  ```bash
  tensorboard --logdir data/
  ```

### 📝 참고 문서
- `solution_guide_section1.md`: Quiz 답변 및 이론 설명
- `solution_guide_section2.md`: 코드 구조 및 replay buffer
- `solution_guide_section3.md`: DQN agent 구현 상세
- `solution_guide_section4.md`: Training loop 구현 상세
- `solution_guide_section5.md`: Double DQN 및 실험 가이드

---

## 진행 상황 추적

- **시작일**: ___________
- **예상 완료일**: ___________
- **실제 완료일**: ___________

### 마일스톤
- [ ] 구현 완료 (Section 3)
- [ ] CartPole 실험 완료 (Section 4)
- [ ] BankHeist 실험 시작 (Section 5)
- [ ] BankHeist 실험 완료 (Section 5)
- [ ] Hyperparameter 실험 완료 (Section 6)
- [ ] 보고서 작성 완료 (Section 7)
- [ ] 최종 제출 완료

---

**Good Luck!** 🚀
