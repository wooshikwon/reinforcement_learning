# VESSL에서 Section 4 실험 실행하기

## 🚀 빠른 실행 (로컬 코드 업로드 방식)

현재 디렉토리에서 다음 명령어를 실행하세요:

```bash
cd /Users/wooshikwon/Desktop/github_wooshikwon/reinforcement_learning/hw3

vessl run create \
  --cluster vessl-kr-a100-80g-sxm \
  --resource gpu-a100-80g-small \
  --image quay.io/vessl-ai/torch:2.3.1-cuda12.1-r5 \
  --upload . \
  --message "pip install gymnasium[classic-control] tensorboard opencv-python pyyaml tqdm ale-py gymnasium[atari,accept-rom-license]" \
  "python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/cartpole.yaml --seed 1 && python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/cartpole_lr_5e-2.yaml --seed 1 && mkdir -p /output && cp -r data /output/"
```

## 📊 실행 상태 확인

### 웹에서 확인:
https://vessl.ai/wooshikwon/gcb6206/runs

### CLI로 확인:
```bash
# 실행 목록 보기
vessl run list

# 특정 run 상태 확인
vessl run get <RUN_NUMBER>

# 로그 실시간 확인
vessl run logs <RUN_NUMBER> -f
```

## 📥 결과 다운로드

실험이 완료되면 (약 15분 소요):

```bash
# run number 확인
vessl run list

# 결과 다운로드
vessl run download <RUN_NUMBER> /output/data ./results/section4/

# 또는 특정 파일만
vessl run download <RUN_NUMBER> /output/data/hw3_dqn_cartpole ./results/section4/cartpole
vessl run download <RUN_NUMBER> /output/data/hw3_dqn_cartpole_lr_5e-2 ./results/section4/cartpole_lr_5e-2
```

## 🔧 문제 해결

### Run이 실패하면:
```bash
# 로그 확인
vessl run logs <RUN_NUMBER>

# 재실행
vessl run create ... (위 명령어 다시 실행)
```

### 업로드 파일 크기 줄이기:
```bash
# .gitignore에 data/ 추가되어 있는지 확인
cat .gitignore | grep data

# 또는 특정 파일만 업로드
vessl run create --upload gcb6206/ --upload experiments/ ...
```

## 📝 예상 결과

실험 완료 후 다음 디렉토리에 결과 저장:
- `results/section4/cartpole/` - 기본 DQN 결과
- `results/section4/cartpole_lr_5e-2/` - 높은 LR 결과

각 디렉토리에는 TensorBoard 로그 파일이 포함:
- `events.out.tfevents.*`

## 🎨 Plot 생성

결과 다운로드 후:
```bash
# TensorBoard로 확인
tensorboard --logdir results/section4/

# 또는 parse 스크립트 사용
python gcb6206/scripts/parse_tensorboard.py results/section4/cartpole
python gcb6206/scripts/parse_tensorboard.py results/section4/cartpole_lr_5e-2
```
