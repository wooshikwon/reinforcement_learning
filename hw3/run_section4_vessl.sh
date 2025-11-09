#!/bin/bash
# Section 4 실험을 VESSL에서 실행하는 스크립트

# 방법 1: YAML 파일 사용 (추천)
echo "=== VESSL Run 시작 (YAML 사용) ==="
vessl run create -f vessl_section4.yaml

# 방법 2: 직접 명령어 사용 (로컬 코드 업로드)
# echo "=== VESSL Run 시작 (로컬 코드 업로드) ==="
# vessl run create \
#   --name "united-senior-run" \
#   --cluster "vessl-kr-a100-80g-sxm" \
#   --resource "gpu-a100-80g-small" \
#   --image "quay.io/vessl-ai/torch:2.3.1-cuda12.1-r5" \
#   --upload . \
#   --message "pip install gymnasium[classic-control] tensorboard opencv-python pyyaml tqdm ale-py gymnasium[atari,accept-rom-license]" \
#   --command "python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/cartpole.yaml --seed 1 && python gcb6206/scripts/run_hw3.py -cfg experiments/dqn/cartpole_lr_5e-2.yaml --seed 1"

echo ""
echo "=== 실행 상태 확인 ==="
echo "VESSL 웹사이트에서 확인: https://vessl.ai/"
echo "또는 CLI로 확인: vessl run list"
echo ""
echo "=== 결과 다운로드 ==="
echo "실험 완료 후 다음 명령어로 결과 다운로드:"
echo "vessl run download <RUN_NUMBER> /output/section4/ ./results/section4/"
