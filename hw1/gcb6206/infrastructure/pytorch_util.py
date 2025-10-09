"""
Functions to edit:
    1. build_mlp (line 26)
"""

from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}


def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation: Activation = "tanh",
    output_activation: Activation = "identity",
) -> nn.Module:
    """
    Builds a feedforward neural network

    arguments:
        n_layers: number of hidden layers
        size: dimension of each hidden layer
        activation: activation of each hidden layer

        input_size: size of the input layer
        output_size: size of the output layer
        output_activation: activation of the output layer

    returns:
        MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    # HINT 1: Take a look at the following link to see how nn.Sequential works:
    # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    # HINT 2: We are only using linear layers and activation layers.
    # HINT 3: You can simple create a list, append nn layers, and convert with nn.Sequential.

    # 신경망 레이어들을 저장할 빈 리스트 생성
    layers = []

    # 입력층 → 첫 번째 은닉층: input_size 차원을 size 차원으로 선형 변환
    layers.append(nn.Linear(input_size, size))
    # 첫 번째 은닉층의 활성화 함수 추가 (비선형성 부여)
    layers.append(activation)

    # 나머지 은닉층들 추가: n_layers-1개의 은닉층 (첫 번째는 이미 추가했으므로)
    for _ in range(n_layers - 1):
        # 은닉층: size 차원을 size 차원으로 변환 (같은 크기 유지)
        layers.append(nn.Linear(size, size))
        # 각 은닉층마다 활성화 함수 추가
        layers.append(activation)

    # 출력층: 마지막 은닉층(size)을 출력 차원(output_size)으로 변환
    layers.append(nn.Linear(size, output_size))
    # 출력층의 활성화 함수 추가 (연속 행동 공간에서는 보통 identity)
    layers.append(output_activation)

    # nn.Sequential로 레이어들을 순차적으로 실행하는 하나의 모듈로 결합
    # *layers는 리스트를 개별 인자들로 언패킹
    return nn.Sequential(*layers)


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device

    # CUDA (NVIDIA GPU) 체크
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))

    # MPS (Apple Silicon GPU) 체크
    elif torch.backends.mps.is_available() and use_gpu:
        device = torch.device("mps")
        print("Using Apple Silicon MPS")

    # CPU fallback
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    """
    CUDA 전용 함수. MPS는 단일 디바이스이므로 set_device 불필요.
    """
    if device.type == "cuda":
        torch.cuda.set_device(gpu_id)
    elif device.type == "mps":
        pass  # MPS는 단일 디바이스, 설정 불필요


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to("cpu").detach().numpy()
