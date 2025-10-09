"""
Some miscellaneous utility functions

Functions to edit:
    1. rollout_trajectory (line 19)
    2. rollout_trajectories (line 67)
    3. rollout_n_trajectories (line 83)
"""

import numpy as np
import time

############################################
############################################

MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]
MJ_ENV_KWARGS = {name: {"render_mode": "rgb_array"} for name in MJ_ENV_NAMES}
MJ_ENV_KWARGS["Ant-v4"]["use_contact_forces"] = True


def rollout_trajectory(env, policy, max_traj_length, render=False):
    """
    Rolls out a policy and generates a trajectories

    :param policy: the policy to roll out
    :param max_traj_length: the number of steps to roll out
    :render: whether to save images from the rollout
    """
    # initialize env for the beginning of a new rollout
    # TODO: implement the following line
    # 환경을 초기 상태로 리셋하고 첫 observation을 받아옴
    # Gymnasium의 reset()은 (observation, info) 튜플을 반환
    # info는 사용하지 않으므로 _로 무시
    ob, _ = env.reset()  # HINT: should be the output of resetting the env

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render image of the simulated env
        if render:
            if hasattr(env, "sim"):
                image_obs.append(
                    env.sim.render(camera_name="track", height=500, width=500)[::-1]
                )
            else:
                image_obs.append(env.render())

        # use the most recent ob to decide what to do
        obs.append(ob)
        # 현재 observation을 정책에 입력하여 행동(action) 선택
        # policy.get_action()은 observation을 받아 행동을 샘플링하여 반환
        ac = policy.get_action(ob)  # HINT: query the policy's get_action function
        acs.append(ac)

        # take that action and record results
        ob, rew, terminated, truncated, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # TODO end the rollout if the rollout ended
        # HINT: rollout can end due to termination or truncation, or due to exceeding or reaching (>=) max_traj_length
        # rollout 종료 조건 3가지:
        # 1. terminated: 환경이 종료 상태 도달 (예: 로봇이 넘어짐, 목표 달성)
        # 2. truncated: 시간 제한 도달 (max episode steps)
        # 3. steps >= max_traj_length: 지정한 최대 trajectory 길이 도달
        # 하나라도 True면 rollout 종료 (1), 모두 False면 계속 (0)
        rollout_done = (terminated or truncated) or (steps >= max_traj_length)  # HINT: this is either 0 or 1

        terminals.append(rollout_done)

        if rollout_done:
            break

    return Traj(obs, image_obs, acs, rewards, next_obs, terminals)


def rollout_trajectories(
    env, policy, min_timesteps_per_batch, max_traj_length, render=False
):
    """
    Collect rollouts until we have collected min_timesteps_per_batch steps.

    TODO implement this function
    Hint1: use `rollout_trajectory` to get each traj (i.e. rollout) that goes into trajs
    Hint2: use `get_trajlength` to count the timesteps collected in each traj
    Hint3: repeat while we have collected at least min_timesteps_per_batch steps
    """
    # 현재까지 수집한 총 timestep 수를 추적하는 카운터 (0부터 시작)
    timesteps_this_batch = 0
    # 수집한 trajectory들을 저장할 리스트
    trajs = []

    # 목표 timestep 수(min_timesteps_per_batch)에 도달할 때까지 반복
    # 예: batch_size=1000이면 1000 timesteps 수집할 때까지 계속
    while timesteps_this_batch < min_timesteps_per_batch:
        # rollout_trajectory를 호출하여 하나의 trajectory 수집
        # env에서 policy를 실행하여 max_traj_length까지 또는 종료시까지 데이터 수집
        traj = rollout_trajectory(env, policy, max_traj_length, render)

        # 수집한 trajectory를 리스트에 추가
        trajs.append(traj)

        # 이번 trajectory의 길이(timestep 수)를 누적
        # get_trajlength()는 trajectory의 reward 배열 길이를 반환 (= step 수)
        timesteps_this_batch += get_trajlength(traj)

    # 수집한 trajectory 리스트와 총 timestep 수 반환
    return trajs, timesteps_this_batch


def rollout_n_trajectories(env, policy, ntraj, max_traj_length, render=False):
    """
    Collect ntraj rollouts.

    TODO implement this function
    Hint1: use rollout_trajectory to get each traj (i.e. rollout) that goes into trajs
    """
    # 수집한 trajectory들을 저장할 리스트
    trajs = []

    # 정확히 ntraj개의 trajectory를 수집
    # rollout_trajectories()는 timestep 기반이지만, 이 함수는 개수 기반
    # 주로 비디오 저장용으로 사용 (예: MAX_NVIDEO=2개의 trajectory)
    for _ in range(ntraj):
        # rollout_trajectory를 호출하여 하나의 trajectory 수집
        # 각 trajectory는 max_traj_length까지 또는 종료시까지 실행
        traj = rollout_trajectory(env, policy, max_traj_length, render)

        # 수집한 trajectory를 리스트에 추가
        trajs.append(traj)

    # ntraj개의 trajectory 리스트 반환
    return trajs


############################################
############################################


def Traj(obs, image_obs, acs, rewards, next_obs, terminals):
    """
    Take info (separate arrays) from a single rollout
    and return it in a single dictionary
    """
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


def convert_listofrollouts(trajs, concat_rew=True):
    """
    Take a list of rollout dictionaries
    and return separate arrays,
    where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    if concat_rew:
        rewards = np.concatenate([traj["reward"] for traj in trajs])
    else:
        rewards = [traj["reward"] for traj in trajs]
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    return observations, actions, rewards, next_observations, terminals


############################################
############################################


def get_trajlength(traj):
    return len(traj["reward"])
