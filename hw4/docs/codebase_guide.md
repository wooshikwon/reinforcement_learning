# GCB6206 Homework 4 Codebase Structure Guide
## A Beginner's Guide to Understanding Soft Actor-Critic Implementation

---

## Table of Contents

1. [Introduction](#introduction)
2. [Reinforcement Learning Fundamentals](#reinforcement-learning-fundamentals)
3. [Soft Actor-Critic (SAC) Overview](#soft-actor-critic-overview)
4. [Project Structure](#project-structure)
5. [Core Components Deep Dive](#core-components-deep-dive)
6. [Data Flow and Training Loop](#data-flow-and-training-loop)
7. [Key Concepts and Design Patterns](#key-concepts-and-design-patterns)
8. [How to Navigate the Codebase](#how-to-navigate-the-codebase)

---

## Introduction

### What is this project?

This codebase implements **Soft Actor-Critic (SAC)**, one of the most successful deep reinforcement learning algorithms for continuous control tasks. It's designed for training agents to solve tasks like making a robotic arm reach a target, teaching a humanoid to walk, or controlling a simulated cheetah to run.

### Who is this guide for?

- **Python beginners** who want to understand reinforcement learning code
- **RL beginners** who want to see how theoretical concepts translate to code
- **Students** working on the homework assignment
- **Anyone** curious about how modern RL algorithms are implemented

### What you'll learn:

1. How RL concepts (states, actions, rewards) map to code
2. The structure and organization of an RL codebase
3. How neural networks are used in actor-critic methods
4. Best practices for RL implementation

---

## Reinforcement Learning Fundamentals

### The Big Picture

Imagine you're training a robot dog to walk:

```
┌─────────────┐
│ Environment │ ← The simulated world (physics, gravity, etc.)
└─────────────┘
      ↓ observation (state)
      ↑ action
┌─────────────┐
│    Agent    │ ← The "brain" that decides what to do
└─────────────┘
```

**The learning loop:**
1. Agent observes the current **state** (joint angles, velocities)
2. Agent chooses an **action** (motor torques)
3. Environment responds with new **state** and **reward**
4. Agent learns to maximize total reward over time

### Key RL Concepts

#### 1. Markov Decision Process (MDP)

An MDP is the mathematical framework for RL:

- **State (s)**: Complete description of the environment
  - Example: [position, velocity, angle, angular_velocity]
- **Action (a)**: What the agent can do
  - Discrete: {left, right, jump}
  - Continuous: [motor_1_torque, motor_2_torque, ...]
- **Reward (r)**: Scalar feedback signal
  - Example: +1 for staying upright, -100 for falling
- **Transition**: p(s'|s,a) - probability of next state
- **Policy (π)**: π(a|s) - agent's strategy (state → action)

#### 2. Value Functions

**Q-function (Action-Value)**: Q(s, a) = Expected total reward starting from state s, taking action a, then following policy π

```
Q(s, a) = E[r_0 + γr_1 + γ²r_2 + γ³r_3 + ...]
```

Where γ (gamma) is the discount factor (0 < γ < 1):
- γ = 0.99: Care about long-term rewards
- γ = 0.0: Only care about immediate reward

**Why Q-functions?**
If we know Q(s, a) for all actions, we can choose the best action:
```
π*(s) = argmax_a Q(s, a)
```

#### 3. Policy Gradient Methods

Instead of learning Q-values, directly learn the policy π_θ(a|s):

**Idea**: Adjust policy parameters θ to increase expected reward

```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) · Q(s,a)]
```

**Intuition**: If action a led to good Q-value, make it more likely!

#### 4. Actor-Critic Methods

Combine both approaches:

- **Actor (Policy)**: π_θ(a|s) - Decides what action to take
- **Critic (Value)**: Q_φ(s, a) - Evaluates how good the action is

**Advantages:**
- Lower variance than pure policy gradient
- More efficient than pure value methods
- Works well for continuous actions

---

## Soft Actor-Critic (SAC) Overview

### What makes SAC "Soft"?

**Standard RL objective**: Maximize total reward
```
max E[∑ γ^t r_t]
```

**SAC objective**: Maximize reward + entropy
```
max E[∑ γ^t (r_t + α·H(π(·|s_t)))]
```

Where H(π) is entropy: H(π) = -E[log π(a|s)]

**Why entropy?**
- **Exploration**: Encourages trying different actions
- **Robustness**: Learns multiple ways to solve task
- **Prevents collapse**: Avoids premature convergence

### SAC Algorithm Components

```
┌──────────────────────────────────────┐
│           SAC Agent                  │
├──────────────────────────────────────┤
│                                      │
│  ┌────────────┐    ┌──────────────┐ │
│  │   Actor    │    │   Critics    │ │
│  │  π_θ(a|s)  │    │  Q_φ(s,a)    │ │
│  └────────────┘    └──────────────┘ │
│        │                  │          │
│        ↓                  ↓          │
│   Sample action      Evaluate Q     │
│                                      │
│  ┌──────────────────────────────┐   │
│  │   Target Critics (slowly     │   │
│  │   updated copy of critics)   │   │
│  └──────────────────────────────┘   │
└──────────────────────────────────────┘
```

**Key features:**
1. **Off-policy**: Learn from old experiences (replay buffer)
2. **Maximum entropy**: Add entropy bonus
3. **Twin critics**: Reduce overestimation bias
4. **Continuous actions**: Handle real-valued actions

### SAC Training Algorithm

```
Repeat:
    1. Sample action: a ~ π_θ(·|s)
    2. Execute action, observe (s, a, r, s', done)
    3. Store in replay buffer

    4. Sample batch from replay buffer

    5. Update critic:
       - Compute target: y = r + γ(Q'(s',a') + α·H(π(·|s')))
       - Minimize: (Q_φ(s,a) - y)²

    6. Update actor:
       - Maximize: Q_φ(s, π_θ(s)) + α·H(π_θ(·|s))

    7. Update target critics (slowly)
```

---

## Project Structure

### Directory Layout

```
hw4/
├── gcb6206/                      # Main package
│   ├── agents/                   # Agent implementations
│   │   └── sac_agent.py         # SAC algorithm
│   ├── networks/                 # Neural network architectures
│   │   ├── mlp_policy.py        # Actor network (policy)
│   │   └── state_action_value_critic.py  # Critic network
│   ├── infrastructure/           # Utilities and helpers
│   │   ├── replay_buffer.py     # Experience replay
│   │   ├── logger.py            # TensorBoard logging
│   │   ├── pytorch_util.py      # PyTorch helpers
│   │   ├── distributions.py     # Custom distributions
│   │   └── utils.py             # General utilities
│   ├── env_configs/             # Environment configurations
│   │   ├── sac_config.py        # SAC hyperparameters
│   │   └── schedule.py          # Learning rate schedules
│   └── scripts/                 # Entry points
│       ├── run_hw4.py           # Main training script
│       └── scripting_utils.py   # Config loading
├── experiments/                  # Experiment configs
│   └── sac/                     # SAC experiment YAML files
│       ├── sanity_pendulum_1.yaml
│       ├── halfcheetah_reinforce1.yaml
│       └── ...
├── docs/                        # Documentation
├── requirements.txt             # Python dependencies
└── setup.py                     # Package installation
```

### Design Philosophy

**Separation of Concerns:**
- **agents/**: Algorithm logic (what to learn)
- **networks/**: Neural network architectures (function approximators)
- **infrastructure/**: Reusable components (replay buffer, logging)
- **env_configs/**: Hyperparameters (how to learn)
- **scripts/**: Execution logic (when to do what)

---

## Core Components Deep Dive

### 1. Agent: `gcb6206/agents/sac_agent.py`

**Purpose**: Implements the SAC learning algorithm

**Class**: `SoftActorCritic`

#### Key Attributes

```python
class SoftActorCritic(nn.Module):
    def __init__(...):
        # Actor: π_θ(a|s) - policy network
        self.actor = make_actor(observation_shape, action_dim)

        # Critics: Q_φ(s,a) - value networks (can be multiple)
        self.critics = nn.ModuleList([
            make_critic(observation_shape, action_dim)
            for _ in range(num_critic_networks)
        ])

        # Target critics: Q_φ'(s,a) - slowly updated copies
        self.target_critics = nn.ModuleList([...])

        # Optimizers
        self.actor_optimizer = make_actor_optimizer(...)
        self.critic_optimizer = make_critic_optimizer(...)

        # Hyperparameters
        self.discount = discount  # γ (gamma)
        self.temperature = temperature  # α (alpha) for entropy
```

#### Key Methods

**`get_action(observation)`**
```python
def get_action(self, observation: np.ndarray) -> np.ndarray:
    """Select action for execution in environment"""
    # 1. Convert numpy → torch
    # 2. Get policy distribution π(·|s)
    # 3. Sample action a ~ π(·|s)
    # 4. Convert torch → numpy
```

**`update_critic(obs, action, reward, next_obs, done)`**
```python
def update_critic(...):
    """One gradient step on critic networks"""
    # 1. Sample next action: a' ~ π(·|s')
    # 2. Compute target: y = r + γ(Q'(s',a') + α·H(π(·|s')))
    # 3. Compute Q-values: Q(s,a)
    # 4. Compute loss: MSE(Q, y)
    # 5. Backprop and update
```

**`update_actor(obs)`**
```python
def update_actor(obs):
    """One gradient step on actor network"""
    # Two variants:
    # REINFORCE: ∇ log π(a|s) · Q(s,a)
    # REPARAMETRIZE: ∇ Q(s, π(s))
```

**`update(observations, actions, rewards, next_observations, dones, step)`**
```python
def update(...):
    """Main update: critic + actor + target networks"""
    # 1. Update critic (multiple times)
    # 2. Update actor (once)
    # 3. Update target networks
```

---

### 2. Actor Network: `gcb6206/networks/mlp_policy.py`

**Purpose**: Neural network that outputs a probability distribution over actions

**Class**: `MLPPolicy`

#### Architecture

```
Input: state (observation)
    ↓
[Linear Layer → Activation] × n_layers
    ↓
Output Layer
    ↓
For continuous actions:
    - Mean: μ(s)
    - Std: σ(s) (optionally state-dependent)
    ↓
Distribution: π(a|s) = N(μ(s), σ(s))  or  Tanh(N(μ(s), σ(s)))
```

#### Key Code

```python
def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
    """
    Input: batch of observations [batch_size, obs_dim]
    Output: action distribution for each observation
    """
    if self.state_dependent_std:
        # Both mean and std depend on state
        mean, std = torch.chunk(self.net(obs), 2, dim=-1)
        std = F.softplus(std) + 1e-2  # Ensure positive
    else:
        # Only mean depends on state
        mean = self.net(obs)
        std = F.softplus(self.std) + 1e-2  # Learnable parameter

    if self.use_tanh:
        # Squash actions to [-1, 1]
        return make_tanh_transformed(mean, std)
    else:
        return make_multi_normal(mean, std)
```

**Why Tanh?**
- Many environments expect actions in [-1, 1]
- Tanh transform: a = tanh(ã) where ã ~ N(μ, σ)
- Prevents extreme actions

---

### 3. Critic Network: `gcb6206/networks/state_action_value_critic.py`

**Purpose**: Neural network that estimates Q(s, a)

**Class**: `StateActionCritic`

#### Architecture

```
Input: concatenated [state, action]
    ↓
[Linear Layer → Activation] × n_layers
    ↓
Output Layer (1 value)
    ↓
Output: Q(s, a) - scalar value
```

#### Key Code

```python
class StateActionCritic(nn.Module):
    def __init__(self, ob_dim, ac_dim, n_layers, size):
        super().__init__()
        # MLP: (obs_dim + action_dim) → hidden → ... → 1
        self.net = ptu.build_mlp(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        )

    def forward(self, obs, acs):
        # Concatenate state and action
        input = torch.cat([obs, acs], dim=-1)
        # Output Q-value
        return self.net(input).squeeze(-1)
```

**Design note**: Unlike DQN which outputs Q(s, a) for all actions, this outputs Q-value for specific (s, a) pair. This is necessary for continuous action spaces.

---

### 4. Replay Buffer: `gcb6206/infrastructure/replay_buffer.py`

**Purpose**: Store and sample past experiences for off-policy learning

**Class**: `ReplayBuffer`

#### Why Replay Buffer?

**Problem**: RL data is highly correlated
- Sequential states are similar
- Leads to overfitting and instability

**Solution**: Store experiences, sample randomly
- Breaks correlation
- Reuses data efficiently
- Enables off-policy learning

#### Key Operations

```python
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.max_size = capacity
        self.observations = None  # Allocated lazily
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dones = None

    def insert(self, observation, action, reward, next_observation, done):
        """Add one transition (s, a, r, s', done)"""
        # Circular buffer: overwrites oldest when full
        idx = self.size % self.max_size
        self.observations[idx] = observation
        # ... store other fields
        self.size += 1

    def sample(self, batch_size):
        """Sample random batch of transitions"""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }
```

**Memory efficiency**: Uses numpy arrays, only allocates once

---

### 5. Configuration: `gcb6206/env_configs/sac_config.py`

**Purpose**: Define all hyperparameters in one place

**Function**: `sac_config(env_name, **kwargs)`

#### Key Hyperparameters

```python
def sac_config(
    env_name: str,

    # Network architecture
    hidden_size: int = 128,
    num_layers: int = 3,

    # Learning rates
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,

    # Training
    total_steps: int = 300000,
    batch_size: int = 128,
    discount: float = 0.99,

    # Exploration
    random_steps: int = 5000,  # Random actions at start
    training_starts: int = 10000,  # Start training after this

    # Target networks
    use_soft_target_update: bool = False,
    target_update_period: int = None,  # Hard update
    soft_target_update_rate: float = None,  # Soft update τ

    # Actor-critic
    actor_gradient_type: str = "reinforce",  # or "reparametrize"
    num_actor_samples: int = 1,
    num_critic_updates: int = 1,
    num_critic_networks: int = 1,  # 2 for Double-Q

    # Entropy
    use_entropy_bonus: bool = True,
    temperature: float = 0.1,  # α for entropy
):
    # Returns dict with all configs
```

**Configuration pattern**:
- Base configurations in Python
- Overrides in YAML files (`experiments/sac/*.yaml`)
- Allows easy experimentation

---

### 6. Training Script: `gcb6206/scripts/run_hw4.py`

**Purpose**: Main training loop, ties everything together

#### Training Loop Structure

```python
def run_training_loop(config, logger, args):
    # 1. Setup
    env = config["make_env"]()
    agent = SoftActorCritic(...)
    replay_buffer = ReplayBuffer(...)

    observation, _ = env.reset()

    # 2. Main loop
    for step in range(config["total_steps"]):

        # 3. Collect data
        if step < config["random_steps"]:
            action = env.action_space.sample()  # Random
        else:
            action = agent.get_action(observation)  # From policy

        next_observation, reward, done, truncated, info = env.step(action)
        replay_buffer.insert(observation, action, reward, next_observation, done)

        # 4. Train agent
        if step >= config["training_starts"]:
            batch = replay_buffer.sample(config["batch_size"])
            batch = ptu.from_numpy(batch)  # numpy → torch
            update_info = agent.update(**batch, step=step)

            # Log training stats
            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)

        # 5. Evaluate
        if step % args.eval_interval == 0:
            eval_returns = evaluate(agent, eval_env)
            logger.log_scalar(np.mean(eval_returns), "eval_return", step)

        # 6. Reset if done
        if done or truncated:
            observation, _ = env.reset()
        else:
            observation = next_observation
```

**Key stages:**
1. **Random exploration** (0 to random_steps): Build initial replay buffer
2. **Learning** (training_starts onwards): Update networks
3. **Evaluation** (periodic): Test without exploration noise

---

## Data Flow and Training Loop

### Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Training Loop                           │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌──────────────────┴──────────────────┐
        ↓                                      ↓
┌──────────────┐                     ┌──────────────┐
│ Environment  │                     │ Replay Buffer│
│              │                     │              │
│ observation  │────────────────────→│ (s,a,r,s',d) │
│   ↓          │                     │              │
│ Agent.get    │                     │              │
│   _action()  │                     │ Sample batch │
│   ↓          │                     │      ↓       │
│ action       │                     └──────┼───────┘
│   ↓          │                            │
│ env.step()   │                            ↓
│   ↓          │                   ┌────────────────┐
│(s',r,done)   │                   │ Agent.update() │
└──────────────┘                   │                │
                                   │  Update critic │
                                   │  Update actor  │
                                   │  Update target │
                                   └────────────────┘
```

### Step-by-Step Execution

**Step 1: Environment Interaction**
```python
observation = env.reset()  # Get initial state
# observation shape: [obs_dim]
```

**Step 2: Action Selection**
```python
action = agent.get_action(observation)
# Inside get_action():
#   obs_torch = ptu.from_numpy(observation)[None]  # [1, obs_dim]
#   dist = self.actor(obs_torch)  # Get π(·|s)
#   action = dist.sample()  # Sample a ~ π(·|s)
#   return ptu.to_numpy(action).squeeze(0)  # [action_dim]
```

**Step 3: Environment Step**
```python
next_obs, reward, done, truncated, info = env.step(action)
# next_obs: [obs_dim]
# reward: scalar
# done: bool
```

**Step 4: Store in Replay Buffer**
```python
replay_buffer.insert(observation, action, reward, next_obs, done)
```

**Step 5: Sample and Train**
```python
batch = replay_buffer.sample(batch_size)
# batch = {
#   "observations": [batch_size, obs_dim],
#   "actions": [batch_size, action_dim],
#   "rewards": [batch_size],
#   "next_observations": [batch_size, obs_dim],
#   "dones": [batch_size],
# }

batch = ptu.from_numpy(batch)  # Convert to torch tensors

update_info = agent.update(**batch, step=step)
# Returns dict of training metrics
```

### Inside `agent.update()`

```python
def update(self, observations, actions, rewards, next_observations, dones, step):
    # 1. Update critic multiple times
    for _ in range(self.num_critic_updates):
        critic_info = self.update_critic(
            observations, actions, rewards, next_observations, dones
        )

    # 2. Update actor once
    actor_info = self.update_actor(observations)

    # 3. Update target networks
    if step % self.target_update_period == 0:  # Hard update
        self.update_target_critic()
    # OR
    self.soft_update_target_critic(tau=0.005)  # Soft update

    return {**actor_info, **critic_info}
```

### Inside `update_critic()`

```python
def update_critic(self, obs, action, reward, next_obs, done):
    # 1. Compute targets (no gradients)
    with torch.no_grad():
        next_action_dist = self.actor(next_obs)
        next_action = next_action_dist.sample()
        next_q = self.target_critic(next_obs, next_action)

        if self.use_entropy_bonus:
            entropy = self.entropy(next_action_dist)
            next_q += self.temperature * entropy

        target = reward + self.discount * (1 - done) * next_q

    # 2. Predict Q-values
    q_values = self.critic(obs, action)

    # 3. Compute loss and update
    loss = self.critic_loss(q_values, target)  # MSE

    self.critic_optimizer.zero_grad()
    loss.backward()
    self.critic_optimizer.step()
```

### Inside `update_actor()`

**REINFORCE version:**
```python
def actor_loss_reinforce(self, obs):
    # 1. Get policy distribution
    action_dist = self.actor(obs)

    # 2. Sample actions (no gradients for actions)
    with torch.no_grad():
        actions = action_dist.sample((num_samples,))
        q_values = self.critic(obs, actions)

    # 3. Compute REINFORCE gradient
    log_probs = action_dist.log_prob(actions)
    loss = -(log_probs * q_values).mean()

    return loss
```

**REPARAMETRIZE version:**
```python
def actor_loss_reparametrize(self, obs):
    # 1. Get policy distribution
    action_dist = self.actor(obs)

    # 2. Sample with reparameterization (gradients flow!)
    action = action_dist.rsample()

    # 3. Compute Q-values (gradients flow through action!)
    q_values = self.critic(obs, action)

    # 4. Loss (negative because we maximize)
    loss = -q_values.mean()

    return loss
```

---

## Key Concepts and Design Patterns

### 1. Separation of Concerns

**Neural Network (networks/)**: Pure function approximation
```python
class MLPPolicy:
    def forward(self, obs):
        # obs → distribution
        # No RL logic, just neural network
```

**Agent (agents/)**: RL algorithm logic
```python
class SoftActorCritic:
    def update_critic(self, ...):
        # Bootstrapping, target networks, loss computation
        # Pure RL logic, delegates NN to networks/
```

**Infrastructure (infrastructure/)**: Reusable utilities
```python
class ReplayBuffer:
    # Generic experience replay
    # Can be used with any off-policy algorithm
```

### 2. Configuration Management

**Hierarchy:**
1. **Base defaults**: In `sac_config.py`
2. **Experiment-specific**: In YAML files
3. **Command-line**: Via argparse

**Example:**
```yaml
# experiments/sac/my_experiment.yaml
base_config: sac
env_name: HalfCheetah-v4
temperature: 0.2  # Override default
```

```bash
python run_hw4.py -cfg experiments/sac/my_experiment.yaml --seed 42
```

### 3. Logging and Monitoring

**TensorBoard integration:**
```python
logger.log_scalar(value, name, step)
logger.log_scalar(q_values.mean().item(), "q_values", step)
```

**View results:**
```bash
tensorboard --logdir data/
```

### 4. Modular Network Creation

**Factory pattern:**
```python
def sac_config(...):
    def make_actor(obs_shape, action_dim):
        return MLPPolicy(
            ob_dim=obs_shape[0],
            ac_dim=action_dim,
            n_layers=num_layers,
            layer_size=hidden_size,
        )

    return {
        "agent_kwargs": {
            "make_actor": make_actor,
            # ...
        }
    }
```

**Benefits:**
- Easy to swap architectures
- Deferred construction
- Configuration flexibility

### 5. PyTorch Utilities

**Device management:**
```python
# pytorch_util.py
device = None  # Global device

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
    else:
        device = torch.device("cpu")

def from_numpy(data):
    """Convert numpy → torch, move to device"""
    return torch.from_numpy(data).float().to(device)

def to_numpy(tensor):
    """Convert torch → numpy"""
    return tensor.to("cpu").detach().numpy()
```

**MLP builder:**
```python
def build_mlp(input_size, output_size, n_layers, size, activation="tanh"):
    layers = []
    for _ in range(n_layers):
        layers += [nn.Linear(in_size, size), activation]
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    return nn.Sequential(*layers)
```

### 6. Target Network Updates

**Hard update (periodic copy):**
```python
def update_target_critic(self):
    """Copy weights completely"""
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)
```

**Soft update (exponential moving average):**
```python
def soft_update_target_critic(self, tau=0.005):
    """Polyak averaging: θ' ← θ' + τ(θ - θ')"""
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
```

---

## How to Navigate the Codebase

### Starting Point: Understanding a New Feature

**Q: "How does SAC select actions?"**

1. **Start with the entry point**: `scripts/run_hw4.py`
   - Find action selection: `action = agent.get_action(observation)`

2. **Follow to agent**: `agents/sac_agent.py`
   - Find method `get_action()`
   - See it calls `self.actor(observation)`

3. **Follow to network**: `networks/mlp_policy.py`
   - Find `forward()` method
   - Understand it returns a distribution

4. **Check utilities**: `infrastructure/distributions.py`
   - See custom distribution implementations

### Reading the Codebase: Recommended Order

**For understanding SAC:**

1. Start: `scripts/run_hw4.py`
   - Overall training loop
   - When things happen

2. Then: `agents/sac_agent.py`
   - `__init__`: What components exist
   - `get_action`: How actions are chosen
   - `update`: Main learning logic

3. Then: `agents/sac_agent.py` (detailed)
   - `update_critic`: How values are learned
   - `update_actor`: How policy is learned

4. Then: Network architectures
   - `networks/mlp_policy.py`: Actor
   - `networks/state_action_value_critic.py`: Critic

5. Finally: Infrastructure
   - `infrastructure/replay_buffer.py`: Data storage
   - `infrastructure/utils.py`: Helper functions

### Debugging Tips

**Q-values exploding/vanishing?**
- Check: `update_critic()` - Are targets computed correctly?
- Check: Target network updates - Are they happening?
- Log: Q-values, target values, critic loss

**Policy not improving?**
- Check: `update_actor()` - Is loss decreasing?
- Check: Entropy - Is it too high/low?
- Log: Actor loss, entropy, policy std

**No learning at all?**
- Check: Replay buffer has enough samples
- Check: `training_starts` parameter
- Check: Learning rates
- Verify: Gradients are flowing (print grad norms)

### Common Modification Patterns

**Change network architecture:**
```python
# In env_configs/sac_config.py
def make_actor(obs_shape, action_dim):
    return MLPPolicy(
        ...,
        n_layers=5,  # Changed from 3
        layer_size=256,  # Changed from 128
    )
```

**Add new hyperparameter:**
```python
# In sac_config.py
def sac_config(..., my_new_param=default_value):
    return {
        "agent_kwargs": {
            ...,
            "my_new_param": my_new_param,
        }
    }

# In sac_agent.py
class SoftActorCritic:
    def __init__(self, ..., my_new_param):
        self.my_new_param = my_new_param
```

**Add new logging:**
```python
# In sac_agent.py
def update_critic(...):
    ...
    return {
        "critic_loss": loss.item(),
        "q_values": q_values.mean().item(),
        "my_new_metric": new_value.item(),  # Add this
    }
```

---

## Advanced Topics

### 1. Reparameterization Trick

**Problem**: Can't backpropagate through sampling
```python
a ~ N(μ(s), σ(s))  # How to get ∇_θ?
```

**Solution**: Reparameterize
```python
ε ~ N(0, 1)  # Standard normal
a = μ(s) + σ(s) · ε  # Now gradients flow through μ and σ!
```

**In code:**
```python
# .sample(): No gradients
action = distribution.sample()

# .rsample(): Reparameterized, gradients flow
action = distribution.rsample()
```

### 2. Tanh Transform for Bounded Actions

**Problem**: Environments expect actions in [-1, 1], but Gaussian is unbounded

**Solution**: Squash through tanh
```python
a_unbounded ~ N(μ, σ)
a = tanh(a_unbounded)  # Now a ∈ (-1, 1)
```

**Correction for probability:**
```python
log π(a|s) = log π(a_unbounded|s) - log|da/da_unbounded|
            = log π(a_unbounded|s) - log(1 - tanh²(a_unbounded))
```

### 3. Multiple Critic Networks

**Why?**
- Overestimation bias in Q-learning
- Single critic tends to be optimistic

**Solutions:**
- **Double-Q**: Two critics, each uses other as target
- **Clipped Double-Q**: Use min(Q1, Q2) for targets
- **Mean**: Average multiple critics

**Implementation:**
```python
self.critics = nn.ModuleList([
    make_critic(...) for _ in range(num_critic_networks)
])

def critic(self, obs, action):
    # Returns: [num_critics, batch_size]
    return torch.stack([critic(obs, action) for critic in self.critics])
```

### 4. Entropy-Regularized RL

**Objective:**
```
J(π) = E[∑ γ^t (r_t + α H(π(·|s_t)))]
```

**Effects:**
- Prevents premature convergence
- Learns robust policies
- Automatic exploration

**Temperature (α):**
- High α: More random (more exploration)
- Low α: More deterministic (more exploitation)
- Can be learned automatically (not in this homework)

---

## Glossary

**Actor**: The policy network π_θ(a|s) that outputs action distributions

**Critic**: The value network Q_φ(s,a) that estimates expected returns

**Bellman Equation**: Recursive relationship: Q(s,a) = r + γE[Q(s',a')]

**Bootstrapping**: Using estimate of future values in learning target

**Discount Factor (γ)**: How much to value future vs immediate rewards

**Entropy**: Measure of randomness: H(π) = -E[log π(a|s)]

**Episode**: Complete sequence from start state to terminal state

**Off-policy**: Learn from data generated by different policy

**On-policy**: Learn from data generated by current policy

**Policy**: Mapping from states to actions: π(a|s)

**Replay Buffer**: Storage for past experiences (s, a, r, s', done)

**Reparameterization**: a = μ + σ·ε technique for gradient flow

**Return**: Sum of discounted rewards: G_t = ∑_{k=0}^∞ γ^k r_{t+k}

**Reward**: Scalar feedback signal from environment

**State**: Complete description of environment at time t

**Target Network**: Slowly-updated copy of value network for stability

**Temperature (α)**: Weight for entropy bonus in SAC

**Trajectory/Rollout**: Sequence of (s, a, r) tuples

**Value Function**: Expected return: V(s) = E[G_t | s_t = s]

---

## Next Steps

### To deepen understanding:

1. **Read the SAC paper**: https://arxiv.org/abs/1801.01290
2. **Implement variants**: Try different architectures, hyperparameters
3. **Experiment**: Try new environments, visualize learned policies
4. **Compare algorithms**: Implement TD3, PPO, and compare

### Resources:

- **OpenAI Spinning Up**: https://spinningup.openai.com/
- **Berkeley CS 285**: http://rail.eecs.berkeley.edu/deeprlcourse/
- **PyTorch tutorials**: https://pytorch.org/tutorials/
- **Gymnasium docs**: https://gymnasium.farama.org/

---

## Summary

This codebase implements a complete, production-quality SAC agent. Key takeaways:

1. **Modular design**: Separate concerns (algorithm, networks, infrastructure)
2. **Configuration-driven**: Easy to experiment with hyperparameters
3. **Best practices**: Target networks, replay buffer, entropy regularization
4. **Extensible**: Easy to modify and extend

Understanding this codebase gives you:
- **Practical RL implementation skills**
- **Deep learning engineering patterns**
- **Foundation for research and production RL**

Happy learning! 🚀
