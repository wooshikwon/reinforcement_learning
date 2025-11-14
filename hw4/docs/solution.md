# GCB6206 Homework 4: Soft Actor-Critic - Complete Solution Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Section 3: Training Loop](#section-3-training-loop)
3. [Section 4: Bootstrapping](#section-4-bootstrapping)
4. [Section 5: Entropy Bonus and SAC](#section-5-entropy-bonus-and-sac)
5. [Section 6: Actor Update with REINFORCE](#section-6-actor-update-with-reinforce)
6. [Section 7: Actor Update with REPARAMETRIZE](#section-7-actor-update-with-reparametrize)
7. [Section 8: Stabilizing Target Values](#section-8-stabilizing-target-values)
8. [Complete Code Solutions](#complete-code-solutions)
9. [Running Experiments](#running-experiments)

---

## Introduction

This guide provides complete solutions for implementing Soft Actor-Critic (SAC), a state-of-the-art off-policy actor-critic algorithm for continuous control tasks.

**Key Concepts:**
- **Actor-Critic**: Combines value-based (critic) and policy-based (actor) methods
- **Off-policy**: Learns from replayed experiences, not just current policy
- **Soft**: Maximizes entropy for better exploration
- **Continuous actions**: Uses policy gradient methods instead of max Q-value

---

## Section 3: Training Loop

### Problem
Complete TODOs in `gcb6206/scripts/run_hw4.py` to implement the training loop.

### Solution

**Location**: `gcb6206/scripts/run_hw4.py`

#### TODO 1: Select action (Line 63-64)
```python
# TODO(student): Select an action
action = agent.get_action(observation)
```

**Explanation**: The agent's `get_action()` method samples an action from the policy distribution. During the first `random_steps` iterations, we use random actions for initial exploration.

#### TODO 2: Sample batch from replay buffer (Line 85-87)
```python
# TODO(student): Sample a batch of config["batch_size"] transitions from the replay buffer
# Please refer to gcb6206/infrastructure/replay_buffer.py
batch = replay_buffer.sample(config["batch_size"])
```

**Explanation**: The replay buffer's `sample()` method returns a dictionary containing:
- `observations`: Current states
- `actions`: Actions taken
- `rewards`: Rewards received
- `next_observations`: Next states
- `dones`: Episode termination flags

#### TODO 3: Train the agent (Line 92-93)
```python
# TODO(student): Train the agent using `update` method. `batch` is a dictionary of torch tensors.
update_info = agent.update(
    observations=batch["observations"],
    actions=batch["actions"],
    rewards=batch["rewards"],
    next_observations=batch["next_observations"],
    dones=batch["dones"],
    step=step,
)
```

**Explanation**: The `update()` method performs one gradient step on both the actor and critic networks.

---

## Section 4: Bootstrapping

### Problem 1: Implement critic update with bootstrapping

**Location**: `gcb6206/agents/sac_agent.py`, method `update_critic` (lines 170-232)

### Solution

#### TODO 1: Sample from the actor (Lines 186-188)
```python
# TODO(student): Sample from the actor
next_action_distribution: torch.distributions.Distribution = self.actor(next_obs)
next_action = next_action_distribution.sample()
```

**Explanation**:
- Get the policy distribution π(a|s') for next states
- Sample next actions a_{t+1} ~ π(·|s_{t+1}) for computing target values
- Use `.sample()` (not `.rsample()`) because we don't need gradients in the target computation

#### TODO 2: Compute next Q-values (Lines 190-192)
```python
# TODO(student)
# Compute the next Q-values using `self.target_critic` for the sampled actions
next_qs = self.target_critic(next_obs, next_action)
```

**Explanation**:
- Use target critic network (not current critic) for stability
- `next_qs` has shape `(num_critic_networks, batch_size)`
- Target networks prevent moving target problem

#### TODO 3: Add entropy bonus (Lines 204-207)
```python
if self.use_entropy_bonus and self.backup_entropy:
    # TODO(student): Add entropy bonus to the target values for SAC
    # NOTE: use `self.entropy()`
    next_action_entropy = self.entropy(next_action_distribution)
    next_qs += self.temperature * next_action_entropy
```

**Explanation**:
- Entropy encourages exploration by rewarding randomness
- `self.temperature` (β) controls the tradeoff between entropy and reward
- Entropy is added to Q-values to create "soft" value function

#### TODO 4: Compute target Q-value (Lines 209-215)
```python
# TODO(student): Compute the target Q-value
# HINT: implement Equation (1) in Homework 4
target_values: torch.Tensor = reward + self.discount * (1 - done) * next_qs
```

**Explanation**:
- Bellman backup: y = r + γ(1 - d)Q'(s', a')
- `(1 - done)` zeros out future rewards when episode ends
- `self.discount` is the discount factor γ

#### TODO 5: Predict Q-values (Lines 217-218)
```python
# TODO(student): Predict Q-values using `self.critic`
q_values = self.critic(obs, action)
```

**Explanation**: Current Q-values for the state-action pairs in the batch.

#### TODO 6: Compute loss (Lines 221-222)
```python
# TODO(student): Compute loss using `self.critic_loss`
loss: torch.Tensor = self.critic_loss(q_values, target_values)
```

**Explanation**: MSE loss between predicted Q-values and target values.

---

### Problem 2: Update critic multiple times

**Location**: `gcb6206/agents/sac_agent.py`, method `update` (lines 339-382)

### Solution

#### TODO: Update critic for num_critic_updates steps (Lines 352-353)
```python
# TODO(student): Update the critic for num_critic_upates steps, and add the output stats to critic_infos
critic_infos = []
for _ in range(self.num_critic_updates):
    critic_info = self.update_critic(
        observations, actions, rewards, next_observations, dones
    )
    critic_infos.append(critic_info)
```

**Explanation**: Multiple critic updates per actor update can improve value function accuracy.

---

### Problem 3: Implement target network updates

**Location**: `gcb6206/agents/sac_agent.py`, method `update` (lines 339-382)

### Solution

#### TODO: Update actor (Lines 355-356)
```python
# TODO(student): Update the actor
actor_info = self.update_actor(observations)
```

#### TODO: Perform hard or soft target updates (Lines 358-365)
```python
# TODO(student): Perform either hard or soft target updates.
# Relevant variables:
#  - step
#  - self.target_update_period (None when using soft updates)
#  - self.soft_target_update_rate (None when using hard updates)
# For hard target updates, you should do it every self.target_update_period step
# For soft target updates, you should do it every step
# HINT: use `self.update_target_critic` or `self.soft_update_target_critic`

if self.target_update_period is not None:
    # Hard update
    if step % self.target_update_period == 0:
        self.update_target_critic()
else:
    # Soft update
    self.soft_update_target_critic(self.soft_target_update_rate)
```

**Explanation**:
- **Hard update**: φ' ← φ every K steps (complete copy)
- **Soft update**: φ' ← φ' + τ(φ - φ') every step (exponential moving average)
- Soft updates are generally more stable

---

### Problem 4: Calculate expected Q-value for "do nothing" policy

**Question**: The "do-nothing" reward for Pendulum-v1 is about −10 per step. Assuming the agent always gets "do nothing" and never terminates, what should be the average Q-value considering discount factor γ = 0.99?

### Answer

For an infinite horizon with constant reward r = -10 and discount γ = 0.99:

```
Q(s, a) = E[∑_{t=0}^∞ γ^t · r]
        = r · ∑_{t=0}^∞ γ^t
        = r / (1 - γ)
        = -10 / (1 - 0.99)
        = -10 / 0.01
        = -1000
```

**Expected Q-value**: approximately **-1000**

This matches the expected stabilization point mentioned in the homework (around -700, which could differ due to sampling variance or the fact that the pendulum might occasionally fall).

---

## Section 5: Entropy Bonus and SAC

### Problem: Implement entropy computation and soft Q-values

**Location**: `gcb6206/agents/sac_agent.py`

### Solution

#### TODO: Implement entropy method (Lines 234-247)
```python
def entropy(self, action_distribution: torch.distributions.Distribution):
    """
    Compute the (approximate) entropy of the action distribution for each batch element.
    """

    # TODO(student): Compute the entropy of the action distribution
    # HINT: use one action sample for estimating the entropy.
    # HINT: use action_distribution.log_prob to get the log probability.
    # NOTE: think about whether to use .rsample() or .sample() here
    action = action_distribution.rsample()
    entropy = -action_distribution.log_prob(action)

    assert entropy.shape == action.shape[:-1]
    return entropy
```

**Explanation**:
- **Entropy**: H(π) = E_a~π[-log π(a|s)]
- Use `.rsample()` for reparameterization trick (enables gradient flow if needed)
- Monte Carlo estimate with single sample: H ≈ -log π(â|s) where â ~ π(·|s)
- Higher entropy = more random policy = better exploration

**Why rsample?** Even though we don't backprop through entropy here, using rsample is consistent with the reparametrization approach and enables gradient flow if needed in actor updates.

#### Note: Entropy bonus already added in update_critic
The entropy term was already added to target values in Section 4, TODO 3.

---

### Expected Result

For Pendulum-v1 with only entropy maximization (no reward maximization), the entropy should reach approximately:

```
H(U[-1, 1]) = -log(1/2) = log(2) ≈ 0.69
```

This is the maximum entropy for a uniform distribution over the 1D action space [-1, 1].

---

## Section 6: Actor Update with REINFORCE

### Problem: Implement REINFORCE gradient estimator

**Location**: `gcb6206/agents/sac_agent.py`, method `actor_loss_reinforce` (lines 249-286)

### Solution

```python
def actor_loss_reinforce(self, obs: torch.Tensor):
    batch_size = obs.shape[0]

    # TODO(student): Generate an action distribution
    action_distribution: torch.distributions.Distribution = self.actor(obs)

    with torch.no_grad():
        # TODO(student): Draw self.num_actor_samples samples from the action distribution for each batch element
        # NOTE: think about whether to use .rsample() or .sample() here
        action = action_distribution.sample((self.num_actor_samples,))
        assert action.shape == (
            self.num_actor_samples,
            batch_size,
            self.action_dim,
        ), action.shape

        # TODO(student): Compute Q-values for the current state-action pair
        # HINT: need to add one dimension with `self.num_actor_samples` at the beginning of `obs`
        # HINT: for this, you can use either `repeat` or `expand`
        q_values = self.critic(
            obs[None].expand(self.num_actor_samples, batch_size, -1),
            action
        )
        assert q_values.shape == (
            self.num_critic_networks,
            self.num_actor_samples,
            batch_size,
        ), q_values.shape

        # Our best guess of the Q-values is the mean of the ensemble
        q_values = torch.mean(q_values, axis=0)

    # Do REINFORCE (without baseline)
    # TODO(student): Calculate log-probs
    log_probs = action_distribution.log_prob(action)
    assert log_probs.shape == q_values.shape

    # TODO(student): Compute policy gradient using log-probs and Q-values
    loss = -(log_probs * q_values).mean()

    return loss, torch.mean(self.entropy(action_distribution))
```

**Explanation**:

1. **Generate action distribution**: π_θ(·|s) for each state in batch
2. **Sample actions**: Use `.sample()` (not `.rsample()`) because REINFORCE doesn't use reparameterization
3. **Expand observations**: Repeat each observation `num_actor_samples` times to match action samples
4. **Compute Q-values**: Q(s, a) for each sampled action
5. **Calculate log probabilities**: log π_θ(a|s)
6. **REINFORCE gradient**: ∇_θ E[Q] ≈ E[∇_θ log π(a|s) · Q(s,a)]
7. **Loss**: Negative because we want to maximize (gradient ascent)

**Why sample() not rsample()?** REINFORCE uses the log-derivative trick (score function), not reparameterization.

**Multiple samples**: Using multiple action samples reduces variance of the gradient estimate.

---

## Section 7: Actor Update with REPARAMETRIZE

### Problem: Implement reparameterized gradient estimator

**Location**: `gcb6206/agents/sac_agent.py`, method `actor_loss_reparametrize` (lines 288-304)

### Solution

```python
def actor_loss_reparametrize(self, obs: torch.Tensor):
    batch_size = obs.shape[0]

    # Sample from the actor
    action_distribution: torch.distributions.Distribution = self.actor(obs)

    # TODO(student): Sample actions
    # Note: Think about whether to use .rsample() or .sample() here...
    action = action_distribution.rsample()

    # TODO(student): Compute Q-values for the sampled state-action pair
    q_values = self.critic(obs, action)

    # TODO(student): Compute the actor loss using Q-values
    loss = -q_values.mean()

    return loss, torch.mean(self.entropy(action_distribution))
```

**Explanation**:

1. **Get action distribution**: π_θ(·|s) from actor network
2. **Sample with reparameterization**: Use `.rsample()` to enable gradient flow through sampling
3. **Compute Q-values**: Q(s, a) where a = μ_θ(s) + σ_θ(s) · ε, ε ~ N(0,1)
4. **Compute loss**: Maximize Q-values (minimize negative Q-values)

**Reparameterization Trick**:
- Instead of: a ~ π_θ(·|s), can't backprop through sampling
- Use: a = μ_θ(s) + σ_θ(s) · ε where ε ~ N(0,1)
- Now gradients flow: ∇_θ Q(s, a) = ∇_θ Q(s, μ_θ(s) + σ_θ(s)·ε)

**Advantages over REINFORCE**:
- Much lower variance
- Works well with single sample
- Enables efficient optimization

---

## Section 8: Stabilizing Target Values

### Problem: Implement Double-Q and Clipped Double-Q

**Location**: `gcb6206/agents/sac_agent.py`, method `q_backup_strategy` (lines 127-168)

### Solution

```python
def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
    """
    Handle Q-values from multiple different target critic networks to produce target values.

    For example:
     - for "vanilla", we can just leave the Q-values as-is (we only have one critic).
     - for double-Q, swap the critics' predictions (so each uses the other as the target).
     - for clip-Q, clip to the minimum of the two critics' predictions.

    Parameters:
        next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size).
            Leading dimension corresponds to target values FROM the different critics.
    Returns:
        torch.Tensor: Target values of shape (num_critics, batch_size).
            Leading dimension corresponds to target values FOR the different critics.
    """

    assert (
        next_qs.ndim == 2
    ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
    num_critic_networks, batch_size = next_qs.shape
    assert num_critic_networks == self.num_critic_networks

    # TODO(student): Implement the different backup strategies.
    if self.target_critic_backup_type == "doubleq":
        # Double-Q: Swap critics - each uses the other as target
        if num_critic_networks == 2:
            next_qs = torch.stack([next_qs[1], next_qs[0]], dim=0)
        else:
            # For more than 2 critics, rotate them
            next_qs = torch.roll(next_qs, shifts=1, dims=0)

    elif self.target_critic_backup_type == "min":
        # Clipped Double-Q: Use minimum across all critics
        next_qs = next_qs.min(dim=0, keepdim=False).values

    else:
        # Default (mean or single critic), we don't need to do anything.
        pass

    # If our backup strategy removed a dimension, add it back in explicitly
    # (assume the target for each critic will be the same)
    if next_qs.shape == (batch_size,):
        next_qs = next_qs[None].expand((self.num_critic_networks, batch_size)).contiguous()

    assert next_qs.shape == (
        self.num_critic_networks,
        batch_size,
    ), next_qs.shape
    return next_qs
```

**Explanation**:

### Double-Q Learning
- **Problem**: Single Q-network tends to overestimate values
- **Solution**: Use two critics Q_A and Q_B
  - Q_A uses Q'_B as target: y_A = r + γQ'_B(s', a')
  - Q_B uses Q'_A as target: y_B = r + γQ'_A(s', a')
- **Implementation**: Swap the Q-values from the two target critics

### Clipped Double-Q (TD3 style)
- **Problem**: Still some overestimation with Double-Q
- **Solution**: Use pessimistic estimate
  - Both critics use: y = r + γ min(Q'_A(s', a'), Q'_B(s', a'))
- **Implementation**: Take minimum across all critics
- **Result**: Same target for all critics, more conservative

### Why this matters
- Overestimation bias compounds over training
- Can lead to unstable learning or divergence
- Clipped Double-Q generally most stable

---

## Complete Code Solutions

### Summary of all code changes:

#### File: `gcb6206/scripts/run_hw4.py`

**Line 63-64**: Select action
```python
action = agent.get_action(observation)
```

**Line 85-87**: Sample batch
```python
batch = replay_buffer.sample(config["batch_size"])
```

**Line 92-93**: Train agent
```python
update_info = agent.update(
    observations=batch["observations"],
    actions=batch["actions"],
    rewards=batch["rewards"],
    next_observations=batch["next_observations"],
    dones=batch["dones"],
    step=step,
)
```

#### File: `gcb6206/agents/sac_agent.py`

All solutions provided in detail above for:
- `update_critic()` method
- `entropy()` method
- `actor_loss_reinforce()` method
- `actor_loss_reparametrize()` method
- `q_backup_strategy()` method
- `update()` method

---

## Running Experiments

### Section 4.2: Bootstrapping Sanity Check

```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/sanity_pendulum_1.yaml
```

**Expected**: Q-values stabilize around -700 to -1000

### Section 5.2: Entropy Sanity Check

```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/sanity_pendulum_2.yaml
```

**Expected**: Entropy increases to ~0.69 (log 2)

### Section 6.2: REINFORCE Experiments

**Inverted Pendulum**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/sanity_invertedpendulum_reinforce.yaml
```
**Expected**: Reward ~1000

**HalfCheetah REINFORCE-1**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/halfcheetah_reinforce1.yaml
```
**Expected**: Positive rewards in 500K steps

**HalfCheetah REINFORCE-10**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/halfcheetah_reinforce10.yaml
```
**Expected**: >500 evaluation return in 200K steps (faster than REINFORCE-1)

### Section 7.2: REPARAMETRIZE Experiments

**Inverted Pendulum**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/sanity_invertedpendulum_reparametrize.yaml
```
**Expected**: Reward ~1000

**HalfCheetah**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/halfcheetah_reparametrize.yaml
```
**Expected**: Best performance among all methods

### Section 8.2: Stabilizing Experiments

**Single-Q Hopper**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/hopper.yaml --seed 48
```

**Double-Q Hopper**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/hopper_doubleq.yaml --seed 48
```

**Clipped Double-Q Hopper**:
```bash
python gcb6206/scripts/run_hw4.py -cfg experiments/sac/hopper_clipq.yaml --seed 48
```

**Expected**: Clipped Double-Q typically performs best with lowest Q-value overestimation

---

## Key Takeaways

1. **SAC combines the best of both worlds**: Value-based stability + policy gradient flexibility
2. **Entropy regularization**: Improves exploration and prevents premature convergence
3. **Reparameterization trick**: Much lower variance than REINFORCE
4. **Target networks**: Essential for stability in off-policy learning
5. **Clipped Double-Q**: Effectively addresses overestimation bias

---

## Common Pitfalls and Debugging Tips

1. **Gradients not flowing**: Make sure to use `.rsample()` for reparameterization, `.sample()` for REINFORCE
2. **Shape mismatches**: Pay attention to batch dimensions, especially with multiple samples/critics
3. **Exploding/vanishing Q-values**: Check target network updates are working
4. **No learning**: Verify replay buffer has enough samples before training starts
5. **Entropy too high/low**: Check temperature coefficient and entropy implementation

---

## Additional Resources

- **SAC Paper**: https://arxiv.org/abs/1801.01290
- **TD3 Paper** (Clipped Double-Q): https://arxiv.org/abs/1802.09477
- **OpenAI Spinning Up**: https://spinningup.openai.com/en/latest/algorithms/sac.html
