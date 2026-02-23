# How Training Works in RetroRacer

## The Big Picture

Training teaches a neural network to drive by trial and error. The AI tries random actions, sees what works (reward goes up) and what doesn't (reward goes down), and gradually builds a strategy.

```
┌─────────┐    action     ┌─────────────┐    reward + next state    ┌──────────┐
│  Agent   │─────────────▶│ Environment │──────────────────────────▶│  Agent   │
│ (brain)  │  [steer,     │  (the game) │   [+1.2 for progress,    │ (learns) │
│          │   throttle]  │             │    -0.5 for off-center]   │          │
└─────────┘              └─────────────┘                           └──────────┘
                               │
                               ▼
                         Physics step,
                         new car position,
                         new observation
```

The loop runs **millions of times**. Each step takes ~0.03ms, so 1.5M steps ≈ 1 hour.

---

## SAC (Soft Actor-Critic)

We use the **SAC** algorithm from Stable-Baselines3. Here's what makes it special:

### Two Neural Networks

SAC maintains two separate networks:

| Network            | Job                                                                             | Analogy        |
| ------------------ | ------------------------------------------------------------------------------- | -------------- |
| **Actor** (policy) | Decides what to do: "given this observation, output [steer, throttle]"          | The driver     |
| **Critic** (value) | Judges how good a state is: "being here at this speed is worth X future reward" | The strategist |

### Why Two Networks?

The **critic** gives the **actor** feedback. Instead of waiting until the end of a lap to know if a decision was good, the critic estimates the long-term value _right now_. This makes learning much faster than simple "try and see what happens."

### The "Soft" Part — Entropy

SAC has a unique trick: it rewards **exploration**. The `ent_coef="auto"` parameter means the agent gets a bonus for trying diverse actions rather than always doing the same thing. This prevents it from getting stuck in a rut early on (like always turning left).

---

## Line-by-Line: `train_sac()`

### The SAC Configuration

```python
model = SAC(
    policy="MlpPolicy",       # ① Neural network type
    env=env,                   # ② The game environment
    learning_rate=3e-4,        # ③ How fast it learns
    buffer_size=500_000,       # ④ Memory size
    batch_size=256,            # ⑤ Samples per learning step
    tau=0.005,                 # ⑥ Target network update speed
    gamma=0.99,                # ⑦ How much it values future reward
    train_freq=1,              # ⑧ Learn every step
    gradient_steps=1,          # ⑨ One gradient update per step
    learning_starts=20_000,    # ⑩ Random exploration period
    ent_coef="auto",           # ⑪ Auto-tuned exploration bonus
)
```

### What Each Parameter Does

**① `MlpPolicy`** — "Multi-Layer Perceptron." A simple feedforward neural network (not a CNN or transformer). Takes the 22 observation values in, outputs 2 action values. Architecture: `[22] → [256] → [256] → [2]`.

**② `env`** — The `RetroRacerEnv`. SAC calls `env.step(action)` millions of times during training to simulate driving.

**③ `learning_rate=3e-4`** — How big each learning step is.

- Too high (1e-2): network oscillates, never converges
- Too low (1e-5): takes forever to learn
- 3e-4 is a sweet spot, like "walk don't run"

**④ `buffer_size=500_000`** — The **Replay Buffer**. This is key to SAC's efficiency:

- Every experience (state, action, reward, next_state) gets stored
- During learning, SAC samples _random_ past experiences to learn from
- This breaks correlation (consecutive frames are too similar to learn from directly)
- 500k means it remembers the last ~500k steps

**⑤ `batch_size=256`** — Each learning update trains on 256 random experiences from the buffer. Larger = more stable but slower.

**⑥ `tau=0.005`** — The critic has a "target network" (a delayed copy). `tau` controls how slowly it updates:

- Each step: `target = 0.995 * target + 0.005 * current_critic`
- This prevents the critic from chasing its own tail

**⑦ `gamma=0.99`** — The **discount factor**. How much the agent values future rewards vs immediate rewards.

- `gamma=0`: only cares about the next step (short-sighted)
- `gamma=0.99`: reward 100 steps from now is worth `0.99^100 ≈ 0.37` of its face value
- This is what makes the agent brake _before_ a corner — it knows crashing in 50 steps is bad even though the corner is still ahead

**⑧ `train_freq=1`** — Learn every single step. More frequent = faster learning but heavier compute.

**⑨ `gradient_steps=1`** — One neural network update per step. This keeps training/playing balanced.

**⑩ `learning_starts=20_000`** — For the first 20k steps, the agent takes **completely random** actions. This fills the replay buffer with diverse experiences before learning begins. Without this, the network would learn from a very narrow set of early experiences.

**⑪ `ent_coef="auto"`** — SAC auto-tunes how much to encourage exploration. Early in training, it explores a lot (random-ish actions). Later, as it gets confident, it exploits what it's learned.

---

## The Training Loop (`model.learn()`)

Here's what happens inside `model.learn()` each step:

```
Step 1: Observe → 22 numbers (speed, heading, curvature, etc.)
            │
            ▼
Step 2: Actor Network → outputs [steer, throttle] + random noise
            │
            ▼
Step 3: env.step(action) → physics sim → new state + reward
            │
            ▼
Step 4: Store (state, action, reward, next_state) in replay buffer
            │
            ▼
Step 5: Sample 256 random experiences from buffer
            │
            ▼
Step 6: Update Critic → "this state was actually worth X reward"
            │
            ▼
Step 7: Update Actor → "in this state, turning right would've been better"
            │
            ▼
Step 8: Update entropy coefficient → "explore more/less"
            │
            ▼
Repeat 1,500,000 times
```

---

## What the Agent Learns Over Time

| Steps     | Behavior                                      | R̄ (avg reward) |
| --------- | --------------------------------------------- | -------------- |
| 0-20k     | Random flailing                               | ~ -5 to 0      |
| 20k-100k  | Learns to go forward                          | ~ 0 to +10     |
| 100k-300k | Learns to stay on track                       | ~ +10 to +50   |
| 300k-700k | Learns to follow the track shape              | ~ +50 to +200  |
| 700k-1.5M | Learns racing line, braking, speed management | ~ +200 to +700 |
| 1.5M+     | Fine-tuning, diminishing returns              | +700+          |

---

## Fine-Tuning (`fine_tune()`)

Fine-tuning loads an existing model and continues training with a **lower learning rate** (1e-4 vs 3e-4):

```python
model = SAC.load(base_model_path, env=env)
model.learning_rate = 1e-4  # gentle adjustments, don't unlearn
```

The lower rate means the agent makes smaller adjustments — it already knows _how_ to race, it just needs to learn _this specific track's_ corners.

---

## Domain Randomization (`train_base()`)

```python
env = RetroRacerEnv(randomize_track=True)
```

Each episode uses a **different random track**. The agent can't memorize any one track — it has to learn general principles:

- "Sharp turn ahead → slow down"
- "Straight → accelerate"
- "Drifting off center → steer back"

This makes the base model work on _any_ track after brief fine-tuning.

---

## The `.zip` Model File

The saved `.zip` contains:

- **Actor network weights** — the driving brain (~200KB)
- **Critic network weights** — the value estimator (~200KB)
- **Optimizer state** — momentum from training (for resuming)
- **Entropy coefficient** — current exploration level
- **Hyperparameters** — learning rate, gamma, etc.

Total: ~2-4MB per model.
