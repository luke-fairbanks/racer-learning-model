import os
import random
from collections import deque
from dataclasses import dataclass
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import ale_py
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation

gym.register_envs(ale_py)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def make_env(render=False):
    env = gym.make("ALE/Breakout-v5", render_mode="human" if render else None)
    env = GrayscaleObservation(env, keep_dim=False)      # (H,W)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStackObservation(env, stack_size=4)       # (4,84,84)
    return env


class QNet(nn.Module):
    # Classic Atari-ish CNN
    def __init__(self, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84)
            n_flat = self.net(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        # x: (B,4,84,84) float in [0,1]
        x = self.net(x)
        return self.head(x)


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def add(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s = np.stack([b.s for b in batch], axis=0)
        a = np.array([b.a for b in batch], dtype=np.int64)
        r = np.array([b.r for b in batch], dtype=np.float32)
        s2 = np.stack([b.s2 for b in batch], axis=0)
        done = np.array([b.done for b in batch], dtype=np.float32)
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buf)


def to_np_obs(obs):
    """
    FrameStackObservation returns a LazyFrames-like object.
    Convert to a real np array (4,84,84) uint8.
    copy=True ensures each observation in the replay buffer
    is an independent copy, not a view into shared memory.
    """
    return np.array(obs, dtype=np.uint8, copy=True)


def epsilon_by_step(step, eps_start=1.0, eps_end=0.02, decay_steps=500_000):
    # Linear decay
    if step >= decay_steps:
        return eps_end
    return eps_start + (eps_end - eps_start) * (step / decay_steps)

def play(checkpoint="dqn_breakout.pt", episodes=5):
    env = make_env(render=True)
    n_actions = env.action_space.n
    q = QNet(n_actions).to(DEVICE)
    ckpt = torch.load(checkpoint, map_location=DEVICE)
    # Support both old (raw state_dict) and new (full checkpoint) formats
    if isinstance(ckpt, dict) and "q" in ckpt:
        q.load_state_dict(ckpt["q"])
    else:
        q.load_state_dict(ckpt)
    q.eval()

    for ep in range(episodes):
        obs, info = env.reset()
        obs, _, _, _, _ = env.step(1)  # press FIRE to launch ball
        obs_np = to_np_obs(obs)
        epR = 0.0

        done = False
        while not done:
            if random.random() < 0.05:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.from_numpy(obs_np).unsqueeze(0).to(DEVICE).float() / 255.0
                    action = int(torch.argmax(q(s_t), dim=1).item())

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            obs_np = to_np_obs(obs)
            epR += reward

        print(f"[play] episode {ep+1}: reward {epR:.1f}")

    env.close()

CHECKPOINT_PATH = "dqn_breakout.pt"

def main():
    env = make_env(render=False)
    n_actions = env.action_space.n

    q = QNet(n_actions).to(DEVICE)
    q_targ = QNet(n_actions).to(DEVICE)
    q_targ.load_state_dict(q.state_dict())
    q_targ.eval()

    optimizer = optim.Adam(q.parameters(), lr=1e-4)
    rb = ReplayBuffer(capacity=200_000)

    gamma = 0.99
    batch_size = 64
    learn_start = 10_000          # fill buffer a bit before learning
    train_freq = 4                # learn every 4 env steps
    target_update_freq = 10_000   # copy weights every N steps

    max_steps = 3_000_000
    global_step = 0
    episode = 0

    # Resume from checkpoint if available
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        q.load_state_dict(ckpt["q"])
        q_targ.load_state_dict(ckpt["q_targ"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt["global_step"]
        episode = ckpt["episode"]
        print(f"resumed from checkpoint at step {global_step}, episode {episode}")

    episode_reward = 0.0
    recent = deque(maxlen=20)

    obs, info = env.reset()
    obs, _, _, _, _ = env.step(1)  # press FIRE to launch ball
    obs_np = to_np_obs(obs)  # (4,84,84)
    current_lives = info.get("lives", 5)

    while global_step < max_steps:
        eps = epsilon_by_step(global_step)

        # Select action
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s_t = torch.from_numpy(obs_np).unsqueeze(0).to(DEVICE).float() / 255.0
                qvals = q(s_t)
                action = int(torch.argmax(qvals, dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        reward = np.clip(reward, -1.0, 1.0)
        done = bool(terminated or truncated)

        # Episodic life tracking: treat life loss as episode boundary for replay
        lives = info.get("lives", 0)
        life_lost = (lives < current_lives)
        current_lives = lives
        done_for_buffer = done or life_lost

        next_obs_np = to_np_obs(next_obs)

        rb.add(Transition(obs_np, action, float(reward), next_obs_np, done_for_buffer))

        obs_np = next_obs_np
        episode_reward += reward
        global_step += 1

        # Fire after losing a life (ball needs re-launch)
        if life_lost and not done:
            obs, _, _, _, _ = env.step(1)  # press FIRE to launch ball
            obs_np = to_np_obs(obs)

        # If episode ended, reset
        if done:
            episode += 1
            recent.append(episode_reward)
            if episode % 10 == 0:
                avg20 = sum(recent) / max(1, len(recent))
                print(f"ep {episode:5d} | step {global_step:7d} | eps {eps:.3f} | epR {episode_reward:.1f} | avg20 {avg20:.2f} | rb {len(rb)}")
            obs, info = env.reset()
            obs, _, _, _, _ = env.step(1)  # press FIRE to launch ball
            obs_np = to_np_obs(obs)
            current_lives = info.get("lives", 5)
            episode_reward = 0.0

        # Learn
        if len(rb) >= learn_start and global_step % train_freq == 0:
            s, a, r, s2, done_f = rb.sample(batch_size)

            s_t = torch.from_numpy(s).to(DEVICE).float() / 255.0
            a_t = torch.from_numpy(a).to(DEVICE).long()
            r_t = torch.from_numpy(r).to(DEVICE)
            s2_t = torch.from_numpy(s2).to(DEVICE).float() / 255.0
            done_t = torch.from_numpy(done_f).to(DEVICE)

            q_sa = q(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                best_actions = q(s2_t).argmax(dim=1)                       # online net picks the action
                max_q_next = q_targ(s2_t).gather(1, best_actions.unsqueeze(1)).squeeze(1)  # target net evaluates it
                target = r_t + gamma * (1.0 - done_t) * max_q_next

            loss = nn.functional.smooth_l1_loss(q_sa, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 1.0)
            optimizer.step()

        # Target net update
        if global_step % target_update_freq == 0 and global_step > 0:
            q_targ.load_state_dict(q.state_dict())
            # Save checkpoint with full training state
            torch.save({
                "q": q.state_dict(),
                "q_targ": q_targ.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "episode": episode,
            }, CHECKPOINT_PATH)
            print(f"saved checkpoint at step {global_step}")

    env.close()
    torch.save(q.state_dict(), "dqn_breakout_final.pt")
    print("done")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "play":
        play()
    else:
        main()