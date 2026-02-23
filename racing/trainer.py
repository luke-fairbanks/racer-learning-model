"""Training, watching, and human driving functionality."""

import os
import sys
from collections import deque
from datetime import datetime

import numpy as np

from racing.env import RetroRacerEnv


# -------------------------
# Model path management
# -------------------------


def _format_steps(steps):
    """Format step count: 800000 -> '800k', 1600000 -> '1.6M'."""
    if steps >= 1_000_000:
        val = steps / 1_000_000
        return f"{val:.1f}M" if val != int(val) else f"{int(val)}M"
    elif steps >= 1_000:
        val = steps / 1_000
        return f"{val:.0f}k" if val == int(val) else f"{val:.1f}k"
    return str(steps)


def make_model_path(track_name, total_steps, models_dir="models"):
    """Generate a timestamped model path: models/{track_name}/{timestamp}_{steps}.zip"""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    steps_str = _format_steps(total_steps)
    dirpath = os.path.join(models_dir, track_name)
    os.makedirs(dirpath, exist_ok=True)
    return os.path.join(dirpath, f"{ts}_{steps_str}")


def list_models(track_name, models_dir="models"):
    """List available models for a given track. Returns list of dicts with name/path."""
    dirpath = os.path.join(models_dir, track_name)
    models = []
    if os.path.isdir(dirpath):
        for f in sorted(os.listdir(dirpath)):
            if f.endswith(".zip"):
                name = os.path.splitext(f)[0]
                models.append({
                    "name": name,
                    "path": os.path.join(dirpath, f),
                })
    return models


def list_all_models(models_dir="models"):
    """List all models across all tracks."""
    all_models = {}
    if os.path.isdir(models_dir):
        for track_name in sorted(os.listdir(models_dir)):
            track_dir = os.path.join(models_dir, track_name)
            if os.path.isdir(track_dir):
                models = list_models(track_name, models_dir)
                if models:
                    all_models[track_name] = models
    return all_models


# -------------------------
# Human control loop
# -------------------------
def human_drive(track_file=None):
    import pygame

    env = RetroRacerEnv(render_mode="human", seed=0, track_seed=123, track_file=track_file)
    obs, info = env.reset()
    env.render()  # force pygame init before event loop

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        if keys[pygame.K_r]:
            obs, info = env.reset()

        steer = 0.0
        if keys[pygame.K_LEFT]:
            steer -= 1.0
        if keys[pygame.K_RIGHT]:
            steer += 1.0

        throttle = 0.0  # no input = not moving
        if keys[pygame.K_UP]:
            throttle = 1.0

        action = np.array([steer, throttle], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


# -------------------------
# Train SAC
# -------------------------
def train_sac(total_steps=1_500_000, save_path=None, track_file=None,
              track_name="default", randomize_track=False):
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback

    # Auto-generate save path if not specified
    if save_path is None:
        save_path = make_model_path(track_name, total_steps)

    class InlineLogger(BaseCallback):
        """Prints a single updating line instead of repeating tables."""
        def __init__(self):
            super().__init__()
            self.ep_rewards = deque(maxlen=20)
            self.ep_lengths = deque(maxlen=20)

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    self.ep_rewards.append(info["episode"]["r"])
                    self.ep_lengths.append(info["episode"]["l"])

            ep_count = len(self.ep_rewards)
            if ep_count > 0 and self.num_timesteps % 500 == 0:
                avg_r = sum(self.ep_rewards) / len(self.ep_rewards)
                avg_l = sum(self.ep_lengths) / len(self.ep_lengths)
                pct = self.num_timesteps / total_steps * 100
                fps = int(self.num_timesteps / (max(1e-6, self.locals.get("time_elapsed", 1))))
                bar_w = 20
                filled = int(bar_w * pct / 100)
                bar = "█" * filled + "░" * (bar_w - filled)
                sys.stdout.write(
                    f"\r  {bar} {pct:5.1f}%  |  step {self.num_timesteps:>7,}/{total_steps:,}  |"
                    f"  ep {ep_count}  |  R̄ {avg_r:+6.1f}  |  len {avg_l:5.0f}  |  {fps} fps   "
                )
                sys.stdout.flush()

            # Checkpoint every 100k steps
            if self.num_timesteps % 100_000 == 0 and self.num_timesteps > 0:
                ckpt = f"{save_path}_ckpt_{_format_steps(self.num_timesteps)}"
                self.model.save(ckpt)
                sys.stdout.write(f"\n  💾 Checkpoint saved: {ckpt}.zip\n")
                sys.stdout.flush()

            return True

        def _on_training_end(self):
            sys.stdout.write("\n")

    env = RetroRacerEnv(
        render_mode=None, seed=0, track_seed=123,
        track_file=track_file, randomize_track=randomize_track,
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        learning_rate=3e-4,
        buffer_size=500_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=20_000,
        ent_coef="auto",
    )

    mode_str = "random tracks (base model)" if randomize_track else f"'{track_name}'"
    print(f"  Training SAC for {total_steps:,} steps on {mode_str}...")
    print(f"  Model will save to: {save_path}.zip")
    print()
    model.learn(total_timesteps=int(total_steps), callback=InlineLogger())
    model.save(save_path)
    env.close()
    print(f"\n  ✓ Saved model to {save_path}.zip")
    return save_path + ".zip"


def train_base(total_steps=2_000_000, save_path=None):
    """Train a base model on randomized tracks for generalization."""
    if save_path is None:
        save_path = make_model_path("_base", total_steps)
    return train_sac(
        total_steps=total_steps,
        save_path=save_path,
        track_name="_base",
        randomize_track=True,
    )


def fine_tune(base_model_path, total_steps=200_000, track_file=None, track_name="default"):
    """Load a pre-trained model and fine-tune it on a specific track."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback

    save_path = make_model_path(track_name, total_steps)

    env = RetroRacerEnv(
        render_mode=None, seed=0, track_seed=123, track_file=track_file,
    )

    # Load existing model and set the new environment
    model = SAC.load(base_model_path, env=env)
    # Reset learning rate for fine-tuning (lower = more stable)
    model.learning_rate = 1e-4

    class InlineLogger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.ep_rewards = deque(maxlen=20)
            self.ep_lengths = deque(maxlen=20)

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    self.ep_rewards.append(info["episode"]["r"])
                    self.ep_lengths.append(info["episode"]["l"])

            ep_count = len(self.ep_rewards)
            if ep_count > 0 and self.num_timesteps % 500 == 0:
                avg_r = sum(self.ep_rewards) / len(self.ep_rewards)
                avg_l = sum(self.ep_lengths) / len(self.ep_lengths)
                pct = self.num_timesteps / total_steps * 100
                fps = int(self.num_timesteps / (max(1e-6, self.locals.get("time_elapsed", 1))))
                bar_w = 20
                filled = int(bar_w * pct / 100)
                bar = "█" * filled + "░" * (bar_w - filled)
                sys.stdout.write(
                    f"\r  {bar} {pct:5.1f}%  |  step {self.num_timesteps:>7,}/{total_steps:,}  |"
                    f"  ep {ep_count}  |  R̄ {avg_r:+6.1f}  |  len {avg_l:5.0f}  |  {fps} fps   "
                )
                sys.stdout.flush()
            return True

        def _on_training_end(self):
            sys.stdout.write("\n")

    print(f"  Fine-tuning from {os.path.basename(base_model_path)} on '{track_name}'...")
    print(f"  Steps: {total_steps:,}  |  LR: {model.learning_rate}")
    print(f"  Saving to: {save_path}.zip")
    print()
    model.learn(total_timesteps=int(total_steps), callback=InlineLogger())
    model.save(save_path)
    env.close()
    print(f"\n  ✓ Fine-tuned model saved to {save_path}.zip")
    return save_path + ".zip"


# -------------------------
# Watch trained agent
# -------------------------
def watch(model_path="retro_racer_sac.zip", episodes=5, track_file=None):
    import pygame
    import torch
    import math
    from stable_baselines3 import SAC

    env = RetroRacerEnv(render_mode="human", seed=0, track_seed=123, track_file=track_file)
    model = SAC.load(model_path)

    show_viz = True  # toggle with V key
    env._delay_flip = True  # we'll flip after drawing viz

    for ep in range(episodes):
        obs, info = env.reset()
        env.render()
        ep_r = 0.0
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                    show_viz = not show_viz

            action, _ = model.predict(obs, deterministic=True)

            # Get Q-value from critic before stepping
            q_value = None
            if show_viz:
                try:
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
                    act_t = torch.FloatTensor(action).unsqueeze(0).to(model.device)
                    with torch.no_grad():
                        q1, q2 = model.critic(obs_t, act_t)
                        q_value = float(torch.min(q1, q2).item())
                except Exception:
                    q_value = None

            obs, reward, terminated, truncated, info = env.step(action)
            ep_r += reward

            if show_viz and env._pygame_inited:
                _draw_ai_viz(env, action, obs, q_value, model)
            pygame.display.flip()

            done = bool(terminated or truncated)

        print(f"  [watch] episode {ep+1}/{episodes} total reward: {ep_r:.2f}")

    env.close()


def _draw_ai_viz(env, action, obs, q_value, model):
    """Draw AI decision visualization overlays."""
    import pygame
    import math

    scr = env._screen
    font = env._font
    zoom = 0.72
    cam = env.pos.copy()
    N = len(env.track.pts)

    # --- 1. Lookahead rays (what the AI sees ahead) ---
    from racing.env import find_nearest_track_point
    i, proj, t, dist, s, tangent = find_nearest_track_point(env.track, env.pos)

    for k in range(1, env.lookahead + 1):
        idx = (i + k * 3) % N
        pt = env.track.pts[idx]
        pt_scr = env._world_to_screen(pt, cam, zoom)
        car_scr = env._world_to_screen(env.pos, cam, zoom)

        # Color by curvature from observation (curvature values are at indices 4+2*lookahead+k-1)
        curv_idx = 4 + 2 * env.lookahead + (k - 1)
        if curv_idx < len(obs):
            curv = abs(obs[curv_idx])  # 0 = straight, ~1 = sharp
            # Green (straight) → Yellow → Red (sharp)
            r = min(255, int(curv * 3 * 255))
            g = max(0, 255 - int(curv * 2 * 255))
            color = (r, g, 40)
        else:
            color = (100, 200, 100)

        # Draw ray line
        pygame.draw.line(scr, (*color, ), car_scr, pt_scr, 1)
        # Draw point dot
        pygame.draw.circle(scr, color, pt_scr, 4)

    # --- 2. Predicted path (simulate forward with current action) ---
    sim_pos = env.pos.copy()
    sim_vel = env.vel
    sim_heading = env.heading
    dt = env.dt

    pred_points = []
    for step in range(15):
        forward = np.array([math.cos(sim_heading), math.sin(sim_heading)], dtype=np.float32)
        sim_pos = sim_pos + forward * float(sim_vel * dt)

        steer = float(action[0])
        speed_factor = min(sim_vel / (env.max_speed * env.steer_speed_scale + 1e-8), 1.0)
        speed_pct = sim_vel / (env.max_speed + 1e-8)
        grip_factor = 1.0 / (1.0 + 2.5 * speed_pct * speed_pct)
        turn_rate = steer * env.steer_rate * (0.25 + 0.75 * speed_factor) * grip_factor
        sim_heading += turn_rate * dt

        pt_scr = env._world_to_screen(sim_pos, cam, zoom)
        pred_points.append(pt_scr)

    # Draw predicted path as dots
    for pi, pt in enumerate(pred_points):
        alpha = max(40, 200 - pi * 12)
        pygame.draw.circle(scr, (0, 220, 230), pt, 2)

    # --- 3. Steer/Throttle gauges (bottom-left) ---
    gauge_x = 14
    gauge_y = scr.get_height() - 60

    steer_val = float(action[0])
    throttle_val = float(action[1])

    # Steer gauge
    pygame.draw.rect(scr, (40, 40, 55), (gauge_x, gauge_y, 120, 12))
    center_x = gauge_x + 60
    bar_w = int(steer_val * 55)
    if bar_w > 0:
        pygame.draw.rect(scr, (255, 180, 40), (center_x, gauge_y + 1, bar_w, 10))
    elif bar_w < 0:
        pygame.draw.rect(scr, (255, 180, 40), (center_x + bar_w, gauge_y + 1, -bar_w, 10))
    pygame.draw.line(scr, (200, 200, 220), (center_x, gauge_y), (center_x, gauge_y + 12), 1)
    steer_label = font.render(f"STEER {steer_val:+.2f}", True, (160, 170, 190))
    scr.blit(steer_label, (gauge_x + 130, gauge_y - 1))

    # Throttle gauge
    gauge_y += 18
    pygame.draw.rect(scr, (40, 40, 55), (gauge_x, gauge_y, 120, 12))
    t_bar_w = int(throttle_val * 120)
    t_color = (60, 200, 80) if throttle_val > 0.3 else (200, 80, 60)
    pygame.draw.rect(scr, t_color, (gauge_x, gauge_y + 1, max(0, t_bar_w), 10))
    throttle_label = font.render(f"THRTL {throttle_val:.2f}", True, (160, 170, 190))
    scr.blit(throttle_label, (gauge_x + 130, gauge_y - 1))

    # --- 4. Q-value confidence meter (bottom-right) ---
    if q_value is not None:
        q_x = scr.get_width() - 180
        q_y = scr.get_height() - 42

        # Normalize Q to a 0-1 bar (typical range: -10 to +50)
        q_norm = max(0.0, min(1.0, (q_value + 10) / 60))

        pygame.draw.rect(scr, (40, 40, 55), (q_x, q_y, 160, 12))
        # Color: red (low confidence) → green (high confidence)
        q_r = max(0, int(255 * (1 - q_norm)))
        q_g = min(255, int(255 * q_norm))
        q_bar_w = int(q_norm * 160)
        pygame.draw.rect(scr, (q_r, q_g, 40), (q_x, q_y + 1, max(0, q_bar_w), 10))

        q_label = font.render(f"Q: {q_value:+.1f}", True, (160, 170, 190))
        scr.blit(q_label, (q_x, q_y - 18))



# -------------------------
# Race: Human vs AI
# -------------------------
def race(model_path, track_file=None):
    """Race the human player against a trained AI agent."""
    import pygame
    from stable_baselines3 import SAC

    # Human env (renders)
    human_env = RetroRacerEnv(render_mode="human", seed=0, track_seed=123, track_file=track_file)
    # AI env (no render — we sync its position to human_env.opponents)
    ai_env = RetroRacerEnv(render_mode=None, seed=0, track_seed=123, track_file=track_file)

    model = SAC.load(model_path)

    def _reset_race():
        h_obs, _ = human_env.reset()
        a_obs, _ = ai_env.reset()
        # Stagger: offset AI to the side
        offset_dir = ai_env.track.pts[1] - ai_env.track.pts[0]
        offset_dir = offset_dir / (float(np.linalg.norm(offset_dir)) + 1e-8)
        perp = np.array([-offset_dir[1], offset_dir[0]], dtype=np.float32)
        ai_env.pos = ai_env.track.pts[0].copy() + perp * 15
        return h_obs, a_obs

    human_obs, ai_obs = _reset_race()
    human_env.render()

    race_frame = 0
    human_finish_frame = None
    ai_finish_frame = None
    race_over = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        if keys[pygame.K_r]:
            human_obs, ai_obs = _reset_race()
            race_frame = 0
            human_finish_frame = None
            ai_finish_frame = None
            race_over = False
            if hasattr(race, '_scoreboard'):
                del race._scoreboard
            continue

        if not race_over:
            race_frame += 1

            # Human input
            steer = 0.0
            if keys[pygame.K_LEFT]:
                steer -= 1.0
            if keys[pygame.K_RIGHT]:
                steer += 1.0
            throttle = 0.0
            if keys[pygame.K_UP]:
                throttle = 1.0

            human_action = np.array([steer, throttle], dtype=np.float32)
            ai_action, _ = model.predict(ai_obs, deterministic=True)

            # Step both
            human_obs, _, h_term, h_trunc, h_info = human_env.step(human_action)
            ai_obs, _, a_term, a_trunc, a_info = ai_env.step(ai_action)

            # Sync AI position for rendering
            human_env.opponents = [(ai_env.pos.copy(), ai_env.heading, ai_env.lap_progress)]

            # Check for finishes
            if human_finish_frame is None and human_env.lap_progress >= human_env.track.total:
                human_finish_frame = race_frame
            if ai_finish_frame is None and ai_env.lap_progress >= ai_env.track.total:
                ai_finish_frame = race_frame

            # Reset crashed cars (but keep racing)
            if a_term or a_trunc:
                if ai_finish_frame is None:
                    ai_obs, _ = ai_env.reset()
            if h_term or h_trunc:
                if human_finish_frame is None:
                    human_obs, _ = human_env.reset()

            # Race ends when both finish or at least one has finished
            if human_finish_frame is not None and ai_finish_frame is not None:
                race_over = True
            elif human_finish_frame is not None or ai_finish_frame is not None:
                # Give the other racer 5 more seconds to finish
                first_finish = human_finish_frame or ai_finish_frame
                if race_frame - first_finish > 150:  # 5 seconds at 30fps
                    race_over = True

        else:
            # Race is over — show scoreboard overlay (cached to avoid flicker)
            if not hasattr(race, '_scoreboard'):
                # Render one final game frame as background
                human_env.render()
                scr = human_env._screen
                font = human_env._font

                # Darken background
                overlay = pygame.Surface(scr.get_size(), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 160))
                scr.blit(overlay, (0, 0))

                cx = scr.get_width() // 2
                cy = scr.get_height() // 2

                # Title
                title = font.render("RACE COMPLETE", True, (255, 255, 255))
                scr.blit(title, (cx - title.get_width() // 2, cy - 80))

                # Determine winner
                if human_finish_frame and ai_finish_frame:
                    if human_finish_frame < ai_finish_frame:
                        winner_text = "YOU WIN!"
                        winner_color = (80, 255, 80)
                    elif ai_finish_frame < human_finish_frame:
                        winner_text = "AI WINS!"
                        winner_color = (100, 160, 255)
                    else:
                        winner_text = "TIE!"
                        winner_color = (255, 255, 100)
                elif human_finish_frame:
                    winner_text = "YOU WIN!"
                    winner_color = (80, 255, 80)
                else:
                    winner_text = "AI WINS!"
                    winner_color = (100, 160, 255)

                wt = font.render(winner_text, True, winner_color)
                scr.blit(wt, (cx - wt.get_width() // 2, cy - 50))

                # Times
                def _fmt_time(frames):
                    if frames is None:
                        return "DNF"
                    secs = frames / 30.0
                    return f"{secs:.1f}s"

                you_line = f"YOU:  {_fmt_time(human_finish_frame)}"
                ai_line = f"AI:   {_fmt_time(ai_finish_frame)}"
                yt = font.render(you_line, True, (220, 60, 60))
                at = font.render(ai_line, True, (60, 120, 240))
                scr.blit(yt, (cx - 60, cy - 10))
                scr.blit(at, (cx - 60, cy + 16))

                # Instructions
                restart = font.render("R = Restart  |  ESC = Quit", True, (140, 140, 160))
                scr.blit(restart, (cx - restart.get_width() // 2, cy + 60))

                # Cache the fully composed screen
                race._scoreboard = scr.copy()
                pygame.display.flip()
            else:
                human_env._screen.blit(race._scoreboard, (0, 0))
                pygame.display.flip()

            human_env._clock.tick(30)
            continue

    human_env.close()
    ai_env.close()

