# RetroRacer Notes

## Project Structure

```
racing/
  __init__.py      Package exports
  track.py         Track dataclass, spline math, save/load, random gen
  env.py           RetroRacerEnv — physics, rewards, rendering
  editor.py        Pygame track editor
  trainer.py       train_sac, train_base, fine_tune, watch, human_drive
  cli.py           TUI menus + CLI flag support
  __main__.py      python -m racing
run.py             Entry point (python run.py)
```

## How to Run

```bash
python run.py              # Interactive TUI
python run.py human        # Drive manually
python run.py train 1.5M   # Train on default track
python run.py watch        # Watch latest model
python run.py edit         # Track editor
```

## Training Pipeline

### 1. Train Base (option 5)

- Trains on a **randomly-generated track every episode** (domain randomization)
- Agent learns general racing: stay on road, brake corners, accelerate straights
- Default **2M steps** — good overnight run
- Saves to `models/_base/`

### 2. Fine-tune (option 6)

- Loads a base (or any) model, continues training on a **specific track**
- Lower learning rate (1e-4 vs 3e-4) for stability
- Default **200k steps** (~5-10 min)
- Saves to `models/{track_name}/`

### 3. Train (option 2)

- Trains from scratch on a single track
- Default **1.5M steps**

## Training Progress Bar Explained

```
██░░░░░░░░░░░░░░░░░░  11.8%  |  step  23,500/200,000  |  ep 20  |  R̄  +27.7  |  len   108  |  23500 fps
```

| Field                 | Meaning                                                                                                                                      |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `██░░░░░░...`         | Visual progress bar                                                                                                                          |
| `11.8%`               | % of total training steps completed                                                                                                          |
| `step 23,500/200,000` | Current step / total steps                                                                                                                   |
| `ep 20`               | Number of episodes completed (one episode = one attempt at the track, ends on crash or lap complete)                                         |
| `R̄ +27.7`             | **Average reward** over last 20 episodes. Higher = better. Negative = crashing a lot. Positive = making progress. Watch this climb over time |
| `len 108`             | Average episode length (steps). Short = crashing early. Long = surviving longer / completing laps                                            |
| `23500 fps`           | Training speed (steps per second). Higher = faster training                                                                                  |

**What to look for:**

- **R̄ climbing** = agent is learning 📈
- **R̄ plateaued** = agent has converged (good or stuck)
- **R̄ dropping** = training degraded — use a checkpoint instead
- **len increasing** = agent surviving longer on track
- **len hitting 4000** (max_steps) = agent completing full runs

## Checkpoints

- Auto-saved every **100k steps** as `{model}_ckpt_100k.zip`, `_ckpt_200k.zip`, etc.
- If training degrades, grab an earlier checkpoint

## Model File Location

```
models/
  _base/                              Base models (random tracks)
    2026-02-22_10-01_2M.zip
  default/                            Default track models
    2026-02-22_00-53_800k.zip
  monaco/                             Per-track models
    2026-02-22_11-00_200k.zip
```

## Physics

- **Max speed**: 280
- **Understeer**: steering effectiveness drops with speed² — must brake before corners
- **Non-linear accel**: punchy at low speed, tapers at high speed
- **No throttle** = engine braking (car decelerates)
- **Off-track grace**: 6 frames (~0.2s) before reset

## Observations (22-dim vector the AI sees)

- Speed, heading error (sin/cos), cross-track error
- 6 lookahead heading errors (sin/cos) — where the track goes
- 6 curvature values — how sharp upcoming turns are

## Reward Components

- **Progress along track** (biggest factor)
- **Speed bonus** (go fast)
- **Centering penalty** (stay on road)
- **Heading penalty** (face the right direction)
- **Steering smoothness** (don't jerk)
- **Curvature × speed penalty** (don't fly into corners)
- **Lap completion bonus** (+30)
- **Off-track penalty** (-5)

## Controls (Human Drive)

- **← →** Steer
- **↑** Throttle
- **R** Reset
- **ESC** Quit
