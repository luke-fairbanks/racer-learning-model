# 🏎️ RetroRacer

A continuous-control, top-down racing game built for reinforcement learning. Train an AI agent to race any track using SAC (Soft Actor-Critic), or drive it yourself.

## Features

- **Physics-based driving** — non-linear acceleration, understeer at high speed, engine braking
- **Visual track editor** — design custom circuits with live spline preview
- **Domain randomization** — train a base model on infinite random tracks, then fine-tune to any specific track in minutes
- **Interactive TUI** — clean console interface for managing tracks, models, and training runs
- **Checkpointing** — auto-saves every 100k steps so you never lose progress

## Quick Start

```bash
# Install dependencies
pip install gymnasium stable-baselines3 pygame numpy

# Launch the interactive menu
python run.py
```

```
  ╭────────────────────────────────╮
  │  🏎️  RetroRacer                │
  ╰────────────────────────────────╯

    1. Drive         Manual control
    2. Train         Train AI on a track
    3. Watch         Watch trained agent
    4. Edit          Track editor
    5. Train Base    Train on random tracks
    6. Fine-tune     Adapt base model to a track
    7. Quit
```

## Training Pipeline

### Train a Base Model

Train on randomly-generated tracks each episode. The agent learns universal racing skills — staying on road, braking into corners, accelerating on straights.

```bash
python run.py   # → option 5 → Enter (2M steps, ~overnight)
```

### Fine-tune for a Specific Track

Load the base model and adapt it to any track in ~200k steps (~5-10 min).

```bash
python run.py   # → option 6 → select base model → select track
```

### Train From Scratch

Train directly on a single track without a base model.

```bash
python run.py   # → option 2 → select track → Enter (1.5M steps)
```

## CLI Flags

For scripting and automation:

```bash
python run.py human                        # Drive manually
python run.py train 1.5M --track tracks/monaco.json
python run.py watch --track tracks/monaco.json
python run.py edit tracks/monaco.json
```

## Track Editor

Design tracks by placing control points. The editor generates smooth Catmull-Rom splines through them in real-time.

**Controls:** Left-click to place points, drag to move, right-click to delete, `S` to save, `T` to test drive.

## Project Structure

```
racing/
  track.py       Track data, spline generation, save/load, random generation
  env.py         Gymnasium environment — physics, rewards, rendering
  editor.py      Pygame track editor
  trainer.py     SAC training, fine-tuning, watching, human driving
  cli.py         Interactive TUI + CLI argument parsing
run.py           Entry point
```

## How It Works

### Observations (22-dim)

The agent sees: its speed, heading error, cross-track error, 6 lookahead heading errors (where the track goes), and 6 curvature values (how sharp upcoming turns are).

### Reward Signal

Progress along the track, speed bonus, centering penalty, heading alignment, steering smoothness, curvature-aware speed penalty (don't fly into corners), and a big lap completion bonus.

### Physics

Non-linear acceleration with speed-dependent power curve. Understeer model reduces steering effectiveness at high speed — the car won't turn at 280, forcing the agent to learn braking strategy.

## Models

Models auto-save with timestamps to prevent overwrites:

```
models/
  _base/          Base models (random tracks)
  default/        Default track
  monaco/         Custom tracks
```

## Tech Stack

- **[Gymnasium](https://gymnasium.farama.org/)** — RL environment interface
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** — SAC algorithm
- **[Pygame](https://www.pygame.org/)** — rendering and track editor
- **[NumPy](https://numpy.org/)** — physics and math
