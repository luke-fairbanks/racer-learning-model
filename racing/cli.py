"""Interactive CLI with TUI menus for RetroRacer."""

import os
import sys

from racing.track import list_tracks
from racing.trainer import (
    human_drive,
    train_sac,
    train_base,
    fine_tune,
    race,
    watch,
    list_models,
    list_all_models,
)
from racing.editor import track_editor


# -------------------------
# ANSI helpers
# -------------------------
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _header():
    print()
    print(f"  {BOLD}{CYAN}╭────────────────────────────────╮{RESET}")
    print(f"  {BOLD}{CYAN}│{RESET}  {BOLD}🏎️  RetroRacer                {CYAN}│{RESET}")
    print(f"  {BOLD}{CYAN}╰────────────────────────────────╯{RESET}")
    print()


def _prompt(msg="  > "):
    try:
        return input(f"{CYAN}{msg}{RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


# -------------------------
# Track selection
# -------------------------
def _select_track():
    """Interactive track selection. Returns (track_name, track_file_or_None)."""
    tracks = list_tracks()

    print(f"  {BOLD}Select a track:{RESET}")
    print()
    for i, t in enumerate(tracks):
        marker = f"{GREEN}●{RESET}" if t["path"] else f"{DIM}●{RESET}"
        print(f"    {marker} {BOLD}{i}{RESET}. {t['name']}")
    print()

    choice = _prompt("  Track number > ")
    if choice is None:
        return None, None

    try:
        idx = int(choice)
        if 0 <= idx < len(tracks):
            t = tracks[idx]
            return t["name"], t["path"]
    except ValueError:
        pass

    print(f"  {RED}Invalid choice.{RESET}")
    return None, None


# -------------------------
# Model selection
# -------------------------
def _select_model(track_name):
    """Interactive model selection for a track. Returns model_path or None."""
    models = list_models(track_name)

    if not models:
        print(f"  {YELLOW}No trained models found for '{track_name}'.{RESET}")
        print(f"  {DIM}Train first with: python run.py train{RESET}")
        return None

    print(f"  {BOLD}Select a model for '{track_name}':{RESET}")
    print()
    for i, m in enumerate(models):
        print(f"    {MAGENTA}●{RESET} {BOLD}{i}{RESET}. {m['name']}")
    print()

    choice = _prompt("  Model number > ")
    if choice is None:
        return None

    try:
        idx = int(choice)
        if 0 <= idx < len(models):
            return models[idx]["path"]
    except ValueError:
        pass

    print(f"  {RED}Invalid choice.{RESET}")
    return None


# -------------------------
# Menu actions
# -------------------------
def _action_drive():
    track_name, track_file = _select_track()
    if track_name is None:
        return
    print(f"\n  {GREEN}Launching on '{track_name}'...{RESET}")
    print(f"  {DIM}Controls: ← → steer | ↑ throttle | R reset | ESC quit{RESET}\n")
    human_drive(track_file=track_file)


def _action_train():
    track_name, track_file = _select_track()
    if track_name is None:
        return

    print()
    steps_input = _prompt(f"  Steps (default 1.5M) > ")
    if steps_input is None:
        return

    if steps_input == "":
        total_steps = 1_500_000
    else:
        try:
            # Support shorthand: 800k, 1.5M, etc.
            s = steps_input.lower().strip()
            if s.endswith("m"):
                total_steps = int(float(s[:-1]) * 1_000_000)
            elif s.endswith("k"):
                total_steps = int(float(s[:-1]) * 1_000)
            else:
                total_steps = int(s)
        except ValueError:
            print(f"  {RED}Invalid step count.{RESET}")
            return

    print(f"\n  {GREEN}Training on '{track_name}' for {total_steps:,} steps...{RESET}\n")
    model_path = train_sac(
        total_steps=total_steps,
        track_file=track_file,
        track_name=track_name,
    )
    print()
    _prompt("  Press Enter to continue...")


def _action_watch():
    track_name, track_file = _select_track()
    if track_name is None:
        return

    model_path = _select_model(track_name)
    if model_path is None:
        _prompt("  Press Enter to continue...")
        return

    print(f"\n  {GREEN}Watching agent on '{track_name}'...{RESET}")
    print(f"  {DIM}Model: {model_path}{RESET}\n")
    watch(model_path=model_path, track_file=track_file)


def _action_edit():
    tracks = list_tracks()
    print(f"  {BOLD}Edit a track:{RESET}")
    print()
    print(f"    {GREEN}●{RESET} {BOLD}0{RESET}. New track")
    for i, t in enumerate(tracks):
        if t["path"]:
            print(f"    {YELLOW}●{RESET} {BOLD}{i}{RESET}. {t['name']}")
    print()

    choice = _prompt("  Choice > ")
    if choice is None:
        return

    try:
        idx = int(choice)
        if idx == 0:
            name = _prompt("  Track name > ")
            if not name:
                return
            save_path = f"tracks/{name}.json"
            track_editor(save_path=save_path)
        elif 1 <= idx <= len(tracks) - 1:
            t = tracks[idx]
            if t["path"]:
                track_editor(load_file=t["path"], save_path=t["path"])
        else:
            print(f"  {RED}Invalid choice.{RESET}")
    except ValueError:
        print(f"  {RED}Invalid choice.{RESET}")


def _action_race():
    track_name, track_file = _select_track()
    if track_name is None:
        return

    model_path = _select_model(track_name)
    if model_path is None:
        # Also check base models
        base_models = list_models("_base")
        if base_models:
            print(f"  {YELLOW}No track-specific model. Use a base model?{RESET}")
            for i, m in enumerate(base_models):
                print(f"    {MAGENTA}●{RESET} {BOLD}{i}{RESET}. [base] {m['name']}")
            print()
            choice = _prompt("  Model number (or Enter to cancel) > ")
            if choice is None or choice == "":
                return
            try:
                idx = int(choice)
                if 0 <= idx < len(base_models):
                    model_path = base_models[idx]["path"]
            except ValueError:
                pass
        if model_path is None:
            _prompt("  Press Enter to continue...")
            return

    print(f"\n  {GREEN}🏁 RACE: You vs AI on '{track_name}'!{RESET}")
    print(f"  {DIM}Controls: ← → steer | ↑ throttle | R restart | ESC quit{RESET}\n")
    race(model_path=model_path, track_file=track_file)


def _action_train_base():
    print(f"  {BOLD}Train a base model on random tracks{RESET}")
    print(f"  {DIM}This creates a generalizable agent that can be fine-tuned for any track.{RESET}")
    print()

    steps_input = _prompt(f"  Steps (default 2M) > ")
    if steps_input is None:
        return

    if steps_input == "":
        total_steps = 2_000_000
    else:
        try:
            s = steps_input.lower().strip()
            if s.endswith("m"):
                total_steps = int(float(s[:-1]) * 1_000_000)
            elif s.endswith("k"):
                total_steps = int(float(s[:-1]) * 1_000)
            else:
                total_steps = int(s)
        except ValueError:
            print(f"  {RED}Invalid step count.{RESET}")
            return

    print(f"\n  {GREEN}Training base model for {total_steps:,} steps on random tracks...{RESET}\n")
    train_base(total_steps=total_steps)
    print()
    _prompt("  Press Enter to continue...")


def _action_fine_tune():
    # First, find available base models
    base_models = list_models("_base")
    all_models = list_all_models()

    # Collect all available models to fine-tune from
    available = []
    if base_models:
        for m in base_models:
            available.append({"name": f"[base] {m['name']}", "path": m["path"]})
    for track_name, models in all_models.items():
        if track_name == "_base":
            continue
        for m in models:
            available.append({"name": f"[{track_name}] {m['name']}", "path": m["path"]})

    if not available:
        print(f"  {YELLOW}No models available to fine-tune from.{RESET}")
        print(f"  {DIM}Train a base model first (option 5).{RESET}")
        _prompt("  Press Enter to continue...")
        return

    print(f"  {BOLD}Select a model to fine-tune from:{RESET}")
    print()
    for i, m in enumerate(available):
        print(f"    {MAGENTA}●{RESET} {BOLD}{i}{RESET}. {m['name']}")
    print()

    choice = _prompt("  Model number > ")
    if choice is None:
        return
    try:
        idx = int(choice)
        if not (0 <= idx < len(available)):
            print(f"  {RED}Invalid choice.{RESET}")
            return
    except ValueError:
        print(f"  {RED}Invalid choice.{RESET}")
        return

    base_path = available[idx]["path"]

    # Select target track
    print()
    track_name, track_file = _select_track()
    if track_name is None:
        return

    print()
    steps_input = _prompt(f"  Steps (default 200k) > ")
    if steps_input is None:
        return

    if steps_input == "":
        total_steps = 200_000
    else:
        try:
            s = steps_input.lower().strip()
            if s.endswith("m"):
                total_steps = int(float(s[:-1]) * 1_000_000)
            elif s.endswith("k"):
                total_steps = int(float(s[:-1]) * 1_000)
            else:
                total_steps = int(s)
        except ValueError:
            print(f"  {RED}Invalid step count.{RESET}")
            return

    print(f"\n  {GREEN}Fine-tuning on '{track_name}' for {total_steps:,} steps...{RESET}\n")
    fine_tune(
        base_model_path=base_path,
        total_steps=total_steps,
        track_file=track_file,
        track_name=track_name,
    )
    print()
    _prompt("  Press Enter to continue...")


# -------------------------
# Main menu
# -------------------------
def main_menu():
    """Interactive main menu."""
    while True:
        _clear()
        _header()

        # Show available tracks and models summary
        tracks = list_tracks()
        all_models = list_all_models()
        track_count = len(tracks) - 1  # minus default
        model_count = sum(len(m) for m in all_models.values())

        print(f"  {DIM}{track_count} custom track{'s' if track_count != 1 else ''}  ·  "
              f"{model_count} trained model{'s' if model_count != 1 else ''}{RESET}")
        print()

        options = [
            ("1", "Drive", "Manual control"),
            ("2", "Race", "Race against the AI"),
            ("3", "Train", "Train AI on a track"),
            ("4", "Watch", "Watch trained agent"),
            ("5", "Edit", "Track editor"),
            ("6", "Train Base", "Train on random tracks (generalizable)"),
            ("7", "Fine-tune", "Adapt base model to specific track"),
            ("8", "Quit", ""),
        ]
        for key, label, desc in options:
            desc_str = f"  {DIM}{desc}{RESET}" if desc else ""
            print(f"    {BOLD}{key}{RESET}. {label}{desc_str}")
        print()

        choice = _prompt()
        if choice is None or choice == "8" or choice.lower() == "q":
            print(f"\n  {DIM}Goodbye!{RESET}\n")
            break

        _clear()
        _header()

        if choice == "1":
            _action_drive()
        elif choice == "2":
            _action_race()
        elif choice == "3":
            _action_train()
        elif choice == "4":
            _action_watch()
        elif choice == "5":
            _action_edit()
        elif choice == "6":
            _action_train_base()
        elif choice == "7":
            _action_fine_tune()
        else:
            print(f"  {RED}Unknown option '{choice}'.{RESET}")
            _prompt("  Press Enter to continue...")


# -------------------------
# CLI entry point (supports both TUI and flags)
# -------------------------
def main():
    """Main entry point. Supports interactive TUI or direct CLI flags."""

    # If no args beyond script name, launch TUI
    if len(sys.argv) <= 1:
        main_menu()
        return

    mode = sys.argv[1]

    # Parse --track flag
    track_file = None
    track_name = "default"
    for i, arg in enumerate(sys.argv):
        if arg == "--track" and i + 1 < len(sys.argv):
            track_file = sys.argv[i + 1]
            track_name = os.path.splitext(os.path.basename(track_file))[0]

    if mode == "human":
        human_drive(track_file=track_file)

    elif mode == "train":
        steps = 800_000
        for arg in sys.argv[2:]:
            if arg.startswith("--"):
                continue
            try:
                s = arg.lower()
                if s.endswith("m"):
                    steps = int(float(s[:-1]) * 1_000_000)
                elif s.endswith("k"):
                    steps = int(float(s[:-1]) * 1_000)
                else:
                    steps = int(s)
                break
            except ValueError:
                continue
        train_sac(total_steps=steps, track_file=track_file, track_name=track_name)

    elif mode == "watch":
        model_path = None
        for arg in sys.argv[2:]:
            if not arg.startswith("--") and arg.endswith(".zip"):
                model_path = arg
                break

        if model_path is None:
            # Try to find latest model for this track
            models = list_models(track_name)
            if models:
                model_path = models[-1]["path"]
                print(f"  Using latest model: {model_path}")
            else:
                print(f"  {RED}No models found for track '{track_name}'.{RESET}")
                print(f"  {DIM}Train first: python run.py train --track {track_file or 'default'}{RESET}")
                return

        watch(model_path=model_path, track_file=track_file)

    elif mode == "edit":
        load = None
        for arg in sys.argv[2:]:
            if not arg.startswith("--"):
                load = arg
                break
        if load is None:
            load = track_file
        track_editor(load_file=load)

    else:
        print(f"  {BOLD}RetroRacer{RESET} — Usage:")
        print()
        print(f"    python run.py              {DIM}# Interactive TUI{RESET}")
        print(f"    python run.py human        {DIM}# Drive manually{RESET}")
        print(f"    python run.py train [800k] {DIM}# Train AI{RESET}")
        print(f"    python run.py watch        {DIM}# Watch trained agent{RESET}")
        print(f"    python run.py edit         {DIM}# Track editor{RESET}")
        print()
        print(f"  {DIM}Flags: --track tracks/file.json{RESET}")
