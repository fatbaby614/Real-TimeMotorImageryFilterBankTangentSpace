"""Real-time motor imagery control with LSL streaming and pygame UI."""
from __future__ import annotations

import argparse
import math
import time
from collections import Counter, deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple, Union

import numpy as np
import pygame
from pylsl import StreamInlet, resolve_byprop

from algorithms.fbcsp import FilterBankCSPClassifier
from algorithms_collection import FilterBankTangentSpace
from config import mi_config as cfg


def load_ea_matrix(model_dir: Path) -> Optional[np.ndarray]:
    """Load EA whitening matrix if available.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        EA whitening matrix or None if not found
    """
    ea_path = Path(model_dir) / "ea_whitening_matrix.npy"
    if ea_path.exists():
        print(f"Loading EA whitening matrix from {ea_path}")
        return np.load(ea_path)
    return None


LABEL_TO_COMMAND = {
    0: cfg.IDLE_COMMAND,
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
}

COMMAND_TO_VECTOR = {
    cfg.IDLE_COMMAND: (0.0, 0.0),
    "UP": (0.0, -1.0),
    "DOWN": (0.0, 1.0),
    "LEFT": (-1.0, 0.0),
    "RIGHT": (1.0, 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time MI control loop")
    parser.add_argument("model_dir", help="Directory containing trained model artifacts")
    parser.add_argument("--stream-name", default=cfg.LSL_STREAM_NAME, help="LSL stream name")
    parser.add_argument(
        "--itr-mode",
        action="store_true",
        help="Enable ITR measurement: display targets and record accuracy / timing to compute ITR.",
    )
    parser.add_argument(
        "--itr-trials",
        type=int,
        default=20,
        help="Number of trials per class to run in ITR mode (default: 20).",
    )
    return parser.parse_args()


# ---------- ITR helpers ----------

def itr_bits_per_trial(n_classes: int, accuracy: float) -> float:
    """Shannon ITR per trial (bits) given N classes and accuracy P."""
    if n_classes < 2:
        return 0.0
    p = max(1e-9, min(1.0 - 1e-9, accuracy))
    q = (1.0 - p) / (n_classes - 1)
    entropy = p * math.log2(p) + (n_classes - 1) * q * math.log2(max(q, 1e-9))
    return math.log2(n_classes) + entropy


def draw_itr_cue(
    screen: pygame.Surface,
    target: str,
    trial_idx: int,
    total: int,
    command: str,
    confidence: float,
    correct: int,
    elapsed: float,
) -> None:
    screen.fill(cfg.BACKGROUND_COLOR)
    w, h = cfg.SCREEN_SIZE
    arrow_map = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→", cfg.IDLE_COMMAND: "○"}
    font_big = pygame.font.SysFont(cfg.FONT_NAME, 120)
    font_mid = pygame.font.SysFont(cfg.FONT_NAME, 36)
    font_sm  = pygame.font.SysFont(cfg.FONT_NAME, 26)

    # Target arrow
    tgt_surf = font_big.render(arrow_map.get(target, target), True, (80, 220, 120))
    screen.blit(tgt_surf, tgt_surf.get_rect(center=(w // 2, h // 2 - 60)))

    # Current prediction
    pred_color = (80, 220, 120) if command == target else (220, 90, 90)
    pred_surf = font_mid.render(f"Predicted: {arrow_map.get(command, command)}", True, pred_color)
    screen.blit(pred_surf, pred_surf.get_rect(center=(w // 2, h // 2 + 60)))

    # Stats row
    stats = f"Trial {trial_idx}/{total}   Correct: {correct}   Conf: {confidence:.2f}   Elapsed: {elapsed:.1f}s"
    stats_surf = font_sm.render(stats, True, (200, 210, 240))
    screen.blit(stats_surf, stats_surf.get_rect(center=(w // 2, h - 40)))

    # Confidence bar
    bar_w = int(w * 0.5)
    bar_x = (w - bar_w) // 2
    bar_y = h - 80
    pygame.draw.rect(screen, (50, 60, 90), (bar_x, bar_y, bar_w, 12), border_radius=6)
    filled = int(bar_w * max(0.0, min(1.0, confidence)))
    if filled:
        pygame.draw.rect(screen, (80, 200, 140), (bar_x, bar_y, filled, 12), border_radius=6)


def run_itr_mode(
    model: Union[FilterBankCSPClassifier, FilterBankTangentSpace],
    inlet: StreamInlet,
    n_classes: int,
    trials_per_class: int,
    ea_matrix: Optional[np.ndarray] = None,
) -> None:
    """Present MI targets one by one and compute ITR at the end."""
    screen, clock = init_pygame()
    commands = ["UP", "DOWN", "LEFT", "RIGHT"][:n_classes]
    schedule: List[str] = commands * trials_per_class
    import random as _random
    _random.shuffle(schedule)
    total_trials = len(schedule)

    buffer: List[List[float]] = []
    samples_since_inference = 0
    window_samples = int(cfg.SLIDING_WINDOW_SEC * cfg.SAMPLE_RATE_HZ)
    step_samples   = int(cfg.WINDOW_STEP_SEC   * cfg.SAMPLE_RATE_HZ)
    max_buffer     = window_samples * 3
    history: Deque[str] = deque(maxlen=cfg.MAJORITY_VOTE_WINDOW)

    correct_count = 0
    trial_times: List[float] = []
    trial_idx = 0
    current_command = cfg.IDLE_COMMAND
    current_confidence = 0.0

    DECISION_WINDOW_SEC = cfg.SLIDING_WINDOW_SEC + cfg.WINDOW_STEP_SEC
    trial_start = time.perf_counter()
    session_start = trial_start

    try:
        while trial_idx < total_trials:
            target = schedule[trial_idx]
            now = time.perf_counter()
            elapsed_session = now - session_start

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise SystemExit

            chunk, _ = inlet.pull_chunk(
                timeout=0.0,
                max_samples=int(cfg.SAMPLE_RATE_HZ * cfg.CHUNK_LENGTH_SEC),
            )
            if chunk:
                buffer.extend(s[: cfg.EXPECTED_CHANNEL_COUNT] for s in chunk)
                samples_since_inference += len(chunk)
                if len(buffer) > max_buffer:
                    buffer = buffer[-max_buffer:]

            if len(buffer) >= window_samples and samples_since_inference >= step_samples:
                window = np.asarray(buffer[-window_samples:])
                # Apply EA whitening if available
                if ea_matrix is not None:
                    window = (ea_matrix @ window.T).T
                trial_data = np.expand_dims(window.T, axis=0)
                probs = model.predict_proba(trial_data)[0]
                best_idx  = int(np.argmax(probs))
                best_label = model.classes_[best_idx]
                best_conf  = float(probs[best_idx])
                if best_conf >= cfg.CONFIDENCE_THRESHOLD:
                    history.append(LABEL_TO_COMMAND.get(int(best_label), cfg.IDLE_COMMAND))
                else:
                    history.append(cfg.IDLE_COMMAND)
                current_command    = majority_vote(history)
                current_confidence = best_conf
                samples_since_inference = 0

            draw_itr_cue(
                screen, target, trial_idx + 1, total_trials,
                current_command, current_confidence, correct_count, elapsed_session,
            )
            pygame.display.flip()
            clock.tick(cfg.REFRESH_RATE_HZ)

            # Advance trial after one decision window has elapsed
            trial_elapsed = now - trial_start
            if trial_elapsed >= DECISION_WINDOW_SEC:
                is_correct = current_command == target
                if is_correct:
                    correct_count += 1
                trial_times.append(trial_elapsed)
                trial_idx += 1
                history.clear()
                current_command    = cfg.IDLE_COMMAND
                current_confidence = 0.0
                trial_start = time.perf_counter()

        # ---- Results ----
        accuracy = correct_count / total_trials if total_trials else 0.0
        mean_trial_sec = float(np.mean(trial_times)) if trial_times else DECISION_WINDOW_SEC
        bits_per_trial = itr_bits_per_trial(n_classes, accuracy)
        itr_bpm = (bits_per_trial / mean_trial_sec) * 60.0

        print("\n===== ITR Results =====")
        print(f"Total trials   : {total_trials}")
        print(f"Correct        : {correct_count}")
        print(f"Accuracy       : {accuracy * 100:.1f}%")
        print(f"Mean trial time: {mean_trial_sec:.2f} s")
        print(f"Bits/trial     : {bits_per_trial:.3f} bits")
        print(f"ITR            : {itr_bpm:.2f} bits/min")

        # Show result screen
        screen.fill(cfg.BACKGROUND_COLOR)
        font = pygame.font.SysFont(cfg.FONT_NAME, 36)
        lines = [
            f"Accuracy: {accuracy * 100:.1f}%  ({correct_count}/{total_trials})",
            f"Mean trial: {mean_trial_sec:.2f} s",
            f"Bits/trial: {bits_per_trial:.3f}",
            f"ITR: {itr_bpm:.2f} bits/min",
            "",
            "Press ESC to exit",
        ]
        for i, line in enumerate(lines):
            surf = font.render(line, True, (220, 230, 255))
            screen.blit(surf, surf.get_rect(center=(cfg.SCREEN_SIZE[0] // 2, 160 + i * 50)))
        pygame.display.flip()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT,):
                    waiting = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    waiting = False
            clock.tick(15)
    finally:
        pygame.quit()


def connect_lsl(stream_name: str) -> StreamInlet:
    streams = resolve_byprop("name", stream_name, timeout=10)
    if not streams:
        raise RuntimeError(f"Could not find LSL stream named {stream_name}")
    inlet = StreamInlet(streams[0], max_buflen=60)
    return inlet


def init_pygame() -> Tuple[pygame.Surface, pygame.time.Clock]:
    pygame.init()
    pygame.display.set_caption("MI Real-Time Control")
    screen = pygame.display.set_mode(cfg.SCREEN_SIZE)
    clock = pygame.time.Clock()
    return screen, clock


def draw_ui(screen: pygame.Surface, command: str, confidence: float, cursor_pos: np.ndarray) -> None:
    screen.fill(cfg.BACKGROUND_COLOR)
    font = pygame.font.SysFont(cfg.FONT_NAME, 64)
    text = font.render(f"Command: {command}", True, (255, 255, 255))
    rect = text.get_rect(center=(cfg.SCREEN_SIZE[0] // 2, cfg.SCREEN_SIZE[1] // 2 - 60))
    screen.blit(text, rect)

    sub_font = pygame.font.SysFont(cfg.FONT_NAME, 30)
    bar_width = int(cfg.SCREEN_SIZE[0] * 0.65)
    bar_height = 12
    bar_x = (cfg.SCREEN_SIZE[0] - bar_width) // 2
    bar_y = cfg.SCREEN_SIZE[1] - 50
    label_text = sub_font.render("Confidence", True, (180, 220, 255))
    label_rect = label_text.get_rect(midbottom=(cfg.SCREEN_SIZE[0] // 2, bar_y - 6))
    screen.blit(label_text, label_rect)
    pygame.draw.rect(screen, (60, 60, 90), (bar_x, bar_y, bar_width, bar_height), border_radius=6)
    filled_width = int(bar_width * max(0.0, min(1.0, confidence)))
    if filled_width > 0:
        pygame.draw.rect(screen, (80, 220, 120), (bar_x, bar_y, filled_width, bar_height), border_radius=6)
    value_text = sub_font.render(f"{confidence:.2f}", True, (255, 255, 255))
    value_rect = value_text.get_rect(midtop=(cfg.SCREEN_SIZE[0] // 2, bar_y + bar_height + 6))
    screen.blit(value_text, value_rect)

    pygame.draw.circle(screen, (80, 100, 140), (cfg.SCREEN_SIZE[0] // 2, cfg.SCREEN_SIZE[1] // 2 + 120), 60, width=1)
    pygame.draw.circle(screen, cfg.CURSOR_COLOR, cursor_pos.astype(int), cfg.CURSOR_RADIUS)


def majority_vote(history: Deque[str]) -> str:
    counts = Counter(history)
    if not counts:
        return cfg.IDLE_COMMAND
    return counts.most_common(1)[0][0]


def run_loop(
    model: Union[FilterBankCSPClassifier, FilterBankTangentSpace], 
    inlet: StreamInlet,
    ea_matrix: Optional[np.ndarray] = None
) -> None:
    screen, clock = init_pygame()
    buffer: List[List[float]] = []
    samples_since_inference = 0
    window_samples = int(cfg.SLIDING_WINDOW_SEC * cfg.SAMPLE_RATE_HZ)
    step_samples = int(cfg.WINDOW_STEP_SEC * cfg.SAMPLE_RATE_HZ)
    max_buffer_samples = window_samples * 3
    prediction_history: Deque[str] = deque(maxlen=cfg.MAJORITY_VOTE_WINDOW)
    current_command = cfg.IDLE_COMMAND
    current_confidence = 0.0
    cursor_pos = np.array([cfg.SCREEN_SIZE[0] / 2, cfg.SCREEN_SIZE[1] / 2], dtype=float)
    cursor_vel = np.zeros(2, dtype=float)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
            chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=int(cfg.SAMPLE_RATE_HZ * cfg.CHUNK_LENGTH_SEC))
            if chunk:
                buffer.extend(sample[: cfg.EXPECTED_CHANNEL_COUNT] for sample in chunk)
                samples_since_inference += len(chunk)
                if len(buffer) > max_buffer_samples:
                    buffer = buffer[-max_buffer_samples:]

            if len(buffer) >= window_samples and samples_since_inference >= step_samples:
                window = np.asarray(buffer[-window_samples:])
                # Apply EA whitening if available
                if ea_matrix is not None:
                    window = (ea_matrix @ window.T).T
                trial = np.expand_dims(window.T, axis=0)
                probs = model.predict_proba(trial)[0]
                best_idx = int(np.argmax(probs))
                best_label = model.classes_[best_idx]
                best_conf = float(probs[best_idx])
                print(f"Prediction: {LABEL_TO_COMMAND.get(best_label, cfg.IDLE_COMMAND)} |  ({best_conf:.2f})")
                if best_conf >= cfg.CONFIDENCE_THRESHOLD:
                    prediction_history.append(LABEL_TO_COMMAND.get(int(best_label), cfg.IDLE_COMMAND))
                else:
                    prediction_history.append(cfg.IDLE_COMMAND)
                current_command = majority_vote(prediction_history)
                current_confidence = best_conf
                samples_since_inference = 0

            direction = np.array(COMMAND_TO_VECTOR.get(current_command, (0.0, 0.0)), dtype=float)
            target_velocity = direction * cfg.CURSOR_SPEED_PX
            cursor_vel = (
                cfg.CURSOR_DAMPING * cursor_vel
                + (1.0 - cfg.CURSOR_DAMPING) * target_velocity
            )
            cursor_pos += cursor_vel
            cursor_pos[0] = np.clip(cursor_pos[0], cfg.CURSOR_RADIUS, cfg.SCREEN_SIZE[0] - cfg.CURSOR_RADIUS)
            cursor_pos[1] = np.clip(cursor_pos[1], cfg.CURSOR_RADIUS, cfg.SCREEN_SIZE[1] - cfg.CURSOR_RADIUS)

            draw_ui(screen, current_command, current_confidence, cursor_pos)
            pygame.display.flip()
            clock.tick(cfg.REFRESH_RATE_HZ)
    finally:
        pygame.quit()


def main() -> None:
    args = parse_args()
    
    # Detect algorithm type from model directory name
    model_dir = Path(args.model_dir)
    if "filterbank_tangent" in model_dir.name:
        print("Loading FilterBankTangentSpace model...")
        model = FilterBankTangentSpace.load_model(model_dir / "model.pkl")
    else:
        print("Loading FilterBankCSP model...")
        model = FilterBankCSPClassifier.load(model_dir)
    
    # Load EA whitening matrix if available
    ea_matrix = load_ea_matrix(model_dir)
    if ea_matrix is not None:
        print("EA alignment enabled")
    else:
        print("No EA matrix found, using raw data")
    
    inlet = connect_lsl(args.stream_name)
    if args.itr_mode:
        n_classes = len(LABEL_TO_COMMAND) - 1  # exclude IDLE
        run_itr_mode(model, inlet, n_classes, args.itr_trials, ea_matrix)
    else:
        run_loop(model, inlet, ea_matrix)


if __name__ == "__main__":
    main()
