"""Motor imagery test with image cues."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque, Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import pygame
from pylsl import StreamInlet, resolve_byprop

from algorithms_collection import FilterBankTangentSpace
from config import mi_config as cfg


LABEL_TO_COMMAND = {
    0: cfg.IDLE_COMMAND,
    1: "UP",      # Tongue
    2: "DOWN",    # Feet
    3: "LEFT",    # Left Hand
    4: "RIGHT",   # Right Hand
}

COMMAND_TO_LABEL = {
    "UP": 1,
    "DOWN": 2,
    "LEFT": 3,
    "RIGHT": 4,
    cfg.IDLE_COMMAND: 0
}

# Trial types and corresponding images
CUE_TYPES = ["LEFT", "RIGHT", "DOWN", "UP"]
CUE_LABELS = {"LEFT": "Left Hand", "RIGHT": "Right Hand", "DOWN": "Feet", "UP": "Tongue"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motor imagery test with image cues")
    parser.add_argument("model_dir", help="Directory containing trained model artifacts")
    parser.add_argument("--stream-name", default=cfg.LSL_STREAM_NAME, help="LSL stream name")
    parser.add_argument("--trials", type=int, default=15, help="Number of trials per class")
    parser.add_argument("--subject-id", type=int, default=1, help="Subject ID")
    parser.add_argument("--save-dir", default="results", help="Directory to save test results")
    return parser.parse_args()

def itr_bits_per_trial(n_classes: int, accuracy: float) -> float:
    """Calculate ITR per trial in bits."""
    if n_classes < 2:
        return 0.0
    p = max(1e-9, min(1.0 - 1e-9, accuracy))
    q = (1.0 - p) / (n_classes - 1)
    entropy = p * math.log2(p) + (n_classes - 1) * q * math.log2(max(q, 1e-9))
    return math.log2(n_classes) + entropy

def connect_lsl(stream_name: str) -> StreamInlet:
    """Connect to LSL stream."""
    print(f"Resolving LSL stream '{stream_name}'...")
    streams = resolve_byprop("name", stream_name, timeout=10)
    if not streams:
        raise RuntimeError(f"Could not find LSL stream named {stream_name}")
    inlet = StreamInlet(streams[0], max_buflen=60)
    return inlet

def init_pygame() -> Tuple[pygame.Surface, pygame.time.Clock]:
    """Initialize pygame."""
    pygame.init()
    pygame.display.set_caption("Motor Imagery Test")
    screen = pygame.display.set_mode(cfg.SCREEN_SIZE)
    clock = pygame.time.Clock()
    return screen, clock

def load_cue_images() -> dict:
    """Load cue images."""
    images = {}
    base_dir = Path(__file__).resolve().parent
    for cue_type in CUE_TYPES:
        if cue_type == "LEFT":
            path = base_dir / "res" / "left.png"
        elif cue_type == "RIGHT":
            path = base_dir / "res" / "right.png"
        elif cue_type == "DOWN":
            path = base_dir / "res" / "down.png"
        elif cue_type == "UP":
            path = base_dir / "res" / "up.png"
        
        if path.exists():
            image = pygame.image.load(str(path)).convert_alpha()
            # Scale image to fit screen
            max_size = (int(cfg.SCREEN_SIZE[0] * 0.4), int(cfg.SCREEN_SIZE[1] * 0.4))
            image = pygame.transform.scale(image, max_size)
            images[cue_type] = image
        else:
            print(f"Warning: Image not found at {path}")
    return images

def draw_ui(screen: pygame.Surface, cue_type: str, cue_label: str, command: str, 
            confidence: float, trial_info: dict, itr_info: dict, 
            state: str, countdown: int, cue_images: dict) -> None:
    """Draw UI elements."""
    screen.fill(cfg.BACKGROUND_COLOR)
    width, height = cfg.SCREEN_SIZE
    center_x, center_y = width // 2, height // 2
    
    # Draw state-specific content
    if state == "ready":
        font_big = pygame.font.SysFont(cfg.FONT_NAME, 80)
        ready_text = font_big.render("Get Ready", True, (255, 165, 0))
        ready_rect = ready_text.get_rect(center=(center_x, center_y - 150))
        screen.blit(ready_text, ready_rect)
        
        font_mid = pygame.font.SysFont(cfg.FONT_NAME, 48)
        next_text = font_mid.render(f"Next: {CUE_LABELS.get(cue_type, cue_type)}", True, (255, 200, 120))
        next_rect = next_text.get_rect(center=(center_x, center_y))
        screen.blit(next_text, next_rect)
        
        # Show next cue image if available
        if cue_type in cue_images:
            image = cue_images[cue_type]
            img_rect = image.get_rect(center=(center_x, center_y + 100))
            screen.blit(image, img_rect)
    elif state == "trial":
        # Draw cue label
        font_big = pygame.font.SysFont(cfg.FONT_NAME, 60)
        cue_text = font_big.render(f"Imagine: {CUE_LABELS.get(cue_type, cue_type)}", True, (80, 220, 120))
        cue_rect = cue_text.get_rect(center=(center_x, center_y - 250))
        screen.blit(cue_text, cue_rect)
        
        # Draw cue image if available
        if cue_type in cue_images:
            image = cue_images[cue_type]
            img_rect = image.get_rect(center=(center_x, center_y))
            screen.blit(image, img_rect)
        
        # Draw current prediction
        pred_color = (80, 220, 120) if command == cue_type else (220, 90, 90)
        font_mid = pygame.font.SysFont(cfg.FONT_NAME, 36)
        pred_text = font_mid.render(f"Prediction: {CUE_LABELS.get(command, command)}", True, pred_color)
        pred_rect = pred_text.get_rect(center=(center_x, center_y + 200))
        screen.blit(pred_text, pred_rect)
        
        # Draw confidence bar
        font_sm = pygame.font.SysFont(cfg.FONT_NAME, 26)
        bar_width = int(width * 0.5)
        bar_x = (width - bar_width) // 2
        bar_y = center_y + 250
        pygame.draw.rect(screen, (50, 60, 90), (bar_x, bar_y, bar_width, 12), border_radius=6)
        filled = int(bar_width * max(0.0, min(1.0, confidence)))
        if filled:
            pygame.draw.rect(screen, (80, 200, 140), (bar_x, bar_y, filled, 12), border_radius=6)
        conf_text = font_sm.render(f"Confidence: {confidence:.2f}", True, (200, 210, 240))
        conf_rect = conf_text.get_rect(center=(center_x, bar_y - 20))
        screen.blit(conf_text, conf_rect)
    elif state == "rest":
        font_big = pygame.font.SysFont(cfg.FONT_NAME, 80)
        rest_text = font_big.render("Rest", True, (100, 180, 220))
        rest_rect = rest_text.get_rect(center=(center_x, center_y))
        screen.blit(rest_text, rest_rect)
    
    # Draw countdown if applicable
    if countdown > 0:
        font_count = pygame.font.SysFont(cfg.FONT_NAME, 100)
        count_text = font_count.render(str(countdown), True, (255, 255, 255))
        count_rect = count_text.get_rect(center=(center_x, height - 100))
        screen.blit(count_text, count_rect)
    
    # Draw trial info
    font_sm = pygame.font.SysFont(cfg.FONT_NAME, 26)
    trial_text = font_sm.render(
        f"Trial: {trial_info['current']}/{trial_info['total']}  "
        f"Correct: {trial_info['correct']}  "
        f"Accuracy: {trial_info['accuracy']:.2f}%", 
        True, (200, 210, 240)
    )
    trial_rect = trial_text.get_rect(center=(center_x, height - 50))
    screen.blit(trial_text, trial_rect)
    
    # Draw ITR info
    if itr_info['valid']:
        itr_text = font_sm.render(
            f"ITR: {itr_info['value']:.2f} bits/min  "
            f"Avg Time: {itr_info['avg_time']:.2f}s", 
            True, (200, 210, 240)
        )
        itr_rect = itr_text.get_rect(center=(center_x, height - 150))
        screen.blit(itr_text, itr_rect)

def majority_vote(history: Deque[str]) -> str:
    """Get majority vote from history."""
    counts = Counter(history)
    if not counts:
        return cfg.IDLE_COMMAND
    return counts.most_common(1)[0][0]

def run_mi_test(model: FilterBankTangentSpace, inlet: StreamInlet, n_trials_per_class: int, 
                subject_id: int, save_dir: Path) -> None:
    """Run motor imagery test with image cues."""
    screen, clock = init_pygame()
    cue_images = load_cue_images()
    
    # Generate trial schedule (15 trials per class by default)
    schedule = []
    for cue_type in CUE_TYPES:
        schedule.extend([cue_type] * n_trials_per_class)
    import random
    random.shuffle(schedule)
    total_trials = len(schedule)
    
    # Initialize variables
    buffer: List[List[float]] = []
    samples_since_inference = 0
    window_samples = int(cfg.SLIDING_WINDOW_SEC * cfg.SAMPLE_RATE_HZ)
    step_samples = int(cfg.WINDOW_STEP_SEC * cfg.SAMPLE_RATE_HZ)
    max_buffer = window_samples * 3
    history: Deque[str] = deque(maxlen=cfg.MAJORITY_VOTE_WINDOW)
    
    correct_count = 0
    trial_times: List[float] = []
    trial_idx = 0
    current_command = cfg.IDLE_COMMAND
    current_confidence = 0.0
    
    # State management
    STATE_READY = "ready"
    STATE_TRIAL = "trial"
    STATE_REST = "rest"
    current_state = STATE_READY
    state_start_time = time.perf_counter()
    session_start = state_start_time
    
    # Time settings (in seconds)
    READY_TIME = 3  # Preparation time
    TRIAL_TIME = 4  # Trial time
    REST_TIME = 2  # Rest time
    
    # Test results
    test_results = {
        "subject_id": subject_id,
        "trials": [],
        "total_trials": total_trials,
        "correct_trials": 0,
        "accuracy": 0.0,
        "itr": 0.0,
        "average_trial_time": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model_dir),
        "trials_per_class": n_trials_per_class,
    }
    
    try:
        while trial_idx < total_trials:
            cue_type = schedule[trial_idx]
            now = time.perf_counter()
            state_elapsed = now - state_start_time
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise SystemExit
            
            # State transitions
            if current_state == STATE_READY:
                if state_elapsed >= READY_TIME:
                    current_state = STATE_TRIAL
                    state_start_time = now
                    # Clear buffer before trial
                    buffer.clear()
                    samples_since_inference = 0
                    history.clear()
                    current_command = cfg.IDLE_COMMAND
                    current_confidence = 0.0
            elif current_state == STATE_TRIAL:
                if state_elapsed >= TRIAL_TIME:
                    # Calculate trial result
                    is_correct = current_command == cue_type
                    if is_correct:
                        correct_count += 1
                    trial_times.append(TRIAL_TIME)
                    
                    # Record trial result
                    test_results["trials"].append({
                        "cue": cue_type,
                        "cue_label": CUE_LABELS.get(cue_type, cue_type),
                        "prediction": current_command,
                        "prediction_label": CUE_LABELS.get(current_command, current_command),
                        "correct": is_correct,
                        "confidence": current_confidence,
                        "duration": TRIAL_TIME
                    })
                    
                    trial_idx += 1
                    current_state = STATE_REST
                    state_start_time = now
            elif current_state == STATE_REST:
                if state_elapsed >= REST_TIME:
                    if trial_idx < total_trials:
                        current_state = STATE_READY
                    state_start_time = now
            
            # Get data from LSL (only during trial state)
            if current_state == STATE_TRIAL:
                chunk, _ = inlet.pull_chunk(
                    timeout=0.0,
                    max_samples=int(cfg.SAMPLE_RATE_HZ * cfg.CHUNK_LENGTH_SEC),
                )
                if chunk:
                    buffer.extend(s[: cfg.EXPECTED_CHANNEL_COUNT] for s in chunk)
                    samples_since_inference += len(chunk)
                    if len(buffer) > max_buffer:
                        buffer = buffer[-max_buffer:]
                
                # Make prediction
                if len(buffer) >= window_samples and samples_since_inference >= step_samples:
                    window = np.asarray(buffer[-window_samples:])
                    trial_data = np.expand_dims(window.T, axis=0)
                    probs = model.predict_proba(trial_data)[0]
                    best_idx = int(np.argmax(probs))
                    best_label = model.classes_[best_idx]
                    best_conf = float(probs[best_idx])
                    if best_conf >= cfg.CONFIDENCE_THRESHOLD:
                        history.append(LABEL_TO_COMMAND.get(int(best_label), cfg.IDLE_COMMAND))
                    else:
                        history.append(cfg.IDLE_COMMAND)
                    current_command = majority_vote(history)
                    current_confidence = best_conf
                    samples_since_inference = 0
            
            # Calculate countdown
            countdown = 0
            if current_state == STATE_READY:
                countdown = max(0, READY_TIME - int(state_elapsed))
            elif current_state == STATE_TRIAL:
                countdown = max(0, TRIAL_TIME - int(state_elapsed))
            elif current_state == STATE_REST:
                countdown = max(0, REST_TIME - int(state_elapsed))
            
            # Draw UI
            trial_info = {
                "current": trial_idx + 1,
                "total": total_trials,
                "correct": correct_count,
                "accuracy": (correct_count / (trial_idx + 1)) * 100 if trial_idx > 0 else 0.0
            }
            
            itr_info = {
                "valid": len(trial_times) > 0,
                "value": 0.0,
                "avg_time": 0.0
            }
            if itr_info['valid']:
                accuracy = correct_count / (trial_idx + 1)
                avg_time = np.mean(trial_times)
                itr_info['value'] = (itr_bits_per_trial(len(CUE_TYPES), accuracy) / avg_time) * 60.0
                itr_info['avg_time'] = avg_time
            
            draw_ui(screen, cue_type, CUE_LABELS.get(cue_type, cue_type), 
                    current_command, current_confidence, trial_info, 
                    itr_info, current_state, countdown, cue_images)
            pygame.display.flip()
            clock.tick(cfg.REFRESH_RATE_HZ)
        
        # Calculate final results
        if total_trials > 0:
            accuracy = correct_count / total_trials
            avg_time = np.mean(trial_times)
            itr = (itr_bits_per_trial(len(CUE_TYPES), accuracy) / avg_time) * 60.0
            
            test_results["correct_trials"] = correct_count
            test_results["accuracy"] = accuracy
            test_results["itr"] = itr
            test_results["average_trial_time"] = avg_time
        
        # Show result screen
        screen.fill(cfg.BACKGROUND_COLOR)
        font = pygame.font.SysFont(cfg.FONT_NAME, 36)
        lines = [
            "=== Test Results ===",
            f"Total Trials: {total_trials}",
            f"Correct Trials: {correct_count}",
            f"Accuracy: {accuracy * 100:.1f}%",
            f"Average Trial Time: {avg_time:.2f} seconds",
            f"ITR: {itr:.2f} bits/min",
            "",
            "Press ESC to exit"
        ]
        for i, line in enumerate(lines):
            surf = font.render(line, True, (220, 230, 255))
            screen.blit(surf, surf.get_rect(center=(cfg.SCREEN_SIZE[0] // 2, 160 + i * 50)))
        pygame.display.flip()
        
        # Save results
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = save_dir / f"mi_test_results_subject{subject_id}_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        print(f"Test results saved to {result_file}")
        
        # Wait for user to exit
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

def main() -> None:
    global args
    args = parse_args()
    
    # Load model
    model_path = Path(args.model_dir) / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = FilterBankTangentSpace.load_model(model_path)
    print("Model loaded successfully!")
    
    # Connect to LSL stream
    inlet = connect_lsl(args.stream_name)
    
    # Run MI test
    save_dir = Path(args.save_dir)
    run_mi_test(model, inlet, args.trials, args.subject_id, save_dir)


if __name__ == "__main__":
    main()
