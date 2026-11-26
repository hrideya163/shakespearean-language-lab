"""Reproduce the whole study with one command (takes a few hours on a laptop CPU).

    python run_all.py               # everything
    python run_all.py --skip_train  # only re-run the experiments on existing checkpoints

Every step is just one of the normal scripts, run in order, so you can also run them by hand.
"""
import argparse
import os
import subprocess
import sys

import config

TRAINING = [
    ["download_data.py"],
    ["style_classifier.py"],
    ["train_generator.py", "--cell", "lstm"],
    ["train_generator.py", "--cell", "gru"],
    ["train_generator.py", "--cell", "rnn"],
    ["train_generator.py", "--cell", "gru", "--hidden", "300", "--name", "gru_matched"],
    ["train_generator.py", "--cell", "rnn", "--hidden", "542", "--name", "rnn_matched"],
    # 3x longer training: which models keep improving? (RQ1, slowest part: about 2.5 hours)
    ["train_generator.py", "--cell", "lstm", "--steps", "9000", "--name", "lstm_long"],
    ["train_generator.py", "--cell", "rnn", "--hidden", "542", "--steps", "9000", "--name", "rnn_matched_long"],
    ["train_generator.py", "--cell", "gru", "--hidden", "300", "--steps", "9000", "--name", "gru_matched_long"],
    ["train_translator.py"],
    ["train_translator.py", "--no_copy"],
    ["train_translator.py", "--extended"],      # pointer-generator extended vocabulary (RQ4 ablation)
    ["train_translator.py", "--dropout", "0"],   # no regularisation, to show overfitting (RQ4 part E)
]

EXPERIMENTS = [
    ["exp_compare_cells.py"],   # RQ1
    ["exp_inside_lstm.py"],     # RQ2
    ["exp_temperature.py"],     # RQ3
    ["exp_copy_gate.py"],       # RQ4
    ["exp_drift.py"],           # RQ5
    ["evaluate.py"],            # baseline vs LSTM (add --llm files for the LLM comparison)
]


def run(step):
    print("\n" + "=" * 70 + "\n>>> python " + " ".join(step) + "\n" + "=" * 70, flush=True)
    subprocess.run([sys.executable] + step, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    for step in ([] if args.skip_train else TRAINING) + EXPERIMENTS:
        run(step)
    print("\nAll done. Results are in", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
