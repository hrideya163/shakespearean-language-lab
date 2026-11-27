import argparse
import os
import subprocess
import sys

from lab import config

GENERATORS = [
    ["--cell", "lstm"],
    ["--cell", "gru"],
    ["--cell", "rnn"],
    ["--cell", "gru", "--hidden", "300", "--name", "gru_matched"],
    ["--cell", "rnn", "--hidden", "542", "--name", "rnn_matched"],
    ["--cell", "lstm", "--steps", "9000", "--name", "lstm_long"],
    ["--cell", "rnn", "--hidden", "542", "--steps", "9000", "--name", "rnn_matched_long"],
    ["--cell", "gru", "--hidden", "300", "--steps", "9000", "--name", "gru_matched_long"],
    ["--cell", "lstm", "--hidden", "512", "--steps", "9000", "--name", "lstm_large"],
]
TRANSLATORS = [[], ["--no_copy"], ["--extended"], ["--dropout", "0"]]

TRAINING = (
    [["download_data.py"], ["-m", "lab.style"]]
    + [["-m", "lab.generator.train", *args] for args in GENERATORS]
    + [["-m", "lab.translator.train", *args] for args in TRANSLATORS]
)
EXPERIMENTS = [
    ["-m", "experiments.rq1_cells"],
    ["-m", "experiments.rq2_inside_lstm"],
    ["-m", "experiments.rq3_sampling"],
    ["-m", "experiments.rq4_copy_gate"],
    ["-m", "experiments.rq4_decoding", "--tune"],
    ["-m", "experiments.rq4_decoding"],
    ["-m", "experiments.rq5_drift"],
    ["-m", "experiments.evaluate"],
]


def run(step):
    print("\n" + "=" * 70 + "\n>>> python " + " ".join(step) + "\n" + "=" * 70, flush=True)
    subprocess.run([sys.executable, *step], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train", action="store_true", help="only re-run the experiments")
    args = parser.parse_args()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    for step in ([] if args.skip_train else TRAINING) + EXPERIMENTS:
        run(step)
    print("\nAll done. Results are in", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
