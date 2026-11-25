"""All project settings in one place. Change values here, not inside the other files."""

# ---------- paths ----------
CORPUS_DIR = "data/corpus"        # full dataset (created by download_data.py)
SAMPLE_DIR = "data/sample"        # tiny hand-made dataset for a quick smoke test
CHECKPOINT_DIR = "checkpoints"
TRANSLATOR_PATH = "checkpoints/translator.pt"
TRANSLATOR_NOCOPY_PATH = "checkpoints/translator_nocopy.pt"
STYLE_CLASSIFIER_PATH = "checkpoints/style_classifier.joblib"
RESULTS_DIR = "results"
SEED = 42


def generator_path(cell):
    """checkpoints/generator_lstm.pt, generator_gru.pt, generator_rnn.pt"""
    return f"{CHECKPOINT_DIR}/generator_{cell}.pt"


# ---------- text generator (character-level language model) ----------
GEN_EMB_DIM = 64
GEN_HID_DIM = 256
GEN_SEQ_LEN = 100        # characters per training example
GEN_BATCH_SIZE = 64
GEN_STEPS = 3000         # training steps (batches)
GEN_EVAL_EVERY = 250     # check validation loss every N steps
GEN_LEARNING_RATE = 2e-3
GEN_DROPOUT = 0.2

# ---------- translator (word-level encoder-decoder) ----------
MAX_LEN = 30          # skip sentences longer than this many tokens
MIN_FREQ = 2          # a word must appear this many times to enter the vocabulary
EMB_DIM = 128         # size of each word vector
HID_DIM = 256         # size of the LSTM hidden state
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
PATIENCE = 2          # early stopping: stop after this many epochs without improvement

# ---------- shared ----------
CLIP = 1.0            # gradient clipping (stops exploding gradients)
LOW_CONFIDENCE = 0.40  # translated words below this probability are flagged as "unsure"
