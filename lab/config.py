CORPUS_DIR = "data/corpus"
SAMPLE_DIR = "data/sample"
CHECKPOINT_DIR = "checkpoints"
TRANSLATOR_PATH = "checkpoints/translator.pt"
TRANSLATOR_NOCOPY_PATH = "checkpoints/translator_nocopy.pt"
TRANSLATOR_EXT_PATH = "checkpoints/translator_ext.pt"
TRANSLATOR_NODROPOUT_PATH = "checkpoints/translator_nodropout.pt"
STYLE_CLASSIFIER_PATH = "checkpoints/style_classifier.joblib"
RESULTS_DIR = "results"
SEED = 42


def generator_path(name):
    return f"{CHECKPOINT_DIR}/generator_{name}.pt"


GEN_EMB_DIM = 64
GEN_HID_DIM = 256
GEN_SEQ_LEN = 100
GEN_BATCH_SIZE = 64
GEN_STEPS = 3000
GEN_EVAL_EVERY = 250
GEN_LEARNING_RATE = 2e-3
GEN_DROPOUT = 0.2

MAX_LEN = 30
MIN_FREQ = 2
EMB_DIM = 128
HID_DIM = 256
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
PATIENCE = 2

CLIP = 1.0
LOW_CONFIDENCE = 0.40
