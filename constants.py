import string
from torchnlp.encoders import LabelEncoder

ALPHABET = [' '] + list(string.ascii_lowercase)
ENCODER = LabelEncoder(ALPHABET, reserved_labels=[''], unknown_index=0)
N_EPOCHS = 200
CHECKPOINT_DIR = "./checkpoints"
NUM_WORKERS = 0
