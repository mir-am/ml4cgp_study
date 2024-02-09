import os

__version__ = "0.1.0"

PROJECT_DATA_PATH = os.environ.get('ML4CG_DATA_PATH')
TESTING_MODE = os.environ.get('TESTING_MODE') # Only for testing/debugging code.
TOKENIZER_BATCH_SIZE = os.environ.get("TOKENIZER_BATCH_SIZE")
if TOKENIZER_BATCH_SIZE is None:
    TOKENIZER_BATCH_SIZE = 100