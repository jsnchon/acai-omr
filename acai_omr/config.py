# this file stores a bunch of useful constants used across many files in one place (also helps avoid circular imports)

GRAND_STAFF_ROOT_DIR = "data/grandstaff-lmx.2024-02-12/grandstaff-lmx"
PRIMUS_PREPARED_ROOT_DIR = "data/primusPrepared"
DOREMI_PREPARED_ROOT_DIR = "data/doReMiPrepared"
OLIMPIC_SYNTHETIC_ROOT_DIR = "data/olimpic-1.0-synthetic.2024-02-12/olimpic-1.0-synthetic"
OLIMPIC_SCANNED_ROOT_DIR = "data/olimpic-1.0-scanned.2024-02-12/olimpic-1.0-scanned"

# weights for a tiny MAE for testing purposes
DEBUG_PRETRAINED_MAE_PATH = "debug_pretrained_mae.pth"

LMX_BOS_TOKEN = "<bos>"
LMX_EOS_TOKEN = "<eos>"
LMX_PAD_TOKEN = "<pad>" # token used for padding lmx sequences

DEFAULT_VITOMR_PATH = "omr_train/vitomr.pth"