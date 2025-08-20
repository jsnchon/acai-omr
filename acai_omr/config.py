from enum import Enum
import json
import pathlib

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

# inference streaming events. Python back-end can import this Enum while javascript in the front-end can
# use the saved .json for more type safety
class InferenceEvent(Enum):
    ENCODING_START = "encoding_start"
    ENCODING_FINISH = "encoding_finish"
    STEP = "step" # each inference step yields both beam(s) and log prob(s)
    INFERENCE_FINISH = "inference_finish" # include the result from the last step which we treat differently (eg only has one sequence, want to stream its score to the ui)

INFERENCE_EVENT_JSON_PATH = pathlib.Path("acai_omr/ui/static/inference_events.json")

inference_events = {e.name: e.value for e in InferenceEvent}
INFERENCE_EVENT_JSON_PATH.write_text(json.dumps(inference_events, indent=2))