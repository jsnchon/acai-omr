import torch
from flask import Blueprint, render_template, request, Response
from acai_omr.inference.vitomr_inference import streamed_inference
from acai_omr.config import INFERENCE_VITOMR_PATH
from acai_omr.train.omr_teacher_force_train import set_up_omr_teacher_force_train
from acai_omr.__init__ import InferenceEvent
from acai_omr.utils.utils import stringify_lmx_seq
import logging  
from PIL import Image
import json
import os

main = Blueprint("main", __name__)
logger = logging.getLogger(__name__)

DEBUG_IMAGE_PATH = "inference_test.png"
MAX_BATCH_SIZE = 1
CACHE_DTYPE = torch.bfloat16

# without this, in debug mode flask runs import statements twice in two processes which means the model is duplicated and
# takes up too much memory
if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    vitomr, base_img_transform, _, device = set_up_omr_teacher_force_train()
    logger.info(f"Enabling caching for decoder with a max batch size of {MAX_BATCH_SIZE} and cache datatype of {CACHE_DTYPE}")
    vitomr.decoder = vitomr.decoder.to_cached_version(MAX_BATCH_SIZE, CACHE_DTYPE)
    
    logger.info(f"Loading state dict from {INFERENCE_VITOMR_PATH}")
    if device == "cpu":
        vitomr_state_dict = torch.load(INFERENCE_VITOMR_PATH, map_location=torch.device("cpu"))
    else:
        vitomr_state_dict = torch.load(INFERENCE_VITOMR_PATH)

    vitomr.load_state_dict(vitomr_state_dict)

    if device == "cpu":
        flush_interval = 25
    else:
        flush_interval = 100 # buffer GPUs more

@main.route("/")
def index():
    return render_template("index.html", weights_path=INFERENCE_VITOMR_PATH)

# SSE wrapper that post-processes and then yields events yielded by inference.
# Again, we assume there's only one sequence being passed here at a time
def stream_inference_wrapper(vitomr, img, device, max_inference_len, flush_interval):
    for event in streamed_inference(vitomr, img, device, max_inference_len, flush_interval):
        if event["type"] == InferenceEvent.STEP.value:
            tokens = event["payload"]["tokens"]
            tokens = tokens[tokens != vitomr.decoder.pad_idx] # last buffer may have leftover pad tokens
            tokens = stringify_lmx_seq(tokens.squeeze(0), vitomr.decoder.idxs_to_tokens)
            event["payload"] = {"tokens": tokens}

        if event["type"] == InferenceEvent.INFERENCE_FINISH.value:
            sequence = event["payload"]["sequence"]
            seq_mask = event["payload"]["mask"]
            sequence = stringify_lmx_seq(sequence.squeeze(0), vitomr.decoder.idxs_to_tokens)
            logger.info("Inference finished")
            seq_log_probs = event["payload"]["log_probs"]
            average_log_prob = seq_log_probs.sum() / seq_mask.sum()
            average_confidence = torch.exp(average_log_prob).item()
            event["payload"] = {"sequence": sequence, "average_confidence": average_confidence}
        
        yield f"data: {json.dumps(event)}\n\n"

@main.route("/inference/stream")
def stream_inference():
    max_inference_len = int(request.args.get("max_inference_len", 1536))

    img = Image.open(DEBUG_IMAGE_PATH).convert("L")
    img = base_img_transform(img).to(device)

    # make sure to transform any images using patch transform
    logger.info(f"Starting inference with max length {max_inference_len} and streaming from endpoint with a flush interval of {flush_interval}")

    return Response(stream_inference_wrapper(vitomr, img, device, max_inference_len, flush_interval), mimetype="text/event-stream")
