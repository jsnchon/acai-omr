import torch
from flask import Blueprint, render_template, request, Response
from acai_omr.inference.vitomr_inference import beam_search
from acai_omr.config import DEFAULT_VITOMR_PATH, LMX_BOS_TOKEN, LMX_EOS_TOKEN, InferenceEvent
from acai_omr.train.omr_train import set_up_omr_train
import logging
from PIL import Image
import json

main = Blueprint("main", __name__)
logger = logging.getLogger(__name__)

DEBUG_IMAGE_PATH = "inference_test.png"

@main.route("/")
def index():
    return render_template("index.html")

# SSE wrapper that 1) post-processes events yielded by beam_search for the UI and 2) yields a final event signalling the 
# inference has ended and carrying the final inference result so the rest of the inference back-end can be ran
def stream_beam_search_wrapper(vitomr, image, bos_token_idx, eos_token_idx, device, beam_width, max_inference_len):
    inference_event = None
    for event in beam_search(vitomr, image, bos_token_idx, eos_token_idx, device, beam_width, max_inference_len):
        inference_event = event

        if event["type"] == InferenceEvent.STEP.value:
            # for intermediate inference steps, process the payload to only include the decoded/stringified beam tensors
            beams_dict = {}
            for i, beam in enumerate(event["payload"]["beams"]):
                decoded_beam = [vitomr.decoder.idxs_to_tokens[idx.item()] for idx in beam]
                decoded_beam = " ".join(decoded_beam)
                beams_dict[i] = decoded_beam
            event["payload"] = beams_dict
        elif event["type"] == InferenceEvent.FINAL_STEP.value:
            # process payload for final yielded inference event before streaming
            decoded_lmx_seq = [vitomr.decoder.idxs_to_tokens[idx.item()] for idx in inference_event["payload"]["beams"].squeeze(0)]
            decoded_lmx_seq = " ".join(decoded_lmx_seq)
            final_payload = {"lmx_seq": decoded_lmx_seq, "score": inference_event["payload"]["log_probs"].item()}
            event["payload"] = final_payload

        yield f"data: {json.dumps(event)}" # treat this endpoint as streaming whole event objects

@main.route("/stream/inference/")
def streamed_inference():
    vitomr, base_img_transform, _, device = set_up_omr_train()

    weights_path = request.args.get("weights_path", DEFAULT_VITOMR_PATH)
    beam_width = int(request.args.get("beam_width", 3))
    max_inference_len = int(request.args.get("max_inference_len", 1536))
    
    logger.info(f"Loading state dict from {weights_path}")
    if device == "cpu":
        vitomr_state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    else:
        vitomr_state_dict = torch.load(weights_path)

    vitomr.load_state_dict(vitomr_state_dict)
    
    bos_token_idx = vitomr.decoder.tokens_to_idxs[LMX_BOS_TOKEN]
    eos_token_idx = vitomr.decoder.tokens_to_idxs[LMX_EOS_TOKEN]

    image = Image.open(DEBUG_IMAGE_PATH).convert("L")
    image = base_img_transform(image)

    # make sure to transform any images using patch transform
    logger.info("Starting inference and streaming from this endpoint")
    logger.info(f"Running beam search with beam width {beam_width} and max inference length {max_inference_len}")

    return Response(stream_beam_search_wrapper(vitomr, image, bos_token_idx, eos_token_idx, device, beam_width, max_inference_len), mimetype="text/event-stream")