import torch
from flask import Blueprint, render_template, request, Response
from acai_omr.inference.vitomr_inference import beam_search
from acai_omr.config import DEFAULT_VITOMR_PATH, LMX_BOS_TOKEN, LMX_EOS_TOKEN
from acai_omr.train.omr_train import set_up_omr_train
import logging
from PIL import Image
import uuid

main = Blueprint("main", __name__)
logger = logging.getLogger(__name__)

DEBUG_IMAGE_PATH = "inference_test.png"

inference_results = {}

@main.route("/")
def index():
    return render_template("index.html")

# wrapper that 1) post-processes active beams yielded by beam_search for the UI and 2) saves the final inference result to
# a buffer and yields an event signaling the inference has ended so the rest of the inference back-end can be ran
def stream_beam_search_wrapper(result_buffer: dict[str, torch.Tensor], job_id: str, vitomr, image, bos_token_idx, eos_token_idx, device, beam_width, max_inference_len):
    inference_result = None
    for beams in beam_search(vitomr, image, bos_token_idx, eos_token_idx, device, beam_width, max_inference_len):
        inference_result = beams
        decoded_tokens = [[vitomr.decoder.idxs_to_tokens[idx.item()] for idx in row] for row in beams]
        yield str(decoded_tokens)
    
    result_buffer[job_id] = inference_result
    yield "event: inference_stream_finished"

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

    return Response(stream_beam_search_wrapper(inference_results, str(uuid.uuid4()), vitomr, image, bos_token_idx, eos_token_idx, device, beam_width, max_inference_len), mimetype="text/event-stream")