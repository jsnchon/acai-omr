import torch
from flask import Blueprint, render_template, request, Response
from acai_omr.inference.vitomr_inference import beam_search
from acai_omr.config import DEFAULT_VITOMR_PATH, LMX_BOS_TOKEN, LMX_EOS_TOKEN
from acai_omr.utils.utils import set_up_logger
from acai_omr.train.omr_train import set_up_omr_train
import logging
from PIL import Image

main = Blueprint("main", __name__)
logger = set_up_logger(__name__, logging.DEBUG)

DEBUG_IMAGE_PATH = "inference_test.png"

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/stream/inference/")
def streamed_inference():
    vitomr, base_img_transform, _, device = set_up_omr_train()

    weights_path = request.args.get("weights_path", DEFAULT_VITOMR_PATH)
    beam_width = request.args.get("beam_width", 3)
    max_inference_len = request.args.get("beam_width", 1536)
    
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
    return Response(beam_search(vitomr, image, bos_token_idx, eos_token_idx, device, beam_width, max_inference_len), mimetype="text/event-stream")