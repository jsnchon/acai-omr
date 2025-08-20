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
        elif event["type"] == InferenceEvent.INFERENCE_FINISH.value:
            # process payload for final yielded inference event before streaming. Including "done": True will signal the stream should be closed
            decoded_lmx_seq = [vitomr.decoder.idxs_to_tokens[idx.item()] for idx in inference_event["payload"]["lmx_seq"].squeeze(0)]
            decoded_lmx_seq = " ".join(decoded_lmx_seq)
            final_payload = {"lmx_seq": decoded_lmx_seq, "score": inference_event["payload"]["score"].item()}
            event["payload"] = final_payload

        yield f"data: {json.dumps(event)}\n\n" # two \n to add a blank line which SSE uses as a signal for the event end

@main.route("/inference/stream")
def stream_inference():
    vitomr, base_img_transform, _, device = set_up_omr_train()

    weights_path = request.args.get("weights_path", DEFAULT_VITOMR_PATH)
    beam_width = int(request.args.get("beam_width", 3))
    max_inference_len = int(request.args.get("max_inference_len", 1536))

    # DEBUG
#    from acai_omr.models.models import FineTuneOMREncoder, OMRDecoder, ViTOMR, MAE
#    from acai_omr.train.pre_train import PE_MAX_HEIGHT, PE_MAX_WIDTH
#    from acai_omr.train.omr_train import PE_MAX_HEIGHT, PE_MAX_WIDTH, MAX_LMX_SEQ_LEN, LMX_VOCAB_PATH
#    from acai_omr.config import DEBUG_PRETRAINED_MAE_PATH
#    DEBUG_KWARGS = {"num_layers": 2, "num_heads": 1, "mlp_dim": 1}
#    DEBUG_PATCH_SIZE = 16
#
#    debug_encoder = FineTuneOMREncoder(DEBUG_PATCH_SIZE, PE_MAX_HEIGHT, PE_MAX_WIDTH, 1, hidden_dim=10, **DEBUG_KWARGS)
#    debug_decoder = OMRDecoder(MAX_LMX_SEQ_LEN, LMX_VOCAB_PATH, hidden_dim=10, **DEBUG_KWARGS)
#    debug_mae_state_dict = torch.load(DEBUG_PRETRAINED_MAE_PATH)
#    debug_vitomr = ViTOMR(debug_encoder, debug_mae_state_dict, debug_decoder)
#    vitomr = debug_vitomr
#    weights_path = "debug_omr_train/debug_vitomr.pth"
    # end debug
    
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
    logger.info("Starting inference and streaming from endpoint")
    logger.info(f"Running beam search with beam width {beam_width} and max inference length {max_inference_len}")

    return Response(stream_beam_search_wrapper(vitomr, image, bos_token_idx, eos_token_idx, device, beam_width, max_inference_len), mimetype="text/event-stream")