import torch
from acai_omr.models.models import ViTOMR
from acai_omr.train.omr_train import set_up_omr_train
from acai_omr.config import LMX_BOS_TOKEN, LMX_EOS_TOKEN, InferenceEvent, INFERENCE_VITOMR_PATH
from torch.amp import autocast
from PIL import Image
import logging
import subprocess
from pathlib import Path
import os

INFERENCE_IMAGE_PATH = "inference_test.png"

logger = logging.getLogger(__name__)

# split seqs and corresponding log_probs into individual sequences, calculate normalized beam scores for each sequence,
# and return a list of (seq, score) tuples
def split_and_score_seqs(seqs, log_probs):
    result = []

    split_seqs = torch.split(seqs, 1, dim=0)
    split_seqs_log_probs = torch.split(log_probs, 1, dim=0)
    for seq, log_prob in zip(split_seqs, split_seqs_log_probs):
        seq = seq
        seq_len = seq.shape[1]
        score = log_prob / seq_len
        result.append((seq, score))
    
    return result

"""
Autoregressive beam search inference using average log probability as length normalization. This is a generator of events (dicts of event_type: payload_dict)
to allow for the possibility of streaming each stage of the inference pipeline to the ui. If this module is ran as a stand-alone,
it'll only be concerned with the final generated result
"""
def beam_search(
    vitomr: ViTOMR, 
    img: torch.Tensor, 
    bos_token_idx: int, 
    eos_token_idx: int, 
    device, 
    beam_width: int, 
    max_inference_len: int): # <bos> token is counted as contributing one step to this limit

    vitomr.eval()
    # encode image latent once to avoid redundant encoder computations
    logger.debug("Encoding image latent")
    yield {"type": InferenceEvent.ENCODING_START.value, "payload": {"message": "Encoding image"}}
    with torch.no_grad():
        with autocast(device_type=device, dtype=torch.bfloat16):
            img_latent = vitomr.encoder.generate(img)
            img_latent = vitomr.transition_head(img_latent)
    yield {"type": InferenceEvent.ENCODING_FINISH.value, "payload": {"message": "Finished encoding image"}}

    completed_beams = [] # tuples of (sequence, score) where score is length-normalized log prob

    # start with just one sequence of <bos> and expand into beam_width beams in the first iteration
    active_seqs = torch.full((1, 1), bos_token_idx, dtype=torch.long, device=device)
    log_probs = torch.zeros(1, device=device)

    logger.debug("Starting inference loop")
    for _ in range(max_inference_len - 1):
        logger.debug(f"Active beams:\n{active_seqs} with log probs {log_probs}")
        with torch.no_grad():
            with autocast(device_type=device, dtype=torch.bfloat16):
                next_token_distr = vitomr.generate(img_latent, active_seqs) # (num_seqs, vocab_size), num_seqs is 1 in first iteration and beam_width otherwise

        expanded_log_probs = log_probs.unsqueeze(1) + next_token_distr # cumulative log probs for all b * v possible extensions
        logger.debug(f"Shape of all candidate extensions: {expanded_log_probs.shape}")

        top_log_probs, top_indices = expanded_log_probs.view(-1).topk(beam_width)
        # convert indices from being for flattened tensor to corresponding un-flattened values
        vocab_size = expanded_log_probs.shape[1]
        beam_indices = top_indices // vocab_size # each beam has vocab_size elements
        token_indices = top_indices % vocab_size
        logger.debug(f"Top k results. Probs: {top_log_probs}, flattened indices: {top_indices}, beam indices: {beam_indices}, vocab token indices: {token_indices}")

        active_seqs = torch.cat([active_seqs.index_select(dim=0, index=beam_indices), token_indices.unsqueeze(1)], dim=1)
        log_probs = top_log_probs

        if torch.any(finished_seqs_filter := (active_seqs[:, -1] == eos_token_idx)):
            finished_seqs = active_seqs[finished_seqs_filter]
            finished_seqs_log_probs = log_probs[finished_seqs_filter]
            logger.debug(f"Newly finished beams: {finished_seqs} with log probs: {finished_seqs_log_probs}")

            completed_beams += split_and_score_seqs(finished_seqs, finished_seqs_log_probs)
            logger.debug(f"completed_beams after appending newly finished beams: {completed_beams}")

        yield {"type": InferenceEvent.STEP.value, "payload": {"beams": active_seqs, "log_probs": log_probs}}

        if len(completed_beams) >= beam_width:
            break
    
    logger.debug(f"Active beams at inference end:\n{active_seqs} with log probs {log_probs}")
    if len(completed_beams) < beam_width: # didn't finish early and instead ran out of inference steps
        completed_beams += split_and_score_seqs(active_seqs, log_probs) # add whatever incomplete beams we have as candidates

    logger.debug(f"completed_beams before sort: {completed_beams}")
    completed_beams.sort(key=lambda x: x[1], reverse=True) # sort sequences by score
    best_seq, score = completed_beams[0] # best_seq is (1 x best_seq_len)
    logger.info(f"INFERENCE RESULT\n{'-' * 20}\n{best_seq}\nScore: {score}")
    yield {"type": InferenceEvent.INFERENCE_FINISH.value, "payload": {"lmx_seq": best_seq, "score": score}}

# convert list of tokens into a single lmx token string. Remove <bos> and <eos> tokens since Olimpic wasn't designed to handle those
def stringify_lmx_seq(lmx_seq: list[str]):
    if lmx_seq[-1] == LMX_EOS_TOKEN:
        lmx_seq.pop(-1)
    lmx_seq = lmx_seq[1: ]
    lmx_seq = " ".join(lmx_seq)
    return lmx_seq

# lmx_seq is the sequence of decoded lmx tokens (excluding <bos> and <eos>), lmx_seq_path and xml_file_path are where the lmx and xml files should be saved to
def delinearize(lmx_seq: str, lmx_seq_path: str, xml_file_path: str):
    logger.info(f"Delinearizing lmx sequence:\n{lmx_seq}")
    logger.info(f"Writing sequence to {lmx_seq_path}")
    lmx_seq_path = Path(lmx_seq_path)
    lmx_seq_path.write_text(lmx_seq)

    logger.info(f"Delinearizing and saving xml file to {xml_file_path}")
    try:
        delinearize_result = subprocess.run(
            ["python3", "-m", "olimpic_app.linearization", "delinearize", lmx_seq_path, xml_file_path], 
            capture_output=True, text=True
        )

        delinearize_result.check_returncode()
        delinearize_problems = delinearize_result.stderr.splitlines()
        if delinearize_problems:
            logger.warning(f"Caught problems with delinearization: {delinearize_problems}")
        return {"ok": True, "xml_file_path": xml_file_path, "delinearize_problems": delinearize_problems}

    except subprocess.CalledProcessError as e:
        logger.warning(f"Olimpic delinearization catastrophically failed: {e.stderr}")
        return {"ok": False, "error": e.stderr}

# show how accurate translation was
def convert_back_to_img(xml_file_path: str, img_file_path: str):
    logger.info(f"Converting {xml_file_path} to image and saving image to {img_file_path}")
    subprocess.run(["musescore3", "-o", "mscore_out.png", xml_file_path])
    # musescore CLI renders images with a transparent background, so treat its output as an intermediate file and convert it
    # to a png with a white background using imagemagick (also note CLI adds numbers to the filename)
    subprocess.run(["convert", "mscore_out-1.png", "-background", "white", "-alpha", "remove", "-alpha", "off", img_file_path])
    os.remove("mscore_out-1.png")
    logger.info("Final image saved!")
    return {"img_file_path": img_file_path}

# non-streamed local (ie back-end only) inference. We just consume the generator until we get the final result
def inference(
    vitomr: ViTOMR, 
    img: torch.Tensor, 
    bos_token_idx: int, 
    eos_token_idx: int, 
    device, 
    beam_width=3, 
    max_inference_len=1536):

    inference_event = None
    for event in beam_search(vitomr, img, bos_token_idx, eos_token_idx, device, beam_width, max_inference_len):
        inference_event = event 

    # last yielded value by beam_search is the best sequence and its normalized score
    return inference_event["payload"]["lmx_seq"], inference_event["payload"]["score"]

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    logger.info(f"Loading state dict from {INFERENCE_VITOMR_PATH}")

    vitomr, base_img_transform, _, device = set_up_omr_train()
    if device == "cpu":
        vitomr_state_dict = torch.load(INFERENCE_VITOMR_PATH, map_location=torch.device("cpu"))
    else:
        vitomr_state_dict = torch.load(INFERENCE_VITOMR_PATH)

    vitomr.load_state_dict(vitomr_state_dict)
    
    bos_token_idx = vitomr.decoder.tokens_to_idxs[LMX_BOS_TOKEN]
    eos_token_idx = vitomr.decoder.tokens_to_idxs[LMX_EOS_TOKEN]

    # note to future self: make sure to convert any images to grayscale/transform using patch transform
    image = Image.open(INFERENCE_IMAGE_PATH).convert("L")
    image = base_img_transform(image)

    # these are low so I can debug on my weak laptop
    beam_width = 2
    max_inference_len = 40

    logger.info("Starting inference")
    lmx_seq, score = inference(vitomr, image, bos_token_idx, eos_token_idx, device, beam_width=beam_width, max_inference_len=max_inference_len)
    lmx_seq = [vitomr.decoder.idxs_to_tokens[idx.item()] for idx in lmx_seq.squeeze(0)]
    lmx_seq = stringify_lmx_seq(lmx_seq)
    logger.info(f"Decoded inference result: {lmx_seq}\nSequence score: {score}")

    response = delinearize(lmx_seq, "inference_result.lmx", "inference_result.musicxml")
    if response["ok"]:
        convert_back_to_img(response["xml_file_path"], "inference_result.png")
    else:
        logger.info("Delinearization failed, skipping conversion into image. You should check the .lmx file")