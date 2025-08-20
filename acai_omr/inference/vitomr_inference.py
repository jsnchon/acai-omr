import torch
from acai_omr.models.models import ViTOMR
from acai_omr.train.omr_train import set_up_omr_train
from acai_omr.config import LMX_BOS_TOKEN, LMX_EOS_TOKEN, InferenceEvent
from torch.amp import autocast
from PIL import Image
import logging

VITOMR_WEIGHTS_PATH = "omr_train/vitomr.pth"
IMAGE_PATH = "inference_test.png"

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

    logger.info(f"Loading state dict from {VITOMR_WEIGHTS_PATH}")

    vitomr, base_img_transform, _, device = set_up_omr_train()
    if device == "cpu":
        vitomr_state_dict = torch.load(VITOMR_WEIGHTS_PATH, map_location=torch.device("cpu"))
    else:
        vitomr_state_dict = torch.load(VITOMR_WEIGHTS_PATH)

    vitomr.load_state_dict(vitomr_state_dict)
    
    bos_token_idx = vitomr.decoder.tokens_to_idxs[LMX_BOS_TOKEN]
    eos_token_idx = vitomr.decoder.tokens_to_idxs[LMX_EOS_TOKEN]

    image = Image.open(IMAGE_PATH).convert("L")
    image = base_img_transform(image)

    # make sure to transform any images using patch transform
    logger.info("Starting inference")
    lmx_seq, score = inference(vitomr, image, bos_token_idx, eos_token_idx, device)
    lmx_seq = [vitomr.decoder.idxs_to_tokens[idx.item()] for idx in lmx_seq.squeeze(0)]
    logger.info(f"Decoded inference result: {' '.join(lmx_seq)}\nSequence score: {score}")