import torch
from models import ViTOMR
from omr_train import vitomr, device, base_img_transform
from config import LMX_BOS_TOKEN, LMX_EOS_TOKEN
from torch.amp import autocast
from PIL import Image
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# split seqs and corresponding log_probs into individual sequences, calculate normalized beam scores for each sequence,
# and return a list of (seq, score) tuples
def split_and_score_seqs(seqs, log_probs):
    result = []

    split_seqs = torch.split(seqs, 1, dim=0)
    split_seqs_log_probs = torch.split(log_probs, 1, dim=0)
    for seq, log_prob in zip(split_seqs, split_seqs_log_probs):
        seq = seq.squeeze(0)
        seq_len = seq.shape[0]
        score = log_prob / seq_len
        result.append((seq, score))
    
    return result

# autoregressive beam search inference using average log probability as length normalization
def beam_search(
    vitomr: ViTOMR, 
    img: torch.Tensor, 
    bos_token_idx: int, 
    eos_token_idx: int, 
    device, 
    beam_width=3, 
    max_inference_len=1536): # <bos> token is counted as contributing one step to this limit

    vitomr.eval()
    logger.debug("Encoding image latent")
    # encode image latent once to avoid redundant encoder computations
    with autocast(device_type=device, dtype=torch.bfloat16):
        img_latent = vitomr.encoder.generate(img)
        img_latent = vitomr.transition_head(img_latent)

    completed_beams = [] # tuples of (sequence, score) where score is length-normalized log prob

    # start with just one sequence of <bos> and expand into beam_width beams in the first iteration
    active_seqs = torch.full((1, 1), bos_token_idx, dtype=torch.long, device=device)
    log_probs = torch.zeros(1, device=device)

    for _ in range(max_inference_len - 1):
        logger.debug(f"Active seqs at step start: {active_seqs}")
        with autocast(device_type=device, dtype=torch.bfloat16):
            next_token_distr = vitomr.generate(img_latent, active_seqs) # (num_seqs, vocab_size), num_seqs is 1 in first iteration and beam_width otherwise

        expanded_log_probs = log_probs.unsqueeze(1) + next_token_distr # cumulative log probs for all b * v possible extensions

        top_log_probs, top_indices = expanded_log_probs.view(-1).topk(beam_width)
        logger.debug(f"Top k results. Probs: {top_log_probs}, flattened indices: {top_indices}")
        # convert indices from being for flattened tensor to corresponding un-flattened values
        vocab_size = expanded_log_probs.shape[1]
        beam_indices = top_indices // vocab_size # each beam has vocab_size elements
        token_indices = top_indices % vocab_size

        active_seqs = torch.cat([active_seqs.index_select(dim=0, index=beam_indices), token_indices.unsqueeze(1)], dim=1)
        log_probs = top_log_probs
        logger.debug(f"Extended sequences: {active_seqs}, new log probs: {log_probs}")

        if torch.any(finished_seqs_filter := (active_seqs[:, -1] == eos_token_idx)):
            finished_seqs = active_seqs[finished_seqs_filter]
            finished_seqs_log_probs = log_probs[finished_seqs_filter]
            logger.debug(f"Newly finished seqs this step: {finished_seqs} with log probs: {finished_seqs_log_probs}")

            completed_beams += split_and_score_seqs(finished_seqs, finished_seqs_log_probs)
            logger.debug(f"completed_beams after adding newly finished seqs: {completed_beams}")

        if len(completed_beams) >= beam_width:
            break
    
    if len(completed_beams) < beam_width: # didn't finish early and instead ran out of inference steps
        completed_beams += split_and_score_seqs(active_seqs, log_probs) # add whatever incomplete beams we have as candidates

    logger.debug(f"completed_beams before sorting: {completed_beams}")
    completed_beams.sort(key=lambda x: x[1], reverse=True) # sort sequences by score
    return completed_beams[0][0]

def infer_from_image(image_path, vitomr_weights_path):
    if device == "cpu":
        vitomr_state_dict = torch.load(vitomr_weights_path, map_location=torch.device("cpu"))
    else:
        vitomr_state_dict = torch.load(vitomr_weights_path)

    vitomr.load_state_dict(vitomr_state_dict)
    
#    # DEBUG purposes
#    from test_vitomr import debug_vitomr
#    vitomr = debug_vitomr
#    # end debug

    bos_token_idx = vitomr.decoder.tokens_to_idxs[LMX_BOS_TOKEN]
    eos_token_idx = vitomr.decoder.tokens_to_idxs[LMX_EOS_TOKEN]

    image = Image.open(image_path).convert("L")
    image = base_img_transform(image)

    # make sure to transform any images using patch transform
    lmx_seq = beam_search(vitomr, image, bos_token_idx, eos_token_idx, device)
    lmx_seq = [vitomr.decoder.idxs_to_tokens[idx.item()] for idx in lmx_seq]
    logger.info(f"INFERENCE RESULT\n{'-' * 20}\n{" ".join(lmx_seq)}")

if __name__ == "__main__":
    VITOMR_WEIGHTS_PATH = "omr_train/vitomr.pth"
    IMAGE_PATH = "inference_test.png"
    infer_from_image(IMAGE_PATH, VITOMR_WEIGHTS_PATH)