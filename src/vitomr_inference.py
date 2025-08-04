import torch
from models import ViTOMR
import argparse
# TODO: implement and test (can use debug trained vitomr for testing)

# autoregressive beam search inference using average log probability as length normalization
def beam_search(
    vitomr: ViTOMR, 
    input_img: torch.Tensor, 
    bos_token_idx: int, 
    eos_token_idx: int, 
    device, 
    beam_width=5, 
    max_inference_len=512):

    # start with just sos tokens

    # expand search


    # at end sum up sequence log probabilities, divide by the sequence lengths (remember can vary)
    pass