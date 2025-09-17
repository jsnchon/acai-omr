import torch
from acai_omr.models.models import ViTOMR
from acai_omr.train.omr_teacher_force_train import set_up_omr_teacher_force_train
from acai_omr.config import INFERENCE_VITOMR_PATH
from acai_omr.utils.utils import stringify_lmx_seq
from acai_omr.__init__ import InferenceEvent
from torch.amp import autocast
from PIL import Image
import logging
import subprocess
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# lmx_seq is the sequence of decoded lmx tokens (excluding <bos> and <eos>), lmx_seq_path and xml_file_path are where the lmx and xml files should be saved to
def delinearize(lmx_seq: str, lmx_seq_path: str, xml_file_path: str):
    logger.info(f"Delinearizing lmx sequence:\n{lmx_seq}")
    logger.info(f"Writing sequence to {lmx_seq_path}")
    lmx_seq_path = Path(lmx_seq_path)
    lmx_seq_path.write_text(lmx_seq)

    logger.info(f"Delinearizing and saving xml file to {xml_file_path}")
    try:
        delinearize_result = subprocess.run(
            ["poetry", "run", "python", "-m", "olimpic_app.linearization", "delinearize", lmx_seq_path, xml_file_path], 
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
    return img_file_path

def streamed_inference(
    img: torch.Tensor, 
    vitomr: ViTOMR, 
    device, 
    max_inference_len=1536,
    flush_interval=25):

    vitomr.eval()
    with torch.no_grad():
        logger.debug("Encoding image into latent")
        yield {"type": InferenceEvent.ENCODING_START.value, "payload": None}
        # don't autocast this because encoder uses some non autocastable path in eval() mode
        img_latent, latent_attention_mask = vitomr.encoder(img)
        with autocast(device_type=device, dtype=torch.bfloat16):
            img_latent = vitomr.transition_head(img_latent)
            yield {"type": InferenceEvent.ENCODING_FINISH.value, "payload": None}
            logger.debug("Starting decoder generation")
            for event in vitomr.streamed_cached_greedy_generate(img_latent, latent_attention_mask, max_len=max_inference_len, flush_interval=flush_interval):
                yield event

# non-streamed back-end only inference
def inference(
    vitomr: ViTOMR, 
    img: torch.Tensor, 
    device, 
    max_inference_len=1536):

    vitomr.eval()
    with torch.no_grad():
        img_latent, latent_attention_mask = vitomr.encoder(img)
        with autocast(device_type=device, dtype=torch.bfloat16):
            img_latent = vitomr.transition_head(img_latent)
            seqs, log_probs, seq_mask = vitomr.cached_greedy_generate(img_latent, latent_attention_mask, max_len=max_inference_len)

    return seqs, log_probs, seq_mask

if __name__ == "__main__":
    INFERENCE_IMAGE_PATH = "inference_test.png"

    logging.basicConfig(level=logging.DEBUG)

    MAX_BATCH_SIZE = 32
    CACHE_DTYPE = torch.bfloat16
    logger.info(f"Loading state dict from {INFERENCE_VITOMR_PATH}")

    vitomr, base_img_transform, _, device = set_up_omr_teacher_force_train()
    vitomr.decoder = vitomr.decoder.to_cached_version(MAX_BATCH_SIZE, CACHE_DTYPE)
    if device == "cpu":
        vitomr_state_dict = torch.load(INFERENCE_VITOMR_PATH, map_location=torch.device("cpu"))
    else:
        vitomr_state_dict = torch.load(INFERENCE_VITOMR_PATH)

    vitomr.load_state_dict(vitomr_state_dict)
    
    # note to future self: make sure to convert any images to grayscale/transform using patch transform
    img = Image.open(INFERENCE_IMAGE_PATH).convert("L")
    img = base_img_transform(img).to(device)

    logger.info("Starting inference")
    lmx_seqs, log_probs, seq_mask = inference(vitomr, img, device)
    for i, lmx_seq in enumerate(lmx_seqs):
        mask = seq_mask[i]
        lmx_seq = lmx_seq[mask]
        lmx_seq = stringify_lmx_seq(lmx_seq, vitomr.decoder.idxs_to_tokens)
        average_log_prob = log_probs[i][mask].sum().item() / mask.sum().item()
        logger.info(f"Decoded inference result: {lmx_seq}\nAverage log prob per token: {average_log_prob}")

        response = delinearize(lmx_seq, "inference_result.lmx", "inference_result.musicxml")
        if response["ok"]:
            convert_back_to_img(response["xml_file_path"], "inference_result.png")
        else:
            logger.info("Delinearization failed, skipping conversion into image. You should check the .lmx file")
